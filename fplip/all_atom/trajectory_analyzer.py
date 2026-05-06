"""
Trajectory Analyzer Module

Provides fast trajectory analysis using MDAnalysis for coordinate loading
and OpenBabel for interaction detection.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from MDAnalysis import Universe
from scipy.spatial import cKDTree

from fplip.all_atom.interaction_detector import UnifiedInteractionDetector
from fplip.all_atom.molecule_complex import MoleculeComplex
from fplip.basic import config


class TrajectoryAnalyzer:
    """Analyzer for MD trajectories using MDAnalysis + OpenBabel.

    This class provides fast trajectory analysis by:
    1. Using MDAnalysis to load trajectory coordinates (efficient)
    2. Using OpenBabel for structure loading and property initialization (once)
    3. Using KD-tree for fast coordinate alignment (once)
    4. Rapidly updating coordinates for each frame (fast)
    """

    def __init__(
        self,
        tpr_file: str,
        xtc_file: str,
        gro_file: Optional[str] = None,
        pdb_str: Optional[str] = None,
        tolerance: float = 1e-4
    ):
        """Initialize TrajectoryAnalyzer.

        Parameters
        ----------
        tpr_file : str
            Path to GROMACS TPR file (provides topology)
        xtc_file : str
            Path to GROMACS XTC file (provides coordinates)
        gro_file : str, optional
            Path to GROMACS GRO file (provides residue IDs, preferred)
        pdb_str : str, optional
            PDB string for OpenBabel loading (if not using gro_file)
        tolerance : float
            KD-tree tolerance for coordinate matching (default 1e-4)
        """
        self.tpr_file = tpr_file
        self.xtc_file = xtc_file
        self.gro_file = gro_file
        self.tolerance = tolerance

        self.u: Optional[Universe] = None
        self.mol: Optional[MoleculeComplex] = None
        self.detector: Optional[UnifiedInteractionDetector] = None
        self.kdtree: Optional[cKDTree] = None
        self._aligned = False
        self._detector_precomputed = False

    def load_universe(self):
        """Load MDA Universe from GROMACS files."""
        if self.gro_file:
            self.u = Universe(self.tpr_file, self.xtc_file)
            self.u2 = Universe(self.gro_file)
            self.u.atoms.residues.resids = self.u2.atoms.residues.resids  # pyright: ignore[reportAttributeAccessIssue]
        else:
            self.u = Universe(self.tpr_file, self.xtc_file)
        return self.u

    def load_molecule(self, pdb_str: str, as_string: bool = True):
        """Load OpenBabel molecule from PDB string.

        Parameters
        ----------
        pdb_str : str
            PDB format string
        as_string : bool
            Whether pdb_str is a string (True) or file path (False)
        """
        config.NOHYDRO = True
        self.mol = MoleculeComplex()
        self.mol.load_pdb(pdb_str, as_string=as_string)
        return self.mol

    def align_with_mda(self, frame: int = 0) -> bool:
        """Align OpenBabel molecule with MDA universe using KD-tree.

        This method builds a KD-tree from MDA coordinates and matches
        OpenBabel atoms to MDA atoms based on coordinate proximity.

        Parameters
        ----------
        frame : int
            Frame index to use for alignment (default 0)

        Returns
        -------
        bool
            True if all atoms were aligned, False otherwise
        """
        if self.u is None:
            raise RuntimeError("Universe not loaded. Call load_universe() first.")
        if self.mol is None:
            raise RuntimeError("Molecule not loaded. Call load_molecule() first.")

        self.u.trajectory[frame]
        mda_coords = self.u.atoms.positions.copy()

        self.kdtree = cKDTree(mda_coords)

        matched = 0
        for atom in self.mol.atom_container:
            dist, idx = self.kdtree.query(atom.coords)
            if dist < self.tolerance:
                atom.mda_idx = idx
                matched += 1
            else:
                atom.mda_idx = None

        self._aligned = (matched == len(self.mol.atom_container))
        return self._aligned

    def setup_detector(self, atom_props=None):
        """Setup interaction detector after alignment.

        Parameters
        ----------
        atom_props : AtomProperties, optional
            AtomProperties instance (created if not provided)
        """
        if self.mol is None:
            raise RuntimeError("Molecule not loaded. Call load_molecule() first.")

        if atom_props is None:
            from fplip.all_atom.atom_properties import AtomProperties
            atom_props = AtomProperties(self.mol.atom_container)

        self.detector = UnifiedInteractionDetector(
            self.mol.atom_container,
            atom_props,
            self.mol.residues
        )
        return self.detector

    def precompute_detector_once(self):
        """One-time initialization of detector.

        Calls the property aggregation and precomputation methods that only
        need to run once. After this, use detect_frame_fast() for subsequent
        frames.
        """
        if self.detector is None:
            raise RuntimeError("Detector not setup. Call setup_detector() first.")

        if self._detector_precomputed:
            return

        self.detector._aggregate_properties_to_residues()
        self.detector._precompute_residue_charge_groups()
        self.detector._precompute_cached_data()

        self._detector_precomputed = True

    def detect_frame_fast(self, frame_idx: int, verbose: bool = False) -> Dict[str, List]:
        """Detect interactions for a frame using cached setup.

        Assumes setup_detector_once() has been called. This method skips
        the one-time setup methods and only does:
        - Coordinate update
        - Per-residue detection
        - Post-processing (dedup, refine, water bridges)

        Parameters
        ----------
        frame_idx : int
            Frame index to process
        verbose : bool
            Whether to show progress bars

        Returns
        -------
        Dict[str, List]
            Dictionary of detected interactions
        """
        if self.detector is None:
            raise RuntimeError("Detector not setup. Call setup_detector() first.")

        if not self._detector_precomputed:
            self.precompute_detector_once()

        self.update_frame(frame_idx)

        self.detector.interactions = {
            'hydrophobic': [],
            'hbond': [],
            'hbond_possible': [],
            'hbond_heavy_atom': [],
            'saltbridge': [],
            'pistacking': [],
            'pication': [],
            'halogen': [],
            'metal': [],
            'water_bridge': [],
            'water_bridge_possible': [],
        }

        for residue in self.detector.residues:
            self.detector._detect_for_residue(residue)

        self.detector._remove_duplicates()
        self.detector._refine_hbonds()
        self.detector._detect_water_bridges()
        self.detector._detect_water_bridges_plip_style()

        return self.detector.interactions

    def update_frame(self, frame_idx: int, filter_waters: bool = True):
        """Update coordinates for a specific frame.

        Parameters
        ----------
        frame_idx : int
            Frame index to load
        """
        if self.detector is None:
            raise RuntimeError("Detector not setup. Call setup_detector() first.")

        self.u.trajectory[frame_idx]
        mda_coords = self.u.atoms.positions
        self.detector.update_coords(mda_coords)
        if filter_waters:
            print(self.filter_distant_waters())

    def filter_distant_waters(self, distance_threshold: float = 5.0) -> Dict[str, int]:
        """Filter out water molecules distant from other molecules.

        This method should be called after update_frame() has been executed.
        It marks water residues with is_skip=True if their oxygen atom is
        farther than distance_threshold from any non-water atom.

        This method does NOT update coordinates - it reuses the coordinates
        that were already updated by update_frame().

        Parameters
        ----------
        distance_threshold : float
            Maximum distance (Angstroms) between water oxygen and
            nearest non-water atom to keep the water (default 5.0)

        Returns
        -------
        Dict[str, int]
            Statistics: {"total": total_water_count, "filtered": filtered_count, "kept": kept_count}
        """
        if self.detector is None:
            raise RuntimeError("Detector not setup. Call setup_detector() first.")

        water_residues = [r for r in self.detector.residues if r.is_water]

        if not water_residues:
            return {"total": 0, "filtered": 0, "kept": 0}

        non_water_coords = []
        for atom in self.mol.atom_container:
            if not atom.residue_obj.is_water:
                non_water_coords.append(atom.coords)

        if not non_water_coords:
            for water_res in water_residues:
                water_res.is_skip = True
            return {"total": len(water_residues), "filtered": len(water_residues), "kept": 0}

        kdtree = cKDTree(np.array(non_water_coords))

        filtered = 0
        kept = 0

        for water_res in water_residues:
            oxygen_coords = None
            for atom in water_res.atoms:
                if atom.atomic_num == 8:
                    oxygen_coords = atom.coords
                    break

            if oxygen_coords is None:
                water_res.is_skip = True
                filtered += 1
                continue

            dist, _ = kdtree.query(oxygen_coords)

            if dist > distance_threshold:
                water_res.is_skip = True
                filtered += 1
            else:
                water_res.is_skip = False
                kept += 1

        return {"total": len(water_residues), "filtered": filtered, "kept": kept}

    def detect_all(self, frame_idx: int, verbose: bool = False) -> Dict[str, List]:
        """Update coordinates and detect interactions for a frame.

        Parameters
        ----------
        frame_idx : int
            Frame index to process
        verbose : bool
            Whether to show progress bars

        Returns
        -------
        Dict[str, List]
            Dictionary of detected interactions
        """
        self.update_frame(frame_idx)
        return self.detector.detect_all(verbose=verbose)

    def iterate_frames(
        self,
        start: int = 0,
        stop: Optional[int] = None,
        step: int = 1,
        progress_callback: Optional[Callable[[int, Dict], None]] = None
    ):
        """Iterate over frames and detect interactions.

        Parameters
        ----------
        start : int
            Starting frame index
        stop : int, optional
            Stopping frame index (exclusive)
        step : int
            Frame step
        progress_callback : Callable, optional
            Callback function called after each frame with (frame_idx, interactions)

        Yields
        ------
        Tuple[int, Dict[str, List]]
            (frame_index, interactions_dict) for each frame
        """
        if self.detector is None:
            raise RuntimeError("Detector not setup. Call setup_detector() first.")

        frame_indices = list(range(start, stop or len(self.u.trajectory), step))
        self.precompute_detector_once()
        for frame_idx in frame_indices:
            interactions = self.detect_frame_fast(frame_idx, verbose=False)
            if progress_callback:
                progress_callback(frame_idx, interactions)
            yield frame_idx, interactions

    def get_alignment_stats(self) -> Dict[str, int]:
        """Get alignment statistics.

        Returns
        -------
        Dict[str, int]
            Statistics about alignment: total_atoms, matched_atoms, unmatched_atoms
        """
        if self.mol is None:
            return {"total_atoms": 0, "matched_atoms": 0, "unmatched_atoms": 0}

        total = len(self.mol.atom_container)
        matched = sum(1 for atom in self.mol.atom_container if atom.mda_idx is not None)

        return {
            "total_atoms": total,
            "matched_atoms": matched,
            "unmatched_atoms": total - matched
        }


if __name__ == "__main__":
    from lazydock.gmx.mda.convert import PDBConverter
    test_data_dir = Path(__file__).parent.parent.parent / 'test_data/pull'
    tpr = str(test_data_dir / "pull.tpr")
    xtc = str(test_data_dir / "pull_center.xtc")
    gro = str(test_data_dir / "pull.gro")
    analyzer = TrajectoryAnalyzer(tpr, xtc, gro, tolerance=1e-4)
    analyzer.load_universe()
    analyzer.u.trajectory[0]
    converter = PDBConverter(analyzer.u.atoms, reindex=False)
    pdb_str = converter.fast_convert()
    analyzer.load_molecule(pdb_str, as_string=True)
    analyzer.align_with_mda(frame=0)
    analyzer.setup_detector()
    analyzer.precompute_detector_once()

    analyzer.update_frame(0)
    interactions = analyzer.detector.detect_all(verbose=True)
    pass