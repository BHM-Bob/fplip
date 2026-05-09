"""
Trajectory Analyzer Module

Provides fast trajectory analysis using MDAnalysis for coordinate loading
and OpenBabel for interaction detection.
"""
import sys
from pathlib import Path
sys.path.insert(0, str((Path(__file__).parent / '../..').resolve()))

from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from MDAnalysis import Universe
from scipy.spatial import cKDTree

from fplip.all_atom.interaction_detector import UnifiedInteractionDetector
from fplip.all_atom.molecule_complex import MoleculeComplex
from fplip.all_atom.trajectory_analyzer import \
    TrajectoryAnalyzer as _TrajectoryAnalyzer
from fplip.all_atom_cuda.backend import ComputeBackend
from fplip.all_atom_cuda.cuda_detector import CudaInteractionDetector
from fplip.all_atom_cuda.numpy_backend import NumPyBackend
from fplip.basic import config
from fplip.basic.logger import logger


class TrajectoryAnalyzer(_TrajectoryAnalyzer):
    """Analyzer for MD trajectories using MDAnalysis + OpenBabel.

    This class provides fast trajectory analysis by:
    1. Using MDAnalysis to load trajectory coordinates (efficient)
    2. Using OpenBabel for structure loading and property initialization (once)
    3. Using KD-tree for fast coordinate alignment (once)
    4. Rapidly updating coordinates for each frame (fast)
    """
    detector: CudaInteractionDetector
    def setup_detector(self, atom_props=None, backend: Optional[ComputeBackend] = None):
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
            
        if backend is None:
            backend = NumPyBackend()

        self.detector = CudaInteractionDetector(
            self.mol.atom_container,
            atom_props,
            self.mol.residues,
            backend
        )
        # pre-collect data for distant water filter
        self.water_residues = [r for r in self.detector.residues if r.is_water]
        ## use index, because indexing in GPU is faster than CPU
        self.non_water_atoms, self.water_o_atoms = [], []
        for i in self.detector.atom_container.sorted_indices:
            if self.detector.atom_container.atoms[i].residue_obj.is_water:
                if self.detector.atom_container.atoms[i].atomic_num == 8:
                    self.water_o_atoms.append(self.detector.atom_container.atoms[i])
            else:
                self.non_water_atoms.append(self.detector.atom_container.atoms[i])
        self.non_water_atom_idxs = self.detector.atom_container.get_atom_coords_idxs_from_atoms(self.non_water_atoms)
        self.water_o_atom_idxs = self.detector.atom_container.get_atom_coords_idxs_from_atoms(self.water_o_atoms)
        return self.detector
    
    def detect_all(self, detect_water_bridges_plip_style: bool = False) -> Dict[str, List]:
        """Update coordinates and detect interactions for a frame.

        Parameters
        ----------
        detect_water_bridges_plip_style : bool, optional
            Whether to detect water bridges in the style of PLIP

        Returns
        -------
        Dict[str, List]
            Dictionary of detected interactions
        """
        # _precompute_cached_data include coords precompute, so no need to call it again
        self.detector._precompute_cached_data()

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

        # Detect interactions for each interaction-type
        ## Hydrophobic interactions
        self.detector._detect_hydrophobic()
        ## Hydrogen bonds
        if self.detector._has_explicit_h:
            self.detector._detect_hbonds_case1_vectorized()
        elif config.ALLOW_HEAVY_ATOM_HBOND:
            # Use distance-only detection for heavy atom H-bonds (optional, less reliable)
            logger.info("Using heavy atom H-bond detection (distance-only). "
                       "Note: Results may be less reliable without explicit hydrogens.")
            self.detector._detect_hbonds_without_h()
        ## Salt bridges
        self.detector._detect_saltbridges()
        ## Pistack interactions
        self.detector._detect_pistacking()
        ## Pication interactions
        self.detector._detect_pication()
        ## Halogen bonds
        self.detector._detect_halogen()
        ## Metal interactions
        self.detector._detect_metal()

        self.detector._remove_duplicates()
        self.detector._refine_hbonds()
        self.detector._detect_water_bridges()
        if detect_water_bridges_plip_style:
            self.detector._all_hba_coords = self.detector.backend.to_numpy(self.detector._all_hba_coords) # type: ignore
            self.detector._all_hbd_don_coords = self.detector.backend.to_numpy(self.detector._all_hbd_don_coords) # type: ignore
            self.detector._all_hbd_h_coords = self.detector.backend.to_numpy(self.detector._all_hbd_h_coords) # type: ignore
            self.detector._detect_water_bridges_plip_style()

        return self.detector.interactions

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
        return self.detect_all(verbose=verbose)

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

        if not self.water_residues:
            return {"total": 0, "filtered": 0, "kept": 0}

        if not self.non_water_atom_idxs:
            for water_res in self.water_residues:
                water_res.is_skip = True
            return {"total": len(self.water_residues), "filtered": len(self.water_residues), "kept": 0}

        w_o_coords = self.detector.atom_container.coords_array[self.water_o_atom_idxs]
        non_water_coords = self.detector.atom_container.coords_array[self.non_water_atom_idxs]
        dist = self.detector.backend.cdist(w_o_coords, non_water_coords)
        close_mask = self.detector.backend.min(dist, dim=1) < distance_threshold
        kept: int = close_mask.sum()
        filtered = len(close_mask) - kept
        skip_atoms_idxs = []
        for r in self.water_residues:
            r.is_skip = False
        skip_indices = np.where(self.detector.backend.to_numpy(~close_mask))[0]
        for i in skip_indices:
            self.water_residues[i].is_skip = True
            skip_atoms_idxs.extend(self.water_residues[i].atom_idxs)

        self.detector.atom_container.remain_atom_mask = np.ones(self.detector.atom_container.coords_array.shape[0], dtype=bool)
        self.detector.atom_container.remain_atom_mask[self.detector.atom_container.idx_to_array_pos_array[skip_atoms_idxs]] = False
        self.detector.atom_container.remain_atom_idxs = np.nonzero(self.detector.atom_container.remain_atom_mask)[0]
        self.detector.atom_container.remain_atom_idxs_set = set(self.detector.atom_container.remain_atom_idxs)
        return {"total": len(self.water_residues), "filtered": filtered, "kept": kept}


if __name__ == "__main__":
    from lazydock.gmx.mda.utils import filter_atoms_by_chains
    from lazydock.gmx.mda.convert import PDBConverter
    from fplip.all_atom_cuda.cupy_backend import CuPyBackend
    from fplip.all_atom_cuda.torch_backend import TorchBackend
    test_data_dir = Path(__file__).parent.parent.parent / 'test_data/pull'
    tpr = str(test_data_dir / "pull.tpr")
    xtc = str(test_data_dir / "pull_center.xtc")
    gro = str(test_data_dir / "pull.gro")
    analyzer = TrajectoryAnalyzer(tpr, xtc, gro, tolerance=1e-4)
    analyzer.load_universe()
    analyzer.u.trajectory[0]
    converter = PDBConverter(filter_atoms_by_chains(analyzer.u.atoms, ['A', 'B', 'CL']))
    pdb_str = converter.fast_convert()
    analyzer.load_molecule(pdb_str, as_string=True)
    analyzer.align_with_mda(frame=0)
    analyzer.load_waters('SOL')
    analyzer.setup_detector(backend=TorchBackend())
    analyzer.precompute_detector_once()
    
    from tqdm import tqdm
    for i in tqdm(range(15)):
        interactions = analyzer.detect_frame_fast(i, verbose=True)
    pass