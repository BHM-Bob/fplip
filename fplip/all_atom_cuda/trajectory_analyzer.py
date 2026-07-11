"""
Trajectory Analyzer Module

Provides fast trajectory analysis using MDAnalysis for coordinate loading
and OpenBabel for interaction detection.
"""
import signal
import time
from multiprocessing import Queue
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from lazydock.gmx.mda.convert import FakeAtomGroup
from mbapy import TaskPool
from MDAnalysis import Universe
from tqdm import tqdm

from fplip.all_atom.trajectory_analyzer import \
    TrajectoryAnalyzer as _TrajectoryAnalyzer
from fplip.all_atom_cuda.backend import ComputeBackend
from fplip.all_atom_cuda.cuda_detector import (CudaInteractionDetector,
                                               Interaction)
from fplip.all_atom_cuda.cupy_backend import CuPyBackend
from fplip.all_atom_cuda.numpy_backend import NumPyBackend
from fplip.all_atom_cuda.torch_backend import TorchBackend
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
        backend : ComputeBackend, optional
            Compute backend instance, choose from NumPyBackend, TorchBackend and CuPyBackend
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
        self.detector._remove_subring_duplicates()
        self.detector._refine_hbonds()
        self.detector._detect_water_bridges()
        if detect_water_bridges_plip_style:
            self.detector._all_hba_coords = self.detector.backend.to_numpy(self.detector._all_hba_coords) # type: ignore
            self.detector._all_hbd_don_coords = self.detector.backend.to_numpy(self.detector._all_hbd_don_coords) # type: ignore
            self.detector._all_hbd_h_coords = self.detector.backend.to_numpy(self.detector._all_hbd_h_coords) # type: ignore
            self.detector._detect_water_bridges_plip_style()

        return self.detector.interactions

    def detect_frame_fast(self, frame_idx: int, filter_waters: Optional[float] = 5.0, verbose: bool = False,
                          detect_water_bridges_plip_style: bool = False) -> Dict[str, List]:
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
        filter_waters : Optional[float]
            Distance threshold to filter out distant molecules (default None, no filter)
        detect_water_bridges_plip_style : bool, optional
            Whether to detect water bridges in the style of PLIP
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

        self.update_frame(frame_idx, filter_waters, precompute_coords=True, verbose=verbose)
        return self.detect_all(detect_water_bridges_plip_style)

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


_mp_frame_que = None
_mp_result_que = None

def _mp_init(frame_que: Queue, result_que: Queue):
    global _mp_frame_que, _mp_result_que
    _mp_frame_que = frame_que
    _mp_result_que = result_que
    # in gsd, it registers a sys.exit handler to catch SIGTERM
    # but it causes an abnormal quit process, so we disable it
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
                
def _analyzer_server(analyzer: TrajectoryAnalyzer,
                    init_opts: List[Tuple[str, Callable, List[Any]]],
                    filter_waters: float = 5,
                    detect_water_bridges_plip_style: bool = True):        
    def update_data(analyzer: TrajectoryAnalyzer, mda_coords: np.ndarray):
        analyzer.detector.update_coords(mda_coords)
        if filter_waters is not None:
            analyzer.filter_distant_waters(distance_threshold=filter_waters)
        analyzer.detector._precompute_cached_data()
        
    for fn_name, args in init_opts:
        if fn_name == "setup_detector":
            if args[1] == 'cupy':
                analyzer.setup_detector(atom_props=args[0], backend=CuPyBackend())
            elif args[1] == 'torch':
                analyzer.setup_detector(atom_props=args[0], backend=TorchBackend())
            elif args[1] == 'numpy':
                analyzer.setup_detector(atom_props=args[0], backend=NumPyBackend())
            else:
                analyzer.setup_detector(*args)
        else:
            getattr(analyzer, fn_name)(*args)

    while True:
        frame_idx, mda_coords = _mp_frame_que.get()
        if frame_idx == -1:
            analyzer.detector.backend.free_mem()
            return
        update_data(analyzer, mda_coords)
        interactions = analyzer.detect_all(detect_water_bridges_plip_style)
        # del objs to avoid Swig obj serialization error
        for inter_data in interactions['hbond'] + interactions['hbond_possible']:
            for atom in ['donor', 'h_atom', 'acceptor']:
                inter_data.objs[atom] = None
        _mp_result_que.put((frame_idx, interactions))

class FakeUniverse:
    def __init__(self, universe: Universe):
        self.atoms = FakeAtomGroup(universe.atoms)
        self.trajectory = np.zeros(len(universe.trajectory))

class TrajectoryParallelAnalyzer(TrajectoryAnalyzer):
    def __init__(
        self,
        n_workers: int,
        tpr_file: str,
        xtc_file: str,
        gro_file: Optional[str] = None,
        pdb_str: Optional[str] = None,
        tolerance: float = 1e-4
    ):
        super().__init__(tpr_file, xtc_file, gro_file, pdb_str, tolerance)
        if n_workers <= 0 or not isinstance(n_workers, int):
            raise ValueError("n_workers must be greater than 0 and an integer value.")
        self.n_workers = n_workers
        self.analyzers: list[TrajectoryAnalyzer] = [
            TrajectoryAnalyzer(tpr_file, xtc_file, gro_file, pdb_str, tolerance)
            for _ in range(n_workers)
        ]
        self._init_opts: List[Tuple[str, Callable, List[Any]]] = []
        self.read_lock = Lock()
        
    def load_universe(self):
        super().load_universe()
        # init workers with same MDA universe
        for i in range(self.n_workers):
            if self.gro_file:
                self.analyzers[i].u = self.u
                self.analyzers[i].u2 = self.u2
                self.analyzers[i].u.atoms.residues.resids = self.analyzers[i].u2.atoms.residues.resids  # pyright: ignore[reportAttributeAccessIssue]
            else:
                self.analyzers[i].u = self.u
                
    def transfer_unverse(self):
        for i in range(self.n_workers):
            if self.gro_file:
                self.analyzers[i].u = FakeUniverse(self.u) # type: ignore
                self.analyzers[i].u2 = FakeUniverse(self.u2) # type: ignore
            else:
                self.analyzers[i].u = FakeUniverse(self.u) # type: ignore
    
    def load_molecule(self, pdb_str: str, as_string: bool = True, fix_pdb: bool = True):
        self._init_opts.append(("load_molecule", [pdb_str, as_string, fix_pdb]))
            
    def load_waters(self, water_chain: Union[str, List[str]]):
        self._init_opts.append(("load_waters", [water_chain]))
            
    def align_with_mda(self, frame: int = 0):
        self._init_opts.append(("align_with_mda", [frame]))
            
    def setup_detector(self, atom_props=None, backend: Optional[ComputeBackend] = None):
        self.backend = backend if backend is not None else TorchBackend()
        self._init_opts.append(("setup_detector", [atom_props, backend.name if backend else None]))
            
    def precompute_detector_once(self):
        self._init_opts.append(("precompute_detector_once", []))
            
    def _task_server(self, frame_indices: List[int], frame_que: Queue):
        for frame_idx in frame_indices:
            # submit task till all worker are busy
            # use 2 * self.n_workers as task buffer to avoid worker is not busy
            self.u.trajectory[frame_idx]
            frame_que.put((frame_idx, self.u.atoms.positions.copy()))
            while frame_que.qsize() > 2 * self.n_workers:
                time.sleep(0.01)
            
    def iterate_frames_parallel(self, start: int = 0,
                                stop: Optional[int] = None,
                                step: int = 1,
                                filter_waters: Optional[float] = 5,
                                detect_water_bridges_plip_style: bool = False,
                                verbose: bool = False):
        """Iterate over frames and detect interactions.

        Parameters
        ----------
        start : int
            Starting frame index
        stop : int, optional
            Stopping frame index (exclusive)
        step : int
            Frame step
        filter_waters : float, optional
            Filter waters with radius less than this value
        detect_water_bridges_plip_style : bool, optional
            Whether to detect water bridges in PLIP style
        verbose : bool
            Whether to show progress bars for each frame

        Yields
        ------
        Tuple[int, Dict[str, List]]
            (frame_index, interactions_dict) for each frame
        """
        # if self.detector is None:
        #     raise RuntimeError("Detector not setup. Call setup_detector() first.")
        # init task pool
        frame_que, result_que = Queue(), Queue()
        pool = TaskPool('process', self.n_workers, report_error=True,
                        mp_pool_init_kwargs={'initializer': _mp_init,
                                             'initargs': (frame_que, result_que)}).start()
        for analyzer in self.analyzers:
            pool.add_task(None, _analyzer_server, analyzer, self._init_opts,
                          filter_waters, detect_water_bridges_plip_style)
        # start processing frames
        frame_indices = list(range(start, stop or len(self.u.trajectory), step))
        task_server = Thread(target=self._task_server, args=(frame_indices, frame_que))
        task_server.start()
        for i in tqdm(frame_indices, desc="Processing frames", disable=not verbose):
            # check result, if has one, yield it
            frame_idx, interactions = result_que.get()
            # clean up after last frame
            if i == frame_indices[-1]:
                for _ in range(self.n_workers):
                    frame_que.put((-1, None))
                pool.wait_till_free()
                task_server.join()
                pool.close(1)
                self.backend.free_mem()
            # yield result
            yield frame_idx, interactions


def set_chainIDs_from_segids_map(analyzer: Union[TrajectoryAnalyzer, TrajectoryParallelAnalyzer],
                                 segid2chain: Dict[str, str]) -> Union[TrajectoryAnalyzer, TrajectoryParallelAnalyzer]:
    '''
    设置AtomGroup的chainIDs

    Args:
        analyzer (TrajectoryAnalyzer | TrajectoryParallelAnalyzer): 原始TrajectoryAnalyzer
        segid2chain (Dict[str, str]): segid到chain的映射关系
    Returns:
        TrajectoryAnalyzer | TrajectoryParallelAnalyzer: 设置后的TrajectoryAnalyzer
    '''
    chainIDs = analyzer.u.atoms.chainIDs
    for segid, chain in segid2chain.items():
        chainIDs[np.isin(analyzer.u.atoms.segids, [segid])] = chain
    analyzer.u.atoms.chainIDs = chainIDs
    return analyzer

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