import os
from typing import List, Union

from lazydock.gmx.mda.convert import PDBConverter
from lazydock.gmx.mda.utils import filter_atoms_by_chains
import numpy as np
import torch
import cupy as cp

from fplip.all_atom.trajectory_analyzer import TrajectoryAnalyzer
from fplip.all_atom_cuda.trajectory_analyzer import TrajectoryAnalyzer as TrajectoryAnalyzerCUDA
from fplip.all_atom_cuda import TorchBackend, CuPyBackend, NumPyBackend


def set_chainIDs(analyzer: TrajectoryAnalyzer) -> TrajectoryAnalyzer:
    '''
    设置AtomGroup的chainIDs

    Args:
        analyzer (TrajectoryAnalyzer): 原始TrajectoryAnalyzer
    '''
    chainIDs = analyzer.u.atoms.chainIDs
    lig_segs = list(filter(lambda s: s.startswith('HET'), np.unique(analyzer.u.atoms.segids)))
    chainIDs[np.isin(analyzer.u.atoms.segids, lig_segs)] = 'L'
    rec_segs = list(filter(lambda s: s.startswith('PRO'), np.unique(analyzer.u.atoms.segids)))
    chainIDs[np.isin(analyzer.u.atoms.segids, rec_segs)] = 'R'
    rec_segs = list(filter(lambda s: s == 'SOLV', np.unique(analyzer.u.atoms.segids)))
    chainIDs[np.isin(analyzer.u.atoms.segids, rec_segs)] = 'S'
    rec_segs = list(filter(lambda s: s == 'IONS', np.unique(analyzer.u.atoms.segids)))
    chainIDs[np.isin(analyzer.u.atoms.segids, rec_segs)] = 'I'
    analyzer.u.atoms.chainIDs = chainIDs
    return analyzer


def analyze_from_traj(root: str, ana_cls: type = TrajectoryAnalyzer, backend = None):
    tpr = os.path.join(root, "step3_input.pdb")
    xtc = os.path.join(root, "step5_production.dcd")
    analyzer = ana_cls(tpr, xtc, tolerance=1e-4)
    analyzer.load_universe()
    analyzer.u.trajectory[0]
    analyzer = set_chainIDs(analyzer)
    
    converter = PDBConverter(filter_atoms_by_chains(analyzer.u.atoms, ['R', 'L']), reindex=False)
    pdb_str = converter.fast_convert()
    analyzer.load_molecule(pdb_str, as_string=True)
    analyzer.align_with_mda(frame=0)
    analyzer.load_waters('S')
    if backend is None:
        analyzer.setup_detector()
    else:
        analyzer.setup_detector(backend=backend)
    analyzer.precompute_detector_once()

    analyzer.update_frame(0)
    for _, interactions in analyzer.iterate_frames(verbose=True):
        pass
    return True

def analyze_from_traj_torch(root: str):
    backend = TorchBackend()
    try:
        return analyze_from_traj(root, ana_cls=TrajectoryAnalyzerCUDA, backend=backend)
    except torch.OutOfMemoryError:
        print("Torch analysis OOM")
        torch.cuda.empty_cache()
        return None

def analyze_from_traj_cupy(root: str):
    backend = CuPyBackend()
    try:
        return analyze_from_traj(root, ana_cls=TrajectoryAnalyzerCUDA, backend=backend)
    except cp.cuda.memory.OutOfMemoryError:
        print("Cupy analysis OOM")
        memory_pool = cp.get_default_memory_pool()
        memory_pool.free_all_blocks()
        return None
    
def analyze_from_traj_numpy(root: str):
    backend = NumPyBackend()
    return analyze_from_traj(root, ana_cls=TrajectoryAnalyzerCUDA, backend=backend)
