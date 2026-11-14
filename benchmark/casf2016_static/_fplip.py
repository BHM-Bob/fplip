from functools import partial

import torch
import cupy as cp

from fplip.all_atom import _analyze_complex
from fplip.all_atom_cuda import _analyze_complex as _analyze_complex_cuda
from fplip.all_atom_cuda import TorchBackend, CuPyBackend, NumPyBackend



analyze_from_pdbstr = partial(_analyze_complex, as_string=True)

analyze_from_pdbstr_numpy = partial(_analyze_complex_cuda, as_string=True, backend=NumPyBackend())

def analyze_from_pdbstr_torch(pdbstr: str):
    try:
        return _analyze_complex_cuda(pdbstr, as_string=True, backend=TorchBackend())
    except torch.OutOfMemoryError:
        print("Torch analysis OOM")
        torch.cuda.empty_cache()
        return None

def analyze_from_pdbstr_cupy(pdbstr: str):
    try:
        return _analyze_complex_cuda(pdbstr, as_string=True, backend=CuPyBackend())
    except cp.cuda.memory.OutOfMemoryError:
        print("Cupy analysis OOM")
        memory_pool = cp.get_default_memory_pool()
        memory_pool.free_all_blocks()
        return None
