"""
All-Atom CUDA Interaction Detection Module for PLIP

This module provides CUDA-accelerated interaction detection capabilities
built on top of the All-Atom module. It supports multiple compute backends:
- CuPyBackend: GPU acceleration via CuPy (NVIDIA CUDA)
- TorchBackend: GPU acceleration via PyTorch (NVIDIA CUDA)

Usage:
    from plip.all_atom_cuda import CudaInteractionDetector

    # GPU mode (CuPy)
    from plip.all_atom_cuda import CuPyBackend
    detector = CudaInteractionDetector(container, props, residues, backend=CuPyBackend())

    # GPU mode (PyTorch)
    from plip.all_atom_cuda import TorchBackend
    detector = CudaInteractionDetector(container, props, residues, backend=TorchBackend())
"""

import sys
from pathlib import Path
sys.path.insert(0, str((Path(__file__).parent / '../..').resolve()))


from fplip.all_atom.atom_container import AtomContainer, AtomInfo
from fplip.all_atom.atom_properties import AtomProperties
from fplip.all_atom.interaction_catalog import InteractionCatalog
from fplip.all_atom.interaction_detector import Interaction, UnifiedInteractionDetector
from fplip.all_atom.molecule_complex import MoleculeComplex
from fplip.all_atom.residue import Residue
from fplip.all_atom.simple_report import SimpleReport

from fplip.all_atom_cuda.backend import ComputeBackend
from fplip.all_atom_cuda.cuda_detector import CudaInteractionDetector
from fplip.all_atom_cuda.numpy_backend import NumPyBackend


def _get_cupy_backend():
    try:
        from fplip.all_atom_cuda.cupy_backend import CuPyBackend
        return CuPyBackend
    except ImportError:
        return None


def _get_torch_backend():
    try:
        from fplip.all_atom_cuda.torch_backend import TorchBackend
        return TorchBackend
    except ImportError:
        return None


CuPyBackend = _get_cupy_backend()
TorchBackend = _get_torch_backend()

__all__ = [
    'AtomContainer',
    'AtomInfo',
    'AtomProperties',
    'MoleculeComplex',
    'Residue',
    'UnifiedInteractionDetector',
    'Interaction',
    'InteractionCatalog',
    'SimpleReport',
    'ComputeBackend',
    'CudaInteractionDetector',
    'CuPyBackend',
    'TorchBackend',
    'NumPyBackend',
]


def _analyze_complex(pdb_file: str, backend=None):
    """Helper method to analyze a PDB file with optional GPU acceleration."""
    mol = MoleculeComplex()
    mol.load_pdb(pdb_file)
    props = AtomProperties(mol.atom_container)
    if backend is not None:
        detector = CudaInteractionDetector(mol.atom_container, props, mol.residues, backend=backend)
    else:
        detector = CudaInteractionDetector(mol.atom_container, props, mol.residues)
    interactions = detector.detect_all()
    return interactions, mol, props


if __name__ == '__main__':
    # interactions, mol, props = _analyze_complex('test_data/GPCR_pep.pdb',
    #                                             backend=TorchBackend())
    interactions, mol, props = _analyze_complex('fplip/test/pdb/1vsn.pdb',
                                                backend=TorchBackend())
    print(interactions)