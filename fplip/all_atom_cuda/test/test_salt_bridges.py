"""
All-Atom CUDA Module Salt Bridge Detection Tests

Tests for salt bridge detection in the all-atom-cuda module,
aligned with original PLIP test cases.

Default backend: torch (CUDA)
"""

import os
import unittest
from pathlib import Path

from fplip.all_atom.atom_properties import AtomProperties
from fplip.all_atom.molecule_complex import MoleculeComplex
from fplip.all_atom_cuda import CudaInteractionDetector

# Try to import Torch backend
try:
    from fplip.all_atom_cuda.torch_backend import TorchBackend
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import CuPy backend
try:
    from fplip.all_atom_cuda.cupy_backend import CuPyBackend
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Default backend for tests - use torch if available, otherwise numpy
DEFAULT_BACKEND = os.environ.get('ALL_ATOM_CUDA_TEST_BACKEND', 'torch' if TORCH_AVAILABLE else 'numpy')


def get_backend(backend_name=None):
    """Get compute backend by name.

    Args:
        backend_name: Backend name ('numpy', 'cupy', 'torch') or None for default

    Returns:
        ComputeBackend instance
    """
    name = backend_name or DEFAULT_BACKEND

    if name == 'cupy':
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is not available")
        return CuPyBackend()
    elif name == 'torch':
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available")
        return TorchBackend()
    else:
        raise ValueError(f"Unknown backend: {name}")

TEST_DIR = Path(__file__).parent.parent.parent / 'test'

class AllAtomCUDASaltBridgeTest(unittest.TestCase):
    """Test salt bridge detection in all-atom-cuda module."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = str(TEST_DIR / 'pdb') + '/'
        self.backend = get_backend()

    def _analyze_complex(self, pdb_file: str):
        """Helper method to analyze a PDB file using CudaInteractionDetector."""
        mol = MoleculeComplex()
        mol.load_pdb(self.test_data_dir + pdb_file)
        props = AtomProperties(mol.atom_container)
        detector = CudaInteractionDetector(
            mol.atom_container, props, mol.residues,
            backend=self.backend
        )
        interactions = detector.detect_all()
        return interactions, mol, props

    def test_4yb0_phosphate_groups(self):
        """Test phosphate group detection for nucleic acid ligand C2E (4yb0).

        C2E (c-di-GMP) contains phosphate groups that should be identified
        as negatively charged.
        """
        mol = MoleculeComplex()
        mol.load_pdb(self.test_data_dir + '4yb0.pdb')
        props = AtomProperties(mol.atom_container)

        # Should identify phosphate groups
        neg_charged = props.get_neg_charged()

        # Find phosphate atoms in C2E
        c2e_phosphate = [atom for atom in neg_charged
                        if atom.resname == 'C2E' and atom.atomic_num == 15]

        self.assertTrue(len(c2e_phosphate) > 0,
                       "Should identify phosphate groups in C2E")

        # Should also identify oxygen atoms attached to phosphate
        c2e_phosphate_o = [atom for atom in neg_charged
                          if atom.resname == 'C2E' and atom.atomic_num == 8]

        self.assertTrue(len(c2e_phosphate_o) > 0,
                       "Should identify phosphate oxygen atoms in C2E")

    def test_1vsn_saltbridge_detection(self):
        """Test salt bridge detection for 1vsn.

        1vsn contains charged residues (ARG, LYS, ASP, GLU)
        that may form salt bridges.
        """
        interactions, _, _ = self._analyze_complex('1vsn.pdb')
        saltbridges = interactions.get('saltbridge', [])

        # Should detect salt bridges
        self.assertTrue(len(saltbridges) > 0,
                       "Should detect salt bridges in 1vsn")

    def test_saltbridge_geometry(self):
        """Test that detected salt bridges have reasonable geometry."""
        interactions, _, _ = self._analyze_complex('1vsn.pdb')
        saltbridges = interactions.get('saltbridge', [])

        for sb in saltbridges:
            # Salt bridge distance should be < 5.5 Å
            self.assertTrue(sb.distance < 5.5,
                          f"Salt bridge distance {sb.distance} too large")

            # Should involve charged residues
            charged_pos = {'ARG', 'LYS', 'HIS'}
            charged_neg = {'ASP', 'GLU'}

            has_pos = sb.res_a_name in charged_pos or sb.res_b_name in charged_pos
            has_neg = sb.res_a_name in charged_neg or sb.res_b_name in charged_neg

            # Note: In all-atom mode, we don't distinguish ligand/receptor
            # So both could be protein residues
            self.assertTrue(has_pos or has_neg,
                          "Salt bridge should involve charged residues")

    def test_charge_identification(self):
        """Test that charged groups are correctly identified."""
        mol = MoleculeComplex()
        mol.load_pdb(self.test_data_dir + '1vsn.pdb')
        props = AtomProperties(mol.atom_container)

        # Should identify positive charges
        pos_charged = props.get_pos_charged()
        self.assertTrue(len(pos_charged) > 0,
                       "Should identify positively charged groups")

        # Should identify negative charges
        neg_charged = props.get_neg_charged()
        self.assertTrue(len(neg_charged) > 0,
                       "Should identify negatively charged groups")

        # Check for specific charged residues
        pos_residues = set(atom.resname for atom in pos_charged)
        neg_residues = set(atom.resname for atom in neg_charged)

        expected_pos = {'ARG', 'LYS', 'HIS'}
        expected_neg = {'ASP', 'GLU'}

        self.assertTrue(len(pos_residues & expected_pos) > 0,
                       "Should identify ARG, LYS, or HIS as positive")
        self.assertTrue(len(neg_residues & expected_neg) > 0,
                       "Should identify ASP or GLU as negative")

    def test_1x0n_saltbridge(self):
        """Test salt bridge detection for 1x0n with DTF ligand.

        Original PLIP test case for salt bridges.
        """
        interactions, _, _ = self._analyze_complex('1x0n_state_1.pdb')
        saltbridges = interactions.get('saltbridge', [])

        # Should detect salt bridges
        self.assertTrue(len(saltbridges) > 0,
                       "Should detect salt bridges in 1x0n")

        # Find salt bridges involving DTF ligand
        dtf_bridges = [sb for sb in saltbridges
                      if sb.res_a_name == 'DTF' or sb.res_b_name == 'DTF']

        # DTF is a charged ligand, should form salt bridges
        self.assertTrue(len(dtf_bridges) > 0,
                       "Should detect salt bridges with DTF ligand")

    def test_saltbridge_consistency(self):
        """Test that salt bridge detection is consistent."""
        counts = []

        for _ in range(3):
            interactions, _, _ = self._analyze_complex('1vsn.pdb')
            saltbridges = interactions.get('saltbridge', [])
            counts.append(len(saltbridges))

        # Should get consistent results
        self.assertEqual(len(set(counts)), 1,
                        "Salt bridge detection should be consistent")


if __name__ == '__main__':
    unittest.main()
