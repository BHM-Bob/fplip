"""
All-Atom-CUDA Module Pi-Stacking Detection Tests

Tests for pi-stacking interaction detection in the all-atom-cuda module,
aligned with original PLIP test cases.
"""

import os
from pathlib import Path
import unittest

from fplip.all_atom.atom_properties import AtomProperties
from fplip.all_atom.molecule_complex import MoleculeComplex
from fplip.all_atom_cuda.cuda_detector import CudaInteractionDetector

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

class AllAtomCUDAPiStackingTest(unittest.TestCase):
    """Test pi-stacking detection in all-atom-cuda module."""

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

    def test_4dst_pistacking_detection(self):
        """Test pi-stacking detection for 4dst with GCP ligand.

        Original PLIP test: Consistent ring detection
        Expected: Pi-stacking between GCP and protein residues
        """
        interactions, mol, props = self._analyze_complex('4dst_protonated.pdb')
        pistacking = interactions.get('pistacking', [])

        # Should detect pi-stacking interactions
        self.assertTrue(len(pistacking) > 0,
                       "Should detect pi-stacking in 4dst")

        # Find pi-stacking involving GCP ligand
        gcp_pistack = [ps for ps in pistacking
                      if ps.res_a_name == 'GCP' or ps.res_b_name == 'GCP']

        self.assertTrue(len(gcp_pistack) > 0,
                       "Should detect pi-stacking with GCP ligand")

    def test_pistacking_geometry(self):
        """Test that detected pi-stacking has reasonable geometry."""
        interactions, mol, props = self._analyze_complex('4dst_protonated.pdb')
        pistacking = interactions.get('pistacking', [])

        for ps in pistacking:
            # Distance between ring centers should be < 7 Å
            self.assertTrue(ps.distance < 7.0,
                          f"Pi-stacking distance {ps.distance} too large")

            # Angle should be between 0 and 180 degrees
            self.assertTrue(0 <= ps.angle <= 180,
                          f"Pi-stacking angle {ps.angle} out of range")

    def test_pistacking_types(self):
        """Test detection of parallel and perpendicular pi-stacking."""
        interactions, mol, props = self._analyze_complex('4dst_protonated.pdb')
        pistacking = interactions.get('pistacking', [])

        has_parallel = False
        has_perpendicular = False

        for ps in pistacking:
            details = ps.details
            if details.get('type') == 'parallel':
                has_parallel = True
            elif details.get('type') == 'perpendicular':
                has_perpendicular = True

        # Note: Not all structures have both types
        # This test mainly checks that type classification works
        self.assertTrue(has_parallel or has_perpendicular,
                       "Should classify pi-stacking type")

    def test_ring_detection(self):
        """Test that aromatic rings are correctly identified."""
        mol = MoleculeComplex()
        mol.load_pdb(self.test_data_dir + '4dst_protonated.pdb')
        props = AtomProperties(mol.atom_container)

        # Should detect aromatic rings
        self.assertTrue(len(props.rings) > 0,
                       "Should detect aromatic rings")

        # Check ring properties
        for ring in props.rings:
            self.assertIn('center', ring, "Ring should have center")
            self.assertIn('normal', ring, "Ring should have normal")
            self.assertIn('indices', ring, "Ring should have atom indices")

    def test_consistent_detection(self):
        """Test that pi-stacking detection is consistent across runs."""
        angles = set()

        for _ in range(5):
            interactions, mol, props = self._analyze_complex('4dst_protonated.pdb')
            pistacking = interactions.get('pistacking', [])

            if pistacking:
                angles.add(pistacking[0].angle)

        # Should get consistent results
        self.assertEqual(len(angles), 1,
                        "Pi-stacking detection should be consistent")

    def test_1vsn_pistacking(self):
        """Test pi-stacking detection for 1vsn.

        1vsn contains aromatic residues that may stack with each other
        or with the NFT ligand.
        """
        interactions, mol, props = self._analyze_complex('1vsn.pdb')
        pistacking = interactions.get('pistacking', [])

        # May or may not detect pi-stacking depending on structure
        # Mainly testing that detection doesn't crash
        self.assertIsInstance(pistacking, list)

    def test_pication_detection(self):
        """Test pi-cation interaction detection."""
        interactions, mol, props = self._analyze_complex('1vsn.pdb')
        pication = interactions.get('pication', [])

        # Pi-cation interactions may or may not be present
        # Test mainly checks that detection logic works
        self.assertIsInstance(pication, list)


if __name__ == '__main__':
    unittest.main()
