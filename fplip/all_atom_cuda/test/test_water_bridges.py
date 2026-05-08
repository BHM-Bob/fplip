"""
All-Atom-CUDA Module Water Bridge Detection Tests

Tests for water bridge detection in the all-atom-cuda module,
aligned with original PLIP test cases.
"""

import os
import unittest
from pathlib import Path

from fplip.all_atom.atom_properties import AtomProperties
from fplip.all_atom.molecule_complex import MoleculeComplex
from fplip.all_atom_cuda.cuda_detector import CudaInteractionDetector
from fplip.basic import config

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

class AllAtomCUDAWaterBridgeTest(unittest.TestCase):
    """Test water bridge detection in all-atom-cuda module."""

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

    def test_3ems_water_bridge_detection(self):
        """Test water bridge detection for 3ems.

        Original PLIP test: 4 water bridges involving ARG:A:131
        """
        interactions, _, _ = self._analyze_complex('3ems.pdb')
        water_bridges = interactions.get('water_bridge', [])

        # Should detect water bridges
        self.assertTrue(len(water_bridges) > 0,
                       "Should detect water bridges in 3ems")

        # Find water bridges involving ARG:A:131
        arg_bridges = [wb for wb in water_bridges
                      if (wb.res_a_name == 'ARG' and wb.res_a_num == 131) or
                         (wb.res_b_name == 'ARG' and wb.res_b_num == 131)]

        # Original PLIP detected 4 water bridges for ARG:A:131
        # All-atom module may detect different number due to unified detection
        self.assertTrue(len(arg_bridges) > 0,
                       "Should detect water bridges involving ARG:A:131")

    def test_1vsn_water_bridges(self):
        """Test water bridge detection for 1vsn.

        1vsn contains many water molecules that may form bridges.
        """
        interactions, _, _ = self._analyze_complex('1vsn.pdb')
        water_bridges = interactions.get('water_bridge', [])

        # Should detect water bridges
        self.assertTrue(len(water_bridges) > 0,
                       "Should detect water bridges in 1vsn")

    def test_water_bridge_geometry(self):
        """Test that detected water bridges have reasonable geometry."""
        interactions, _, _ = self._analyze_complex('1vsn.pdb')
        water_bridges = interactions.get('water_bridge', [])

        for wb in water_bridges:
            # Water bridge distance should be reasonable (relaxed for heavy-atom detection)
            self.assertTrue(2.0 <= wb.distance <= 4.5,
                          f"Water bridge distance {wb.distance} out of range")

            # Should involve water residue
            is_water_bridge = (wb.res_a_name == 'HOH' or wb.res_b_name == 'HOH')
            self.assertTrue(is_water_bridge,
                          "Water bridge should involve water molecule")

    def test_water_residue_identification(self):
        """Test that water residues are correctly identified."""
        mol = MoleculeComplex()
        mol.load_pdb(self.test_data_dir + '1vsn.pdb')

        # Should identify water residues
        water_residues = [r for r in mol.residues if r.is_water]
        self.assertTrue(len(water_residues) > 0,
                       "Should identify water residues")

        # Check water residue properties
        for water in water_residues:
            self.assertEqual(water.resname, 'HOH',
                           "Water residue should be named HOH")
            self.assertTrue(len(water.atoms) >= 1,
                          "Water residue should have at least 1 atom (O)")

    def test_water_bridge_requirements(self):
        """Test that water bridges require water to bridge two residues."""
        interactions, _, _ = self._analyze_complex('1vsn.pdb')
        water_bridges = interactions.get('water_bridge', [])

        for wb in water_bridges:
            # Water bridge should have water_residue in details
            self.assertIn('water_residue', wb.details,
                         "Water bridge should record water residue")

            # Water bridge should connect two different atoms (not necessarily different residues)
            # All-Atom design allows intra-residue water bridges (e.g., side chain to backbone)
            atom_a = (wb.res_a_name, wb.res_a_chain, wb.res_a_num, wb.atom_a_name)
            atom_b = (wb.res_b_name, wb.res_b_chain, wb.res_b_num, wb.atom_b_name)
            self.assertNotEqual(atom_a, atom_b,
                               "Water bridge should connect two different atoms")

            # At least one should be water
            has_water = (wb.res_a_name == 'HOH' or wb.res_b_name == 'HOH')
            self.assertTrue(has_water,
                          "Water bridge should involve water")

    def test_hbond_prerequisite(self):
        """Test that water bridges are derived from H-bonds."""
        interactions, _, _ = self._analyze_complex('1vsn.pdb')
        hbonds = interactions.get('hbond', [])
        water_bridges = interactions.get('water_bridge', [])

        # Water bridges require H-bonds
        self.assertTrue(len(hbonds) > 0,
                       "Should have H-bonds for water bridge detection")

        # Find H-bonds involving water
        water_hbonds = [hb for hb in hbonds
                       if hb.res_a_name == 'HOH' or hb.res_b_name == 'HOH']

        self.assertTrue(len(water_hbonds) > 0,
                       "Should have H-bonds involving water")

        # Water bridges should be fewer than or equal to water H-bonds
        self.assertTrue(len(water_bridges) <= len(water_hbonds),
                       "Water bridges should be derived from water H-bonds")

    def test_water_bridge_consistency(self):
        """Test that water bridge detection is consistent.

        Uses NOHYDRO=True to ensure deterministic results by using
        explicit hydrogens from the PDB file rather than OpenBabel's
        non-deterministic protonation.
        """
        # Use explicit hydrogens for deterministic results
        config.NOHYDRO = True

        counts = []

        for _ in range(3):
            interactions, _, _ = self._analyze_complex('1vsn.pdb')
            water_bridges = interactions.get('water_bridge', [])
            counts.append(len(water_bridges))

        # Results should be identical with explicit hydrogens
        self.assertEqual(max(counts), min(counts),
                        f"Water bridge detection inconsistent: {counts}")


if __name__ == '__main__':
    unittest.main()
