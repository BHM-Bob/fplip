"""
All-Atom-CUDA Backend Correctness Tests

Tests to verify that different compute backends (NumPy, CuPy, Torch)
produce consistent and correct results across different interaction types
and molecule sizes.
"""

import unittest
from pathlib import Path

import numpy as np

from fplip.basic import config

from fplip.all_atom.atom_properties import AtomProperties
from fplip.all_atom.molecule_complex import MoleculeComplex
from fplip.all_atom_cuda.cuda_detector import CudaInteractionDetector

# Import NumPy backend (always available)
from fplip.all_atom_cuda.numpy_backend import NumPyBackend
NUMPY_AVAILABLE = True

# Try to import Torch, skip tests if not available
try:
    from fplip.all_atom_cuda.torch_backend import TorchBackend
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import CuPy, skip tests if not available or no CUDA device
try:
    import cupy as cp

    # Test if CUDA is actually available
    cp.cuda.Device(0).use()
    from fplip.all_atom_cuda.cupy_backend import CuPyBackend
    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False

TEST_DIR = Path(__file__).parent.parent.parent / 'test'

class BackendCorrectnessTest(unittest.TestCase):
    """Test that all backends produce consistent results."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = str(TEST_DIR / 'pdb') + '/'
        self.backends = {}
        # NumPy backend is always available as the reference implementation
        if NUMPY_AVAILABLE:
            self.backends['numpy'] = NumPyBackend()
        if TORCH_AVAILABLE:
            self.backends['torch'] = TorchBackend()
        if CUPY_AVAILABLE:
            self.backends['cupy'] = CuPyBackend()
        if len(self.backends) == 0:
            self.skipTest("No available backends to test.")
        self.backends = dict(self.backends)
        
    @staticmethod
    def setUpClass():
        # Use hydrogen atoms from PDB file for deterministic results
        # OpenBabel's AddPolarHydrogens() produces non-deterministic hydrogen positions
        config.NOHYDRO = True
        
    @staticmethod
    def tearDownClass():
        # Reset the hydrogen flag to the original value
        config.NOHYDRO = False

    def _to_numpy_array(self, arr):
        """Convert backend array to numpy array for comparison.

        Handles GPU tensors from PyTorch/CuPy backends.
        """
        if hasattr(arr, 'cpu'):  # PyTorch tensor
            return arr.cpu().numpy()
        elif hasattr(arr, 'get'):  # CuPy array
            return arr.get()
        else:
            return np.asarray(arr)

    def _analyze_complex_with_backend(self, pdb_file: str, backend):
        """Helper method to analyze a PDB file with a specific backend.

        Args:
            pdb_file: PDB file name
            backend: ComputeBackend instance

        Returns:
            Tuple of (interactions, mol, props)
        """
        mol = MoleculeComplex()
        mol.load_pdb(self.test_data_dir + pdb_file)
        props = AtomProperties(mol.atom_container)
        detector = CudaInteractionDetector(
            mol.atom_container, props, mol.residues,
            backend=backend
        )
        interactions = detector.detect_all()
        return interactions, mol, props

    def _count_interactions(self, interactions: dict) -> dict:
        """Count interactions by type.

        Args:
            interactions: Dictionary of interaction lists

        Returns:
            Dictionary with interaction counts
        """
        counts = {}
        for interaction_type, interaction_list in interactions.items():
            counts[interaction_type] = len(interaction_list)
        return counts

    def _compare_interaction_counts(self, counts_by_backend: dict, tolerance: int = 0):
        """Compare interaction counts across backends.

        Args:
            counts_by_backend: Dict mapping backend name to counts dict
            tolerance: Allowed difference in counts (default 0 for exact match)

        Returns:
            True if all backends match within tolerance, False otherwise
        """
        backend_names = list(counts_by_backend.keys())
        if len(backend_names) < 2:
            return True

        # Get all interaction types
        all_types = set()
        for counts in counts_by_backend.values():
            all_types.update(counts.keys())

        # Compare each interaction type
        for interaction_type in all_types:
            counts = [
                counts_by_backend[backend].get(interaction_type, 0)
                for backend in backend_names
            ]
            max_count = max(counts)
            min_count = min(counts)

            if max_count - min_count > tolerance:
                self.fail(
                    f"Backend mismatch for {interaction_type}: "
                    f"{dict(zip(backend_names, counts))}"
                )

        return True

    def test_small_molecule_hydrogen_bonds(self):
        """Test H-bond detection consistency on small molecule (4dst).

        Covers: Hydrogen bonds, small molecule size
        """
        pdb_file = '4dst_protonated.pdb'
        counts_by_backend = {}

        for name, backend in self.backends.items():
            interactions, _, _ = self._analyze_complex_with_backend(pdb_file, backend)
            counts_by_backend[name] = self._count_interactions(interactions)

        # All backends should produce identical counts
        self._compare_interaction_counts(counts_by_backend)

    def test_medium_molecule_salt_bridges(self):
        """Test salt bridge detection consistency on medium molecule (1bma).

        Covers: Salt bridges, medium molecule size
        """
        pdb_file = '1bma.pdb'
        counts_by_backend = {}

        for name, backend in self.backends.items():
            interactions, _, _ = self._analyze_complex_with_backend(pdb_file, backend)
            counts_by_backend[name] = self._count_interactions(interactions)

        self._compare_interaction_counts(counts_by_backend)

    def test_pi_stacking_detection(self):
        """Test pi-stacking detection consistency (4dst with GCP ligand).

        Covers: Pi-stacking, aromatic ring detection
        """
        pdb_file = '4dst_protonated.pdb'
        counts_by_backend = {}

        for name, backend in self.backends.items():
            interactions, _, _ = self._analyze_complex_with_backend(pdb_file, backend)
            counts_by_backend[name] = self._count_interactions(interactions)

        self._compare_interaction_counts(counts_by_backend)

    def test_metal_coordination_detection(self):
        """Test metal coordination detection consistency (1rmd with Zn).

        Covers: Metal coordination, metal ion detection
        """
        pdb_file = '1rmd.pdb'
        counts_by_backend = {}

        for name, backend in self.backends.items():
            interactions, _, _ = self._analyze_complex_with_backend(pdb_file, backend)
            counts_by_backend[name] = self._count_interactions(interactions)

        self._compare_interaction_counts(counts_by_backend)

    def test_water_bridge_detection(self):
        """Test water bridge detection consistency (1vsn with waters).

        Covers: Water bridges, water-mediated interactions
        """
        pdb_file = '1vsn.pdb'
        counts_by_backend = {}

        for name, backend in self.backends.items():
            interactions, _, _ = self._analyze_complex_with_backend(pdb_file, backend)
            counts_by_backend[name] = self._count_interactions(interactions)

        self._compare_interaction_counts(counts_by_backend)

    def test_large_molecule_comprehensive(self):
        """Test comprehensive detection on large molecule (1vsn).

        Covers: Multiple interaction types, large molecule size
        """
        pdb_file = '1vsn.pdb'
        counts_by_backend = {}

        for name, backend in self.backends.items():
            interactions, _, _ = self._analyze_complex_with_backend(pdb_file, backend)
            counts_by_backend[name] = self._count_interactions(interactions)

        self._compare_interaction_counts(counts_by_backend)

    def test_halogen_bond_detection(self):
        """Test halogen bond detection consistency (4dst with fluorine).

        Covers: Halogen bonds, halogen detection
        """
        pdb_file = '4dst_protonated.pdb'
        counts_by_backend = {}

        for name, backend in self.backends.items():
            interactions, _, _ = self._analyze_complex_with_backend(pdb_file, backend)
            counts_by_backend[name] = self._count_interactions(interactions)

        self._compare_interaction_counts(counts_by_backend)

    def test_hydrophobic_interaction_detection(self):
        """Test hydrophobic interaction detection consistency.

        Covers: Hydrophobic interactions
        """
        pdb_file = '4dst_protonated.pdb'
        counts_by_backend = {}

        for name, backend in self.backends.items():
            interactions, _, _ = self._analyze_complex_with_backend(pdb_file, backend)
            counts_by_backend[name] = self._count_interactions(interactions)

        self._compare_interaction_counts(counts_by_backend)

    def test_backend_numerical_operations(self):
        """Test that backend numerical operations produce consistent results.

        Tests basic numerical operations used in interaction detection.
        """
        # Test data
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        coords2 = np.array([[1.0, 1.0, 0.0], [2.0, 0.0, 0.0]])

        for name, backend in self.backends.items():
            # Test cdist
            dist_matrix = backend.cdist(coords1, coords2)
            self.assertEqual(dist_matrix.shape, (3, 2))

            # Test sum
            arr = backend.to_device(np.array([[1.0, 2.0], [3.0, 4.0]]))
            row_sum = backend.sum(arr, axis=1)

            # Convert back for comparison
            row_sum_np = self._to_numpy_array(row_sum)
            expected = np.array([3.0, 7.0])
            self.assertTrue(
                np.allclose(row_sum_np, expected),
                f"{name} backend sum mismatch: {row_sum_np} vs {expected}"
            )

    def test_backend_conditional_operations(self):
        """Test that backend conditional operations (where) work correctly."""
        for name, backend in self.backends.items():
            # Test where
            condition = backend.to_device(np.array([True, False, True]))
            x = 1.0
            y = 0.0
            result = backend.where(condition, x, y)
            result_np = self._to_numpy_array(result)
            expected = np.array([1.0, 0.0, 1.0])
            self.assertTrue(
                np.allclose(result_np, expected),
                f"{name} backend where mismatch: {result_np} vs {expected}"
            )

    def test_backend_angle_operations(self):
        """Test that backend angle operations (degrees) work correctly."""
        for name, backend in self.backends.items():
            # Test degrees conversion
            radians = backend.to_device(np.array([0.0, np.pi/2, np.pi]))
            degrees = backend.degrees(radians)
            degrees_np = self._to_numpy_array(degrees)
            expected = np.array([0.0, 90.0, 180.0])
            self.assertTrue(
                np.allclose(degrees_np, expected, atol=1e-5),
                f"{name} backend degrees mismatch: {degrees_np} vs {expected}"
            )

    def test_detailed_interaction_properties(self):
        """Test that interaction properties are consistent across backends.

        Verifies that detected interactions have the same geometric properties
        regardless of backend.
        """
        pdb_file = '4dst_protonated.pdb'

        interactions_by_backend = {}
        for name, backend in self.backends.items():
            interactions, _, _ = self._analyze_complex_with_backend(pdb_file, backend)
            interactions_by_backend[name] = interactions

        # Compare interaction counts for each type
        for interaction_type in ['hbond', 'saltbridge', 'pistacking']:
            counts = {
                name: len(interactions.get(interaction_type, []))
                for name, interactions in interactions_by_backend.items()
            }

            # All backends should have the same count
            unique_counts = set(counts.values())
            self.assertEqual(
                len(unique_counts), 1,
                f"{interaction_type} counts differ across backends: {counts}"
            )


if __name__ == '__main__':
    unittest.main()
