"""
All-Atom Module Basic Function Tests

Tests for basic functionality in the all-atom module,
aligned with original PLIP test cases.
"""

import random
import unittest

import numpy

from plip.all_atom.molecule_complex import MoleculeComplex
from plip.basic.supplemental import (centroid, cluster_doubles, euclidean3d,
                                     normalize_vector, projection, vecangle,
                                     vector)


class TestLigandSupport(unittest.TestCase):
    """Test for support of different ligands"""

    def test_dna_rna(self):
        """Test if DNA and RNA is correctly processed"""
        mol = MoleculeComplex()
        mol.load_pdb('/home/pcmd36/Desktop/BHM/My_Progs/fplip/plip/test/pdb/1tf6.pdb')

        # Should load the structure
        self.assertTrue(len(mol.atom_container.atoms) > 0,
                       "Should load DNA-containing structure")

        # Check for DNA bases
        dna_bases = {'DG', 'DC', 'DA', 'DT'}
        residue_names = set(r.resname for r in mol.residues)

        # Should contain DNA bases
        self.assertTrue(len(residue_names & dna_bases) > 0,
                       "Should identify DNA residues")


class GeometryTest(unittest.TestCase):
    """Tests for geometrical calculations"""

    @staticmethod
    def vector_magnitude(v):
        return numpy.sqrt(sum(x ** 2 for x in v))

    def setUp(self):
        """Generate random data for the tests"""
        # Generate two random n-dimensional float vectors, with -100 <= n <= 100 and values 0 <= i <= 1
        dim = random.randint(1, 100)
        self.rnd_vec = [random.uniform(-100, 100) for _ in range(dim)]

    def test_euclidean(self):
        """Tests for mathematics.euclidean"""
        # Are the results correct?
        self.assertEqual(euclidean3d([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), 0)
        self.assertEqual(euclidean3d([2.0, 3.0, 4.0], [2.0, 3.0, 4.0]), 0)
        self.assertEqual(euclidean3d([4.0, 5.0, 6.0], [4.0, 5.0, 8.0]), 2.0)
        # Does the function take vectors or tuples as an input? What about integers?
        self.assertEqual(euclidean3d((4.0, 5.0, 6.0), [4.0, 5.0, 8.0]), 2.0)
        self.assertEqual(euclidean3d((4.0, 5.0, 6.0), (4.0, 5.0, 8.0)), 2.0)
        self.assertEqual(euclidean3d((4, 5, 6), (4.0, 5.0, 8.0)), 2.0)
        # Is the output a float?
        self.assertIsInstance(euclidean3d([2.0, 3.0, 4.0], [2.0, 3.0, 4.0]), float)

    def test_vector(self):
        """Tests for mathematics.vector"""
        # Are the results correct?
        self.assertEqual(list(vector([1, 1, 1], [0, 1, 0])), [-1, 0, -1])
        self.assertEqual(list(vector([0, 0, 10], [0, 0, 4])), [0, 0, -6])
        # Do I get an Numpy Array?
        self.assertIsInstance(vector([1, 1, 1], [0, 1, 0]), numpy.ndarray)
        # Do I get 'None' if the points have different dimensions?
        self.assertEqual(vector([1, 1, 1], [0, 1, 0, 1]), None)

    def test_vecangle(self):
        """Tests for mathematics.vecangle"""
        # Are the results correct?
        self.assertEqual(vecangle([3, 4], [-8, 6], deg=False), numpy.radians(90.0))
        self.assertEqual(vecangle([3, 4], [-8, 6]), 90.0)
        self.assertAlmostEqual(vecangle([-1, -1], [1, 1], deg=False), numpy.pi)
        # Correct if both vectors are equal?
        self.assertEqual(vecangle([3, 3], [3, 3]), 0.0)

    def test_centroid(self):
        """Tests for mathematics.centroid"""
        # Are the results correct?
        self.assertEqual(centroid([[0, 0, 0], [2, 2, 2]]), [1.0, 1.0, 1.0])
        self.assertEqual(centroid([[-5, 1, 2], [10, 2, 2]]), [2.5, 1.5, 2.0])

    def test_normalize_vector(self):
        """Tests for mathematics.normalize_vector"""
        # Are the results correct?
        self.assertAlmostEqual(self.vector_magnitude(normalize_vector(self.rnd_vec)), 1)

    def test_projection(self):
        """Tests for mathematics.projection"""
        # Are the results correct?
        self.assertEqual(projection([-1, 0, 0], [3, 3, 3], [1, 1, 1]), [3, 1, 1])

    def test_cluster_doubles(self):
        """Tests for mathematics.cluster_doubles"""
        # Are the results correct?
        self.assertEqual(set(cluster_doubles([(1, 3), (4, 1), (5, 6), (7, 5)])), {(1, 3, 4), (5, 6, 7)})


if __name__ == '__main__':
    unittest.main()
