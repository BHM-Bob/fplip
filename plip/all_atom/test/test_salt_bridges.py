"""
All-Atom Module Salt Bridge Detection Tests

Tests for salt bridge detection in the all-atom module,
aligned with original PLIP test cases.
"""

import unittest

from plip.all_atom.atom_properties import AtomProperties
from plip.all_atom.interaction_detector import UnifiedInteractionDetector
from plip.all_atom.molecule_complex import MoleculeComplex


class AllAtomSaltBridgeTest(unittest.TestCase):
    """Test salt bridge detection in all-atom module."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = '/home/pcmd36/Desktop/BHM/My_Progs/fplip/plip/test/pdb/'

    def _analyze_complex(self, pdb_file: str):
        """Helper method to analyze a PDB file."""
        mol = MoleculeComplex()
        mol.load_pdb(self.test_data_dir + pdb_file)
        props = AtomProperties(mol.atom_container)
        detector = UnifiedInteractionDetector(mol.atom_container, props, mol.residues)
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
        interactions, mol, props = self._analyze_complex('1vsn.pdb')
        saltbridges = interactions.get('saltbridge', [])

        # Should detect salt bridges
        self.assertTrue(len(saltbridges) > 0,
                       "Should detect salt bridges in 1vsn")

    def test_saltbridge_geometry(self):
        """Test that detected salt bridges have reasonable geometry."""
        interactions, mol, props = self._analyze_complex('1vsn.pdb')
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
        interactions, mol, props = self._analyze_complex('1x0n_state_1.pdb')
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
            interactions, mol, props = self._analyze_complex('1vsn.pdb')
            saltbridges = interactions.get('saltbridge', [])
            counts.append(len(saltbridges))

        # Should get consistent results
        self.assertEqual(len(set(counts)), 1,
                        "Salt bridge detection should be consistent")


if __name__ == '__main__':
    unittest.main()
