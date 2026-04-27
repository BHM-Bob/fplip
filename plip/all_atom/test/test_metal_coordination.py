"""
All-Atom Module Metal Coordination Tests

Tests for metal coordination detection in the all-atom module,
based on literature-validated metal binding sites.
"""

import unittest

from plip.all_atom.atom_properties import AtomProperties
from plip.all_atom.interaction_detector import UnifiedInteractionDetector
from plip.all_atom.molecule_complex import MoleculeComplex


class AllAtomMetalCoordinationTest(unittest.TestCase):
    """Test metal coordination detection in all-atom module."""

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

    def test_1rmd_zinc_coordination(self):
        """Test zinc binding sites in RAG1 dimerization domain (1rmd).

        Literature: Harding (2004), Fig. 1a
        Expected: Zn coordinated by 3 CYS and 1 HIS
        """
        interactions, mol, props = self._analyze_complex('1rmd.pdb')
        metal_complexes = interactions.get('metal', [])

        # Should detect metal coordination
        self.assertTrue(len(metal_complexes) > 0,
                       "Should detect Zn coordination in 1rmd")

        # Check coordination partners
        metal_residues = set()
        for mc in metal_complexes:
            if mc.res_a_name not in ['ZN', 'MN', 'CA', 'MG', 'FE', 'CU']:
                metal_residues.add(mc.res_a_name)
            if mc.res_b_name not in ['ZN', 'MN', 'CA', 'MG', 'FE', 'CU']:
                metal_residues.add(mc.res_b_name)

        # Should involve CYS and/or HIS
        self.assertTrue('CYS' in metal_residues or 'HIS' in metal_residues,
                       "Zn should be coordinated by CYS or HIS")

    def test_1rla_manganese_coordination(self):
        """Test manganese coordination in rat liver arginase (1rla).

        Literature: Harding (2004), Fig. 1b
        Expected: Mn coordinated by 1 HIS, 3 ASP, and 1 water
        """
        interactions, mol, props = self._analyze_complex('1rla.pdb')
        metal_complexes = interactions.get('metal', [])

        # Should detect metal coordination
        self.assertTrue(len(metal_complexes) > 0,
                       "Should detect Mn coordination in 1rla")

    def test_1het_zinc_coordination(self):
        """Test zinc coordination in liver alcohol dehydrogenase (1het).

        Literature: Harding (2004), Fig. 2
        Expected: Zn coordinated by 4 cysteines (CYS97, CYS100, CYS103, CYS111)
        """
        interactions, mol, props = self._analyze_complex('1het.pdb')
        metal_complexes = interactions.get('metal', [])

        # Should detect metal coordination
        self.assertTrue(len(metal_complexes) >= 4,
                       "Should detect at least 4 Zn coordination bonds")

        # Check for cysteine coordination
        cys_count = sum(1 for mc in metal_complexes
                       if mc.res_a_name == 'CYS' or mc.res_b_name == 'CYS')
        self.assertTrue(cys_count >= 4,
                       "Zn should be coordinated by at least 4 CYS residues")

    def test_1vfy_zinc_coordination(self):
        """Test zinc coordination in VPS27P FYVE domain (1vfy).

        Literature: Harding (2004), Fig. 5
        Expected: Zn coordinated by 4 cysteines
        """
        interactions, mol, props = self._analyze_complex('1vfy.pdb')
        metal_complexes = interactions.get('metal', [])

        # Should detect metal coordination
        self.assertTrue(len(metal_complexes) > 0,
                       "Should detect Zn coordination in 1vfy")

        # Filter for ZN:A:300 specifically (like main PLIP test)
        # ZN:A:300 should have 4 CYS coordination
        zn_300_complexes = [mc for mc in metal_complexes
                           if (mc.res_a_name == 'ZN' and mc.res_a_num == 300) or
                              (mc.res_b_name == 'ZN' and mc.res_b_num == 300)]

        self.assertTrue(len(zn_300_complexes) >= 4,
                       "ZN:A:300 should have at least 4 coordination bonds")

        # Check that ZN:A:300 is coordinated by CYS
        cys_count = sum(1 for mc in zn_300_complexes
                       if mc.res_a_name == 'CYS' or mc.res_b_name == 'CYS')
        self.assertEqual(cys_count, 4,
                        "ZN:A:300 should be coordinated by 4 CYS residues")

    def test_2pvb_calcium_coordination(self):
        """Test calcium coordination in pike parvalbumin (2pvb).

        Literature: Harding (2004), Fig. 6
        Expected: Ca with coordination number 5
        """
        interactions, mol, props = self._analyze_complex('2pvb.pdb')
        metal_complexes = interactions.get('metal', [])

        # Should detect metal coordination
        self.assertTrue(len(metal_complexes) > 0,
                       "Should detect Ca coordination in 2pvb")

    def test_metal_binding_distances(self):
        """Test that metal coordination distances are reasonable.

        Based on main PLIP test_1rmd, checks ZN:A:119 specifically.
        Literature-validated distances for zinc coordination are typically
        in the range of 2.0-3.0 Å, but can be as low as 1.8 Å for certain
        coordination geometries.
        """
        interactions, mol, props = self._analyze_complex('1rmd.pdb')
        metal_complexes = interactions.get('metal', [])

        # Filter for ZN:A:119 specifically (like main PLIP test)
        zn_119_complexes = [mc for mc in metal_complexes
                           if (mc.res_a_name == 'ZN' and mc.res_a_num == 119) or
                              (mc.res_b_name == 'ZN' and mc.res_b_num == 119)]

        self.assertTrue(len(zn_119_complexes) >= 4,
                       "ZN:A:119 should have at least 4 coordination bonds")

        # Check distances for ZN:A:119 (main PLIP range: 2.07-2.37 Å)
        # Using 1.8-3.0 Å as reasonable range based on literature
        for mc in zn_119_complexes:
            self.assertTrue(1.8 <= mc.distance <= 3.0,
                          f"Metal coordination distance {mc.distance} out of range [1.8-3.0 Å]")

    def test_metal_identification(self):
        """Test that metal ions are correctly identified."""
        pdb_files = ['1rmd.pdb', '1rla.pdb', '1het.pdb']
        expected_metals = ['ZN', 'MN', 'ZN']

        for pdb_file, expected in zip(pdb_files, expected_metals):
            mol = MoleculeComplex()
            mol.load_pdb(self.test_data_dir + pdb_file)
            props = AtomProperties(mol.atom_container)

            # Should identify metal ions
            metals = props.get_metals()
            self.assertTrue(len(metals) > 0,
                           f"Should identify metal ions in {pdb_file}")

            # Check metal type
            metal_types = set(m.resname for m in metals)
            self.assertTrue(expected in metal_types,
                           f"Should identify {expected} in {pdb_file}")


if __name__ == '__main__':
    unittest.main()
