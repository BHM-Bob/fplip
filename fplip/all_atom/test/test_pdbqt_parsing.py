"""
PDBQT Parser Tests

Tests for PDBQT to PDB conversion functionality in both Python and Cython implementations.
Ensures consistent handling of AutoDock Vina PDBQT files.
"""

import unittest

from fplip.structure._pdb_parser import fix_pdbline_str
from fplip.structure.pdb import PDBParser


class TestPDBQTParsing(unittest.TestCase):
    """Test PDBQT atom type conversion"""

    def setUp(self):
        """Set up test fixtures"""
        self.parser = PDBParser.__new__(PDBParser)
        self.parser.num_fixed_lines = 0

    def test_aromatic_carbon_conversion(self):
        """Test conversion of 'A' (aromatic carbon) to 'C'"""
        line = 'ATOM    168  CG  PHE A  18      -3.629   3.400  14.009  1.00  0.00     0.011 A'
        result, _ = fix_pdbline_str(line, 167)
        self.assertIn(' C\n', result)
        self.assertNotIn(' A\n', result)

    def test_oxygen_conversion(self):
        """Test conversion of 'OA' (aromatic oxygen) to 'O'"""
        line = 'ATOM    168  CG  PHE A  18      -3.629   3.400  14.009  1.00  0.00     0.011 OA'
        result, _ = fix_pdbline_str(line, 167)
        self.assertIn(' O\n', result)
        self.assertNotIn('OA', result)

    def test_nitrogen_conversion(self):
        """Test conversion of 'NA' (aromatic nitrogen) to 'N'"""
        line = 'ATOM    168  CG  PHE A  18      -3.629   3.400  14.009  1.00  0.00     0.011 NA'
        result, _ = fix_pdbline_str(line, 167)
        self.assertIn(' N\n', result)
        self.assertNotIn('NA', result)

    def test_sulfur_conversion(self):
        """Test conversion of 'SA' (aromatic sulfur) to 'S'"""
        line = 'ATOM    168  CG  PHE A  18      -3.629   3.400  14.009  1.00  0.00     0.011 SA'
        result, _ = fix_pdbline_str(line, 167)
        self.assertIn(' S\n', result)
        self.assertNotIn('SA', result)

    def test_hydrogen_conversion_hd(self):
        """Test conversion of 'HD' (hydrogen donor) to 'H'"""
        line = 'ATOM    168  CG  PHE A  18      -3.629   3.400  14.009  1.00  0.00     0.011 HD'
        result, _ = fix_pdbline_str(line, 167)
        self.assertIn(' H\n', result)
        self.assertNotIn('HD', result)

    def test_aromatic_carbon_with_trailing_space(self):
        """Test conversion of 'A ' (aromatic carbon with trailing space) to 'C'"""
        line = 'ATOM    168  CG  PHE A  18      -3.629   3.400  14.009  1.00  0.00     0.011 A '
        result, _ = fix_pdbline_str(line, 167)
        # Should convert 'A ' to 'C ' (preserving the trailing space)
        # Check the end of the line (PDBQT type position)
        self.assertTrue(result.rstrip().endswith(' C'), f"Expected line to end with ' C', got: {repr(result)}")

    def test_oxygen_with_trailing_space(self):
        """Test conversion of 'OA ' (aromatic oxygen with trailing space) to 'O'"""
        line = 'ATOM    168  CG  PHE A  18      -3.629   3.400  14.009  1.00  0.00     0.011 OA '
        result, _ = fix_pdbline_str(line, 167)
        # Should convert 'OA ' to 'O ' (preserving one trailing space)
        # Check the end of the line (PDBQT type position)
        self.assertTrue(result.rstrip().endswith(' O'), f"Expected line to end with ' O', got: {repr(result)}")

    def test_no_conversion_for_standard_carbon(self):
        """Test that standard 'C' is not modified"""
        line = 'ATOM    168  CG  PHE A  18      -3.629   3.400  14.009  1.00  0.00          C'
        result, _ = fix_pdbline_str(line, 167)
        self.assertIn(' C\n', result)

    def test_no_conversion_for_calcium(self):
        """Test that calcium 'CA' is not mistakenly converted to 'C'
        
        This is a regression test for a bug where the PDBQT pattern 'A' (aromatic carbon)
        would match the 'A' in 'CA' (calcium element symbol), causing calcium atoms to be
        incorrectly identified as carbon atoms.
        """
        line = 'HETATM  878 CA    CA A 110      17.153  34.014  11.880  1.00  5.67          CA  '
        result, _ = fix_pdbline_str(line, 877)
        # Calcium should remain as 'CA', not be converted to 'C'
        self.assertIn('CA', result)
        self.assertNotIn(' C  \n', result)
        # Check the element symbol column (76-77) still contains 'CA'
        self.assertIn('CA', result[76:78])

    def test_python_cython_consistency_basic(self):
        """Test that Python and Cython implementations produce consistent results"""
        test_lines = [
            'ATOM    168  CG  PHE A  18      -3.629   3.400  14.009  1.00  0.00     0.011 A',
            'ATOM    168  CG  PHE A  18      -3.629   3.400  14.009  1.00  0.00     0.011 OA',
            'ATOM    168  CG  PHE A  18      -3.629   3.400  14.009  1.00  0.00     0.011 NA',
            'ATOM    168  CG  PHE A  18      -3.629   3.400  14.009  1.00  0.00     0.011 HD',
        ]

        for line in test_lines:
            py_result, _ = self.parser._fix_pdbline_python(line, 167)
            cy_result, _ = fix_pdbline_str(line, 167)

            # Both should convert the PDBQT type to standard element
            # (allowing for minor whitespace differences)
            py_element = py_result.strip()[-1] if py_result else None
            cy_element = cy_result.strip()[-1] if cy_result else None
            self.assertEqual(py_element, cy_element,
                           f"Element mismatch for line: {line}")

    def test_atom_numbering_update(self):
        """Test that atom numbering is correctly updated"""
        line = 'ATOM    200  CG  PHE A  18      -3.629   3.400  14.009  1.00  0.00     0.011 A'
        _, new_num = fix_pdbline_str(line, 167)
        # new_num should be 168 (lastnum + 1)
        self.assertEqual(new_num, 168)

    def test_hetatm_pdbqt_conversion(self):
        """Test PDBQT conversion for HETATM records"""
        line = 'HETATM    1  C1  LIG Z   1      -3.629   3.400  14.009  1.00  0.00     0.011 A'
        result, _ = fix_pdbline_str(line, 0)
        self.assertIn(' C\n', result)
        self.assertNotIn(' A\n', result)


class TestPDBQTParsingEdgeCases(unittest.TestCase):
    """Test edge cases for PDBQT parsing"""

    def test_empty_line(self):
        """Test handling of empty lines"""
        line = ''
        result, _ = fix_pdbline_str(line, 0)
        # Empty lines return None or newline depending on implementation
        self.assertIn(result, [None, '\n', ''])

    def test_short_line(self):
        """Test handling of very short lines"""
        line = 'ATOM'
        result, _ = fix_pdbline_str(line, 0)
        self.assertIn('ATOM', result)

    def test_no_pdbqt_type(self):
        """Test lines without PDBQT types are not modified"""
        line = 'ATOM    168  CG  PHE A  18      -3.629   3.400  14.009  1.00  0.00           '
        result, _ = fix_pdbline_str(line, 167)
        # Should be returned as-is (with newline added)
        self.assertIn('ATOM', result)


if __name__ == '__main__':
    unittest.main()
