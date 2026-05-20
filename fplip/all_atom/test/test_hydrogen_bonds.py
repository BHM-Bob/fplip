"""
All-Atom Module Hydrogen Bond Detection Tests

Tests for hydrogen bond detection in the all-atom module,
aligned with original PLIP test cases.
"""

import unittest
from pathlib import Path

from fplip.all_atom.atom_properties import AtomProperties
from fplip.all_atom.interaction_detector import UnifiedInteractionDetector
from fplip.all_atom.molecule_complex import MoleculeComplex

TEST_DIR = Path(__file__).parent.parent.parent / 'test'

class AllAtomHydrogenBondTest(unittest.TestCase):
    """Test hydrogen bond detection in all-atom module."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = str(TEST_DIR / 'pdb') + '/'

    def _parse_binding_site_id(self, binding_site_id: str):
        """Parse binding site ID in format 'RESNAME:CHAIN:NUMBER'.
        
        Args:
            binding_site_id: String in format 'GCP:A:202'
            
        Returns:
            Tuple of (resname, chain, resnum)
        """
        parts = binding_site_id.split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid binding_site_id format: {binding_site_id}. Expected 'RESNAME:CHAIN:NUMBER'")
        return parts[0], parts[1], int(parts[2])

    def _is_protein_residue(self, res_name: str) -> bool:
        """Check if a residue is a protein residue (standard amino acids).
        
        Args:
            res_name: Residue name
            
        Returns:
            True if it's a standard amino acid, False otherwise
        """
        protein_residues = {
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
        }
        return res_name in protein_residues

    def _filter_interactions_by_ligand(self, interactions: dict, ligand_resname: str,
                                       ligand_chain: str, ligand_resnum: int):
        """Filter interactions to only include protein-ligand interactions.

        This mimics the behavior of original PLIP which only considers interactions
        between the ligand and protein binding site residues (excluding water).

        Args:
            interactions: Dictionary of interaction lists from detector.detect_all()
            ligand_resname: Ligand residue name (e.g., 'GCP')
            ligand_chain: Ligand chain (e.g., 'A')
            ligand_resnum: Ligand residue number (e.g., 202)

        Returns:
            Filtered interactions dictionary containing only protein-ligand interactions
        """
        filtered = {}
        for interaction_type, interaction_list in interactions.items():
            filtered_list = []
            for interaction in interaction_list:
                # Check if either side involves the ligand
                is_ligand_a = (interaction.res_a_name == ligand_resname and
                              interaction.res_a_chain == ligand_chain and
                              interaction.res_a_num == ligand_resnum)
                is_ligand_b = (interaction.res_b_name == ligand_resname and
                              interaction.res_b_chain == ligand_chain and
                              interaction.res_b_num == ligand_resnum)

                # Only keep interactions where one side is the ligand and the other is protein
                if is_ligand_a or is_ligand_b:
                    # Determine the other residue
                    if is_ligand_a:
                        other_res_name = interaction.res_b_name
                    else:
                        other_res_name = interaction.res_a_name

                    # Only include if the other residue is a protein residue
                    if self._is_protein_residue(other_res_name):
                        filtered_list.append(interaction)
            filtered[interaction_type] = filtered_list
        return filtered

    def _count_ligand_hbonds(self, interactions: dict, ligand_resname: str,
                             ligand_chain: str, ligand_resnum: int) -> tuple:
        """Count hydrogen bonds involving the ligand.

        This method counts ligand-involved hydrogen bonds, returning both confirmed
        and possible counts separately. This reflects all-atom's design of detecting
        ALL interactions (including protein-protein) and marking weaker ones as
        "possible" when a donor forms multiple H-bonds.

        Note: Due to all-atom's broader interaction detection (including protein-protein
        and protein-water), some ligand-receptor H-bonds may be marked as "possible"
        when the same donor has a stronger H-bond to another protein residue or water.
        This is expected behavior and differs from main PLIP which only considers
        ligand-receptor interactions.

        Args:
            interactions: Dictionary of interaction lists from detector.detect_all()
            ligand_resname: Ligand residue name (e.g., 'GCP')
            ligand_chain: Ligand chain (e.g., 'A')
            ligand_resnum: Ligand residue number (e.g., 202)

        Returns:
            Tuple of (confirmed_count, possible_count) for ligand-involved H-bonds
        """
        def is_ligand_protein_hbond(hbond):
            """Check if H-bond is between ligand and protein."""
            is_ligand_a = (hbond.res_a_name == ligand_resname and
                          hbond.res_a_chain == ligand_chain and
                          hbond.res_a_num == ligand_resnum)
            is_ligand_b = (hbond.res_b_name == ligand_resname and
                          hbond.res_b_chain == ligand_chain and
                          hbond.res_b_num == ligand_resnum)

            if is_ligand_a or is_ligand_b:
                other_res_name = hbond.res_b_name if is_ligand_a else hbond.res_a_name
                return self._is_protein_residue(other_res_name)
            return False

        confirmed = [hb for hb in interactions.get('hbond', []) if is_ligand_protein_hbond(hb)]
        possible = [hb for hb in interactions.get('hbond_possible', []) if is_ligand_protein_hbond(hb)]

        return len(confirmed), len(possible)

    def _analyze_complex(self, pdb_file: str, binding_site_id: str = None):
        """Helper method to analyze a PDB file.
        
        Args:
            pdb_file: PDB file name
            binding_site_id: Optional binding site ID in format 'RESNAME:CHAIN:NUMBER'
                           If provided, only interactions involving this ligand are returned.
        """
        mol = MoleculeComplex()
        mol.load_pdb(self.test_data_dir + pdb_file)
        props = AtomProperties(mol.atom_container)
        detector = UnifiedInteractionDetector(mol.atom_container, props, mol.residues)
        interactions = detector.detect_all()
        
        # Filter by ligand if binding_site_id is provided
        if binding_site_id:
            ligand_resname, ligand_chain, ligand_resnum = self._parse_binding_site_id(binding_site_id)
            interactions = self._filter_interactions_by_ligand(
                interactions, ligand_resname, ligand_chain, ligand_resnum
            )
        
        return interactions, mol, props

    def test_4dst_nondeterministic_protonation(self):
        """Test hydrogen bond detection with automatic protonation.

        Note: This test uses NOHYDRO=False, which allows OpenBabel to automatically
        add hydrogen atoms. Due to the nature of OpenBabel's hydrogen addition
        algorithm, the results can be non-deterministic, leading to varying
        numbers of hydrogen bonds.

        Attribution for Update (Positively Charged N Filter):
        1. LYS NZ and ARG NE/NH1/NH2 atoms are now correctly marked as positively charged
        2. Positively charged N atoms cannot be H-bond acceptors (no lone pairs available)
        3. This reduces the number of possible H-bonds significantly
        4. The change reflects chemically correct behavior, not a regression
        5. In 4dst, LYS16, LYS147, and several ARG residues lose their acceptor capability

        Attribution for Previous Update (Scheme B - Unified Charge Detection):
        1. LYS NZ atoms are now correctly marked as 'ammonium' instead of 'tertamine'
        2. This leads to more accurate salt bridge identification
        3. Salt bridges filter out H-bonds where donor/acceptor atoms are involved

        Note on all-atom vs main PLIP behavior:
        - See test_4dst_deterministic_protonation for detailed explanation.
        - When comparing with main PLIP, we count both confirmed and possible H-bonds.
        """
        for _ in range(0, 10):
            interactions, _, _ = self._analyze_complex('4dst.pdb', binding_site_id='GCP:A:202')
            # Count ligand-involved H-bonds (see test_4dst_deterministic_protonation for details)
            confirmed, possible = self._count_ligand_hbonds(interactions, 'GCP', 'A', 202)
            # Note: Due to all-atom's broader interaction detection (including protein-protein),
            # some ligand-receptor H-bonds may be marked as "possible" when the same donor
            # has stronger H-bonds to other protein residues. This is expected behavior.
            # The exact numbers may vary due to OpenBabel's non-deterministic protonation.
            #
            # Original expectations (before any updates):
            # - confirmed=12-14, possible=14-18, total=26-32
            #
            # After Scheme B (unified charge detection):
            # - confirmed=12-14, possible=14-18, total=26-32 (unchanged)
            #
            # Updated expectations after positively charged N filter:
            # - confirmed=10-11: Slightly reduced due to improved charge detection
            # - possible=9: Significantly reduced (was 14-18) due to positively charged N filter
            # - total=19-20: Reduced from 26-32 due to chemically correct acceptor filtering
            #
            # Change breakdown:
            # - Original total: 26-32 (confirmed 12-14 + possible 14-18)
            # - New total: 19-21 (confirmed 10-11 + possible 9)
            # - Reduction: ~7-11 H-bonds lost due to positively charged N filter
            total = confirmed + possible
            self.assertTrue(19 <= total <= 21,
                          f"Expected 19-21 total hydrogen bonds, got {total} "
                          f"({confirmed} confirmed + {possible} possible). "
                          f"The reduction is due to positively charged Lys/Arg N atoms "
                          f"no longer being H-bond acceptors.")

    def test_4dst_deterministic_protonation(self):
        """Test hydrogen bond detection with pre-protonated structure.

        This test uses NOHYDRO=True and a pre-protonated PDB file, which ensures
        deterministic results as no automatic hydrogen addition is performed.

        All-atom Design Philosophy:
        - All-atom detects ALL interactions (protein-ligand, protein-protein,
          protein-water, etc.), not just ligand-receptor interactions like main PLIP.
        - H-bond refinement follows PLIP's rule: keep only the strongest H-bond
          per donor (largest angle). Weaker H-bonds from the same donor are
          marked as "possible" rather than discarded.
        - This provides more comprehensive information but may result in fewer
          "confirmed" ligand-receptor H-bonds when the donor has stronger H-bonds
          to other residues (e.g., protein-protein or protein-water).

        Expected Results for 4dst_protonated.pdb with GCP:A:202:
        - Confirmed H-bonds (ligand-protein): 11
          (Unchanged from previous update. Salt bridge detection still filters
           some H-bonds involving GCP-LYS16 and GCP-LYS147.)
        - Possible H-bonds (ligand-protein): 9
          (Updated from 18 after positively charged N filter.
           LYS NZ and ARG NE/NH1/NH2 atoms are no longer H-bond acceptors.)
        - Total ligand-protein H-bonds: 20
          (Updated from 29 due to chemically correct acceptor filtering.)
        - Main PLIP count: 16 (doesn't detect protein-protein interactions)

        Attribution for Update (Positively Charged N Filter):
        1. LYS NZ and ARG NE/NH1/NH2 atoms are now correctly marked as positively charged
        2. Positively charged N atoms cannot be H-bond acceptors (no lone pairs available)
        3. In 4dst_protonated.pdb, this affects:
           - LYS:A:16 NZ, LYS:A:147 NZ (ammonium groups)
           - Multiple ARG residues (NE, NH1, NH2 in guanidinium groups)
        4. The reduction in possible H-bonds (18 -> 9) reflects chemically correct behavior
        5. This is not a regression but an improvement in chemical accuracy

        Attribution for Previous Update (Scheme B - Unified Charge Detection):
        1. LYS NZ atoms are now correctly marked as 'ammonium' instead of 'tertamine',
           leading to more accurate salt bridge detection (GCP-LYS16, GCP-LYS147).
        2. Salt bridges filter out H-bonds where donor/acceptor atoms are
           involved in the salt bridge.

        Note: The numbers reflect ligand-protein interactions only (water excluded).
        The "possible" H-bonds include those filtered by salt bridge detection
        and weaker H-bonds from donors with multiple interactions.
        """
        for _ in range(0, 10):
            interactions, _, _ = self._analyze_complex('4dst_protonated.pdb', binding_site_id='GCP:A:202')
            # Count ligand-involved H-bonds
            confirmed, possible = self._count_ligand_hbonds(interactions, 'GCP', 'A', 202)
            # All-atom detects ALL interactions including protein-protein and
            # protein-water, so some ligand-receptor H-bonds may be marked as
            # "possible" when the same donor has stronger H-bonds to other residues.
            # This is expected all-atom behavior - it provides comprehensive
            # information about all potential interactions in the system.
            #
            # Original expectations (before any updates):
            # - confirmed=13, possible=16, total=29
            #
            # After Scheme B (unified charge detection):
            # - confirmed=11 (was 13): Improved salt bridge detection filters some H-bonds
            # - possible=18 (was 16): More comprehensive detection of possible interactions
            # - total=29 (unchanged): Overall detection accuracy maintained
            #
            # Updated expectations after positively charged N filter:
            # - confirmed=11 (unchanged): Salt bridge detection still filters same H-bonds
            # - possible=9 (was 18): Significantly reduced due to positively charged N filter
            # - total=20 (was 29): Reduced due to chemically correct acceptor filtering
            #
            # Change breakdown:
            # - Original total: 29 (confirmed 13 + possible 16)
            # - After Scheme B: 29 (confirmed 11 + possible 18)
            # - After N filter: 20 (confirmed 11 + possible 9)
            # - Total reduction: 9 H-bonds lost due to positively charged N filter
            total = confirmed + possible
            self.assertEqual(confirmed, 11,
                           f"Expected 11 confirmed hydrogen bonds, got {confirmed}")
            self.assertEqual(possible, 9,
                           f"Expected 9 possible hydrogen bonds, got {possible}. "
                           f"The reduction from 18 is due to positively charged Lys/Arg N atoms "
                           f"no longer being H-bond acceptors.")
            self.assertEqual(total, 20,
                           f"Expected 20 total hydrogen bonds, got {total}")

    def test_no_protonation(self):
        """Test ligand-donated hydrogen bonds with pre-protonated structure.

        This test uses a pre-protonated PDB file (1x0n_state_1.pdb) to ensure
        deterministic results.

        All-atom Design Philosophy:
        - All-atom detects ALL interactions, so the count includes both confirmed
          and possible H-bonds involving the ligand (DTF).
        - H-bond refinement filters out H-bonds involved in salt bridges and
          keeps only the strongest H-bond per donor.

        Expected Results for 1x0n_state_1.pdb with DTF:A:174:
        - Confirmed H-bonds (ligand-protein): 3
          (Unchanged from previous update. Salt bridge detection still filters
           H-bonds involving ARG67-DTF, ARG86-DTF, HIS107-DTF.)
        - Possible H-bonds (ligand-protein): 1
          (Updated from 8 after positively charged N filter.
           LYS NZ and ARG NE/NH1/NH2 atoms are no longer H-bond acceptors.
           This significantly reduces possible H-bonds involving these residues.)
        - Total ligand-protein H-bonds: 4
          (Updated from 11 due to chemically correct acceptor filtering.)

        Attribution for Update (Positively Charged N Filter):
        1. LYS NZ and ARG NE/NH1/NH2 atoms are now correctly marked as positively charged
        2. Positively charged N atoms cannot be H-bond acceptors (no lone pairs available)
        3. In 1x0n_state_1.pdb, this affects multiple ARG residues:
           - ARG:A:67 (NE, NH1, NH2)
           - ARG:A:86 (NE, NH1, NH2)
           - Other ARG residues in the binding site
        4. The reduction in possible H-bonds (8 -> 1) reflects chemically correct behavior
        5. This is not a regression but an improvement in chemical accuracy

        Attribution for Previous Update (Scheme B - Unified Charge Detection):
        1. Improved guanidinium detection in ARG residues leads to more accurate
           salt bridge identification (ARG67-DTF, ARG86-DTF).
        2. Salt bridges filter out H-bonds where donor/acceptor atoms are
           involved in the salt bridge.

        Note: The exact numbers reflect all-atom's comprehensive interaction
        detection and may differ from main PLIP which only considers
        ligand-receptor interactions.
        """
        interactions1, _, _ = self._analyze_complex('1x0n_state_1.pdb', binding_site_id='DTF:A:174')
        # Count ligand-involved H-bonds (both confirmed and possible)
        confirmed, possible = self._count_ligand_hbonds(interactions1, 'DTF', 'A', 174)
        total = confirmed + possible

        # Expected values for 1x0n_state_1.pdb with DTF:A:174
        #
        # Original expectations (before any updates):
        # - confirmed=7, possible=4, total=11
        #
        # After Scheme B (unified charge detection):
        # - confirmed=3 (was 7): Improved salt bridge detection filters more H-bonds
        # - possible=8 (was 4): More comprehensive detection of possible interactions
        # - total=11 (unchanged): Overall detection accuracy maintained
        #
        # Updated after positively charged N filter:
        # - confirmed=3 (unchanged): Salt bridge detection still filters same H-bonds
        # - possible=1 (was 8): Significantly reduced due to positively charged N filter
        # - total=4 (was 11): Reduced due to chemically correct acceptor filtering
        #
        # Change breakdown:
        # - Original total: 11 (confirmed 7 + possible 4)
        # - After Scheme B: 11 (confirmed 3 + possible 8)
        # - After N filter: 4 (confirmed 3 + possible 1)
        # - Total reduction: 7 H-bonds lost due to positively charged N filter
        #
        # Salt bridges affecting refinement:
        # - ARG67 <-> DTF174
        # - ARG86 <-> DTF174
        # - HIS107 <-> DTF174
        self.assertEqual(confirmed, 3,
                        f"Expected 3 confirmed hydrogen bonds, got {confirmed}")
        self.assertEqual(possible, 1,
                        f"Expected 1 possible hydrogen bond, got {possible}. "
                        f"The reduction from 8 is due to positively charged Lys/Arg N atoms "
                        f"no longer being H-bond acceptors.")
        self.assertEqual(total, 4,
                        f"Expected 4 total hydrogen bonds, got {total}")


if __name__ == '__main__':
    unittest.main()
