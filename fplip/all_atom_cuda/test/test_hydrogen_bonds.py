"""
All-Atom-CUDA Module Hydrogen Bond Detection Tests

Tests for hydrogen bond detection in the all-atom-cuda module,
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

class AllAtomCUDAHydrogenBondTest(unittest.TestCase):
    """Test hydrogen bond detection in all-atom-cuda module."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = str(TEST_DIR / 'pdb') + '/'
        self.backend = get_backend()

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
        detector = CudaInteractionDetector(
            mol.atom_container, props, mol.residues,
            backend=self.backend
        )
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
        numbers of hydrogen bonds (17-18 in optimized version vs 16-17 in original).

        The optimized version uses vectorized distance calculations which may
        detect slightly different binding site atoms compared to the original
        two-step filtering approach (residue-based then atom-based).

        Note on all-atom vs main PLIP behavior:
        - See test_4dst_deterministic_protonation for detailed explanation.
        - When comparing with main PLIP, we count both confirmed and possible H-bonds.
        """
        state = config.NOHYDRO
        config.NOHYDRO = False
        for _ in range(0, 10):
            interactions, _, _ = self._analyze_complex('4dst.pdb', binding_site_id='GCP:A:202')
            # Count ligand-involved H-bonds (see test_4dst_deterministic_protonation for details)
            confirmed, possible = self._count_ligand_hbonds(interactions, 'GCP', 'A', 202)
            # Note: Due to all-atom's broader interaction detection (including protein-protein),
            # some ligand-receptor H-bonds may be marked as "possible" when the same donor
            # has stronger H-bonds to other protein residues. This is expected behavior.
            # The exact numbers may vary due to OpenBabel's non-deterministic protonation.
            # Expected range: confirmed=12-14, possible=14-18, total=26-32
            total = confirmed + possible
            self.assertTrue(26 <= total <= 32,
                          f"Expected 26-32 total hydrogen bonds, got {total} "
                          f"({confirmed} confirmed + {possible} possible)")
        config.NOHYDRO = state

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
          (Updated from 13 after unified charge detection in Scheme B.
           LYS ammonium detection improvement leads to more accurate
           salt bridge identification (GCP-LYS16, GCP-LYS147) and H-bond refinement.
           Some H-bonds are filtered due to salt bridge involvement.)
        - Possible H-bonds (ligand-protein): 18
          (Updated from 16 after unified charge detection in Scheme B.
           More H-bonds are now correctly identified and refined.
           These include H-bonds filtered by salt bridge detection and
           weaker H-bonds from donors with multiple interactions.)
        - Total ligand-protein H-bonds: 29
        - Main PLIP count: 16 (doesn't detect protein-protein interactions)

        Attribution for Update (Scheme B - Unified Charge Detection):
        1. LYS NZ atoms are now correctly marked as 'ammonium' instead of 'tertamine',
           leading to more accurate salt bridge detection (GCP-LYS16, GCP-LYS147).
        2. Salt bridges filter out H-bonds where donor/acceptor atoms are
           involved in the salt bridge, reducing confirmed count from 13 to 11.
        3. More comprehensive detection of possible H-bonds reflects all-atom's
           goal of providing complete interaction information.
        4. The change in counts reflects improved chemical accuracy in charge
           detection, not a regression in detection capability.

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
            # Updated expectations after Scheme B (unified charge detection):
            # - confirmed=11 (was 13): Improved salt bridge detection filters some H-bonds
            # - possible=18 (was 16): More comprehensive detection of possible interactions
            # - total=29 (unchanged): Overall detection accuracy maintained
            total = confirmed + possible
            self.assertEqual(confirmed, 11,
                           f"Expected 11 confirmed hydrogen bonds, got {confirmed}")
            self.assertEqual(possible, 18,
                           f"Expected 18 possible hydrogen bonds, got {possible}")
            self.assertEqual(total, 29,
                           f"Expected 29 total hydrogen bonds, got {total}")

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
          (Updated from 7 after unified charge detection in Scheme B.
           Improved salt bridge detection (ARG67-DTF, ARG86-DTF, HIS107-DTF)
           leads to more H-bonds being filtered or marked as "possible".
           The 3 confirmed H-bonds are the strongest ones per donor after
           salt bridge filtering and donor uniqueness enforcement.)
        - Possible H-bonds (ligand-protein): 8
          (Updated from 4 after unified charge detection in Scheme B.
           More H-bonds are now correctly identified and refined.
           These include H-bonds filtered by salt bridge detection and
           weaker H-bonds from donors with multiple interactions.)
        - Total ligand-protein H-bonds: 11

        Attribution for Update (Scheme B - Unified Charge Detection):
        1. Improved guanidinium detection in ARG residues leads to more accurate
           salt bridge identification (ARG67-DTF, ARG86-DTF).
        2. Salt bridges filter out H-bonds where donor/acceptor atoms are
           involved in the salt bridge, reducing confirmed count from 7 to 3.
        3. More comprehensive detection of possible H-bonds reflects all-atom's
           goal of providing complete interaction information.
        4. The change from 7 confirmed to 3 confirmed is due to improved
           salt bridge detection and stricter refinement, not a loss of
           detection capability.

        Note: The exact numbers reflect all-atom's comprehensive interaction
        detection and may differ from main PLIP which only considers
        ligand-receptor interactions.
        """
        interactions1, _, _ = self._analyze_complex('1x0n_state_1.pdb', binding_site_id='DTF:A:174')
        # Count ligand-involved H-bonds (both confirmed and possible)
        confirmed, possible = self._count_ligand_hbonds(interactions1, 'DTF', 'A', 174)
        total = confirmed + possible

        # Expected values for 1x0n_state_1.pdb with DTF:A:174
        # Updated after Scheme B (unified charge detection):
        # - confirmed=3 (was 7): Improved salt bridge detection filters more H-bonds
        # - possible=8 (was 4): More comprehensive detection of possible interactions
        # - total=11 (unchanged): Overall detection accuracy maintained
        #
        # Salt bridges affecting refinement:
        # - ARG67 <-> DTF174
        # - ARG86 <-> DTF174
        # - HIS107 <-> DTF174
        self.assertEqual(confirmed, 3,
                        f"Expected 3 confirmed hydrogen bonds, got {confirmed}")
        self.assertEqual(possible, 8,
                        f"Expected 8 possible hydrogen bonds, got {possible}")
        self.assertEqual(total, 11,
                        f"Expected 11 total hydrogen bonds, got {total}")


if __name__ == '__main__':
    unittest.main()
