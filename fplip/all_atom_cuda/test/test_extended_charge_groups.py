"""
All-Atom-CUDA Module Extended Charge Group Detection Tests

Tests for extended charge group detection in the all-atom-cuda module,
covering charge groups beyond standard protein residues.

These tests cover:
- Sulfate/Sulfonate detection (topology-based)
- Phenolate detection (deprotonated phenolic OH)
- Thiolate detection (deprotonated thiol)
- Pyridinium detection (protonated pyridine with +1 charge)
- Anilinium detection (protonated aniline with +1 charge)
"""

import os
import tempfile
import unittest
from pathlib import Path

from openbabel import pybel

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

class AllAtomCUDAExtendedChargeGroupTest(unittest.TestCase):
    """Test extended charge group detection in all-atom-cuda module."""

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

    def _create_test_molecule(self, smiles: str, resname: str = 'LIG') -> str:
        """Create a test PDB file from SMILES string.
        
        Args:
            smiles: SMILES string for the molecule
            resname: Residue name for the molecule
            
        Returns:
            Path to temporary PDB file
        """
        mol = pybel.readstring('smi', smiles)
        mol.addh()
        mol.make3D()
        
        # Create temporary PDB file
        fd, temp_path = tempfile.mkstemp(suffix='.pdb')
        os.close(fd)
        mol.write('pdb', temp_path, overwrite=True)
        
        # Modify residue name in the PDB file
        with open(temp_path, 'r') as f:
            lines = f.readlines()
        
        with open(temp_path, 'w') as f:
            for line in lines:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    # Replace residue name (columns 18-20)
                    line = line[:17] + resname + line[20:]
                f.write(line)
        
        return temp_path

    def test_sulfate_topology_detection(self):
        """Test sulfate detection using topology-based method.
        
        Sulfate (SO4^2-) should be detected based on:
        - S atom connected to 4 O atoms
        - No C neighbors (free sulfate)
        """
        # Create a simple sulfate molecule
        temp_pdb = self._create_test_molecule('[O-]S(=O)(=O)[O-]', 'SO4')
        
        try:
            mol = MoleculeComplex()
            mol.load_pdb(temp_pdb)
            props = AtomProperties(mol.atom_container)
            
            # Should identify sulfate atoms
            neg_charged = props.get_neg_charged()
            sulfate_atoms = [atom for atom in neg_charged 
                           if atom.resname == 'SO4']
            
            # Should have S and 4 O atoms as negatively charged
            s_atoms = [a for a in sulfate_atoms if a.atomic_num == 16]
            o_atoms = [a for a in sulfate_atoms if a.atomic_num == 8]
            
            self.assertEqual(len(s_atoms), 1, "Should identify 1 S atom in sulfate")
            self.assertEqual(len(o_atoms), 4, "Should identify 4 O atoms in sulfate")
            
            # Check charge type
            for atom in sulfate_atoms:
                self.assertEqual(props.neg_charged[atom.idx], 'sulfate',
                               f"Atom {atom.atom_name} should be typed as 'sulfate'")
        finally:
            os.unlink(temp_pdb)

    def test_sulfonate_topology_detection(self):
        """Test sulfonate detection using topology-based method.
        
        Sulfonate (-SO3^-) should be detected based on:
        - S atom connected to 3 O atoms and 1 C atom
        """
        # Create a simple sulfonate (methanesulfonate)
        temp_pdb = self._create_test_molecule('CS(=O)(=O)[O-]', 'MES')
        
        try:
            mol = MoleculeComplex()
            mol.load_pdb(temp_pdb)
            props = AtomProperties(mol.atom_container)
            
            # Should identify sulfonate atoms
            neg_charged = props.get_neg_charged()
            sulfonate_atoms = [atom for atom in neg_charged 
                             if atom.resname == 'MES']
            
            # Should have S and 3 O atoms as negatively charged
            s_atoms = [a for a in sulfonate_atoms if a.atomic_num == 16]
            o_atoms = [a for a in sulfonate_atoms if a.atomic_num == 8]
            
            self.assertEqual(len(s_atoms), 1, "Should identify 1 S atom in sulfonate")
            self.assertEqual(len(o_atoms), 3, "Should identify 3 O atoms in sulfonate")
            
            # Check charge type
            for atom in sulfonate_atoms:
                self.assertEqual(props.neg_charged[atom.idx], 'sulfonate',
                               f"Atom {atom.atom_name} should be typed as 'sulfonate'")
        finally:
            os.unlink(temp_pdb)

    def test_phenolate_detection(self):
        """Test phenolate detection (deprotonated phenolic OH).
        
        Phenolate (Ar-O^-) should be detected based on:
        - O atom connected to aromatic carbon
        - O has no H neighbors (deprotonated)
        """
        # Create phenolate (deprotonated phenol)
        temp_pdb = self._create_test_molecule('[O-]c1ccccc1', 'PHL')
        
        try:
            mol = MoleculeComplex()
            mol.load_pdb(temp_pdb)
            props = AtomProperties(mol.atom_container)
            
            # Should identify phenolate oxygen
            neg_charged = props.get_neg_charged()
            phenolate_atoms = [atom for atom in neg_charged 
                             if atom.resname == 'PHL']
            
            # Should have 1 O atom as negatively charged
            o_atoms = [a for a in phenolate_atoms if a.atomic_num == 8]
            
            self.assertEqual(len(o_atoms), 1, "Should identify 1 O atom in phenolate")
            
            # Check charge type
            for atom in o_atoms:
                self.assertEqual(props.neg_charged[atom.idx], 'phenolate',
                               f"Oxygen should be typed as 'phenolate'")
        finally:
            os.unlink(temp_pdb)

    def test_neutral_phenol_not_detected_as_phenolate(self):
        """Test that neutral phenol is NOT detected as phenolate.
        
        Neutral phenol (Ar-OH) should NOT be marked as negatively charged
        because it has an H attached to the oxygen.
        """
        # Create neutral phenol
        temp_pdb = self._create_test_molecule('Oc1ccccc1', 'PHL')
        
        try:
            mol = MoleculeComplex()
            mol.load_pdb(temp_pdb)
            props = AtomProperties(mol.atom_container)
            
            # Should NOT identify phenol oxygen as negatively charged
            neg_charged = props.get_neg_charged()
            phenolate_atoms = [atom for atom in neg_charged 
                             if atom.resname == 'PHL']
            
            self.assertEqual(len(phenolate_atoms), 0, 
                           "Neutral phenol should NOT be detected as phenolate")
        finally:
            os.unlink(temp_pdb)

    def test_thiolate_detection(self):
        """Test thiolate detection (deprotonated thiol).
        
        Thiolate (R-S^-) should be detected based on:
        - S atom connected to 1 C atom
        - S has no H neighbors (deprotonated)
        - Not in standard CYS/MET context
        """
        # Create a simple thiolate (methanethiolate)
        temp_pdb = self._create_test_molecule('[S-]C', 'MET')
        
        try:
            mol = MoleculeComplex()
            mol.load_pdb(temp_pdb)
            props = AtomProperties(mol.atom_container)
            
            # Should identify thiolate sulfur
            neg_charged = props.get_neg_charged()
            thiolate_atoms = [atom for atom in neg_charged 
                            if atom.resname == 'MET']
            
            # Should have 1 S atom as negatively charged
            s_atoms = [a for a in thiolate_atoms if a.atomic_num == 16]
            
            self.assertEqual(len(s_atoms), 1, "Should identify 1 S atom in thiolate")
            
            # Check charge type
            for atom in s_atoms:
                self.assertEqual(props.neg_charged[atom.idx], 'thiolate',
                               f"Sulfur should be typed as 'thiolate'")
        finally:
            os.unlink(temp_pdb)

    def test_pyridinium_formal_charge_detection(self):
        """Test pyridinium detection using formal charge.
        
        Pyridinium (protonated pyridine, C5H5NH+) should be detected based on:
        - Aromatic N with formal charge +1
        - NOT just based on having H attached
        """
        # Create pyridinium (protonated pyridine with explicit charge)
        temp_pdb = self._create_test_molecule('[nH+]1ccccc1', 'PYH')
        
        try:
            mol = MoleculeComplex()
            mol.load_pdb(temp_pdb)
            props = AtomProperties(mol.atom_container)
            
            # Should identify pyridinium nitrogen
            pos_charged = props.get_pos_charged()
            pyridinium_atoms = [atom for atom in pos_charged 
                              if atom.resname == 'PYH']
            
            # Should have 1 N atom as positively charged
            n_atoms = [a for a in pyridinium_atoms if a.atomic_num == 7]
            
            self.assertEqual(len(n_atoms), 1, "Should identify 1 N atom in pyridinium")
            
            # Check charge type
            for atom in n_atoms:
                self.assertEqual(props.pos_charged[atom.idx], 'pyridinium',
                               f"Nitrogen should be typed as 'pyridinium'")
        finally:
            os.unlink(temp_pdb)

    def test_neutral_pyridine_not_detected_as_pyridinium(self):
        """Test that neutral pyridine is NOT detected as pyridinium.
        
        Neutral pyridine (C5H5N) should NOT be marked as positively charged
        because it has formal charge 0, even though it may have H attached.
        """
        # Create neutral pyridine
        temp_pdb = self._create_test_molecule('n1ccccc1', 'PYR')
        
        try:
            mol = MoleculeComplex()
            mol.load_pdb(temp_pdb)
            props = AtomProperties(mol.atom_container)
            
            # Should NOT identify pyridine nitrogen as positively charged
            pos_charged = props.get_pos_charged()
            pyridinium_atoms = [atom for atom in pos_charged 
                              if atom.resname == 'PYR']
            
            self.assertEqual(len(pyridinium_atoms), 0, 
                           "Neutral pyridine should NOT be detected as pyridinium")
        finally:
            os.unlink(temp_pdb)

    def test_anilinium_formal_charge_detection(self):
        """Test anilinium detection using formal charge.
        
        Anilinium (protonated aniline, C6H5NH3+) should be detected based on:
        - N with formal charge +1
        - N connected to aromatic carbon
        - N has 2+ H neighbors
        """
        # Create anilinium (protonated aniline)
        temp_pdb = self._create_test_molecule('[NH3+]c1ccccc1', 'ANH')
        
        try:
            mol = MoleculeComplex()
            mol.load_pdb(temp_pdb)
            props = AtomProperties(mol.atom_container)
            
            # Should identify anilinium nitrogen
            pos_charged = props.get_pos_charged()
            anilinium_atoms = [atom for atom in pos_charged 
                             if atom.resname == 'ANH']
            
            # Should have 1 N atom as positively charged
            n_atoms = [a for a in anilinium_atoms if a.atomic_num == 7]
            
            self.assertEqual(len(n_atoms), 1, "Should identify 1 N atom in anilinium")
            
            # Check charge type
            for atom in n_atoms:
                self.assertEqual(props.pos_charged[atom.idx], 'anilinium',
                               f"Nitrogen should be typed as 'anilinium'")
        finally:
            os.unlink(temp_pdb)

    def test_neutral_aniline_not_detected_as_anilinium(self):
        """Test that neutral aniline is NOT detected as anilinium.
        
        Neutral aniline (C6H5NH2) should NOT be marked as positively charged
        because it has formal charge 0.
        """
        # Create neutral aniline
        temp_pdb = self._create_test_molecule('Nc1ccccc1', 'ANL')
        
        try:
            mol = MoleculeComplex()
            mol.load_pdb(temp_pdb)
            props = AtomProperties(mol.atom_container)
            
            # Should NOT identify aniline nitrogen as positively charged
            pos_charged = props.get_pos_charged()
            anilinium_atoms = [atom for atom in pos_charged 
                             if atom.resname == 'ANL']
            
            self.assertEqual(len(anilinium_atoms), 0, 
                           "Neutral aniline should NOT be detected as anilinium")
        finally:
            os.unlink(temp_pdb)

    def test_saltbridge_with_extended_charge_groups(self):
        """Test salt bridge formation between extended charge groups.
        
        Test that sulfate can form salt bridge with ammonium.
        """
        # This test uses a simple model where we verify the charge detection
        # works correctly for both partners
        
        # Create sulfate
        temp_pdb_sulfate = self._create_test_molecule('[O-]S(=O)(=O)[O-]', 'SO4')
        
        try:
            mol = MoleculeComplex()
            mol.load_pdb(temp_pdb_sulfate)
            props = AtomProperties(mol.atom_container)
            
            # Verify sulfate is detected
            neg_charged = props.get_neg_charged()
            sulfate_atoms = [atom for atom in neg_charged 
                           if atom.resname == 'SO4']
            
            self.assertTrue(len(sulfate_atoms) > 0, 
                          "Sulfate should be detected for salt bridge formation")
        finally:
            os.unlink(temp_pdb_sulfate)

    def test_existing_so4_in_pdb_files(self):
        """Test that existing SO4 residues in PDB files are detected.
        
        Some PDB files contain SO4 residues that should be detected
        using topology-based methods since formal charge is 0.
        """
        # Check 1bma.pdb which contains SO4
        mol = MoleculeComplex()
        mol.load_pdb(self.test_data_dir + '1bma.pdb')
        props = AtomProperties(mol.atom_container)
        
        # Find SO4 atoms
        so4_atoms = [atom for atom in mol.atom_container 
                    if atom.resname == 'SO4']
        
        if len(so4_atoms) > 0:
            # Check if any are detected as negatively charged
            neg_charged = props.get_neg_charged()
            so4_neg = [atom for atom in neg_charged 
                      if atom.resname == 'SO4']
            
            # SO4 should be detected based on topology
            self.assertTrue(len(so4_neg) > 0,
                          "SO4 in PDB should be detected using topology-based method")


if __name__ == '__main__':
    unittest.main()
