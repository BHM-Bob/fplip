"""
Atom Container Module

Provides unified storage and management of atom information
for all molecular components (proteins, ligands, DNA/RNA, etc.)
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from plip.basic import config


class AtomInfo:
    """Lightweight class to store atom information"""

    def __init__(self, idx: int, obatom, orig_idx: int):
        self.idx = idx
        self.obatom = obatom
        self.orig_idx = orig_idx
        self.coords = np.array([obatom.GetX(), obatom.GetY(), obatom.GetZ()])

        # Residue information (from OpenBabel)
        residue = obatom.GetResidue()
        self.residue = residue
        self.resname = residue.GetName() if residue else "UNK"
        self.resnum = residue.GetNum() if residue else 0
        self.chain = residue.GetChain() if residue else " "

        # Back-reference to Residue object (set by InteractionDetector after residue creation)
        self.residue_obj = None
        
        # Atom properties
        self.atom_type = obatom.GetType()
        self.atomic_num = obatom.GetAtomicNum()
        self.is_hydrogen = self.atomic_num == 1
        
        # Atom name from residue
        if residue:
            self.atom_name = residue.GetAtomID(obatom).strip()
        else:
            self.atom_name = self.atom_type
        
        # Component type (protein, ligand, dna, rna, water, ion)
        self.component_type = self._determine_component_type()
        
    def _determine_component_type(self) -> str:
        """Determine the type of molecular component this atom belongs to"""
        if not self.residue:
            return "unknown"
        
        res_name = self.resname.strip()
        
        # Check for water
        if self.residue.GetResidueProperty(9):  # 9 is water
            return "water"
        
        # Check for standard amino acids
        if self.residue.GetResidueProperty(0):  # 0 is amino acid
            return "protein"
        
        # Check for DNA/RNA
        if res_name in config.DNA:
            return "dna"
        if res_name in config.RNA:
            return "rna"
        
        # Check for metal ions
        if res_name in config.METAL_IONS:
            return "ion"
        
        # Everything else is considered a ligand
        return "ligand"
    
    def __repr__(self):
        return f"AtomInfo(idx={self.idx}, orig_idx={self.orig_idx}, {self.resname}:{self.chain}:{self.resnum})"


class AtomContainer:
    """
    Container for managing all atoms in a molecular structure.
    Provides unified access to atoms regardless of their component type.
    """
    
    def __init__(self):
        # All atoms storage
        self.atoms: Dict[int, AtomInfo] = {}  # idx -> AtomInfo
        self.atoms_by_orig_idx: Dict[int, AtomInfo] = {}  # orig_idx -> AtomInfo
        
        # Component-based organization
        self.component_atoms: Dict[str, List[int]] = {
            "protein": [],
            "ligand": [],
            "dna": [],
            "rna": [],
            "water": [],
            "ion": [],
            "unknown": []
        }
        
        # Residue-based organization
        ## residue-atom access is proveided in MoleculeComplex.residue_groups
        
        # Chain-based organization
        self.chain_atoms: Dict[str, Set[int]] = defaultdict(set)
        
        # Coordinates array for vectorized operations
        self.coords_array: Optional[np.ndarray] = None
        self.idx_to_array_pos: Dict[int, int] = {}
        # Array-based index mapping for O(1) lookup: idx_to_array_pos_array[ob_idx] = array_pos
        # -1 means the atom doesn't exist
        self.idx_to_array_pos_array: Optional[np.ndarray] = None
        self.array_pos_to_idx_array: Optional[np.ndarray] = None
        
    def add_atom(self, atom_info: AtomInfo):
        """Add an atom to the container"""
        self.atoms[atom_info.idx] = atom_info
        self.atoms_by_orig_idx[atom_info.orig_idx] = atom_info
        
        # Add to component sets
        self.component_atoms[atom_info.component_type].append(atom_info.idx)
        
        # Add to residue sets
        ## residue-atom access is proveided in MoleculeComplex.residue_groups
    
    def build_coordinate_array(self):
        """Build numpy array of coordinates for vectorized operations"""
        sorted_indices = sorted(self.atoms.keys())
        self.coords_array = np.array([self.atoms[idx].coords for idx in sorted_indices])
        self.idx_to_array_pos = {idx: i for i, idx in enumerate(sorted_indices)}
        
        # Build array-based index mapping for O(1) lookup
        if sorted_indices:
            max_idx = max(sorted_indices)
            self.idx_to_array_pos_array = np.full(max_idx + 1, -1, dtype=np.int32)
            for array_pos, ob_idx in enumerate(sorted_indices):
                self.idx_to_array_pos_array[ob_idx] = array_pos
        # Build array-based index mapping for arr-pos-idx to ob-idx
        self.array_pos_to_idx_array = np.array(sorted_indices)
    
    def get_atoms_by_component(self, component_type: str) -> List[AtomInfo]:
        """Get all atoms of a specific component type"""
        return [self.atoms[idx] for idx in sorted(self.component_atoms.get(component_type, set()))]
    
    def get_heavy_atoms(self) -> List[AtomInfo]:
        """Get all non-hydrogen atoms"""
        return [self.atoms[idx] for idx in sorted(self.atoms.keys()) if not self.atoms[idx].is_hydrogen]
    
    def get_atom_coords_array(self, indices: Optional[List[int]] = None) -> Optional[np.ndarray]:
        """Get coordinates array for specified atoms (or all atoms if None)"""
        if indices is None:
            return self.coords_array

        positions = [self.idx_to_array_pos[idx] for idx in indices if idx in self.idx_to_array_pos]
        if not positions:
            return None
        return self.coords_array[positions]
    
    def get_atom_coords_array_from_atoms(self, atoms: List[AtomInfo]) -> Optional[np.ndarray]:
        """Get coordinates array for specified atoms"""
        return self.get_atom_coords_array([atom.idx for atom in atoms])
    
    def __len__(self):
        return len(self.atoms)
    
    def __iter__(self):
        return iter([self.atoms[idx] for idx in sorted(self.atoms.keys())])
    
    def __getitem__(self, idx: int) -> AtomInfo:
        return self.atoms[idx]
