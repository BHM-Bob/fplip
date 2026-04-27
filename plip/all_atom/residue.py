"""
Residue Module

Defines the Residue class as the basic unit for interaction detection.
Each residue contains atoms and their aggregated properties.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from plip.basic import config


class Residue:
    """
    Residue class representing a group of atoms.
    
    For proteins: standard amino acid residue
    For ligands: the entire small molecule as one residue
    For DNA/RNA: each nucleotide as a residue
    For ions: each ion as a residue
    For water: each water molecule as a residue
    """
    
    def __init__(self, resname: str, chain: str, resnum: int):
        self.resname = resname
        self.chain = chain
        self.resnum = resnum
        
        # Unique identifier
        self.resid = f"{resname}:{chain}:{resnum}"
        
        # Atoms in this residue
        self.atoms: List = []
        
        # Geometric center
        self.center: Optional[np.ndarray] = None
        
        # Residue type flags
        self.is_protein = False
        self.is_peptide = False
        self.is_ligand = False
        self.is_water = False
        self.is_ion = False
        self.is_dna = False
        self.is_rna = False
        
        # Aggregated properties (populated by AtomProperties)
        self.hbond_acceptors: List = []
        self.hbond_donors: List[Tuple] = []  # (donor_atom, [h_atoms])
        self.pos_charged: List = []
        self.neg_charged: List = []
        self.hydrophobic_atoms: List = []
        self.rings: List[Dict] = []
        self.metal_atoms: List = []
        self.metal_binding_atoms: List = []
        self.halogen_donors: List = []
        self.halogen_acceptors: List = []
        
        # Pre-computed charge groups for salt bridge detection (populated by finalize)
        # Format: {residue_key: (atoms_list, charge_center)}
        self.pos_charged_groups: Dict = {}
        self.neg_charged_groups: Dict = {}
    
    def add_atom(self, atom_info):
        """Add an atom to this residue"""
        self.atoms.append(atom_info)
    
    def finalize(self):
        """Finalize residue after all atoms are added"""
        if self.atoms:
            self.center = np.mean([a.coords for a in self.atoms], axis=0)
            self._determine_residue_type()
            self._precompute_charge_groups()
    
    def _precompute_charge_groups(self):
        """Pre-compute charge groups for salt bridge detection.
        
        Groups charged atoms by residue key and pre-calculates charge centers.
        This avoids repeated calculations during interaction detection.
        """
        from openbabel import pybel
        
        # Group positive charges
        if self.pos_charged:
            groups = defaultdict(list)
            for atom in self.pos_charged:
                key = (atom.resname, atom.chain, atom.resnum)
                groups[key].append(atom)
            
            # Calculate charge centers
            for key, atoms in groups.items():
                center = np.mean([a.coords for a in atoms], axis=0)
                self.pos_charged_groups[key] = (atoms, center)
        
        # Group negative charges (with special handling for phosphate groups)
        if self.neg_charged:
            groups = defaultdict(list)
            for atom in self.neg_charged:
                key = (atom.resname, atom.chain, atom.resnum)
                
                # Special handling for phosphate groups: group by P atom
                if atom.atomic_num == 15:
                    key = (atom.resname, atom.chain, atom.resnum, atom.idx)
                else:
                    # For oxygen atoms in phosphate groups, find their parent P atom
                    for neighbor in pybel.ob.OBAtomAtomIter(atom.obatom):
                        if neighbor.GetAtomicNum() == 15:
                            key = (atom.resname, atom.chain, atom.resnum, neighbor.GetIdx())
                            break
                
                groups[key].append(atom)
            
            # Calculate charge centers
            for key, atoms in groups.items():
                # For phosphate groups, use P atom's coordinates as center
                p_atoms = [a for a in atoms if a.atomic_num == 15]
                if p_atoms:
                    center = p_atoms[0].coords
                else:
                    center = np.mean([a.coords for a in atoms], axis=0)
                self.neg_charged_groups[key] = (atoms, center)
    
    def _determine_residue_type(self):
        """Determine the type of this residue"""
        # Check first atom's component type
        if self.atoms:
            comp_type = self.atoms[0].component_type
            
            if comp_type == 'protein':
                self.is_protein = True
                self.is_peptide = True
            elif comp_type == 'ligand':
                self.is_ligand = True
            elif comp_type == 'water':
                self.is_water = True
            elif comp_type == 'ion':
                self.is_ion = True
            elif comp_type == 'dna':
                self.is_dna = True
            elif comp_type == 'rna':
                self.is_rna = True
    
    def should_filter_self(self) -> bool:
        """
        Check if this residue should filter interactions with itself.
        
        Protein/peptide residues filter self (no intra-residue interactions).
        Ligands do not filter self (detect intra-ligand interactions).
        """
        return self.is_protein or self.is_peptide
    
    def get_atom_indices(self) -> Set[int]:
        """Get all atom indices in this residue"""
        return {atom.idx for atom in self.atoms}
    
    def get_atom_by_name(self, atom_name: str):
        """Get atom by name (e.g., 'CA', 'N', 'O')"""
        for atom in self.atoms:
            if atom.obatom.GetResidue().GetAtomID(atom.obatom).strip() == atom_name:
                return atom
        return None
    
    def __repr__(self):
        return f"Residue({self.resid}, atoms={len(self.atoms)}, type={'protein' if self.is_protein else 'ligand' if self.is_ligand else 'other'})"
    
    def __hash__(self):
        return hash(self.resid)
    
    def __eq__(self, other):
        if isinstance(other, Residue):
            return self.resid == other.resid
        return False
