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
            self._determine_residue_type()
            self._precompute_charge_groups()
    
    def _precompute_charge_groups(self):
        """Pre-compute charge groups for salt bridge detection.
        
        Groups charged atoms by functional group and pre-calculates charge centers.
        This ensures each functional group (e.g., each carboxylate, each guanidinium)
        is treated as a separate charge center, aligning with main PLIP's approach.
        
        This is critical for small molecules with multiple charged groups that may be
        spatially distant (e.g., citrate with three carboxylates).
        
        For protein residues:
        - ARG: All 3 N atoms (NE, NH1, NH2) form one guanidinium group, centered at CZ
        - LYS: NZ forms its own group
        - ASP/GLU: Both O atoms form one carboxylate group, centered at their centroid
        """
        from openbabel import pybel
        
        # Group positive charges by functional group
        if self.pos_charged:
            groups = defaultdict(list)
            assigned_atoms = set()
            
            # First, identify guanidinium groups by looking at N atoms
            # In ARG, the 3 N atoms (NE, NH1, NH2) are all connected to CZ
            # We need to find C atoms that are connected to 3 N atoms
            guanidinium_centers = {}  # C atom idx -> list of N atom indices
            
            # Check all atoms in this residue (not just pos_charged) to find CZ
            for atom in self.atoms:
                if atom.atomic_num == 6:  # Carbon
                    neighbors = list(pybel.ob.OBAtomAtomIter(atom.obatom))
                    n_neighbors = [n for n in neighbors if n.GetAtomicNum() == 7]
                    if len(n_neighbors) >= 3:
                        # This C is the center of a guanidinium group
                        n_indices = [n.GetIdx() for n in n_neighbors]
                        guanidinium_centers[atom.idx] = n_indices
            
            # Group guanidinium N atoms together by their parent C atom
            for c_idx, n_indices in guanidinium_centers.items():
                key = (self.resname, self.chain, self.resnum, 'guanidinium', c_idx)
                # Find the C atom object
                c_atom = None
                for atom in self.atoms:
                    if atom.idx == c_idx:
                        c_atom = atom
                        break
                
                # Add all N atoms that are in pos_charged and connected to this C
                for atom in self.pos_charged:
                    if atom.idx in n_indices:
                        groups[key].append(atom)
                        assigned_atoms.add(atom.idx)
                
                # Also add the C atom itself if it's in pos_charged (for small molecules)
                if c_atom and c_atom.idx in [a.idx for a in self.pos_charged]:
                    groups[key].append(c_atom)
                    assigned_atoms.add(c_atom.idx)
            
            # Identify imidazolium groups (His sidechain or small molecule imidazole)
            # Imidazolium is a 5-membered ring with 2 N atoms sharing one positive charge
            imidazolium_groups = []  # List of (n_atom1, n_atom2) tuples
            imidazolium_n_atoms = set()
            
            # Find all imidazolium N atoms (marked as 'imidazolium' type)
            imidazolium_atoms = [a for a in self.pos_charged 
                                 if a.idx not in assigned_atoms]
            
            # Group them by ring membership
            # For standard HIS: ND1 and NE2 are the two ring N atoms
            # For small molecules: need to check if they are in the same 5-membered ring
            if len(imidazolium_atoms) >= 2:
                # Try to pair N atoms that are part of the same ring
                # In a 5-membered imidazole ring, the two N atoms are separated by one C atom
                for i, atom1 in enumerate(imidazolium_atoms):
                    if atom1.idx in imidazolium_n_atoms:
                        continue
                    neighbors1 = list(pybel.ob.OBAtomAtomIter(atom1.obatom))
                    # Find neighbors that are also imidazolium atoms
                    for atom2 in imidazolium_atoms[i+1:]:
                        if atom2.idx in imidazolium_n_atoms:
                            continue
                        # Check if they share a common ring (both connected to same C or part of 5-membered ring)
                        neighbors2 = list(pybel.ob.OBAtomAtomIter(atom2.obatom))
                        # Get C neighbors of each N
                        c_neighbors1 = {n.GetIdx() for n in neighbors1 if n.GetAtomicNum() == 6}
                        c_neighbors2 = {n.GetIdx() for n in neighbors2 if n.GetAtomicNum() == 6}
                        # If they share a C neighbor or are in same ring context, group them
                        if c_neighbors1 & c_neighbors2 or self._are_in_same_imidazole_ring(atom1, atom2):
                            imidazolium_groups.append((atom1, atom2))
                            imidazolium_n_atoms.add(atom1.idx)
                            imidazolium_n_atoms.add(atom2.idx)
                            break
                
                # Also handle single N atoms (e.g., if only one is marked as charged)
                for atom in imidazolium_atoms:
                    if atom.idx not in imidazolium_n_atoms:
                        # Single imidazolium N - treat as its own group
                        imidazolium_groups.append((atom,))
                        imidazolium_n_atoms.add(atom.idx)
            elif len(imidazolium_atoms) == 1:
                # Single imidazolium N atom
                imidazolium_groups.append((imidazolium_atoms[0],))
                imidazolium_n_atoms.add(imidazolium_atoms[0].idx)
            
            # Group imidazolium atoms together
            for n_atoms in imidazolium_groups:
                # Use the first N atom's index as the group key, but include all N atoms
                key = (self.resname, self.chain, self.resnum, 'imidazolium', n_atoms[0].idx)
                for atom in n_atoms:
                    groups[key].append(atom)
                    assigned_atoms.add(atom.idx)
            
            # Group remaining atoms by individual atom (LYS NZ, etc.)
            for atom in self.pos_charged:
                if atom.idx in assigned_atoms:
                    continue
                
                # Each non-guanidinium, non-imidazolium charged atom forms its own group
                key = (self.resname, self.chain, self.resnum, 'other', atom.idx)
                groups[key].append(atom)
            
            # Calculate charge centers
            for key, atoms in groups.items():
                # For guanidinium, use the carbon's coordinates as center (following main PLIP)
                if 'guanidinium' in key:
                    c_idx = key[4]  # The C atom index from the key
                    # Find the C atom in this residue's atoms
                    center = None
                    for atom in self.atoms:
                        if atom.idx == c_idx:
                            center = atom.coords
                            break
                    if center is None:
                        # Fallback: use mean of N atoms
                        center = np.mean([a.coords for a in atoms if a.atomic_num == 7], axis=0)
                elif 'imidazolium' in key:
                    # For imidazolium, use the geometric center of the N atoms
                    # The positive charge is delocalized over both N atoms in the ring
                    n_atoms = [a for a in atoms if a.atomic_num == 7]
                    if len(n_atoms) >= 2:
                        center = np.mean([a.coords for a in n_atoms], axis=0)
                    elif n_atoms:
                        center = n_atoms[0].coords
                    else:
                        center = np.mean([a.coords for a in atoms], axis=0)
                else:
                    # For other groups, use the atom's own coordinates
                    center = atoms[0].coords if len(atoms) == 1 else np.mean([a.coords for a in atoms], axis=0)
                
                self.pos_charged_groups[key] = (atoms, center)
        
        # Group negative charges by functional group
        if self.neg_charged:
            groups = defaultdict(list)
            assigned_atoms = set()
            
            # First, identify carboxylate groups by looking at O atoms
            # In ASP/GLU, the 2 O atoms (OD1, OD2 or OE1, OE2) are connected to the same C (CG or CD)
            carboxylate_centers = {}  # C atom idx -> list of O atom indices
            
            # Check all atoms in this residue to find carboxylate C atoms
            for atom in self.atoms:
                if atom.atomic_num == 6:  # Carbon
                    neighbors = list(pybel.ob.OBAtomAtomIter(atom.obatom))
                    o_neighbors = [n for n in neighbors if n.GetAtomicNum() == 8]
                    if len(o_neighbors) == 2:
                        # This C is the center of a carboxylate group
                        o_indices = [o.GetIdx() for o in o_neighbors]
                        carboxylate_centers[atom.idx] = o_indices
            
            # Group carboxylate O atoms together by their parent C atom
            for c_idx, o_indices in carboxylate_centers.items():
                key = (self.resname, self.chain, self.resnum, 'carboxylate', c_idx)
                
                # Find the C atom object
                c_atom = None
                for atom in self.atoms:
                    if atom.idx == c_idx:
                        c_atom = atom
                        break
                
                # Add all O atoms that are in neg_charged and connected to this C
                for atom in self.neg_charged:
                    if atom.idx in o_indices:
                        groups[key].append(atom)
                        assigned_atoms.add(atom.idx)
                
                # Also add the C atom itself if it's in neg_charged (for small molecules with charged C)
                if c_atom and c_atom.idx in [a.idx for a in self.neg_charged]:
                    groups[key].append(c_atom)
                    assigned_atoms.add(c_atom.idx)
            
            # Group phosphate groups by their P atom
            for atom in self.neg_charged:
                if atom.idx in assigned_atoms:
                    continue
                    
                if atom.atomic_num == 15:  # Phosphorus
                    # P atom defines the phosphate group
                    key = (self.resname, self.chain, self.resnum, 'phosphate', atom.idx)
                    groups[key].append(atom)
                    assigned_atoms.add(atom.idx)
                    
                    # Find all O atoms connected to this P
                    for neighbor in pybel.ob.OBAtomAtomIter(atom.obatom):
                        if neighbor.GetAtomicNum() == 8:
                            # Find the corresponding atom object
                            for o_atom in self.neg_charged:
                                if o_atom.idx == neighbor.GetIdx():
                                    groups[key].append(o_atom)
                                    assigned_atoms.add(o_atom.idx)
                                    break
            
            # Group any remaining atoms individually
            for atom in self.neg_charged:
                if atom.idx in assigned_atoms:
                    continue
                key = (self.resname, self.chain, self.resnum, 'other', atom.idx)
                groups[key].append(atom)
            
            # Calculate charge centers
            for key, atoms in groups.items():
                group_type = key[3] if len(key) > 3 else 'other'
                
                if group_type == 'carboxylate':
                    # For carboxylate, use the centroid of the two O atoms (following main PLIP)
                    o_atoms = [a for a in atoms if a.atomic_num == 8]
                    if len(o_atoms) >= 2:
                        center = np.mean([a.coords for a in o_atoms[:2]], axis=0)
                    elif o_atoms:
                        center = o_atoms[0].coords
                    else:
                        center = np.mean([a.coords for a in atoms], axis=0)
                elif group_type == 'phosphate':
                    # For phosphate, use P atom's coordinates (following main PLIP)
                    p_atoms = [a for a in atoms if a.atomic_num == 15]
                    if p_atoms:
                        center = p_atoms[0].coords
                    else:
                        center = np.mean([a.coords for a in atoms], axis=0)
                else:
                    # For other groups, use mean coordinates
                    center = np.mean([a.coords for a in atoms], axis=0)
                
                self.neg_charged_groups[key] = (atoms, center)
    
    def _are_in_same_imidazole_ring(self, atom1, atom2):
        """Check if two N atoms are part of the same 5-membered imidazole ring.
        
        An imidazole ring is a 5-membered aromatic ring with exactly 2 N atoms.
        The two N atoms are typically separated by one C atom in the ring.
        
        Args:
            atom1, atom2: AtomInfo objects for the two N atoms
            
        Returns:
            bool: True if they are in the same imidazole ring
        """
        from openbabel import pybel
        
        # Get the parent molecule
        if not atom1.obatom or not atom2.obatom:
            return False
        
        obmol = atom1.obatom.GetParent()
        if not obmol:
            return False
        
        # Get all rings in the molecule
        rings = list(obmol.GetSSSR())
        
        for ring in rings:
            ring_indices = list(ring._path)
            
            # Check if both atoms are in this ring
            if atom1.idx in ring_indices and atom2.idx in ring_indices:
                # Check if it's a 5-membered ring
                if len(ring_indices) == 5:
                    # Count N atoms in the ring
                    n_count = sum(1 for idx in ring_indices 
                                  if obmol.GetAtom(idx).GetAtomicNum() == 7)
                    # Imidazole has exactly 2 N atoms
                    if n_count == 2:
                        return True
        
        return False
    
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
