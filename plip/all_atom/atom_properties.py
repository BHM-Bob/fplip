"""
Atom Properties Module

Provides identification of atom properties for interaction detection:
- Hydrogen bond donors and acceptors
- Charged groups (positive and negative)
- Hydrophobic atoms
- Aromatic rings
- Metal binding sites
- Halogen bond donors
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from openbabel import pybel

from plip.basic import config
from plip.basic.logger import logger
from plip.basic.supplemental import ring_is_planar

from .atom_container import AtomContainer, AtomInfo


class AtomProperties:
    """
    Identifies and stores atom properties for interaction detection.
    """
    
    def __init__(self, atom_container: AtomContainer):
        self.atom_container = atom_container
        
        # Property storage
        self.hbond_acceptors: Set[int] = set()
        self.hbond_donors: Dict[int, List[int]] = {}  # donor_idx -> [h_idx1, h_idx2, ...]
        
        self.pos_charged: Dict[int, str] = {}  # idx -> charge type
        self.neg_charged: Dict[int, str] = {}  # idx -> charge type
        
        self.hydrophobic_atoms: Set[int] = set()
        
        self.rings: List[Dict] = []  # List of ring information dicts
        
        self.metal_binding: Set[int] = set()
        self.metals: Set[int] = set()
        
        self.halogen_donors: Dict[int, str] = {}  # idx -> halogen type (Cl, Br, I)
        self.halogen_acceptors: Set[int] = set()
        
        # Initialize
        self._identify_all_properties()
    
    def _identify_all_properties(self):
        """Identify all atom properties"""
        logger.info('Identifying atom properties...')
        
        self._identify_hbond_properties()
        self._identify_charges()
        self._identify_hydrophobic()
        self._identify_rings()
        self._identify_metals()
        self._identify_halogen()
        
        logger.info(f'Properties identified: '
                   f'{len(self.hbond_acceptors)} HBA, '
                   f'{len(self.hbond_donors)} HBD, '
                   f'{len(self.pos_charged)} pos, '
                   f'{len(self.neg_charged)} neg, '
                   f'{len(self.hydrophobic_atoms)} hydrophobic, '
                   f'{len(self.rings)} rings, '
                   f'{len(self.metals)} metals')
    
    def _identify_hbond_properties(self):
        """Identify hydrogen bond donors and acceptors"""
        for atom in self.atom_container:
            obatom = atom.obatom
            
            # Skip hydrogens
            if atom.is_hydrogen:
                continue
            
            # Check if atom is H-bond acceptor
            if obatom.IsHbondAcceptor():
                self.hbond_acceptors.add(atom.idx)
            
            # Check if atom is H-bond donor
            if obatom.IsHbondDonor():
                # Find attached hydrogens
                attached_h = []
                for neighbor in pybel.ob.OBAtomAtomIter(obatom):
                    if neighbor.GetAtomicNum() == 1:  # Hydrogen
                        attached_h.append(neighbor.GetIdx())
                if attached_h:
                    self.hbond_donors[atom.idx] = sorted(attached_h)
            else:
                # Fallback: For PDB files without hydrogens, identify potential donors
                # N-H and O-H groups where hydrogens are not explicitly present
                self._identify_potential_donors(atom)
    
    def _identify_potential_donors(self, atom):
        """
        Identify potential H-bond donors for PDB files without explicit hydrogens.

        This method identifies N and O atoms that would typically have H atoms attached
        in a complete structure.
        """
        # Skip if already identified as donor
        if atom.idx in self.hbond_donors:
            return

        atomic_num = atom.atomic_num
        resname = atom.resname
        atom_name = atom.atom_name
        
        # Nitrogen donors (backbone and side chain)
        if atomic_num == 7:  # Nitrogen
            # Backbone amide N (except proline)
            if atom_name == 'N' and resname != 'PRO':
                self.hbond_donors[atom.idx] = []
                return
            
            # Side chain donors
            if resname == 'ASN' and atom_name == 'ND2':
                self.hbond_donors[atom.idx] = []
            elif resname == 'GLN' and atom_name == 'NE2':
                self.hbond_donors[atom.idx] = []
            elif resname == 'ARG' and atom_name in ['NE', 'NH1', 'NH2']:
                self.hbond_donors[atom.idx] = []
            elif resname == 'HIS' and atom_name in ['ND1', 'NE2']:
                self.hbond_donors[atom.idx] = []
            elif resname == 'LYS' and atom_name == 'NZ':
                self.hbond_donors[atom.idx] = []
            elif resname == 'TRP' and atom_name == 'NE1':
                self.hbond_donors[atom.idx] = []
        
        # Oxygen donors (hydroxyl groups)
        elif atomic_num == 8:  # Oxygen
            # Serine, Threonine, Tyrosine hydroxyl
            if resname == 'SER' and atom_name == 'OG':
                self.hbond_donors[atom.idx] = []
            elif resname == 'THR' and atom_name == 'OG1':
                self.hbond_donors[atom.idx] = []
            elif resname == 'TYR' and atom_name == 'OH':
                self.hbond_donors[atom.idx] = []
            # Terminal carboxyl (can be donor in some contexts)
            elif atom_name == 'OXT':
                self.hbond_donors[atom.idx] = []
    
    def _identify_charges(self):
        """Identify charged groups"""
        # Group atoms by residue
        residue_atoms = defaultdict(list)
        for atom in self.atom_container:
            res_key = (atom.resname, atom.chain, atom.resnum)
            residue_atoms[res_key].append(atom)
        
        for res_key, atoms in residue_atoms.items():
            resname = res_key[0]
            
            # Identify positive charges
            self._find_positive_charges(atoms, resname)
            
            # Identify negative charges
            self._find_negative_charges(atoms, resname)
    
    def _find_positive_charges(self, atoms: List[AtomInfo], resname: str):
        """Find positively charged groups in a residue.
        
        All-atom design: Unified detection based on chemical topology.
        Uses residue context for standard amino acids to ensure correct
        protonation state detection even when H atoms are missing in PDB.
        
        Detection priority:
        1. Guanidinium: C connected to 3 N atoms (Arg, ligands)
        2. Ammonium: N with 4 non-H neighbors OR standard Lys NZ
        3. Tertamine: sp3 N with 3+ neighbors (can be protonated)
        4. Sulfonium: S with 3 non-H neighbors
        5. Imidazolium: Aromatic 5-membered ring with 2 N atoms (His)
        """
        # Group atoms by element for efficient processing
        carbon_atoms = [a for a in atoms if a.atomic_num == 6]
        nitrogen_atoms = [a for a in atoms if a.atomic_num == 7]
        sulfur_atoms = [a for a in atoms if a.atomic_num == 16]
        
        # 1. Detect Guanidinium (C-centered)
        # Chemical definition: C connected to 3 N atoms, at least one terminal N
        for atom in carbon_atoms:
            if atom.idx in self.pos_charged:
                continue
            neighbors = list(pybel.ob.OBAtomAtomIter(atom.obatom))
            n_neighbors = [n for n in neighbors if n.GetAtomicNum() == 7]
            if len(n_neighbors) == 3:
                # Check for terminal N (only connected to this C, can pick up H)
                has_terminal_n = any(
                    len([nb for nb in pybel.ob.OBAtomAtomIter(n) 
                         if nb.GetAtomicNum() != 1]) == 1
                    for n in n_neighbors
                )
                if has_terminal_n:
                    self.pos_charged[atom.idx] = 'guanidinium'
                    for n in n_neighbors:
                        self.pos_charged[n.GetIdx()] = 'guanidinium'
        
        # 2. Detect Ammonium (N-centered)
        # For standard residues: Lys NZ is always ammonium (protonated at physiological pH)
        # For non-standard: use topology-based detection
        if resname == 'LYS':
            # In standard Lys, the NZ atom is the ammonium nitrogen
            for atom in nitrogen_atoms:
                if atom.atom_name == 'NZ' and atom.idx not in self.pos_charged:
                    self.pos_charged[atom.idx] = 'ammonium'
        
        # Topology-based detection for all N atoms (including non-standard residues)
        for atom in nitrogen_atoms:
            if atom.idx in self.pos_charged:
                continue
            neighbors = list(pybel.ob.OBAtomAtomIter(atom.obatom))
            non_h_neighbors = [n for n in neighbors if n.GetAtomicNum() != 1]
            
            if len(non_h_neighbors) == 4:
                # Quaternary ammonium - permanently charged
                self.pos_charged[atom.idx] = 'ammonium'
            elif atom.obatom.GetHyb() == 3 and len(neighbors) >= 3:
                # Tertiary amine - can pick up H to become ammonium
                self.pos_charged[atom.idx] = 'tertamine'
        
        # 3. Detect Sulfonium (S-centered)
        for atom in sulfur_atoms:
            if atom.idx in self.pos_charged:
                continue
            neighbors = list(pybel.ob.OBAtomAtomIter(atom.obatom))
            non_h_neighbors = [n for n in neighbors if n.GetAtomicNum() != 1]
            if len(non_h_neighbors) == 3:
                self.pos_charged[atom.idx] = 'sulfonium'
        
        # 4. Detect Imidazolium (ring-based)
        # For standard His, only mark ND1 and NE2 (ring nitrogens) as imidazolium
        if resname == 'HIS':
            for atom in nitrogen_atoms:
                if atom.idx not in self.pos_charged and atom.atom_name in ['ND1', 'NE2']:
                    self.pos_charged[atom.idx] = 'imidazolium'
        # For non-standard residues, use ring detection
        elif self._is_imidazolium_ring(atoms):
            ring_nitrogens = [a for a in atoms if a.atomic_num == 7 
                              and a.idx not in self.pos_charged]
            if len(ring_nitrogens) >= 2:
                for n in ring_nitrogens:
                    self.pos_charged[n.idx] = 'imidazolium'

        # N-terminus detection would require additional context

    def _is_imidazolium_ring(self, atoms: List[AtomInfo]) -> bool:
        """Check if atoms form an imidazolium ring (5-membered, 2 N, aromatic).

        Uses OpenBabel's ring detection for accurate identification.
        """
        # Get all atoms in this residue as OpenBabel atoms
        ob_atoms = [a.obatom for a in atoms]
        if not ob_atoms:
            return False

        # Get the OBMol from the first atom
        obmol = ob_atoms[0].GetParent()
        if not obmol:
            return False

        # Find rings in the molecule
        ring_data = obmol.GetSSSR()  # Smallest Set of Smallest Rings

        for ring in ring_data:
            ring_size = ring.Size()
            if ring_size != 5:
                continue

            # Count N atoms in this ring
            ring_atom_indices = set(ring._path)
            n_count = sum(1 for oba in ob_atoms if oba.GetIdx() in ring_atom_indices and oba.GetAtomicNum() == 7)

            # Imidazolium has exactly 2 N atoms in a 5-membered ring
            if n_count == 2:
                return True

        return False
    
    def _find_negative_charges(self, atoms: List[AtomInfo], resname: str):
        """Find negatively charged groups in a residue"""
        # Carboxylate (Asp, Glu)
        if resname in ['ASP', 'GLU']:
            # Find carboxyl carbons
            for atom in atoms:
                if atom.atomic_num == 6:  # Carbon
                    neighbors = [n for n in pybel.ob.OBAtomAtomIter(atom.obatom)]
                    o_neighbors = [n for n in neighbors if n.GetAtomicNum() == 8]
                    if len(o_neighbors) == 2:
                        self.neg_charged[atom.idx] = f'{resname.lower()}_carboxylate'
                        for o in o_neighbors:
                            self.neg_charged[o.GetIdx()] = f'{resname.lower()}_carboxylate'
        
        # Phosphate groups
        phosphate_atoms = [a for a in atoms if a.atomic_num == 15]  # Phosphorus
        for p_atom in phosphate_atoms:
            self.neg_charged[p_atom.idx] = 'phosphate'
            neighbors = [n for n in pybel.ob.OBAtomAtomIter(p_atom.obatom)]
            for n in neighbors:
                if n.GetAtomicNum() == 8:  # Oxygen
                    self.neg_charged[n.GetIdx()] = 'phosphate'
    
    def _identify_hydrophobic(self):
        """Identify hydrophobic atoms"""
        for atom in self.atom_container:
            # Skip hydrogens
            if atom.is_hydrogen:
                continue
            
            obatom = atom.obatom
            
            # Carbon atoms are generally hydrophobic
            if atom.atomic_num == 6:
                # Exclude carbons in polar groups
                is_polar = False
                for neighbor in pybel.ob.OBAtomAtomIter(obatom):
                    n_atomic_num = neighbor.GetAtomicNum()
                    # Check if connected to electronegative atoms
                    if n_atomic_num in [7, 8, 15, 16]:  # N, O, P, S
                        is_polar = True
                        break
                
                if not is_polar:
                    self.hydrophobic_atoms.add(atom.idx)
            
            # Sulfur in some contexts
            elif atom.atomic_num == 16:  # Sulfur
                self.hydrophobic_atoms.add(atom.idx)
    
    def _identify_rings(self):
        """Identify aromatic rings"""
        # Get all atoms as OBAtom list
        obatoms = [atom.obatom for atom in self.atom_container]
        if not obatoms:
            return
        
        # Get the OBMol
        obmol = obatoms[0].GetParent()
        
        # Find all rings
        rings = []
        for ring in obmol.GetSSSR():
            ring_atoms = [obmol.GetAtom(i + 1) for i in range(ring.Size())]
            ring_indices = [atom.GetIdx() for atom in ring_atoms]
            
            # Check if all atoms are in our container
            if not all(idx in self.atom_container.atoms for idx in ring_indices):
                continue
            
            # Check aromaticity
            is_aromatic = all(atom.IsAromatic() for atom in ring_atoms)

            # Check planarity using simplified method
            coords = np.array([self.atom_container[idx].coords for idx in ring_indices])
            is_planar = self._check_ring_planarity(coords)

            if is_aromatic or is_planar:
                # Calculate ring center
                center = np.mean(coords, axis=0)
                
                # Calculate normal vector
                if len(coords) >= 3:
                    v1 = coords[1] - coords[0]
                    v2 = coords[2] - coords[1]
                    normal = np.cross(v1, v2)
                    normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else np.array([0, 0, 1])
                else:
                    normal = np.array([0, 0, 1])
                
                rings.append({
                    'indices': ring_indices,
                    'center': center,
                    'normal': normal,
                    'is_aromatic': is_aromatic,
                    'size': len(ring_indices)
                })
        
        self.rings = rings

    def _check_ring_planarity(self, coords: np.ndarray) -> bool:
        """
        Check if a ring is planar by computing the deviation from the best-fit plane.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates of ring atoms [n, 3]

        Returns
        -------
        bool
            True if the ring is planar enough to be considered aromatic
        """
        if len(coords) < 4:
            return True  # Small rings are always considered planar

        # Compute the best-fit plane using SVD
        centroid = np.mean(coords, axis=0)
        centered_coords = coords - centroid

        # SVD to find the normal of the best-fit plane
        _, _, vh = np.linalg.svd(centered_coords)
        normal = vh[-1]  # Last row is the normal vector

        # Compute distances from all points to the plane
        distances = np.abs(np.dot(centered_coords, normal))
        max_distance = np.max(distances)

        # Check if all atoms are within the planarity threshold
        return max_distance < config.AROMATIC_PLANARITY

    def _identify_metals(self):
        """Identify metal ions and metal-binding atoms"""
        for atom in self.atom_container:
            # Metal ions
            if atom.resname in config.METAL_IONS:
                self.metals.add(atom.idx)
            
            # Metal-binding atoms (O, N, S with lone pairs)
            if atom.atomic_num in [7, 8, 16]:  # N, O, S
                self.metal_binding.add(atom.idx)
    
    def _identify_halogen(self):
        """Identify halogen bond donors and acceptors"""
        for atom in self.atom_container:
            atomic_num = atom.atomic_num

            # Halogen donors (F, Cl, Br, I bonded to carbon)
            # Note: F is weaker but can still form halogen bonds in some contexts
            if atomic_num in [9, 17, 35, 53]:  # F, Cl, Br, I
                # Check if bonded to carbon
                for neighbor in pybel.ob.OBAtomAtomIter(atom.obatom):
                    if neighbor.GetAtomicNum() == 6:  # Carbon
                        halogen_type = {9: 'F', 17: 'Cl', 35: 'Br', 53: 'I'}.get(atomic_num, 'X')
                        self.halogen_donors[atom.idx] = halogen_type
                        break

            # Halogen acceptors (O, N, S, aromatic systems)
            if atomic_num in [7, 8, 16]:  # N, O, S
                self.halogen_acceptors.add(atom.idx)
    
    # Accessor methods
    def get_hba(self) -> List[AtomInfo]:
        """Get all hydrogen bond acceptors"""
        return [self.atom_container[idx] for idx in sorted(self.hbond_acceptors)]
    
    def get_hbd(self) -> List[Tuple[AtomInfo, List[AtomInfo]]]:
        """Get all hydrogen bond donors with their attached hydrogens"""
        result = []
        for donor_idx in sorted(self.hbond_donors.keys()):
            h_indices = self.hbond_donors[donor_idx]
            donor = self.atom_container[donor_idx]
            hydrogens = [self.atom_container[h_idx] for h_idx in sorted(h_indices)]
            result.append((donor, hydrogens))
        return result
    
    def get_pos_charged(self) -> List[AtomInfo]:
        """Get all positively charged atoms"""
        return [self.atom_container[idx] for idx in sorted(self.pos_charged)]
    
    def get_neg_charged(self) -> List[AtomInfo]:
        """Get all negatively charged atoms"""
        return [self.atom_container[idx] for idx in sorted(self.neg_charged)]
    
    def get_hydrophobic(self) -> List[AtomInfo]:
        """Get all hydrophobic atoms"""
        return [self.atom_container[idx] for idx in sorted(self.hydrophobic_atoms)]
    
    def get_metals(self) -> List[AtomInfo]:
        """Get all metal ions"""
        return [self.atom_container[idx] for idx in sorted(self.metals)]
    
    def get_metal_binding(self) -> List[AtomInfo]:
        """Get all metal-binding atoms"""
        return [self.atom_container[idx] for idx in sorted(self.metal_binding)]
