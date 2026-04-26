"""
Interaction Detector Module - Residue-based Unified Detection

Provides unified interaction detection using a residue-based approach:
- Iterate over each residue
- Build MxN distance matrix (M=residue atoms, N=all atoms)
- Detect interactions with smart self-filtering
"""

import sys
from collections import namedtuple
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/home/pcmd36/Desktop/BHM/My_Progs/fplip/')
from openbabel import pybel

from plip.basic import config
from plip.basic.logger import logger
from plip.basic.supplemental import euclidean3d, projection, vecangle, vector

from plip.all_atom.atom_container import AtomContainer
from plip.all_atom.atom_properties import AtomProperties
from plip.all_atom.residue import Residue

# Unified interaction record
Interaction = namedtuple('Interaction', [
    'type',              # Interaction type
    'res_a_name',        # Residue A name
    'res_a_chain',       # Residue A chain
    'res_a_num',         # Residue A number
    'res_b_name',        # Residue B name
    'res_b_chain',       # Residue B chain
    'res_b_num',         # Residue B number
    'atom_a_name',       # Atom A name
    'atom_a_idx',        # Atom A index
    'atom_b_name',       # Atom B name
    'atom_b_idx',        # Atom B index
    'distance',          # Distance
    'angle',             # Angle (if applicable)
    'details',           # Type-specific details
])


class UnifiedInteractionDetector:
    """
    Unified interaction detector using residue-based approach.
    
    For each residue:
    1. Build MxN distance matrix (M=residue atoms, N=all atoms)
    2. Apply distance thresholds
    3. Smart self-filtering (protein residues filter self, ligands don't)
    4. Detect all interaction types
    """
    
    def __init__(self, atom_container: AtomContainer, atom_props: AtomProperties, residues: List[Residue]):
        self.atom_container = atom_container
        self.atom_props = atom_props
        self.residues = residues
        
        # Pre-compute all atom coordinates for fast distance calculation
        self.all_coords = atom_container.coords_array
        self.idx_to_pos = atom_container.idx_to_array_pos
        self.pos_to_idx = {v: k for k, v in self.idx_to_pos.items()}
        
        # Results storage
        self.interactions: Dict[str, List[Interaction]] = {
            'hydrophobic': [],
            'hbond': [],
            'saltbridge': [],
            'pistacking': [],
            'pication': [],
            'halogen': [],
            'metal': [],
            'water_bridge': [],
        }
    
    def detect_all(self, verbose: bool = False) -> Dict[str, List[Interaction]]:
        """Main entry point for detection"""
        logger.info(f'Starting unified interaction detection for {len(self.residues)} residues...')
        
        # First, aggregate properties to residues
        self._aggregate_properties_to_residues()
        
        # Detect interactions for each residue
        for residue in tqdm(self.residues, desc='Processing residues', disable=not verbose):            
            self._detect_for_residue(residue)
        
        # Remove duplicates (each interaction detected twice: A-B and B-A)
        self._remove_duplicates()
        
        # Detect water bridges
        self._detect_water_bridges()
        
        self._log_summary()
        
        return self.interactions
    
    def _aggregate_properties_to_residues(self):
        """Aggregate atom properties to residue level"""
        for residue in self.residues:
            for atom in residue.atoms:
                atom_idx = atom.idx
                
                # H-bond acceptors
                if atom_idx in self.atom_props.hbond_acceptors:
                    residue.hbond_acceptors.append(atom)
                
                # H-bond donors
                if atom_idx in self.atom_props.hbond_donors:
                    h_atoms = [self.atom_container[h_idx] for h_idx in self.atom_props.hbond_donors[atom_idx]]
                    residue.hbond_donors.append((atom, h_atoms))
                
                # Charges
                if atom_idx in self.atom_props.pos_charged:
                    residue.pos_charged.append(atom)
                if atom_idx in self.atom_props.neg_charged:
                    residue.neg_charged.append(atom)
                
                # Hydrophobic
                if atom_idx in self.atom_props.hydrophobic_atoms:
                    residue.hydrophobic_atoms.append(atom)
                
                # Rings (check if atom is part of a ring)
                for ring in self.atom_props.rings:
                    if atom_idx in ring['indices']:
                        # Check if this ring is already added (compare by first atom index)
                        ring_id = ring['indices'][0]
                        existing_ids = [r['indices'][0] for r in residue.rings]
                        if ring_id not in existing_ids:
                            residue.rings.append(ring)
                
                # Metals
                if atom_idx in self.atom_props.metals:
                    residue.metal_atoms.append(atom)
                    residue.is_ion = True
                
                # Metal binding
                if atom_idx in self.atom_props.metal_binding:
                    residue.metal_binding_atoms.append(atom)
                
                # Halogen
                if atom_idx in self.atom_props.halogen_donors:
                    halogen_type = self.atom_props.halogen_donors[atom_idx]
                    residue.halogen_donors.append((atom, halogen_type))
                if atom_idx in self.atom_props.halogen_acceptors:
                    residue.halogen_acceptors.append(atom)
    
    def _detect_for_residue(self, residue: Residue):
        """Detect all interactions for a single residue"""
        if not residue.atoms:
            return
        
        # Build MxN distance matrix
        res_coords = np.array([a.coords for a in residue.atoms])
        dist_matrix = self._build_distance_matrix(res_coords)
        
        # Apply self-filtering if needed
        if residue.should_filter_self():
            self._apply_self_filter(dist_matrix, residue)
        
        # Detect each interaction type
        self._detect_hydrophobic(residue, dist_matrix)
        self._detect_hbonds(residue, dist_matrix)
        self._detect_saltbridges(residue, dist_matrix)
        self._detect_pistacking(residue, dist_matrix)
        self._detect_pication(residue, dist_matrix)
        self._detect_halogen(residue, dist_matrix)
        self._detect_metal(residue, dist_matrix)
    
    def _build_distance_matrix(self, res_coords: np.ndarray) -> np.ndarray:
        """Build MxN distance matrix"""
        return np.sqrt(
            np.sum((res_coords[:, np.newaxis, :] - self.all_coords[np.newaxis, :, :]) ** 2, axis=2)
        )
    
    def _apply_self_filter(self, dist_matrix: np.ndarray, residue: Residue):
        """Set distances to self atoms to infinity"""
        for atom in residue.atoms:
            if atom.idx in self.idx_to_pos:
                pos = self.idx_to_pos[atom.idx]
                dist_matrix[:, pos] = np.inf
    
    def _get_close_atoms(self, residue: Residue, target_atoms: List, 
                         dist_matrix: np.ndarray, max_dist: float) -> List[Tuple]:
        """Get close atom pairs within distance threshold"""
        result = []
        
        # Get positions of target atoms in the global array
        target_positions = []
        target_atom_map = {}
        for atom in target_atoms:
            if atom.idx in self.idx_to_pos:
                pos = self.idx_to_pos[atom.idx]
                target_positions.append(pos)
                target_atom_map[pos] = atom
        
        if not target_positions:
            return result
        
        # Find close pairs
        for i, res_atom in enumerate(residue.atoms):
            distances = dist_matrix[i, target_positions]
            close_mask = (distances < max_dist) & (distances > config.MIN_DIST)
            
            for j, is_close in enumerate(close_mask):
                if is_close:
                    target_atom = target_atom_map[target_positions[j]]
                    result.append((res_atom, target_atom, distances[j]))
        
        return result
    
    def _detect_hydrophobic(self, residue: Residue, dist_matrix: np.ndarray):
        """Detect hydrophobic interactions"""
        if not residue.hydrophobic_atoms:
            return
        
        # Get all hydrophobic atoms
        all_hydrophobic = self.atom_props.get_hydrophobic()
        
        pairs = self._get_close_atoms(
            residue, all_hydrophobic, dist_matrix, config.HYDROPH_DIST_MAX
        )
        
        for atom_a, atom_b, distance in pairs:
            # Skip if same residue (unless it's a ligand)
            if (atom_a.resname == atom_b.resname and 
                atom_a.chain == atom_b.chain and 
                atom_a.resnum == atom_b.resnum and
                residue.should_filter_self()):
                continue
            
            interaction = Interaction(
                type='hydrophobic',
                res_a_name=atom_a.resname,
                res_a_chain=atom_a.chain,
                res_a_num=atom_a.resnum,
                res_b_name=atom_b.resname,
                res_b_chain=atom_b.chain,
                res_b_num=atom_b.resnum,
                atom_a_name=self._get_atom_name(atom_a),
                atom_a_idx=atom_a.idx,
                atom_b_name=self._get_atom_name(atom_b),
                atom_b_idx=atom_b.idx,
                distance=distance,
                angle=None,
                details={}
            )
            self.interactions['hydrophobic'].append(interaction)
    
    def _detect_hbonds(self, residue: Residue, dist_matrix: np.ndarray):
        """Detect hydrogen bonds"""
        if not (residue.hbond_acceptors or residue.hbond_donors):
            return

        # Get all H-bond donors and acceptors
        all_hba = self.atom_props.get_hba()
        all_hbd = self.atom_props.get_hbd()  # List of (donor, [h_atoms])

        # Check if we have explicit hydrogens
        has_explicit_h = any(len(h_atoms) > 0 for _, h_atoms in all_hbd)

        if has_explicit_h:
            # Use explicit hydrogen geometry
            self._detect_hbonds_with_h(residue, all_hba, all_hbd)
        else:
            # Use hydrogen-free detection (distance-based only)
            self._detect_hbonds_without_h(residue, all_hba, all_hbd)

    def _detect_hbonds_with_h(self, residue: Residue, all_hba: List, all_hbd: List):
        """Detect H-bonds using explicit hydrogen coordinates"""
        # Case 1: Residue is donor, other is acceptor
        if residue.hbond_donors:
            for donor, h_atoms in residue.hbond_donors:
                for hba in all_hba:
                    for h_atom in h_atoms:
                        # Calculate distances
                        dist_ad = euclidean3d(donor.coords, hba.coords)
                        dist_ah = euclidean3d(h_atom.coords, hba.coords)

                        if dist_ad > config.HBOND_DIST_MAX:
                            continue

                        # Calculate angle
                        vec_hd = vector(h_atom.coords, donor.coords)
                        vec_ha = vector(h_atom.coords, hba.coords)
                        angle = vecangle(vec_hd, vec_ha)

                        if angle < config.HBOND_DON_ANGLE_MIN:
                            continue

                        interaction = Interaction(
                            type='hbond',
                            res_a_name=donor.resname,
                            res_a_chain=donor.chain,
                            res_a_num=donor.resnum,
                            res_b_name=hba.resname,
                            res_b_chain=hba.chain,
                            res_b_num=hba.resnum,
                            atom_a_name=self._get_atom_name(donor),
                            atom_a_idx=donor.idx,
                            atom_b_name=self._get_atom_name(hba),
                            atom_b_idx=hba.idx,
                            distance=dist_ad,
                            angle=angle,
                            details={
                                'h_atom': self._get_atom_name(h_atom),
                                'h_idx': h_atom.idx,
                                'dist_ah': dist_ah,
                                'type': 'strong' if dist_ad < 3.2 and angle > 140 else 'weak'
                            }
                        )
                        self.interactions['hbond'].append(interaction)

        # Case 2: Residue is acceptor, other is donor
        if residue.hbond_acceptors:
            for acc in residue.hbond_acceptors:
                for donor, h_atoms in all_hbd:
                    for h_atom in h_atoms:
                        dist_ad = euclidean3d(acc.coords, donor.coords)
                        dist_ah = euclidean3d(acc.coords, h_atom.coords)

                        if dist_ad > config.HBOND_DIST_MAX:
                            continue

                        vec_hd = vector(h_atom.coords, donor.coords)
                        vec_ha = vector(h_atom.coords, acc.coords)
                        angle = vecangle(vec_hd, vec_ha)

                        if angle < config.HBOND_DON_ANGLE_MIN:
                            continue

                        interaction = Interaction(
                            type='hbond',
                            res_a_name=acc.resname,
                            res_a_chain=acc.chain,
                            res_a_num=acc.resnum,
                            res_b_name=donor.resname,
                            res_b_chain=donor.chain,
                            res_b_num=donor.resnum,
                            atom_a_name=self._get_atom_name(acc),
                            atom_a_idx=acc.idx,
                            atom_b_name=self._get_atom_name(donor),
                            atom_b_idx=donor.idx,
                            distance=dist_ad,
                            angle=angle,
                            details={
                                'h_atom': self._get_atom_name(h_atom),
                                'h_idx': h_atom.idx,
                                'dist_ah': dist_ah,
                                'type': 'strong' if dist_ad < 3.2 and angle > 140 else 'weak'
                            }
                        )
                        self.interactions['hbond'].append(interaction)

    def _detect_hbonds_without_h(self, residue: Residue, all_hba: List, all_hbd: List):
        """
        Detect H-bonds without explicit hydrogens (for standard PDB files).
        Uses distance-only criteria between donor and acceptor heavy atoms.
        """
        # Case 1: Residue is donor, other is acceptor
        if residue.hbond_donors:
            for donor, _ in residue.hbond_donors:
                for hba in all_hba:
                    # Skip if same atom
                    if donor.idx == hba.idx:
                        continue

                    dist_ad = euclidean3d(donor.coords, hba.coords)

                    # Use relaxed distance criteria for H-bond detection
                    # Typical H-bond: D...A distance 2.5-3.5 Å
                    # Without H: D...A distance 2.5-3.5 Å (slightly relaxed)
                    if dist_ad < 2.5 or dist_ad > 3.5:
                        continue

                    interaction = Interaction(
                        type='hbond',
                        res_a_name=donor.resname,
                        res_a_chain=donor.chain,
                        res_a_num=donor.resnum,
                        res_b_name=hba.resname,
                        res_b_chain=hba.chain,
                        res_b_num=hba.resnum,
                        atom_a_name=self._get_atom_name(donor),
                        atom_a_idx=donor.idx,
                        atom_b_name=self._get_atom_name(hba),
                        atom_b_idx=hba.idx,
                        distance=dist_ad,
                        angle=None,
                        details={
                            'type': 'heavy_atom',
                            'note': 'No explicit H, distance-only criteria'
                        }
                    )
                    self.interactions['hbond'].append(interaction)

        # Case 2: Residue is acceptor, other is donor
        if residue.hbond_acceptors:
            for acc in residue.hbond_acceptors:
                for donor, _ in all_hbd:
                    # Skip if same atom
                    if acc.idx == donor.idx:
                        continue

                    dist_ad = euclidean3d(acc.coords, donor.coords)

                    if dist_ad < 2.5 or dist_ad > 3.5:
                        continue

                    interaction = Interaction(
                        type='hbond',
                        res_a_name=acc.resname,
                        res_a_chain=acc.chain,
                        res_a_num=acc.resnum,
                        res_b_name=donor.resname,
                        res_b_chain=donor.chain,
                        res_b_num=donor.resnum,
                        atom_a_name=self._get_atom_name(acc),
                        atom_a_idx=acc.idx,
                        atom_b_name=self._get_atom_name(donor),
                        atom_b_idx=donor.idx,
                        distance=dist_ad,
                        angle=None,
                        details={
                            'type': 'heavy_atom',
                            'note': 'No explicit H, distance-only criteria'
                        }
                    )
                    self.interactions['hbond'].append(interaction)
    
    def _detect_saltbridges(self, residue: Residue, dist_matrix: np.ndarray):
        """Detect salt bridges"""
        if not (residue.pos_charged or residue.neg_charged):
            return
        
        all_pos = self.atom_props.get_pos_charged()
        all_neg = self.atom_props.get_neg_charged()
        
        # Case 1: Residue is positive, other is negative
        if residue.pos_charged:
            for pos_atom in residue.pos_charged:
                for neg_atom in all_neg:
                    distance = euclidean3d(pos_atom.coords, neg_atom.coords)
                    
                    if distance < config.SALTBRIDGE_DIST_MAX:
                        interaction = Interaction(
                            type='saltbridge',
                            res_a_name=pos_atom.resname,
                            res_a_chain=pos_atom.chain,
                            res_a_num=pos_atom.resnum,
                            res_b_name=neg_atom.resname,
                            res_b_chain=neg_atom.chain,
                            res_b_num=neg_atom.resnum,
                            atom_a_name=self._get_atom_name(pos_atom),
                            atom_a_idx=pos_atom.idx,
                            atom_b_name=self._get_atom_name(neg_atom),
                            atom_b_idx=neg_atom.idx,
                            distance=distance,
                            angle=None,
                            details={'charge_type': 'pos-neg'}
                        )
                        self.interactions['saltbridge'].append(interaction)
        
        # Case 2: Residue is negative, other is positive
        if residue.neg_charged:
            for neg_atom in residue.neg_charged:
                for pos_atom in all_pos:
                    distance = euclidean3d(neg_atom.coords, pos_atom.coords)
                    
                    if distance < config.SALTBRIDGE_DIST_MAX:
                        interaction = Interaction(
                            type='saltbridge',
                            res_a_name=neg_atom.resname,
                            res_a_chain=neg_atom.chain,
                            res_a_num=neg_atom.resnum,
                            res_b_name=pos_atom.resname,
                            res_b_chain=pos_atom.chain,
                            res_b_num=pos_atom.resnum,
                            atom_a_name=self._get_atom_name(neg_atom),
                            atom_a_idx=neg_atom.idx,
                            atom_b_name=self._get_atom_name(pos_atom),
                            atom_b_idx=pos_atom.idx,
                            distance=distance,
                            angle=None,
                            details={'charge_type': 'neg-pos'}
                        )
                        self.interactions['saltbridge'].append(interaction)
    
    def _detect_pistacking(self, residue: Residue, dist_matrix: np.ndarray):
        """Detect pi-stacking interactions"""
        if not residue.rings:
            return

        all_rings = self.atom_props.rings

        for ring_a in residue.rings:
            for ring_b in all_rings:
                # Skip same ring (compare first atom index)
                if ring_a['indices'][0] == ring_b['indices'][0]:
                    continue

                # Calculate distance between centers
                distance = euclidean3d(ring_a['center'], ring_b['center'])
                
                if distance > config.PISTACK_DIST_MAX:
                    continue
                
                # Calculate angle between normals
                angle = np.arccos(np.clip(
                    np.dot(ring_a['normal'], ring_b['normal']), -1.0, 1.0
                )) * 180 / np.pi
                
                # Determine type
                if angle < config.PISTACK_ANG_DEV or angle > (180 - config.PISTACK_ANG_DEV):
                    ptype = 'parallel'
                elif (90 - config.PISTACK_ANG_DEV) < angle < (90 + config.PISTACK_ANG_DEV):
                    ptype = 'perpendicular'
                else:
                    continue
                
                # Calculate offset (distance between ring centers projected onto ring A plane)
                vec_centers = ring_b['center'] - ring_a['center']
                # Offset is the distance from the center of ring B to the plane of ring A
                offset = np.abs(np.dot(vec_centers, ring_a['normal'])) / np.linalg.norm(ring_a['normal'])
                
                if offset > config.PISTACK_OFFSET_MAX:
                    continue
                
                # Get representative atoms for residue info
                atom_a = self.atom_container[ring_a['indices'][0]]
                atom_b = self.atom_container[ring_b['indices'][0]]
                
                interaction = Interaction(
                    type='pistacking',
                    res_a_name=atom_a.resname,
                    res_a_chain=atom_a.chain,
                    res_a_num=atom_a.resnum,
                    res_b_name=atom_b.resname,
                    res_b_chain=atom_b.chain,
                    res_b_num=atom_b.resnum,
                    atom_a_name='RING',
                    atom_a_idx=ring_a['indices'][0],
                    atom_b_name='RING',
                    atom_b_idx=ring_b['indices'][0],
                    distance=distance,
                    angle=angle,
                    details={
                        'type': ptype,
                        'offset': offset,
                        'ring_a_atoms': ring_a['indices'],
                        'ring_b_atoms': ring_b['indices']
                    }
                )
                self.interactions['pistacking'].append(interaction)
    
    def _detect_pication(self, residue: Residue, dist_matrix: np.ndarray):
        """Detect pi-cation interactions"""
        if not (residue.rings or residue.pos_charged):
            return
        
        all_rings = self.atom_props.rings
        all_pos = self.atom_props.get_pos_charged()
        
        # Case 1: Residue has ring, other has positive charge
        if residue.rings:
            for ring in residue.rings:
                for pos_atom in all_pos:
                    distance = euclidean3d(ring['center'], pos_atom.coords)
                    
                    if distance < config.PICATION_DIST_MAX:
                        atom_a = self.atom_container[ring['indices'][0]]
                        
                        interaction = Interaction(
                            type='pication',
                            res_a_name=atom_a.resname,
                            res_a_chain=atom_a.chain,
                            res_a_num=atom_a.resnum,
                            res_b_name=pos_atom.resname,
                            res_b_chain=pos_atom.chain,
                            res_b_num=pos_atom.resnum,
                            atom_a_name='RING',
                            atom_a_idx=ring['indices'][0],
                            atom_b_name=self._get_atom_name(pos_atom),
                            atom_b_idx=pos_atom.idx,
                            distance=distance,
                            angle=None,
                            details={'ring_center': ring['center']}
                        )
                        self.interactions['pication'].append(interaction)
        
        # Case 2: Residue has positive charge, other has ring
        if residue.pos_charged:
            for pos_atom in residue.pos_charged:
                for ring in all_rings:
                    distance = euclidean3d(pos_atom.coords, ring['center'])
                    
                    if distance < config.PICATION_DIST_MAX:
                        atom_b = self.atom_container[ring['indices'][0]]
                        
                        interaction = Interaction(
                            type='pication',
                            res_a_name=pos_atom.resname,
                            res_a_chain=pos_atom.chain,
                            res_a_num=pos_atom.resnum,
                            res_b_name=atom_b.resname,
                            res_b_chain=atom_b.chain,
                            res_b_num=atom_b.resnum,
                            atom_a_name=self._get_atom_name(pos_atom),
                            atom_a_idx=pos_atom.idx,
                            atom_b_name='RING',
                            atom_b_idx=ring['indices'][0],
                            distance=distance,
                            angle=None,
                            details={'ring_center': ring['center']}
                        )
                        self.interactions['pication'].append(interaction)
    
    def _detect_halogen(self, residue: Residue, dist_matrix: np.ndarray):
        """Detect halogen bonds"""
        if not (residue.halogen_donors or residue.halogen_acceptors):
            return
        
        all_donors = [(self.atom_container[idx], htype) for idx, htype in self.atom_props.halogen_donors.items()]
        all_acceptors = [self.atom_container[idx] for idx in self.atom_props.halogen_acceptors]
        
        # Case 1: Residue is donor
        if residue.halogen_donors:
            for donor, htype in residue.halogen_donors:
                for acc in all_acceptors:
                    distance = euclidean3d(donor.coords, acc.coords)
                    
                    if distance > config.HALOGEN_DIST_MAX:
                        continue
                    
                    # Find carbon bonded to halogen
                    c_atom = None
                    for neighbor in pybel.ob.OBAtomAtomIter(donor.obatom):
                        if neighbor.GetAtomicNum() == 6:
                            c_atom = neighbor
                            break
                    
                    if c_atom is None:
                        continue
                    
                    # Calculate angle
                    c_coords = np.array([c_atom.GetX(), c_atom.GetY(), c_atom.GetZ()])
                    vec_cd = vector(c_coords, donor.coords)
                    vec_ca = vector(c_coords, acc.coords)
                    angle = vecangle(vec_cd, vec_ca)
                    
                    if angle < (config.HALOGEN_DON_ANGLE - config.HALOGEN_ANGLE_DEV):
                        continue
                    
                    interaction = Interaction(
                        type='halogen',
                        res_a_name=donor.resname,
                        res_a_chain=donor.chain,
                        res_a_num=donor.resnum,
                        res_b_name=acc.resname,
                        res_b_chain=acc.chain,
                        res_b_num=acc.resnum,
                        atom_a_name=self._get_atom_name(donor),
                        atom_a_idx=donor.idx,
                        atom_b_name=self._get_atom_name(acc),
                        atom_b_idx=acc.idx,
                        distance=distance,
                        angle=angle,
                        details={'halogen_type': htype}
                    )
                    self.interactions['halogen'].append(interaction)
        
        # Case 2: Residue is acceptor
        if residue.halogen_acceptors:
            for acc in residue.halogen_acceptors:
                for donor, htype in all_donors:
                    distance = euclidean3d(acc.coords, donor.coords)
                    
                    if distance > config.HALOGEN_DIST_MAX:
                        continue
                    
                    # Find carbon bonded to halogen
                    c_atom = None
                    for neighbor in pybel.ob.OBAtomAtomIter(donor.obatom):
                        if neighbor.GetAtomicNum() == 6:
                            c_atom = neighbor
                            break
                    
                    if c_atom is None:
                        continue
                    
                    c_coords = np.array([c_atom.GetX(), c_atom.GetY(), c_atom.GetZ()])
                    vec_cd = vector(c_coords, donor.coords)
                    vec_ca = vector(c_coords, acc.coords)
                    angle = vecangle(vec_cd, vec_ca)
                    
                    if angle < (config.HALOGEN_DON_ANGLE - config.HALOGEN_ANGLE_DEV):
                        continue
                    
                    interaction = Interaction(
                        type='halogen',
                        res_a_name=acc.resname,
                        res_a_chain=acc.chain,
                        res_a_num=acc.resnum,
                        res_b_name=donor.resname,
                        res_b_chain=donor.chain,
                        res_b_num=donor.resnum,
                        atom_a_name=self._get_atom_name(acc),
                        atom_a_idx=acc.idx,
                        atom_b_name=self._get_atom_name(donor),
                        atom_b_idx=donor.idx,
                        distance=distance,
                        angle=angle,
                        details={'halogen_type': htype}
                    )
                    self.interactions['halogen'].append(interaction)
    
    def _detect_metal(self, residue: Residue, dist_matrix: np.ndarray):
        """Detect metal complexation"""
        if not (residue.metal_atoms or residue.metal_binding_atoms):
            return
        
        all_metals = self.atom_props.get_metals()
        all_binding = self.atom_props.get_metal_binding()
        
        # Case 1: Residue is metal
        if residue.metal_atoms:
            for metal in residue.metal_atoms:
                for binding in all_binding:
                    distance = euclidean3d(metal.coords, binding.coords)
                    
                    if distance < config.METAL_DIST_MAX:
                        interaction = Interaction(
                            type='metal',
                            res_a_name=metal.resname,
                            res_a_chain=metal.chain,
                            res_a_num=metal.resnum,
                            res_b_name=binding.resname,
                            res_b_chain=binding.chain,
                            res_b_num=binding.resnum,
                            atom_a_name=self._get_atom_name(metal),
                            atom_a_idx=metal.idx,
                            atom_b_name=self._get_atom_name(binding),
                            atom_b_idx=binding.idx,
                            distance=distance,
                            angle=None,
                            details={}
                        )
                        self.interactions['metal'].append(interaction)
        
        # Case 2: Residue has metal-binding atoms
        if residue.metal_binding_atoms:
            for binding in residue.metal_binding_atoms:
                for metal in all_metals:
                    distance = euclidean3d(binding.coords, metal.coords)
                    
                    if distance < config.METAL_DIST_MAX:
                        interaction = Interaction(
                            type='metal',
                            res_a_name=binding.resname,
                            res_a_chain=binding.chain,
                            res_a_num=binding.resnum,
                            res_b_name=metal.resname,
                            res_b_chain=metal.chain,
                            res_b_num=metal.resnum,
                            atom_a_name=self._get_atom_name(binding),
                            atom_a_idx=binding.idx,
                            atom_b_name=self._get_atom_name(metal),
                            atom_b_idx=metal.idx,
                            distance=distance,
                            angle=None,
                            details={}
                        )
                        self.interactions['metal'].append(interaction)
    
    def _detect_water_bridges(self):
        """Detect water bridges (water mediating between two molecules)"""
        # Find water residues
        water_residues = [r for r in self.residues if r.is_water]
        
        for water_res in water_residues:
            # Collect all H-bonds involving this water
            water_hbonds = []
            for hbond in self.interactions['hbond']:
                if (hbond.res_a_name == water_res.resname and 
                    hbond.res_a_chain == water_res.chain and
                    hbond.res_a_num == water_res.resnum):
                    water_hbonds.append(hbond)
                elif (hbond.res_b_name == water_res.resname and 
                      hbond.res_b_chain == water_res.chain and
                      hbond.res_b_num == water_res.resnum):
                    water_hbonds.append(hbond)
            
            # Check if water bridges two different residues
            if len(water_hbonds) >= 2:
                partner_residues = set()
                for hb in water_hbonds:
                    if hb.res_a_name != water_res.resname:
                        partner_residues.add((hb.res_a_name, hb.res_a_chain, hb.res_a_num))
                    if hb.res_b_name != water_res.resname:
                        partner_residues.add((hb.res_b_name, hb.res_b_chain, hb.res_b_num))
                
                if len(partner_residues) >= 2:
                    # This is a water bridge
                    for hb in water_hbonds:
                        interaction = Interaction(
                            type='water_bridge',
                            res_a_name=hb.res_a_name,
                            res_a_chain=hb.res_a_chain,
                            res_a_num=hb.res_a_num,
                            res_b_name=hb.res_b_name,
                            res_b_chain=hb.res_b_chain,
                            res_b_num=hb.res_b_num,
                            atom_a_name=hb.atom_a_name,
                            atom_a_idx=hb.atom_a_idx,
                            atom_b_name=hb.atom_b_name,
                            atom_b_idx=hb.atom_b_idx,
                            distance=hb.distance,
                            angle=hb.angle,
                            details={**hb.details, 'water_residue': water_res.resid}
                        )
                        self.interactions['water_bridge'].append(interaction)
    
    def _remove_duplicates(self):
        """Remove duplicate interactions (A-B and B-A)"""
        for itype in self.interactions:
            seen = set()
            unique = []
            for inter in self.interactions[itype]:
                # Create canonical key (sorted by atom indices)
                key = tuple(sorted([inter.atom_a_idx, inter.atom_b_idx]))
                if key not in seen:
                    seen.add(key)
                    unique.append(inter)
            self.interactions[itype] = unique
    
    def _get_atom_name(self, atom_info) -> str:
        """Get atom name from OBAtom"""
        residue = atom_info.obatom.GetResidue()
        if residue:
            return residue.GetAtomID(atom_info.obatom).strip()
        return str(atom_info.atomic_num)
    
    def _log_summary(self):
        """Log summary of detected interactions"""
        logger.info('Interaction detection complete:')
        for itype, interactions in self.interactions.items():
            if interactions:
                logger.info(f'  {itype}: {len(interactions)}')