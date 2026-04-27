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
            'hbond_possible': [],  # H-bonds filtered by refinement (salt bridge, duplicate donors)
            'hbond_heavy_atom': [],  # H-bonds detected without explicit hydrogens (less reliable)
            'saltbridge': [],
            'pistacking': [],
            'pication': [],
            'halogen': [],
            'metal': [],
            'water_bridge': [],
            'water_bridge_possible': [],  # PLIP-style water bridges (distance-based)
        }
    
    def detect_all(self, verbose: bool = False) -> Dict[str, List[Interaction]]:
        """Main entry point for detection"""
        logger.info(f'Starting unified interaction detection for {len(self.residues)} residues...')
        
        # First, aggregate properties to residues
        self._aggregate_properties_to_residues()
        
        # Pre-compute cached data for performance
        self._precompute_cached_data()
        
        # Detect interactions for each residue
        for residue in tqdm(self.residues, desc='Processing residues', disable=not verbose):            
            self._detect_for_residue(residue)
        
        # Remove duplicates (each interaction detected twice: A-B and B-A)
        self._remove_duplicates()

        # Refine hydrogen bonds (filter by salt bridges and duplicate donors)
        self._refine_hbonds()

        # Detect water bridges (H-bond based)
        self._detect_water_bridges()

        # Detect PLIP-style water bridges (distance+angle based)
        self._detect_water_bridges_plip_style()

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
    
    def _precompute_cached_data(self):
        """Pre-compute and cache data for performance optimization.
        
        This method is called once before processing all residues to cache
        expensive-to-compute data structures.
        """
        # Get the array-based index mapping from atom_container
        self._idx_array = self.atom_container.idx_to_array_pos_array
        
        # Cache hydrophobic atoms for _detect_hydrophobic
        all_hydrophobic = self.atom_props.get_hydrophobic()
        self._hydrophobic_atoms_list = all_hydrophobic
        self._hydrophobic_coords = np.array([atom.coords for atom in all_hydrophobic])
        # Pre-compute hydrophobic mask using array-based indexing
        self._hydrophobic_mask = self._create_atom_mask(
            [atom.idx for atom in all_hydrophobic]
        )
        
        # Cache H-bond donors and acceptors for _detect_hbonds
        self._all_hba = self.atom_props.get_hba()
        self._all_hbd = self.atom_props.get_hbd()
        # Pre-compute whether we have explicit hydrogens
        self._has_explicit_h = any(len(h_atoms) > 0 for _, h_atoms in self._all_hbd)
        
        # Pre-compute H-bond related coordinates for vectorized operations
        if self._has_explicit_h:
            # Cache all_hba coordinates for Case 1
            self._all_hba_coords = np.array([hba.coords for hba in self._all_hba])
            # Pre-compute HBA mask
            self._hba_mask = self._create_atom_mask([atom.idx for atom in self._all_hba])
            
            # Flatten and cache all_hbd pairs for Case 2
            self._all_hbd_pairs = []
            for donor, h_atoms in self._all_hbd:
                for h_atom in h_atoms:
                    self._all_hbd_pairs.append((donor, h_atom))
            
            if self._all_hbd_pairs:
                self._all_hbd_don_coords = np.array([pair[0].coords for pair in self._all_hbd_pairs])
                self._all_hbd_h_coords = np.array([pair[1].coords for pair in self._all_hbd_pairs])
            else:
                self._all_hbd_don_coords = np.array([]).reshape(0, 3)
                self._all_hbd_h_coords = np.array([]).reshape(0, 3)
        
        # Pre-compute salt bridge atom masks
        all_pos = self.atom_props.get_pos_charged()
        all_neg = self.atom_props.get_neg_charged()
        self._pos_charged_mask = self._create_atom_mask([atom.idx for atom in all_pos])
        self._neg_charged_mask = self._create_atom_mask([atom.idx for atom in all_neg])
        
        # Pre-compute metal and metal-binding atom masks
        all_metals = self.atom_props.get_metals()
        all_metal_binding = self.atom_props.get_metal_binding()
        self._metal_mask = self._create_atom_mask([atom.idx for atom in all_metals])
        self._metal_binding_mask = self._create_atom_mask([atom.idx for atom in all_metal_binding])
    
    def _create_atom_mask(self, atom_idxs: List[int]) -> np.ndarray:
        """Create a boolean mask for specified atom indices using array-based indexing.
        
        Args:
            atom_idxs: List of OpenBabel atom indices
            
        Returns:
            Boolean array where True indicates the atom is in the list
        """
        if self._idx_array is None or len(atom_idxs) == 0:
            return np.array([], dtype=bool)
        
        mask = np.zeros(len(self._idx_array), dtype=bool)
        # Filter valid indices
        valid_idxs = [idx for idx in atom_idxs if idx < len(self._idx_array)]
        if valid_idxs:
            mask[valid_idxs] = True
        return mask
    
    def _detect_for_residue(self, residue: Residue):
        """Detect all interactions for a single residue"""
        if not residue.atoms:
            return
        
        # Get residue coordinates
        res_coords = np.array([a.coords for a in residue.atoms])
        
        # Detect each interaction type using vectorized operations
        self._detect_hydrophobic(residue, res_coords)
        self._detect_hbonds(residue, res_coords)
        self._detect_saltbridges(residue, res_coords)
        self._detect_pistacking(residue, res_coords)
        self._detect_pication(residue, res_coords)
        self._detect_halogen(residue, res_coords)
        self._detect_metal(residue, res_coords)
    
    def _is_same_residue(self, atom_a, atom_b) -> bool:
        """Check if two atoms belong to the same residue"""
        return (atom_a.resname == atom_b.resname and 
                atom_a.chain == atom_b.chain and 
                atom_a.resnum == atom_b.resnum)
    
    def _should_skip_interaction(self, residue: Residue, atom_a, atom_b) -> bool:
        """Determine if an interaction should be skipped due to self-filtering"""
        if not residue.should_filter_self():
            return False  # Ligands don't filter self
        return self._is_same_residue(atom_a, atom_b)
    
    def _detect_hydrophobic(self, residue: Residue, res_coords: np.ndarray):
        """Detect hydrophobic interactions
        
        Optimized: Uses vectorized numpy operations and pre-computed masks
        for fast distance calculations without euclidean3d calls.
        """
        if not residue.hydrophobic_atoms:
            return
        
        # Get coordinates for residue's hydrophobic atoms
        res_hydrophobic_coords = np.array([atom.coords for atom in residue.hydrophobic_atoms])
        
        # Compute distances: vectorized operation [n_res_hydrophobic, n_all_hydrophobic]
        dist_matrix_hydrophobic = np.sqrt(
            np.sum((res_hydrophobic_coords[:, np.newaxis, :] - 
                   self._hydrophobic_coords[np.newaxis, :, :]) ** 2, axis=2)
        )
        
        # Apply distance filter using vectorized operations
        valid_mask = (dist_matrix_hydrophobic < config.HYDROPH_DIST_MAX) & \
                     (dist_matrix_hydrophobic > config.MIN_DIST)
        
        # Get indices of valid pairs
        valid_pairs = np.argwhere(valid_mask)
        
        # Process valid pairs
        for i, j in valid_pairs:
            atom_a = residue.hydrophobic_atoms[i]
            atom_b = self._hydrophobic_atoms_list[j]
            
            # Skip if same residue (unless it's a ligand)
            if self._should_skip_interaction(residue, atom_a, atom_b):
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
                distance=dist_matrix_hydrophobic[i, j],
                angle=None,
                details={}
            )
            self.interactions['hydrophobic'].append(interaction)
    
    def _detect_hbonds(self, residue: Residue, res_coords: np.ndarray):
        """Detect hydrogen bonds"""
        if not (residue.hbond_acceptors or residue.hbond_donors):
            return

        # Use cached H-bond donors and acceptors (pre-computed in _precompute_cached_data)
        all_hba = self._all_hba
        all_hbd = self._all_hbd

        if self._has_explicit_h:
            # Use explicit hydrogen geometry
            self._detect_hbonds_with_h(residue, all_hba, all_hbd)
        elif config.ALLOW_HEAVY_ATOM_HBOND:
            # Use distance-only detection for heavy atom H-bonds (optional, less reliable)
            logger.info("Using heavy atom H-bond detection (distance-only). "
                       "Note: Results may be less reliable without explicit hydrogens.")
            self._detect_hbonds_without_h(residue, all_hba, all_hbd)
        else:
            # Skip H-bond detection without explicit hydrogens (default behavior)
            # H-bond detection requires explicit hydrogens for scientific reliability
            logger.warning("No explicit hydrogens found in structure. "
                          "H-bond detection skipped. "
                          "Please provide a protonated PDB file or use NOHYDRO=False for automatic protonation. "
                          "Alternatively, set config.ALLOW_HEAVY_ATOM_HBOND=True for distance-only detection.")
            return

    def _detect_hbonds_with_h(self, residue: Residue, all_hba: List, all_hbd: List):
        """Detect H-bonds using explicit hydrogen coordinates.
        
        Optimized: Uses vectorized numpy operations for distance and angle calculations
        instead of individual euclidean3d and vecangle calls.
        """
        # Case 1: Residue is donor, other is acceptor
        if residue.hbond_donors:
            self._detect_hbonds_case1_vectorized(residue, all_hba)
        
        # Case 2: Residue is acceptor, other is donor
        if residue.hbond_acceptors:
            self._detect_hbonds_case2_vectorized(residue, all_hbd)
    
    def _detect_hbonds_case1_vectorized(self, residue: Residue, all_hba: List):
        """Case 1: Residue is donor, other is acceptor (vectorized)"""
        # Flatten all (donor, h_atom) pairs from residue
        donor_h_pairs = []
        for donor, h_atoms in residue.hbond_donors:
            for h_atom in h_atoms:
                donor_h_pairs.append((donor, h_atom))
        
        if not donor_h_pairs or not all_hba:
            return
        
        # Pre-extract coordinates as numpy arrays (residue-specific)
        don_coords = np.array([pair[0].coords for pair in donor_h_pairs])  # [n_pairs, 3]
        h_coords = np.array([pair[1].coords for pair in donor_h_pairs])    # [n_pairs, 3]
        # Use cached all_hba coordinates (global, pre-computed)
        acc_coords = self._all_hba_coords  # [n_acc, 3]
        
        # Vectorized distance calculation using broadcasting
        # dist_ad[i, j] = distance between donor i and acceptor j
        diff_ad = don_coords[:, np.newaxis, :] - acc_coords[np.newaxis, :, :]  # [n_pairs, n_acc, 3]
        dist_ad_matrix = np.sqrt(np.sum(diff_ad ** 2, axis=2))  # [n_pairs, n_acc]
        
        # dist_ah[i, j] = distance between hydrogen i and acceptor j
        diff_ah = h_coords[:, np.newaxis, :] - acc_coords[np.newaxis, :, :]  # [n_pairs, n_acc, 3]
        dist_ah_matrix = np.sqrt(np.sum(diff_ah ** 2, axis=2))  # [n_pairs, n_acc]
        
        # Filter by distance criteria
        dist_mask = (dist_ad_matrix > config.MIN_DIST) & (dist_ad_matrix < config.HBOND_DIST_MAX)
        
        # Vectorized angle calculation
        # Vector from H to D: don_coords - h_coords
        vec_hd = don_coords - h_coords  # [n_pairs, 3]
        # Vector from H to A: acc_coords - h_coords (for each pair)
        vec_ha = acc_coords[np.newaxis, :, :] - h_coords[:, np.newaxis, :]  # [n_pairs, n_acc, 3]
        
        # Compute angles using dot product
        norm_hd = np.linalg.norm(vec_hd, axis=1)  # [n_pairs]
        norm_ha = np.linalg.norm(vec_ha, axis=2)  # [n_pairs, n_acc]
        
        # Avoid division by zero
        norm_hd_safe = np.where(norm_hd == 0, 1, norm_hd)
        norm_ha_safe = np.where(norm_ha == 0, 1, norm_ha)
        
        # Dot product
        dot_product = np.sum(vec_ha * vec_hd[:, np.newaxis, :], axis=2)  # [n_pairs, n_acc]
        
        cos_angle = dot_product / (norm_hd_safe[:, np.newaxis] * norm_ha_safe)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clip to avoid numerical errors
        angle_matrix = np.degrees(np.arccos(cos_angle))  # [n_pairs, n_acc]
        
        # Filter by angle criteria
        angle_mask = angle_matrix > config.HBOND_DON_ANGLE_MIN
        
        # Combine masks
        valid_mask = dist_mask & angle_mask
        
        # Get indices of valid pairs
        valid_indices = np.argwhere(valid_mask)  # [N, 2] where each row is [pair_idx, acc_idx]
        
        # Process only valid pairs
        for pair_idx, acc_idx in valid_indices:
            donor, h_atom = donor_h_pairs[pair_idx]
            hba = all_hba[acc_idx]
            
            # Skip if same residue (unless it's a ligand)
            if self._should_skip_interaction(residue, donor, hba):
                continue
            
            dist_ad = dist_ad_matrix[pair_idx, acc_idx]
            dist_ah = dist_ah_matrix[pair_idx, acc_idx]
            angle = angle_matrix[pair_idx, acc_idx]
            
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
                    'type': 'strong' if dist_ad < 3.2 and angle > 140 else 'weak',
                    'donor_idx': donor.idx,
                    'acceptor_idx': hba.idx
                }
            )
            self.interactions['hbond'].append(interaction)
    
    def _detect_hbonds_case2_vectorized(self, residue: Residue, all_hbd: List):
        """Case 2: Residue is acceptor, other is donor (vectorized)"""
        # Use cached global donor-hydrogen pairs (pre-computed in _precompute_cached_data)
        donor_h_pairs = self._all_hbd_pairs
        
        if not donor_h_pairs or not residue.hbond_acceptors:
            return
        
        # Use cached global donor/hydrogen coordinates (pre-computed)
        don_coords = self._all_hbd_don_coords  # [n_pairs, 3]
        h_coords = self._all_hbd_h_coords      # [n_pairs, 3]
        # Extract residue-specific acceptor coordinates
        acc_coords = np.array([acc.coords for acc in residue.hbond_acceptors])  # [n_acc, 3]
        
        # Vectorized distance calculation
        # dist_ad[i, j] = distance between donor i and acceptor j
        diff_ad = don_coords[:, np.newaxis, :] - acc_coords[np.newaxis, :, :]  # [n_pairs, n_acc, 3]
        dist_ad_matrix = np.sqrt(np.sum(diff_ad ** 2, axis=2))  # [n_pairs, n_acc]
        
        # dist_ah[i, j] = distance between hydrogen i and acceptor j
        diff_ah = h_coords[:, np.newaxis, :] - acc_coords[np.newaxis, :, :]  # [n_pairs, n_acc, 3]
        dist_ah_matrix = np.sqrt(np.sum(diff_ah ** 2, axis=2))  # [n_pairs, n_acc]
        
        # Filter by distance criteria
        dist_mask = (dist_ad_matrix > config.MIN_DIST) & (dist_ad_matrix < config.HBOND_DIST_MAX)
        
        # Vectorized angle calculation
        vec_hd = don_coords - h_coords  # [n_pairs, 3]
        vec_ha = acc_coords[np.newaxis, :, :] - h_coords[:, np.newaxis, :]  # [n_pairs, n_acc, 3]
        
        norm_hd = np.linalg.norm(vec_hd, axis=1)  # [n_pairs]
        norm_ha = np.linalg.norm(vec_ha, axis=2)  # [n_pairs, n_acc]
        
        norm_hd_safe = np.where(norm_hd == 0, 1, norm_hd)
        norm_ha_safe = np.where(norm_ha == 0, 1, norm_ha)
        
        dot_product = np.sum(vec_ha * vec_hd[:, np.newaxis, :], axis=2)  # [n_pairs, n_acc]
        
        cos_angle = dot_product / (norm_hd_safe[:, np.newaxis] * norm_ha_safe)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_matrix = np.degrees(np.arccos(cos_angle))  # [n_pairs, n_acc]
        
        # Filter by angle criteria
        angle_mask = angle_matrix > config.HBOND_DON_ANGLE_MIN
        
        # Combine masks
        valid_mask = dist_mask & angle_mask
        
        # Get indices of valid pairs
        valid_indices = np.argwhere(valid_mask)
        
        # Process only valid pairs
        for pair_idx, acc_idx in valid_indices:
            donor, h_atom = donor_h_pairs[pair_idx]
            acc = residue.hbond_acceptors[acc_idx]
            
            # Skip if same residue (unless it's a ligand)
            if self._should_skip_interaction(residue, acc, donor):
                continue
            
            dist_ad = dist_ad_matrix[pair_idx, acc_idx]
            dist_ah = dist_ah_matrix[pair_idx, acc_idx]
            angle = angle_matrix[pair_idx, acc_idx]
            
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
                    'type': 'strong' if dist_ad < 3.2 and angle > 140 else 'weak',
                    'donor_idx': donor.idx,
                    'acceptor_idx': acc.idx
                }
            )
            self.interactions['hbond'].append(interaction)

    def _detect_hbonds_without_h(self, residue: Residue, all_hba: List, all_hbd: List):
        """
        Detect H-bonds without explicit hydrogens (for standard PDB files).
        Uses distance-only criteria between donor and acceptor heavy atoms.
        Results are stored separately in 'hbond_heavy_atom' to distinguish from
        standard H-bonds with explicit hydrogens.
        """
        # Case 1: Residue is donor, other is acceptor
        if residue.hbond_donors:
            for donor, _ in residue.hbond_donors:
                for hba in all_hba:
                    # Skip if same atom
                    if donor.idx == hba.idx:
                        continue

                    # Skip if same residue (unless it's a ligand)
                    if self._should_skip_interaction(residue, donor, hba):
                        continue

                    dist_ad = euclidean3d(donor.coords, hba.coords)

                    # Use relaxed distance criteria for H-bond detection
                    # Typical H-bond: D...A distance 2.5-3.5 Å
                    # Without H: D...A distance 2.5-3.5 Å (slightly relaxed)
                    if dist_ad < 2.5 or dist_ad > 3.5:
                        continue

                    interaction = Interaction(
                        type='hbond_heavy_atom',
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
                            'note': 'No explicit H, distance-only criteria (less reliable)',
                            'donor_idx': donor.idx,
                            'acceptor_idx': hba.idx
                        }
                    )
                    self.interactions['hbond_heavy_atom'].append(interaction)

        # Case 2: Residue is acceptor, other is donor
        if residue.hbond_acceptors:
            for acc in residue.hbond_acceptors:
                for donor, _ in all_hbd:
                    # Skip if same atom
                    if acc.idx == donor.idx:
                        continue

                    # Skip if same residue (unless it's a ligand)
                    if self._should_skip_interaction(residue, acc, donor):
                        continue

                    dist_ad = euclidean3d(acc.coords, donor.coords)

                    if dist_ad < 2.5 or dist_ad > 3.5:
                        continue

                    interaction = Interaction(
                        type='hbond_heavy_atom',
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
                            'note': 'No explicit H, distance-only criteria (less reliable)',
                            'donor_idx': donor.idx,
                            'acceptor_idx': acc.idx
                        }
                    )
                    self.interactions['hbond_heavy_atom'].append(interaction)
    
    def _detect_saltbridges(self, residue: Residue, res_coords: np.ndarray):
        """Detect salt bridges between charged residues.

        Stores complete lists of positive and negative atoms for each salt bridge
        to enable atom-level filtering during H-bond refinement (matching PLIP behavior).
        """
        if not (residue.pos_charged or residue.neg_charged):
            return

        all_pos = self.atom_props.get_pos_charged()
        all_neg = self.atom_props.get_neg_charged()

        # Group charged atoms by residue for proper salt bridge detection
        # For phosphate groups, group by P atom to match main PLIP's behavior
        def group_by_residue_smart(atoms, charge_type='positive'):
            """Group atoms by (resname, chain, resnum) with special handling for phosphate groups"""
            groups = {}
            for atom in atoms:
                key = (atom.resname, atom.chain, atom.resnum)

                # Special handling for phosphate groups: group by P atom
                if charge_type == 'negative' and atom.atomic_num == 15:
                    # This is a phosphorus atom - create a sub-group for this phosphate
                    key = (atom.resname, atom.chain, atom.resnum, atom.idx)
                elif charge_type == 'negative':
                    # For oxygen atoms in phosphate groups, find their parent P atom
                    # Check if this atom is connected to a P atom
                    for neighbor in pybel.ob.OBAtomAtomIter(atom.obatom):
                        if neighbor.GetAtomicNum() == 15:  # Phosphorus
                            # Use the P atom's idx as part of the key
                            key = (atom.resname, atom.chain, atom.resnum, neighbor.GetIdx())
                            break

                if key not in groups:
                    groups[key] = []
                groups[key].append(atom)
            return groups

        res_pos_charged = group_by_residue_smart(residue.pos_charged, 'positive')
        res_neg_charged = group_by_residue_smart(residue.neg_charged, 'negative')
        all_pos_grouped = group_by_residue_smart(all_pos, 'positive')
        all_neg_grouped = group_by_residue_smart(all_neg, 'negative')
        
        # Helper function to calculate charge center
        def calc_charge_center(atoms, charge_type='positive'):
            """Calculate charge center, with special handling for phosphate groups"""
            if charge_type == 'negative':
                # For phosphate groups, use P atom's coordinates as center (matching main PLIP)
                p_atoms = [a for a in atoms if a.atomic_num == 15]
                if p_atoms:
                    return p_atoms[0].coords
            # Default: use mean of all atoms
            return np.mean([a.coords for a in atoms], axis=0)

        # Case 1: Residue is positive, other is negative
        if res_pos_charged:
            for res_key, pos_atoms in res_pos_charged.items():
                for other_key, neg_atoms in all_neg_grouped.items():
                    # Skip if same residue
                    if res_key == other_key:
                        continue

                    # Calculate distance between charge centers
                    pos_center = calc_charge_center(pos_atoms, 'positive')
                    neg_center = calc_charge_center(neg_atoms, 'negative')
                    distance = np.linalg.norm(pos_center - neg_center)
                    
                    if distance < config.SALTBRIDGE_DIST_MAX:
                        # Get representative atoms for residue info
                        pos_atom = pos_atoms[0]
                        neg_atom = neg_atoms[0]
                        
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
                            details={
                                'charge_type': 'pos-neg',
                                'positive_atoms': [a.idx for a in pos_atoms],
                                'negative_atoms': [a.idx for a in neg_atoms]
                            }
                        )
                        self.interactions['saltbridge'].append(interaction)
        
        # Case 2: Residue is negative, other is positive
        if res_neg_charged:
            for res_key, neg_atoms in res_neg_charged.items():
                for other_key, pos_atoms in all_pos_grouped.items():
                    # Skip if same residue
                    if res_key == other_key:
                        continue

                    # Calculate distance between charge centers
                    neg_center = calc_charge_center(neg_atoms, 'negative')
                    pos_center = calc_charge_center(pos_atoms, 'positive')
                    distance = np.linalg.norm(neg_center - pos_center)
                    
                    if distance < config.SALTBRIDGE_DIST_MAX:
                        # Get representative atoms for residue info
                        neg_atom = neg_atoms[0]
                        pos_atom = pos_atoms[0]
                        
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
                            details={
                                'charge_type': 'neg-pos',
                                'positive_atoms': [a.idx for a in pos_atoms],
                                'negative_atoms': [a.idx for a in neg_atoms]
                            }
                        )
                        self.interactions['saltbridge'].append(interaction)
    
    def _detect_pistacking(self, residue: Residue, res_coords: np.ndarray):
        """Detect pi-stacking interactions"""
        if not residue.rings:
            return

        all_rings = self.atom_props.rings

        for ring_a in residue.rings:
            for ring_b in all_rings:
                # Skip same ring (compare first atom index)
                if ring_a['indices'][0] == ring_b['indices'][0]:
                    continue
                
                # Get representative atoms for residue info and self-filtering
                atom_a = self.atom_container[ring_a['indices'][0]]
                atom_b = self.atom_container[ring_b['indices'][0]]
                
                # Skip if same residue (unless it's a ligand)
                if self._should_skip_interaction(residue, atom_a, atom_b):
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
                
                # Calculate offset (min of both projection directions)
                # Project ring A center onto ring B plane and measure distance to ring B center
                proj1 = projection(ring_b['normal'], ring_b['center'], ring_a['center'])
                offset1 = euclidean3d(proj1, ring_b['center'])
                # Project ring B center onto ring A plane and measure distance to ring A center
                proj2 = projection(ring_a['normal'], ring_a['center'], ring_b['center'])
                offset2 = euclidean3d(proj2, ring_a['center'])
                # Use the minimum offset (symmetric measure)
                offset = min(offset1, offset2)
                
                if offset > config.PISTACK_OFFSET_MAX:
                    continue
                
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
    
    def _detect_pication(self, residue: Residue, res_coords: np.ndarray):
        """Detect pi-cation interactions

        Geometric criteria:
        - Distance: < PICATION_DIST_MAX (6.0 Å)
        - Offset: < PISTACK_OFFSET_MAX (2.0 Å)
          Offset is the distance from the projection of the charge onto the ring plane
          to the ring center. This ensures the charge is positioned above the ring face.
        """
        if not (residue.rings or residue.pos_charged):
            return

        all_rings = self.atom_props.rings
        all_pos = self.atom_props.get_pos_charged()

        # Case 1: Residue has ring, other has positive charge
        if residue.rings:
            for ring in residue.rings:
                # Only consider aromatic rings for pi-cation interactions
                if not ring.get('is_aromatic', False):
                    continue
                for pos_atom in all_pos:
                    # Skip if same residue (unless it's a ligand)
                    atom_a = self.atom_container[ring['indices'][0]]
                    if self._should_skip_interaction(residue, atom_a, pos_atom):
                        continue

                    distance = euclidean3d(ring['center'], pos_atom.coords)

                    if distance >= config.PICATION_DIST_MAX:
                        continue

                    # Calculate offset: projection of charge onto ring plane to ring center
                    proj = projection(ring['normal'], ring['center'], pos_atom.coords)
                    offset = euclidean3d(proj, ring['center'])

                    if offset >= config.PISTACK_OFFSET_MAX:
                        continue

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
                        details={'ring_center': ring['center'], 'offset': offset}
                    )
                    self.interactions['pication'].append(interaction)

        # Case 2: Residue has positive charge, other has ring
        if residue.pos_charged:
            for pos_atom in residue.pos_charged:
                for ring in all_rings:
                    # Only consider aromatic rings for pi-cation interactions
                    if not ring.get('is_aromatic', False):
                        continue
                    # Skip if same residue (unless it's a ligand)
                    atom_b = self.atom_container[ring['indices'][0]]
                    if self._should_skip_interaction(residue, pos_atom, atom_b):
                        continue

                    distance = euclidean3d(pos_atom.coords, ring['center'])

                    if distance >= config.PICATION_DIST_MAX:
                        continue

                    # Calculate offset: projection of charge onto ring plane to ring center
                    proj = projection(ring['normal'], ring['center'], pos_atom.coords)
                    offset = euclidean3d(proj, ring['center'])

                    if offset >= config.PISTACK_OFFSET_MAX:
                        continue

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
                        details={'ring_center': ring['center'], 'offset': offset}
                    )
                    self.interactions['pication'].append(interaction)
    
    def _detect_halogen(self, residue: Residue, res_coords: np.ndarray):
        """Detect halogen bonds"""
        if not (residue.halogen_donors or residue.halogen_acceptors):
            return
        
        all_donors = [(self.atom_container[idx], htype) for idx, htype in sorted(self.atom_props.halogen_donors.items())]
        all_acceptors = [self.atom_container[idx] for idx in sorted(self.atom_props.halogen_acceptors)]
        
        # Case 1: Residue is donor
        if residue.halogen_donors:
            for donor, htype in residue.halogen_donors:
                for acc in all_acceptors:
                    # Skip if same residue (unless it's a ligand)
                    if self._should_skip_interaction(residue, donor, acc):
                        continue
                    
                    distance = euclidean3d(donor.coords, acc.coords)
                    
                    if distance > config.HALOGEN_DIST_MAX:
                        continue
                    
                    # Find carbon bonded to halogen (sort by index for determinism)
                    c_atoms = []
                    for neighbor in pybel.ob.OBAtomAtomIter(donor.obatom):
                        if neighbor.GetAtomicNum() == 6:
                            c_atoms.append(neighbor)
                    
                    if not c_atoms:
                        continue
                    
                    # Use the carbon with lowest index for determinism
                    c_atom = min(c_atoms, key=lambda x: x.GetIdx())
                    
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
                    # Skip if same residue (unless it's a ligand)
                    if self._should_skip_interaction(residue, acc, donor):
                        continue
                    
                    distance = euclidean3d(acc.coords, donor.coords)
                    
                    if distance > config.HALOGEN_DIST_MAX:
                        continue
                    
                    # Find carbon bonded to halogen (sort by index for determinism)
                    c_atoms = []
                    for neighbor in pybel.ob.OBAtomAtomIter(donor.obatom):
                        if neighbor.GetAtomicNum() == 6:
                            c_atoms.append(neighbor)
                    
                    if not c_atoms:
                        continue
                    
                    # Use the carbon with lowest index for determinism
                    c_atom = min(c_atoms, key=lambda x: x.GetIdx())
                    
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
    
    def _detect_metal(self, residue: Residue, res_coords: np.ndarray):
        """Detect metal complexation"""
        if not (residue.metal_atoms or residue.metal_binding_atoms):
            return
        
        all_metals = self.atom_props.get_metals()
        all_binding = self.atom_props.get_metal_binding()
        
        # Case 1: Residue is metal
        if residue.metal_atoms:
            for metal in residue.metal_atoms:
                for binding in all_binding:
                    # Skip if same residue (unless it's a ligand)
                    if self._should_skip_interaction(residue, metal, binding):
                        continue
                    
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
                    # Skip if same residue (unless it's a ligand)
                    if self._should_skip_interaction(residue, binding, metal):
                        continue
                    
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
        # Find water residues (sorted for deterministic ordering)
        water_residues = sorted([r for r in self.residues if r.is_water],
                                key=lambda r: (r.chain, r.resnum))

        # Combine standard H-bonds and heavy atom H-bonds for water bridge detection
        all_hbonds = self.interactions['hbond'] + self.interactions['hbond_heavy_atom']

        for water_res in water_residues:
            # Collect all H-bonds involving this water
            # Sort by atom indices for deterministic ordering
            water_hbonds = []
            for hbond in sorted(all_hbonds,
                                key=lambda h: (h.atom_a_idx, h.atom_b_idx)):
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

    def _detect_water_bridges_plip_style(self):
        """Detect water bridges using PLIP-style distance+angle criteria

        This method implements the PLIP water bridge detection approach:
        - Acceptor-water: distance check only (2.5-4.1 Å)
        - Donor-water: distance + angle check
        - Water bridge: same water molecule mediates between two residues

        Results are stored in 'water_bridge_possible' to distinguish from
        the stricter H-bond-based water_bridge detection.
        """
        from plip.basic.supplemental import euclidean3d, vecangle, vector

        # Find water residues (sorted for deterministic ordering)
        water_residues = sorted([r for r in self.residues if r.is_water],
                                key=lambda r: (r.chain, r.resnum))

        # Get H-bond acceptors and donors
        acceptors = self.atom_props.hbond_acceptors
        donors = self.atom_props.hbond_donors

        for water_res in water_residues:
            # Get water oxygen atom
            water_o = None
            water_h = []
            for atom in water_res.atoms:
                if atom.atomic_num == 8:  # Oxygen
                    water_o = atom
                elif atom.is_hydrogen:
                    water_h.append(atom)

            if water_o is None:
                continue

            # Find acceptor-water pairs (distance only)
            acc_water_pairs = []
            for acc_idx in acceptors:
                acc_atom = self.atom_container[acc_idx]
                # Skip if same residue
                if (acc_atom.resname == water_res.resname and
                    acc_atom.chain == water_res.chain and
                    acc_atom.resnum == water_res.resnum):
                    continue
                dist = euclidean3d(acc_atom.coords, water_o.coords)
                if config.WATER_BRIDGE_MINDIST <= dist <= config.WATER_BRIDGE_MAXDIST:
                    acc_water_pairs.append((acc_atom, dist))

            # Find donor-water pairs (distance + angle)
            don_water_pairs = []
            for don_idx in donors:
                don_atom = self.atom_container[don_idx]
                # Skip if same residue
                if (don_atom.resname == water_res.resname and
                    don_atom.chain == water_res.chain and
                    don_atom.resnum == water_res.resnum):
                    continue

                # Find hydrogen attached to donor
                don_h = None
                for atom in self.atom_container:
                    if (atom.is_hydrogen and
                        atom.resname == don_atom.resname and
                        atom.chain == don_atom.chain and
                        atom.resnum == don_atom.resnum):
                        # Check if this H is bonded to donor (distance < 1.2 Å)
                        if euclidean3d(atom.coords, don_atom.coords) < 1.2:
                            don_h = atom
                            break

                if don_h is None:
                    continue

                dist = euclidean3d(don_atom.coords, water_o.coords)
                if dist < config.WATER_BRIDGE_MINDIST or dist > config.WATER_BRIDGE_MAXDIST:
                    continue

                # Calculate donor angle (D-H···O)
                d_angle = vecangle(vector(don_h.coords, don_atom.coords),
                                  vector(don_h.coords, water_o.coords))
                if d_angle > config.WATER_BRIDGE_THETA_MIN:
                    don_water_pairs.append((don_atom, don_h, dist, d_angle))

            # Check for water bridges: acceptor-water-donor combinations
            for acc, dist_aw in acc_water_pairs:
                for don, don_h, dist_dw, d_angle in don_water_pairs:
                    # Calculate water angle (A-O-H)
                    if len(water_h) > 0:
                        # Use the closest H to acceptor
                        closest_h = min(water_h,
                                       key=lambda h: euclidean3d(h.coords, acc.coords))
                        w_angle = vecangle(vector(water_o.coords, acc.coords),
                                          vector(water_o.coords, closest_h.coords))
                    else:
                        w_angle = 90.0  # Default if no H found

                    # Check water angle criteria
                    if not (config.WATER_BRIDGE_OMEGA_MIN < w_angle < config.WATER_BRIDGE_OMEGA_MAX):
                        continue

                    # Create interaction
                    interaction = Interaction(
                        type='water_bridge_possible',
                        res_a_name=acc.resname,
                        res_a_chain=acc.chain,
                        res_a_num=acc.resnum,
                        res_b_name=don.resname,
                        res_b_chain=don.chain,
                        res_b_num=don.resnum,
                        atom_a_name=acc.atom_name,
                        atom_a_idx=acc.idx,
                        atom_b_name=don.atom_name,
                        atom_b_idx=don.idx,
                        distance=dist_aw + dist_dw,  # Total distance
                        angle=w_angle,
                        details={
                            'distance_aw': dist_aw,
                            'distance_dw': dist_dw,
                            'd_angle': d_angle,
                            'w_angle': w_angle,
                            'water_residue': water_res.resid,
                            'protisdon': True  # Protein is donor
                        }
                    )
                    self.interactions['water_bridge_possible'].append(interaction)

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

    def _refine_hbonds(self):
        """Refine hydrogen bonds by filtering out those involved in salt bridges and duplicate donors.

        This implements all-atom's unified approach to H-bond refinement:
        1. Filter out H-bonds where donor/acceptor atoms are involved in salt bridges
           - If donor is in a salt bridge's positive atoms AND acceptor is in the same
             salt bridge's negative atoms (or vice versa), filter it out
        2. Keep only one H-bond per donor (the one with largest angle)

        Unlike PLIP, all-atom does NOT distinguish between ligand and protein.
        It treats all residues uniformly, filtering H-bonds based on whether the
        donor/acceptor pair forms a salt bridge, regardless of residue type.

        Filtered H-bonds are moved to 'hbond_possible' instead of being deleted.
        """
        if not self.interactions['hbond']:
            return

        # Build salt bridge lookup: atom_idx -> (partner_residue_atoms, is_positive)
        # For each atom in a salt bridge, store the partner's atoms and whether this atom is positive
        saltbridge_lookup = {}  # atom_idx -> {'partner_neg': set(), 'partner_pos': set()}

        for sb in self.interactions['saltbridge']:
            pos_atoms = set(sb.details.get('positive_atoms', []))
            neg_atoms = set(sb.details.get('negative_atoms', []))

            # For each positive atom, the partner negative atoms are the other side
            for pos_atom in pos_atoms:
                if pos_atom not in saltbridge_lookup:
                    saltbridge_lookup[pos_atom] = {'partner_neg': set(), 'partner_pos': set()}
                saltbridge_lookup[pos_atom]['partner_neg'].update(neg_atoms)

            # For each negative atom, the partner positive atoms are the other side
            for neg_atom in neg_atoms:
                if neg_atom not in saltbridge_lookup:
                    saltbridge_lookup[neg_atom] = {'partner_neg': set(), 'partner_pos': set()}
                saltbridge_lookup[neg_atom]['partner_pos'].update(pos_atoms)

        # First pass: mark H-bonds involved in salt bridges
        marked_hbonds = []
        for hbond in self.interactions['hbond']:
            is_filtered = False
            donor_idx = hbond.details.get('donor_idx')
            acceptor_idx = hbond.details.get('acceptor_idx')

            # Check if donor is in a salt bridge
            if donor_idx in saltbridge_lookup:
                # If donor is positive and acceptor is in partner negative atoms -> filter
                if acceptor_idx in saltbridge_lookup[donor_idx]['partner_neg']:
                    is_filtered = True
                # If donor is negative and acceptor is in partner positive atoms -> filter
                elif acceptor_idx in saltbridge_lookup[donor_idx]['partner_pos']:
                    is_filtered = True

            marked_hbonds.append((hbond, is_filtered))

        # Second pass: keep only one H-bond per donor (largest angle)
        # Sort marked_hbonds by atom indices to ensure deterministic ordering
        marked_hbonds_sorted = sorted(marked_hbonds, key=lambda x: (x[0].atom_a_idx, x[0].atom_b_idx))

        donor_best = {}  # donor_idx -> (angle, hbond)
        for hbond, is_filtered in marked_hbonds_sorted:
            if is_filtered:
                continue
            donor_idx = hbond.details.get('donor_idx')
            if donor_idx is None:
                # Skip H-bonds without donor_idx (heavy atom only H-bonds)
                continue

            current_angle = hbond.angle if hbond.angle is not None else 0.0

            if donor_idx not in donor_best:
                donor_best[donor_idx] = (current_angle, hbond)
            else:
                if donor_best[donor_idx][0] < current_angle:
                    donor_best[donor_idx] = (current_angle, hbond)

        confirmed_hbonds = [hb[1] for hb in donor_best.values()]
        confirmed_set = set(id(hb) for hb in confirmed_hbonds)
        possible_hbonds = [hb for hb, is_filtered in marked_hbonds_sorted if is_filtered or
                          (id(hb) not in confirmed_set)]

        # Update interactions
        self.interactions['hbond'] = confirmed_hbonds
        self.interactions['hbond_possible'] = possible_hbonds

        if possible_hbonds:
            logger.info(f'  Refined H-bonds: {len(confirmed_hbonds)} confirmed, {len(possible_hbonds)} possible')

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