"""
Interaction Detector Module - Residue-based Unified Detection

Provides unified interaction detection using a residue-based approach:
- Iterate over each residue
- Build MxN distance matrix (M=residue atoms, N=all atoms)
- Detect interactions with smart self-filtering
"""

from collections import defaultdict, namedtuple
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from openbabel import pybel
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from tqdm import tqdm

from fplip.all_atom.atom_container import AtomContainer, AtomInfo
from fplip.all_atom.atom_properties import AtomProperties
from fplip.all_atom.residue import Residue
from fplip.basic import config
from fplip.basic.logger import logger
from fplip.basic.supplemental import euclidean3d, projection, vecangle, vector

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
    'objs',              # Python object reference (for API call)
], defaults=(None,))


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
        self.idx_to_pos_array = atom_container.idx_to_array_pos_array
        
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
        
        # Pre-compute charge groups for all residues (for salt bridge detection)
        self._precompute_residue_charge_groups()
        
        # Pre-compute cached data for performance
        self._precompute_cached_data()

        # Detect interactions for each residue
        for residue in tqdm(self.residues, desc='Processing residues', disable=not verbose, leave=False):
            if residue.is_skip:
                continue
            self._detect_for_residue(residue)

        # Remove duplicates (each interaction detected twice: A-B and B-A)
        self._remove_duplicates()

        # Refine hydrogen bonds (filter by salt bridges and duplicate donors)
        self._refine_hbonds()

        # Detect water bridges (H-bond based)
        self._detect_water_bridges(verbose)

        # Detect PLIP-style water bridges (distance+angle based)
        self._detect_water_bridges_plip_style(verbose)

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

        # Aggregate rings using efficient reverse mapping via atom's back-reference
        # This is O(N_rings) instead of O(N_atoms × N_rings)
        self._aggregate_rings_to_residues()

    def _aggregate_rings_to_residues(self):
        """Aggregate rings to residues using atom's back-reference.

        Instead of checking each atom against all rings (O(N_atoms × N_rings)),
        we iterate over rings and assign them to residues containing their atoms.
        A ring may belong to multiple residues if it spans across them.
        """
        # Track which rings have been added to each residue (by first atom index)
        residue_ring_ids = {residue: set() for residue in self.residues}

        for ring in self.atom_props.rings:
            ring_id = ring['indices'][0]  # Use first atom as ring identifier

            # Find all residues that contain atoms from this ring
            ring_residues = set()
            for atom_idx in ring['indices']:
                atom = self.atom_container[atom_idx]
                if atom.residue_obj:
                    ring_residues.add(atom.residue_obj)

            # Add ring to each residue that contains at least one of its atoms
            for residue in ring_residues:
                if ring_id not in residue_ring_ids[residue]:
                    residue.rings.append(ring)
                    residue_ring_ids[residue].add(ring_id)

    def _precompute_residue_charge_groups(self):
        """Pre-compute charge groups for all residues.
        
        This must be called after _aggregate_properties_to_residues() 
        so that residue.pos_charged and residue.neg_charged are populated.
        """
        for residue in self.residues:
            if residue.pos_charged or residue.neg_charged:
                residue._precompute_charge_groups()
    
    def _precompute_cached_data(self):
        """Pre-compute and cache data for performance optimization.
        
        This method is called once before processing all residues to cache
        expensive-to-compute data structures.
        """        
        # Cache hydrophobic atoms for _detect_hydrophobic
        all_hydrophobic = self.atom_props.get_hydrophobic()
        self._hydrophobic_atoms_list = all_hydrophobic
        self._hydrophobic_coords = self.atom_container.get_atom_coords_array_from_atoms(all_hydrophobic)
        
        # Cache H-bond donors and acceptors for _detect_hbonds
        self._all_hba = self.atom_props.get_hba()
        self._all_hbd = self.atom_props.get_hbd()
        # Pre-compute whether we have explicit hydrogens
        self._has_explicit_h = any(len(h_atoms) > 0 for _, h_atoms in self._all_hbd)

        # Pre-compute H-bond related coordinates for vectorized operations
        # Cache all_hba coordinates (used by both _detect_hbonds_with_h and _detect_hbonds_without_h)
        if self._all_hba:
            self._all_hba_coords = self.atom_container.get_atom_coords_array_from_atoms(self._all_hba)
        else:
            self._all_hba_coords = np.array([]).reshape(0, 3)

        # Pre-compute HBA mask
        self._hba_mask = self._create_atom_mask([atom.idx for atom in self._all_hba])

        # Pre-compute all_hbd donor coordinates for _detect_hbonds_without_h Case 2
        # Extract donor atoms (first element of each tuple) from all_hbd
        if self._all_hbd:
            self._all_hbd_donor_atoms = [donor for donor, _ in self._all_hbd]
            self._all_hbd_donor_coords = self.atom_container.get_atom_coords_array_from_atoms(self._all_hbd_donor_atoms)
        else:
            self._all_hbd_donor_atoms = []
            self._all_hbd_donor_coords = np.array([]).reshape(0, 3)

        self._all_hbd_pairs = []
        if self._has_explicit_h:
            # Flatten and cache all_hbd pairs for Case 2
            for donor, h_atoms in self._all_hbd:
                for h_atom in h_atoms:
                    self._all_hbd_pairs.append((donor, h_atom))

            if self._all_hbd_pairs:
                self._all_hbd_don_coords = self.atom_container.get_atom_coords_array_from_atoms([pair[0] for pair in self._all_hbd_pairs])
                self._all_hbd_h_coords = self.atom_container.get_atom_coords_array_from_atoms([pair[1] for pair in self._all_hbd_pairs])
            else:
                self._all_hbd_don_coords = np.array([]).reshape(0, 3)
                self._all_hbd_h_coords = np.array([]).reshape(0, 3)
        
        # Pre-compute salt bridge data
        self.all_pos_atoms = self.atom_props.get_pos_charged()
        self.all_neg_atoms = self.atom_props.get_neg_charged()
        
        # Pre-compute grouped charged atoms and their centers for salt bridge detection
        # This avoids recomputing these for every residue in _detect_saltbridges
        self._all_pos_grouped = self._group_charged_atoms_by_residue(self.all_pos_atoms, 'positive')
        self._all_neg_grouped = self._group_charged_atoms_by_residue(self.all_neg_atoms, 'negative')
        
        # # Pre-compute metal and metal-binding atom masks
        self.all_metals_atoms = self.atom_props.get_metals()
        self.all_metal_binding_atoms = self.atom_props.get_metal_binding()
        
        # Pre-compute halogen bond donors and acceptors
        # This avoids recomputing these for every residue in _detect_halogen
        self._all_halogen_donors = [(self.atom_container[idx], htype) 
                                     for idx, htype in self.atom_props.halogen_donors.items()]
        self._all_halogen_acceptors = [self.atom_container[idx] 
                                        for idx in self.atom_props.halogen_acceptors]
        
        # Pre-compute halogen donor and acceptor coordinates for vectorized operations
        if self._all_halogen_acceptors:
            self._all_halogen_acceptor_coords = self.atom_container.get_atom_coords_array_from_atoms(self._all_halogen_acceptors)
        else:
            self._all_halogen_acceptor_coords = np.array([])
        
        if self._all_halogen_donors:
            self._all_halogen_donor_coords = self.atom_container.get_atom_coords_array_from_atoms([donor for donor, _ in self._all_halogen_donors])
        else:
            self._all_halogen_donor_coords = np.array([])
        
        # Pre-compute metal and metal-binding atom coordinates
        if self.all_metals_atoms:
            self._metals_coords = self.atom_container.get_atom_coords_array_from_atoms(self.all_metals_atoms)
        else:
            self._metals_coords = np.array([]).reshape(0, 3)
        if self.all_metal_binding_atoms:
            self._binding_coords = self.atom_container.get_atom_coords_array_from_atoms(self.all_metal_binding_atoms)
        else:
            self._binding_coords = np.array([]).reshape(0, 3)
        
        # Pre-compute ring data for pistacking and pication detection
        all_rings = self.atom_props.rings
        if all_rings:
            self._all_ring_centers = np.array([r['center'] for r in all_rings], dtype=np.float64)
            self._all_ring_normals = np.array([r['normal'] for r in all_rings], dtype=np.float64)
            self._aromatic_ring_mask = np.array([r.get('is_aromatic', False) for r in all_rings], dtype=bool)
            # Pre-compute aromatic rings list to avoid rebuilding in _detect_pication
            self._aromatic_rings = [r for r in all_rings if r.get('is_aromatic', False)]

            # Pre-compute aromatic ring membership for hydrophobic interaction filtering
            # atom_idx -> set of aromatic ring indices (for filtering intra-ring hydrophobic interactions)
            self._aromatic_ring_atom_sets = {}
            for ring_idx, ring in enumerate(self._aromatic_rings):
                for atom_idx in ring['indices']:
                    if atom_idx not in self._aromatic_ring_atom_sets:
                        self._aromatic_ring_atom_sets[atom_idx] = set()
                    self._aromatic_ring_atom_sets[atom_idx].add(ring_idx)
        else:
            self._all_ring_centers = np.array([]).reshape(0, 3)
            self._all_ring_normals = np.array([]).reshape(0, 3)
            self._aromatic_ring_mask = np.array([])
            self._aromatic_rings = []
            self._aromatic_ring_atom_sets = {}
        # Pre-compute positive charge coordinates for pication detection
        all_pos = self.atom_props.get_pos_charged()
        if all_pos:
            self._pos_coords = np.array([p.coords for p in all_pos], dtype=np.float64)
        else:
            self._pos_coords = np.array([]).reshape(0, 3)
        
        # Pre-compute salt bridge data as arrays for vectorized operations
        # This avoids rebuilding arrays in _detect_saltbridges for each residue
        # For negative charge groups: keys, centers, and atoms list
        if self._all_neg_grouped:
            self._neg_grouped_keys = list(self._all_neg_grouped.keys())
            self._neg_grouped_centers = np.array([center for _, center in self._all_neg_grouped.values()], dtype=np.float64)
            self._neg_grouped_atoms = [atoms for atoms, _ in self._all_neg_grouped.values()]
        else:
            self._neg_grouped_keys = []
            self._neg_grouped_centers = np.array([]).reshape(0, 3)
            self._neg_grouped_atoms = []
        
        # For positive charge groups: keys, centers, and atoms list
        if self._all_pos_grouped:
            self._pos_grouped_keys = list(self._all_pos_grouped.keys())
            self._pos_grouped_centers = np.array([center for _, center in self._all_pos_grouped.values()], dtype=np.float64)
            self._pos_grouped_atoms = [atoms for atoms, _ in self._all_pos_grouped.values()]
        else:
            self._pos_grouped_keys = []
            self._pos_grouped_centers = np.array([]).reshape(0, 3)
            self._pos_grouped_atoms = []
    
    def _create_atom_mask(self, atom_idxs: List[int]) -> np.ndarray:
        """Create a boolean mask for specified atom indices using array-based indexing.
        
        Args:
            atom_idxs: List of OpenBabel atom indices
            
        Returns:
            Boolean array where True indicates the atom is in the list
        """
        if self.idx_to_pos_array is None or len(atom_idxs) == 0:
            return np.array([], dtype=bool)
        
        mask = np.zeros(len(self.idx_to_pos_array), dtype=bool)
        # Filter valid indices
        valid_idxs = [idx for idx in atom_idxs if idx < len(self.idx_to_pos_array)]
        if valid_idxs:
            mask[valid_idxs] = True
        return mask
    
    def _group_charged_atoms_by_residue(self, atoms: List[AtomInfo], charge_type: str) -> Dict:
        """Group charged atoms by residue with special handling for phosphate groups.
        
        Also pre-computes charge centers for each group to avoid repeated calculations.
        
        Args:
            atoms: List of charged atoms
            charge_type: 'positive' or 'negative'
            
        Returns:
            Dictionary mapping residue key to (atoms_list, charge_center)
        """
        # First pass: group atoms by residue key
        groups = defaultdict(list)
        for atom in atoms:
            key = (atom.resname, atom.chain, atom.resnum)
            
            # Special handling for phosphate groups: group by P atom
            if charge_type == 'negative' and atom.atomic_num == 15:
                # This is a phosphorus atom - create a sub-group for this phosphate
                key = (atom.resname, atom.chain, atom.resnum, atom.idx)
            elif charge_type == 'negative':
                # For oxygen atoms in phosphate groups, find their parent P atom
                for neighbor in pybel.ob.OBAtomAtomIter(atom.obatom):
                    if neighbor.GetAtomicNum() == 15:  # Phosphorus
                        key = (atom.resname, atom.chain, atom.resnum, neighbor.GetIdx())
                        break
            
            groups[key].append(atom)
        
        # Second pass: pre-compute charge centers for each group
        result = {}
        for key, atom_list in groups.items():
            # Calculate charge center
            if charge_type == 'negative':
                # For phosphate groups, use P atom's coordinates as center
                p_atoms = [a for a in atom_list if a.atomic_num == 15]
                if p_atoms:
                    center = p_atoms[0].coords
                else:
                    center = np.mean([a.coords for a in atom_list], axis=0)
            else:
                # Default: use mean of all atoms
                center = np.mean([a.coords for a in atom_list], axis=0)
            
            result[key] = (atom_list, center)
        
        return result
    
    def _detect_for_residue(self, residue: Residue):
        """Detect all interactions for a single residue"""
        if not residue.atoms:
            return
        
        # Detect each interaction type using vectorized operations
        self._detect_hydrophobic(residue)
        self._detect_hbonds(residue)
        self._detect_saltbridges(residue)
        self._detect_pistacking(residue)
        self._detect_pication(residue)
        self._detect_halogen(residue)
        self._detect_metal(residue)
    
    def _is_same_residue(self, atom_a, atom_b) -> bool:
        """Check if two atoms belong to the same residue
        
        Optimization: Uses residue_obj reverse reference for O(1) comparison
        instead of O(3) attribute comparisons (resname, chain, resnum).
        Falls back to attribute comparison if residue_obj is not available.
        """
        # Fast path: use object identity if both atoms have residue_obj
        if atom_a.residue_obj is not None and atom_b.residue_obj is not None:
            return atom_a.residue_obj is atom_b.residue_obj
        
        # Fallback: compare by attributes (should not happen in practice)
        return (atom_a.resname == atom_b.resname and 
                atom_a.chain == atom_b.chain and 
                atom_a.resnum == atom_b.resnum)
    
    def _should_skip_interaction(self, residue: Residue, atom_a, atom_b) -> bool:
        """Determine if an interaction should be skipped due to self-filtering"""
        # check if is ligand (not a standered amino acid), if is ligand, do not skip self-interaction
        if not residue.should_filter_self():
            return False  # Ligands don't filter self
        # if is standard amino acid, skip self-interaction if same residue
        return self._is_same_residue(atom_a, atom_b)
    
    def _detect_hydrophobic(self, residue: Residue):
        """Detect hydrophobic interactions
        
        Optimized: Uses vectorized numpy operations and pre-computed masks
        for fast distance calculations without euclidean3d calls.
        """
        if not residue.hydrophobic_atoms:
            return
        
        # Get coordinates for residue's hydrophobic atoms
        res_hydrophobic_coords = self.atom_container.get_atom_coords_array_from_atoms(residue.hydrophobic_atoms)

        # Compute distances using cdist for better performance
        dist_matrix_hydrophobic = cdist(res_hydrophobic_coords, self._hydrophobic_coords)
        
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

            # Skip if both atoms belong to the same aromatic ring
            # (intra-aromatic-ring carbon interactions are covalent bonds, not hydrophobic)
            rings_a = self._aromatic_ring_atom_sets.get(atom_a.idx, set())
            rings_b = self._aromatic_ring_atom_sets.get(atom_b.idx, set())
            if rings_a & rings_b:  # intersection - same aromatic ring
                continue

            # Skip if one atom is connected to the other's aromatic ring via chemical bond
            # (e.g., S attached to benzene ring - S cannot form hydrophobic interaction with ring atoms)
            # A connected to ring R means: A is not in R, but A has a neighbor atom that is in R
            # This filters out 1-3 and 1-4 invalid interaction
            if rings_b:
                neighbors_a = self.atom_props.atom_neighbors.get(atom_a.idx, set())
                ring_atoms_b = set()
                for ring_idx in rings_b:
                    ring_atoms_b.update(self._aromatic_rings[ring_idx]['indices'])
                if neighbors_a & ring_atoms_b:
                    continue
            if rings_a:
                neighbors_b = self.atom_props.atom_neighbors.get(atom_b.idx, set())
                ring_atoms_a = set()
                for ring_idx in rings_a:
                    ring_atoms_a.update(self._aromatic_rings[ring_idx]['indices'])
                if neighbors_b & ring_atoms_a:
                    continue

            interaction = Interaction(
                type='hydrophobic',
                res_a_name=atom_a.resname,
                res_a_chain=atom_a.chain,
                res_a_num=atom_a.resnum,
                res_b_name=atom_b.resname,
                res_b_chain=atom_b.chain,
                res_b_num=atom_b.resnum,
                atom_a_name=atom_a.atom_name,
                atom_a_idx=atom_a.idx,
                atom_b_name=atom_b.atom_name,
                atom_b_idx=atom_b.idx,
                distance=dist_matrix_hydrophobic[i, j],
                angle=None,
                details={}
            )
            self.interactions['hydrophobic'].append(interaction)
    
    def _detect_hbonds(self, residue: Residue):
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
            # Use debug level to avoid spamming logs - this is expected behavior for many structures
            logger.debug("No explicit hydrogens found in residue %s %s %s. "
                        "H-bond detection skipped for this residue. "
                        "Provide a protonated PDB or use NOHYDRO=False for automatic protonation. "
                        "Set config.ALLOW_HEAVY_ATOM_HBOND=True for distance-only detection.",
                        residue.resname, residue.chain, residue.resnum)
            return

    def _detect_hbonds_with_h(self, residue: Residue, all_hba: List, all_hbd: List):
        """Detect H-bonds using explicit hydrogen coordinates.
        
        Optimized: Uses vectorized numpy operations for distance and angle calculations
        instead of individual euclidean3d and vecangle calls.
        """
        # Case 1: Residue is donor, other is acceptor
        if residue.hbond_donors:
            self._detect_hbonds_case1_vectorized(residue, all_hba)
        
        # DEV NOTE: case 2 is redundant with case 1
        # # Case 2: Residue is acceptor, other is donor
        # if residue.hbond_acceptors:
        #     self._detect_hbonds_case2_vectorized(residue, all_hbd)
    
    def _detect_hbonds_case1_vectorized(self, residue: Residue, all_hba: List):
        """Case 1: Residue is donor, other is acceptor (vectorized with sparse angle calculation)"""
        # Flatten all (donor, h_atom) pairs from residue
        donor_h_pairs = []
        for donor, h_atoms in residue.hbond_donors:
            for h_atom in h_atoms:
                donor_h_pairs.append((donor, h_atom))

        if not donor_h_pairs or not all_hba:
            return

        # Pre-extract coordinates as numpy arrays (residue-specific)
        don_coords = self.atom_container.get_atom_coords_array_from_atoms([pair[0] for pair in donor_h_pairs])  # [n_pairs, 3]
        h_coords = self.atom_container.get_atom_coords_array_from_atoms([pair[1] for pair in donor_h_pairs])    # [n_pairs, 3]
        # Use cached all_hba coordinates (global, pre-computed)
        acc_coords = self._all_hba_coords  # [n_acc, 3]

        # Vectorized distance calculation using cdist
        # dist_ad[i, j] = distance between donor i and acceptor j
        dist_ad_matrix = cdist(don_coords, acc_coords)  # [n_pairs, n_acc]

        # Filter by distance criteria
        dist_mask = (dist_ad_matrix > config.MIN_DIST) & (dist_ad_matrix < config.HBOND_DIST_MAX)

        # Get indices of pairs that passed distance filter
        pair_indices, acc_indices = np.where(dist_mask)

        if len(pair_indices) == 0:
            return

        # Sparse angle calculation: only for pairs that passed distance filter
        # Vector from H to D: don_coords - h_coords
        vec_hd = don_coords - h_coords  # [n_pairs, 3]

        # Extract only the vectors needed for angle calculation
        vec_hd_sparse = vec_hd[pair_indices]  # [N, 3]
        vec_ha_sparse = acc_coords[acc_indices] - h_coords[pair_indices]  # [N, 3]

        # Compute H-A distance from vec_ha_sparse (reusing the vector, no extra cdist needed)
        dist_ah_sparse = np.linalg.norm(vec_ha_sparse, axis=1)  # [N]

        # Compute angles using dot product (only for sparse pairs)
        norm_hd_sparse = np.linalg.norm(vec_hd_sparse, axis=1)  # [N]
        norm_ha_sparse = dist_ah_sparse  # Reuse H-A distance as norm of vec_ha_sparse

        # Avoid division by zero
        norm_hd_safe = np.where(norm_hd_sparse == 0, 1, norm_hd_sparse)
        norm_ha_safe = np.where(norm_ha_sparse == 0, 1, norm_ha_sparse)

        # Dot product
        dot_product_sparse = np.sum(vec_ha_sparse * vec_hd_sparse, axis=1)  # [N]

        cos_angle_sparse = dot_product_sparse / (norm_hd_safe * norm_ha_safe)
        cos_angle_sparse = np.clip(cos_angle_sparse, -1.0, 1.0)  # Clip to avoid numerical errors
        angle_sparse = np.degrees(np.arccos(cos_angle_sparse))  # [N]

        # Filter by angle criteria
        angle_mask_sparse = angle_sparse > config.HBOND_DON_ANGLE_MIN

        # Get final valid indices
        valid_pair_indices = pair_indices[angle_mask_sparse]
        valid_acc_indices = acc_indices[angle_mask_sparse]
        valid_angles = angle_sparse[angle_mask_sparse]
        valid_dist_ah = dist_ah_sparse[angle_mask_sparse]

        # Process only valid pairs
        for i, (pair_idx, acc_idx) in enumerate(zip(valid_pair_indices, valid_acc_indices)):
            donor, h_atom = donor_h_pairs[pair_idx]
            hba = all_hba[acc_idx]

            # Skip if same residue (unless it's a ligand)
            if self._should_skip_interaction(residue, donor, hba):
                continue

            dist_ad = dist_ad_matrix[pair_idx, acc_idx]
            dist_ah = valid_dist_ah[i]
            angle = valid_angles[i]

            interaction = Interaction(
                type='hbond',
                res_a_name=donor.resname,
                res_a_chain=donor.chain,
                res_a_num=donor.resnum,
                res_b_name=hba.resname,
                res_b_chain=hba.chain,
                res_b_num=hba.resnum,
                atom_a_name=donor.atom_name,
                atom_a_idx=donor.idx,
                atom_b_name=hba.atom_name,
                atom_b_idx=hba.idx,
                distance=dist_ad,
                angle=angle,
                details={
                    'h_atom': h_atom.atom_name,
                    'h_idx': h_atom.idx,
                    'dist_ah': dist_ah,
                    'type': 'strong' if dist_ad < 3.2 and angle > 140 else 'weak',
                    'donor_idx': donor.idx,
                    'acceptor_idx': hba.idx
                },
                objs={'donor': donor, 'h_atom': h_atom, 'acceptor': hba}
            )
            self.interactions['hbond'].append(interaction)
    
    def _detect_hbonds_case2_vectorized(self, residue: Residue, all_hbd: List):
        """Case 2: Residue is acceptor, other is donor (vectorized with sparse angle calculation)"""
        # Use cached global donor-hydrogen pairs (pre-computed in _precompute_cached_data)
        donor_h_pairs = self._all_hbd_pairs

        if not donor_h_pairs or not residue.hbond_acceptors:
            return

        # Use cached global donor/hydrogen coordinates (pre-computed)
        don_coords = self._all_hbd_don_coords  # [n_pairs, 3]
        h_coords = self._all_hbd_h_coords      # [n_pairs, 3]
        # Extract residue-specific acceptor coordinates
        acc_coords = self.atom_container.get_atom_coords_array_from_atoms(residue.hbond_acceptors)  # [n_acc, 3]

        # Vectorized distance calculation using cdist
        # dist_ad[i, j] = distance between donor i and acceptor j
        dist_ad_matrix = cdist(don_coords, acc_coords)  # [n_pairs, n_acc]

        # dist_ah[i, j] = distance between hydrogen i and acceptor j
        dist_ah_matrix = cdist(h_coords, acc_coords)  # [n_pairs, n_acc]

        # Filter by distance criteria
        dist_mask = (dist_ad_matrix > config.MIN_DIST) & (dist_ad_matrix < config.HBOND_DIST_MAX)

        # Get indices of pairs that passed distance filter
        pair_indices, acc_indices = np.where(dist_mask)

        if len(pair_indices) == 0:
            return

        # Sparse angle calculation: only for pairs that passed distance filter
        # Vector from H to D: don_coords - h_coords
        vec_hd = don_coords - h_coords  # [n_pairs, 3]

        # Extract only the vectors needed for angle calculation
        vec_hd_sparse = vec_hd[pair_indices]  # [N, 3]
        vec_ha_sparse = acc_coords[acc_indices] - h_coords[pair_indices]  # [N, 3]

        # Compute angles using dot product (only for sparse pairs)
        norm_hd_sparse = np.linalg.norm(vec_hd_sparse, axis=1)  # [N]
        norm_ha_sparse = np.linalg.norm(vec_ha_sparse, axis=1)  # [N]

        # Avoid division by zero
        norm_hd_safe = np.where(norm_hd_sparse == 0, 1, norm_hd_sparse)
        norm_ha_safe = np.where(norm_ha_sparse == 0, 1, norm_ha_sparse)

        # Dot product
        dot_product_sparse = np.sum(vec_ha_sparse * vec_hd_sparse, axis=1)  # [N]

        cos_angle_sparse = dot_product_sparse / (norm_hd_safe * norm_ha_safe)
        cos_angle_sparse = np.clip(cos_angle_sparse, -1.0, 1.0)
        angle_sparse = np.degrees(np.arccos(cos_angle_sparse))  # [N]

        # Filter by angle criteria
        angle_mask_sparse = angle_sparse > config.HBOND_DON_ANGLE_MIN

        # Get final valid indices
        valid_pair_indices = pair_indices[angle_mask_sparse]
        valid_acc_indices = acc_indices[angle_mask_sparse]
        valid_angles = angle_sparse[angle_mask_sparse]

        # Process only valid pairs
        for i, (pair_idx, acc_idx) in enumerate(zip(valid_pair_indices, valid_acc_indices)):
            donor, h_atom = donor_h_pairs[pair_idx]
            acc = residue.hbond_acceptors[acc_idx]

            # Skip if same residue (unless it's a ligand)
            if self._should_skip_interaction(residue, acc, donor):
                continue

            dist_ad = dist_ad_matrix[pair_idx, acc_idx]
            dist_ah = dist_ah_matrix[pair_idx, acc_idx]
            angle = valid_angles[i]

            interaction = Interaction(
                type='hbond',
                res_a_name=acc.resname,
                res_a_chain=acc.chain,
                res_a_num=acc.resnum,
                res_b_name=donor.resname,
                res_b_chain=donor.chain,
                res_b_num=donor.resnum,
                atom_a_name=acc.atom_name,
                atom_a_idx=acc.idx,
                atom_b_name=donor.atom_name,
                atom_b_idx=donor.idx,
                distance=dist_ad,
                angle=angle,
                details={
                    'h_atom': h_atom.atom_name,
                    'h_idx': h_atom.idx,
                    'dist_ah': dist_ah,
                    'type': 'strong' if dist_ad < 3.2 and angle > 140 else 'weak',
                    'donor_idx': donor.idx,
                    'acceptor_idx': acc.idx
                },
                objs={'donor': donor, 'h_atom': h_atom, 'acceptor': acc}
            )
            self.interactions['hbond'].append(interaction)

    def _detect_hbonds_without_h(self, residue: Residue, all_hba: List, all_hbd: List):
        """
        Detect H-bonds without explicit hydrogens (for standard PDB files).
        Uses distance-only criteria between donor and acceptor heavy atoms.
        Results are stored separately in 'hbond_heavy_atom' to distinguish from
        standard H-bonds with explicit hydrogens.

        Optimized: Uses vectorized distance calculation with scipy.spatial.distance.cdist
        for efficient batch processing of all donor-acceptor pairs.
        """
        # Case 1: Residue is donor, other is acceptor
        if residue.hbond_donors and all_hba:
            # Extract donor atoms (first element of each tuple)
            residue_donors = [donor for donor, _ in residue.hbond_donors]

            # Get coordinates for vectorized calculation
            donor_coords = self.atom_container.get_atom_coords_array_from_atoms(residue_donors)
            # Use pre-computed HBA coordinates from _precompute_cached_data
            hba_coords = self._all_hba_coords

            # Vectorized distance calculation using cdist
            dist_matrix = cdist(donor_coords, hba_coords)

            # Apply distance filter: 2.5-3.5 Å
            valid_mask = (dist_matrix >= 2.5) & (dist_matrix <= 3.5)
            valid_indices = np.argwhere(valid_mask)

            for i, j in valid_indices:
                donor = residue_donors[i]
                hba = all_hba[j]

                # Skip if same atom
                if donor.idx == hba.idx:
                    continue

                # Skip if same residue (unless it's a ligand)
                if self._should_skip_interaction(residue, donor, hba):
                    continue

                interaction = Interaction(
                    type='hbond_heavy_atom',
                    res_a_name=donor.resname,
                    res_a_chain=donor.chain,
                    res_a_num=donor.resnum,
                    res_b_name=hba.resname,
                    res_b_chain=hba.chain,
                    res_b_num=hba.resnum,
                    atom_a_name=donor.atom_name,
                    atom_a_idx=donor.idx,
                    atom_b_name=hba.atom_name,
                    atom_b_idx=hba.idx,
                    distance=float(dist_matrix[i, j]),
                    angle=None,
                    details={
                        'type': 'heavy_atom',
                        'note': 'No explicit H, distance-only criteria (less reliable)',
                        'donor_idx': donor.idx,
                        'acceptor_idx': hba.idx
                    }
                )
                self.interactions['hbond_heavy_atom'].append(interaction)

        # DEV NOTE: case 2 is redundant with case 1
        # # Case 2: Residue is acceptor, other is donor
        # if residue.hbond_acceptors and all_hbd:
        #     # Use pre-computed donor atoms and coordinates from _precompute_cached_data
        #     all_donors = self._all_hbd_donor_atoms

        #     # Get coordinates for vectorized calculation
        #     acc_coords = np.array([acc.coords for acc in residue.hbond_acceptors])
        #     # Use pre-computed donor coordinates
        #     donor_coords = self._all_hbd_donor_coords

        #     # Vectorized distance calculation using cdist
        #     dist_matrix = cdist(acc_coords, donor_coords)

        #     # Apply distance filter: 2.5-3.5 Å
        #     valid_mask = (dist_matrix >= 2.5) & (dist_matrix <= 3.5)
        #     valid_indices = np.argwhere(valid_mask)

        #     for i, j in valid_indices:
        #         acc = residue.hbond_acceptors[i]
        #         donor = all_donors[j]

        #         # Skip if same atom
        #         if acc.idx == donor.idx:
        #             continue

        #         # Skip if same residue (unless it's a ligand)
        #         if self._should_skip_interaction(residue, acc, donor):
        #             continue

        #         interaction = Interaction(
        #             type='hbond_heavy_atom',
        #             res_a_name=acc.resname,
        #             res_a_chain=acc.chain,
        #             res_a_num=acc.resnum,
        #             res_b_name=donor.resname,
        #             res_b_chain=donor.chain,
        #             res_b_num=donor.resnum,
        #             atom_a_name=acc.atom_name,
        #             atom_a_idx=acc.idx,
        #             atom_b_name=donor.atom_name,
        #             atom_b_idx=donor.idx,
        #             distance=float(dist_matrix[i, j]),
        #             angle=None,
        #             details={
        #                 'type': 'heavy_atom',
        #                 'note': 'No explicit H, distance-only criteria (less reliable)',
        #                 'donor_idx': donor.idx,
        #                 'acceptor_idx': acc.idx
        #             }
        #         )
        #         self.interactions['hbond_heavy_atom'].append(interaction)
    
    def _detect_saltbridges(self, residue: Residue):
        """Detect salt bridges between charged residues using vectorized distance calculation.

        Uses pre-computed grouped charged atoms and charge centers from:
        1. Residue.pos_charged_groups / Residue.neg_charged_groups (per-residue, set in finalize)
        2. self._all_pos_grouped / self._all_neg_grouped (global, set in _precompute_cached_data)
        
        Vectorized distance calculation: computes all distances at once using NumPy.
        
        NOTE: Each functional group pair generates only ONE salt bridge record.
        For example, if ARG's guanidinium group interacts with ASP's carboxylate group,
        only one salt bridge is recorded (not multiple atom-pair salt bridges).
        This aligns with the chemical reality that salt bridges are interactions between
        charge centers, not individual atoms.
        """
        if not (residue.pos_charged or residue.neg_charged):
            return

        # Only detect from positive residues to avoid duplicates
        # Each functional group pair should generate only one salt bridge
        if residue.pos_charged and residue.pos_charged_groups and len(self._neg_grouped_centers) > 0:
            for pos_key, (pos_atoms, pos_center) in residue.pos_charged_groups.items():
                # Vectorized distance calculation (all at once)
                distances = np.linalg.norm(self._neg_grouped_centers - pos_center, axis=1)

                # Find all pairs within distance threshold
                valid_indices = np.where(distances < config.SALTBRIDGE_DIST_MAX)[0]

                for idx in valid_indices:
                    neg_atoms = self._neg_grouped_atoms[idx]
                    neg_key = self._neg_grouped_keys[idx]
                    distance = distances[idx]
                    
                    # Get representative atoms for residue/atom identification
                    pos_atom = pos_atoms[0]
                    neg_atom = neg_atoms[0]

                    # Skip if same residue (unless it's a ligand) - consistent with other interaction types
                    if self._should_skip_interaction(residue, pos_atom, neg_atom):
                        continue

                    interaction = Interaction(
                        type='saltbridge',
                        res_a_name=pos_atom.resname,
                        res_a_chain=pos_atom.chain,
                        res_a_num=pos_atom.resnum,
                        res_b_name=neg_atom.resname,
                        res_b_chain=neg_atom.chain,
                        res_b_num=neg_atom.resnum,
                        atom_a_name=pos_atom.atom_name,
                        atom_a_idx=pos_atom.idx,
                        atom_b_name=neg_atom.atom_name,
                        atom_b_idx=neg_atom.idx,
                        distance=float(distance),
                        angle=None,
                        details={
                            'charge_type': 'pos-neg',
                            'positive_atoms': [a.idx for a in pos_atoms],
                            'negative_atoms': [a.idx for a in neg_atoms],
                            'positive_group_key': str(pos_key),
                            'negative_group_key': str(neg_key)
                        }
                    )
                    self.interactions['saltbridge'].append(interaction)
    
    def _detect_pistacking(self, residue: Residue):
        """Detect pi-stacking interactions using vectorized calculations.
        
        Optimized: Uses pre-computed ring data and vectorized numpy operations
        for distance, angle, and offset calculations.
        """
        if not residue.rings:
            return

        all_rings = self.atom_props.rings
        if not all_rings:
            return

        # Pre-compute residue ring data
        res_ring_centers = np.array([r['center'] for r in residue.rings])
        res_ring_normals = np.array([r['normal'] for r in residue.rings])
        
        # Use pre-computed all rings data (cached in _precompute_cached_data)
        all_ring_centers = self._all_ring_centers
        all_ring_normals = self._all_ring_normals
        
        # Vectorized distance calculation using cdist
        dist_matrix = cdist(res_ring_centers, all_ring_centers)
        
        # Filter by distance
        valid_dist_mask = dist_matrix <= config.PISTACK_DIST_MAX
        
        # Vectorized angle calculation between normals
        # dot_products[i, j] = dot(res_ring_normals[i], all_ring_normals[j])
        dot_products = np.sum(res_ring_normals[:, np.newaxis, :] * 
                              all_ring_normals[np.newaxis, :, :], axis=2)
        angles = np.arccos(np.clip(dot_products, -1.0, 1.0)) * 180 / np.pi
        
        # Determine stacking type
        parallel_mask = (angles < config.PISTACK_ANG_DEV) | (angles > (180 - config.PISTACK_ANG_DEV))
        perp_mask = ((90 - config.PISTACK_ANG_DEV) < angles) & (angles < (90 + config.PISTACK_ANG_DEV))
        valid_angle_mask = parallel_mask | perp_mask
        
        # Combined mask
        valid_mask = valid_dist_mask & valid_angle_mask
        
        # Get valid pairs
        valid_indices = np.argwhere(valid_mask)
        
        for i, j in valid_indices:
            ring_a = residue.rings[i]
            ring_b = all_rings[j]
            
            # Skip same ring
            if ring_a['indices'][0] == ring_b['indices'][0]:
                continue
            
            # Get representative atoms
            atom_a = self.atom_container[ring_a['indices'][0]]
            atom_b = self.atom_container[ring_b['indices'][0]]
            
            # Skip if same residue
            if self._should_skip_interaction(residue, atom_a, atom_b):
                continue
            
            distance = float(dist_matrix[i, j])
            angle = float(angles[i, j])
            
            # Determine type
            if angle < config.PISTACK_ANG_DEV or angle > (180 - config.PISTACK_ANG_DEV):
                ptype = 'parallel'
            else:
                ptype = 'perpendicular'
            
            # Calculate offset (vectorized projection)
            # Project ring A center onto ring B plane
            vec_ab = ring_a['center'] - ring_b['center']
            proj_dist = np.dot(vec_ab, ring_b['normal'])
            proj1 = ring_a['center'] - proj_dist * ring_b['normal']
            offset1 = np.linalg.norm(proj1 - ring_b['center'])
            
            # Project ring B center onto ring A plane
            vec_ba = ring_b['center'] - ring_a['center']
            proj_dist = np.dot(vec_ba, ring_a['normal'])
            proj2 = ring_b['center'] - proj_dist * ring_a['normal']
            offset2 = np.linalg.norm(proj2 - ring_a['center'])
            
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
                    'offset': float(offset),
                    'ring_a_atoms': ring_a['indices'],
                    'ring_b_atoms': ring_b['indices']
                }
            )
            self.interactions['pistacking'].append(interaction)
    
    def _detect_pication(self, residue: Residue):
        """Detect pi-cation interactions using vectorized calculations.

        Geometric criteria:
        - Distance: < PICATION_DIST_MAX (6.0 Å)
        - Offset: < PISTACK_OFFSET_MAX (2.0 Å)
          Offset is the distance from the projection of the charge onto the ring plane
          to the ring center. This ensures the charge is positioned above the ring face.
        
        Optimized: Uses pre-computed data and vectorized numpy operations
        for distance and offset calculations.
        """
        if not (residue.rings or residue.pos_charged):
            return

        all_rings = self.atom_props.rings
        all_pos = self.atom_props.get_pos_charged()
        
        if not all_rings or not all_pos:
            return

        # Use pre-computed aromatic ring data (cached in _precompute_cached_data)
        aromatic_rings = self._aromatic_rings
        if not aromatic_rings:
            return

        # # Use pre-computed filtered ring centers (aromatic only)
        # ring_centers = self._all_ring_centers[self._aromatic_ring_mask]

        # Case 1: Residue has ring, other has positive charge
        if residue.rings:
            res_aromatic_rings = [r for r in residue.rings if r.get('is_aromatic', False)]
            if res_aromatic_rings and all_pos:
                res_ring_centers = np.array([r['center'] for r in res_aromatic_rings])
                
                # Use pre-computed positive charge coordinates (cached in _precompute_cached_data)
                pos_coords = self._pos_coords
                
                # Vectorized distance calculation using cdist
                dist_matrix = cdist(res_ring_centers, pos_coords)
                
                # Filter by distance
                valid_dist_mask = dist_matrix < config.PICATION_DIST_MAX
                valid_indices = np.argwhere(valid_dist_mask)
                
                for i, j in valid_indices:
                    ring = res_aromatic_rings[i]
                    pos_atom = all_pos[j]
                    
                    # Skip if same residue
                    atom_a = self.atom_container[ring['indices'][0]]
                    if self._should_skip_interaction(residue, atom_a, pos_atom):
                        continue
                    
                    distance = float(dist_matrix[i, j])
                    
                    # Calculate offset: projection of charge onto ring plane to ring center
                    vec = pos_atom.coords - ring['center']
                    proj_dist = np.dot(vec, ring['normal'])
                    proj = pos_atom.coords - proj_dist * ring['normal']
                    offset = np.linalg.norm(proj - ring['center'])
                    
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
                        atom_b_name=pos_atom.atom_name,
                        atom_b_idx=pos_atom.idx,
                        distance=distance,
                        angle=None,
                        details={'ring_center': ring['center'], 'offset': float(offset)}
                    )
                    self.interactions['pication'].append(interaction)
        
        # Case 2 is redundant with Case 1
        # # Case 2: Residue has positive charge, other has ring
        # if residue.pos_charged:
        #     if residue.pos_charged and aromatic_rings:
        #         res_pos_coords = np.array([p.coords for p in residue.pos_charged])
                
        #         # Vectorized distance calculation using cdist
        #         dist_matrix = cdist(res_pos_coords, ring_centers)
                
        #         # Filter by distance
        #         valid_dist_mask = dist_matrix < config.PICATION_DIST_MAX
        #         valid_indices = np.argwhere(valid_dist_mask)
                
        #         for i, j in valid_indices:
        #             pos_atom = residue.pos_charged[i]
        #             ring = aromatic_rings[j]
                    
        #             # Skip if same residue
        #             atom_b = self.atom_container[ring['indices'][0]]
        #             if self._should_skip_interaction(residue, pos_atom, atom_b):
        #                 continue
                    
        #             distance = float(dist_matrix[i, j])
                    
        #             # Calculate offset: projection of charge onto ring plane to ring center
        #             vec = pos_atom.coords - ring['center']
        #             proj_dist = np.dot(vec, ring['normal'])
        #             proj = pos_atom.coords - proj_dist * ring['normal']
        #             offset = np.linalg.norm(proj - ring['center'])
                    
        #             if offset >= config.PISTACK_OFFSET_MAX:
        #                 continue
                    
        #             interaction = Interaction(
        #                 type='pication',
        #                 res_a_name=pos_atom.resname,
        #                 res_a_chain=pos_atom.chain,
        #                 res_a_num=pos_atom.resnum,
        #                 res_b_name=atom_b.resname,
        #                 res_b_chain=atom_b.chain,
        #                 res_b_num=atom_b.resnum,
        #                 atom_a_name=pos_atom.atom_name,
        #                 atom_a_idx=pos_atom.idx,
        #                 atom_b_name='RING',
        #                 atom_b_idx=ring['indices'][0],
        #                 distance=distance,
        #                 angle=None,
        #                 details={'ring_center': ring['center'], 'offset': float(offset)}
        #             )
        #             self.interactions['pication'].append(interaction)
    
    def _get_halogen_bond_angles(self, donor, acceptor):
        """Calculate donor and acceptor angles for halogen bond.

        Following PLIP's approach:
        - Donor angle: C-X⋯O (angle between X->C vector and X->O vector)
        - Acceptor angle: Y-O⋯X (angle between O->Y vector and O->X vector)

        NOTE: This implementation uses pre-computed C and Y atoms from atom_properties,
        which are determined during initialization following PLIP's approach of only
        considering atoms with exactly one proximal neighbor. This ensures:
        1. Unambiguous angle calculation (no need to choose between multiple candidates)
        2. Alignment with main PLIP's behavior
        3. Scientific validity (halogen bonds typically involve well-defined single bonds)

        Returns:
            (don_angle, acc_angle) or (None, None) if angles cannot be calculated
        """
        # Get pre-computed C atom for the halogen donor
        c_atom_idx = self.atom_props.halogen_donor_c_atoms.get(donor.idx)
        if c_atom_idx is None:
            # No pre-computed C atom (donor has 0 or >1 C neighbors)
            return None, None

        c_atom = self.atom_container[c_atom_idx]
        c_coords = c_atom.coords

        # Calculate donor angle (C-X⋯O)
        # Vector from X to C (donor to carbon)
        vec_xc = vector(donor.coords, c_coords)
        # Vector from X to O (donor to acceptor)
        vec_xo = vector(donor.coords, acceptor.coords)
        don_angle = vecangle(vec_xc, vec_xo)

        # Get pre-computed Y atom for the acceptor
        y_atom_idx = self.atom_props.halogen_acceptor_y_atoms.get(acceptor.idx)
        if y_atom_idx is None:
            # No pre-computed Y atom (acceptor has 0 or >1 proximal atoms)
            return don_angle, None

        y_atom = self.atom_container[y_atom_idx]
        y_coords = y_atom.coords

        # Calculate acceptor angle (Y-O⋯X)
        # Vector from O to Y (acceptor to proximal)
        vec_oy = vector(acceptor.coords, y_coords)
        # Vector from O to X (acceptor to donor)
        vec_ox = vector(acceptor.coords, donor.coords)
        acc_angle = vecangle(vec_oy, vec_ox)

        return don_angle, acc_angle

    def _detect_halogen(self, residue: Residue):
        """Detect halogen bonds following PLIP's criteria.
        
        Uses pre-computed halogen bond donors and acceptors from _precompute_cached_data.
        Optimized: Uses vectorized distance calculation to filter pairs before angle calculation.
        
        Halogen bond criteria:
        - Distance: X···O < HALOGEN_DIST_MAX
        - Donor angle: C-X···O > HALOGEN_DON_ANGLE - HALOGEN_ANGLE_DEV
        - Acceptor angle: Y-O···X within HALOGEN_ACC_ANGLE ± HALOGEN_ANGLE_DEV
        """
        if not (residue.halogen_donors or residue.halogen_acceptors):
            return
        
        # Use pre-computed donors, acceptors, and their coordinates
        all_donors = self._all_halogen_donors
        all_acceptors = self._all_halogen_acceptors
        acceptor_coords = self._all_halogen_acceptor_coords
        donor_coords = self._all_halogen_donor_coords
        
        # Case 1: Residue is donor
        if residue.halogen_donors and all_acceptors and len(acceptor_coords) > 0:
            res_donor_coords = np.array([donor.coords for donor, _ in residue.halogen_donors])
            
            # Vectorized distance calculation using cdist
            dist_matrix = cdist(res_donor_coords, acceptor_coords)
            
            # Find pairs within distance threshold
            valid_mask = (dist_matrix > config.MIN_DIST) & (dist_matrix < config.HALOGEN_DIST_MAX)
            valid_indices = np.argwhere(valid_mask)
            
            for i, j in valid_indices:
                donor, htype = residue.halogen_donors[i]
                acc = all_acceptors[j]
                
                # Skip if same residue (unless it's a ligand)
                if self._should_skip_interaction(residue, donor, acc):
                    continue
                
                distance = float(dist_matrix[i, j])
                
                # Calculate both angles
                don_angle, acc_angle = self._get_halogen_bond_angles(donor, acc)
                
                if don_angle is None:
                    continue
                
                # Check donor angle
                if not (config.HALOGEN_DON_ANGLE - config.HALOGEN_ANGLE_DEV < don_angle):
                    continue
                
                # Check acceptor angle (if available)
                if acc_angle is not None:
                    if not (config.HALOGEN_ACC_ANGLE - config.HALOGEN_ANGLE_DEV < acc_angle <
                            config.HALOGEN_ACC_ANGLE + config.HALOGEN_ANGLE_DEV):
                        continue
                
                interaction = Interaction(
                    type='halogen',
                    res_a_name=donor.resname,
                    res_a_chain=donor.chain,
                    res_a_num=donor.resnum,
                    res_b_name=acc.resname,
                    res_b_chain=acc.chain,
                    res_b_num=acc.resnum,
                    atom_a_name=donor.atom_name,
                    atom_a_idx=donor.idx,
                    atom_b_name=acc.atom_name,
                    atom_b_idx=acc.idx,
                    distance=distance,
                    angle=don_angle,
                    details={
                        'halogen_type': htype,
                        'don_angle': don_angle,
                        'acc_angle': acc_angle
                    }
                )
                self.interactions['halogen'].append(interaction)
        
        # Case 2: Residue is acceptor
        if residue.halogen_acceptors and all_donors and len(donor_coords) > 0:
            res_acceptor_coords = np.array([acc.coords for acc in residue.halogen_acceptors])
            
            # Vectorized distance calculation using cdist
            dist_matrix = cdist(res_acceptor_coords, donor_coords)
            
            # Find pairs within distance threshold
            valid_mask = (dist_matrix > config.MIN_DIST) & (dist_matrix < config.HALOGEN_DIST_MAX)
            valid_indices = np.argwhere(valid_mask)
            
            for i, j in valid_indices:
                acc = residue.halogen_acceptors[i]
                donor, htype = all_donors[j]
                
                # Skip if same residue (unless it's a ligand)
                if self._should_skip_interaction(residue, acc, donor):
                    continue
                
                distance = float(dist_matrix[i, j])
                
                # Calculate both angles
                don_angle, acc_angle = self._get_halogen_bond_angles(donor, acc)
                
                if don_angle is None:
                    continue
                
                # Check donor angle
                if not (config.HALOGEN_DON_ANGLE - config.HALOGEN_ANGLE_DEV < don_angle):
                    continue
                
                # Check acceptor angle (if available)
                if acc_angle is not None:
                    if not (config.HALOGEN_ACC_ANGLE - config.HALOGEN_ANGLE_DEV < acc_angle <
                            config.HALOGEN_ACC_ANGLE + config.HALOGEN_ANGLE_DEV):
                        continue
                    
                    interaction = Interaction(
                        type='halogen',
                        res_a_name=acc.resname,
                        res_a_chain=acc.chain,
                        res_a_num=acc.resnum,
                        res_b_name=donor.resname,
                        res_b_chain=donor.chain,
                        res_b_num=donor.resnum,
                        atom_a_name=acc.atom_name,
                        atom_a_idx=acc.idx,
                        atom_b_name=donor.atom_name,
                        atom_b_idx=donor.idx,
                        distance=distance,
                        angle=don_angle,
                        details={
                            'halogen_type': htype,
                            'don_angle': don_angle,
                            'acc_angle': acc_angle
                        }
                    )
                    self.interactions['halogen'].append(interaction)
    
    def _detect_metal(self, residue: Residue):
        """Detect metal complexation using vectorized distance calculation.
        
        Optimized: Uses pre-computed coordinates and vectorized numpy operations
        for distance calculations instead of individual euclidean3d calls.
        """
        if not (residue.metal_atoms or residue.metal_binding_atoms):
            return
        
        all_metals = self.atom_props.get_metals()
        all_binding = self.atom_props.get_metal_binding()
        
        # Early return if no metals or binding atoms in structure
        if not all_metals or not all_binding:
            return
        
        # Use pre-computed coordinates (cached in _precompute_cached_data)
        # metals_coords = self._metals_coords
        binding_coords = self._binding_coords
        
        # Case 1: Residue is metal
        if residue.metal_atoms:
            res_metals_coords = self.atom_container.get_atom_coords_array_from_atoms(residue.metal_atoms)
            
            # Vectorized distance calculation using cdist
            dist_matrix = cdist(res_metals_coords, binding_coords)
            
            # Find valid pairs within distance threshold
            valid_mask = dist_matrix < config.METAL_DIST_MAX
            valid_indices = np.argwhere(valid_mask)
            
            for i, j in valid_indices:
                metal = residue.metal_atoms[i]
                binding = all_binding[j]
                
                # Skip if same residue (unless it's a ligand)
                if self._should_skip_interaction(residue, metal, binding):
                    continue
                
                interaction = Interaction(
                    type='metal',
                    res_a_name=metal.resname,
                    res_a_chain=metal.chain,
                    res_a_num=metal.resnum,
                    res_b_name=binding.resname,
                    res_b_chain=binding.chain,
                    res_b_num=binding.resnum,
                    atom_a_name=metal.atom_name,
                    atom_a_idx=metal.idx,
                    atom_b_name=binding.atom_name,
                    atom_b_idx=binding.idx,
                    distance=float(dist_matrix[i, j]),
                    angle=None,
                    details={}
                )
                self.interactions['metal'].append(interaction)
        
        # Case 2 is redundant with Case 1
        # # Case 2: Residue has metal-binding atoms
        # if residue.metal_binding_atoms:
        #     res_binding_coords = self.atom_container.get_atom_coords_array_from_atoms(residue.metal_binding_atoms)
            
        #     # Vectorized distance calculation using cdist
        #     dist_matrix = cdist(res_binding_coords, metals_coords)
            
        #     # Find valid pairs within distance threshold
        #     valid_mask = dist_matrix < config.METAL_DIST_MAX
        #     valid_indices = np.argwhere(valid_mask)
            
        #     for i, j in valid_indices:
        #         binding = residue.metal_binding_atoms[i]
        #         metal = all_metals[j]
                
        #         # Skip if same residue (unless it's a ligand)
        #         if self._should_skip_interaction(residue, binding, metal):
        #             continue
                
        #         interaction = Interaction(
        #             type='metal',
        #             res_a_name=binding.resname,
        #             res_a_chain=binding.chain,
        #             res_a_num=binding.resnum,
        #             res_b_name=metal.resname,
        #             res_b_chain=metal.chain,
        #             res_b_num=metal.resnum,
        #             atom_a_name=binding.atom_name,
        #             atom_a_idx=binding.idx,
        #             atom_b_name=metal.atom_name,
        #             atom_b_idx=metal.idx,
        #             distance=float(dist_matrix[i, j]),
        #             angle=None,
        #             details={}
        #         )
        #         self.interactions['metal'].append(interaction)

    def _create_water_bridge_interaction(self, water_res, water_o, water_o_coords,
                                         pa_key, pa_info, pb_key, pb_info):
        """Create a water bridge interaction between two partner atoms.

        Parameters
        ----------
        water_res : Residue
            The water residue mediating the bridge
        water_o : AtomInfo
            The water oxygen atom
        water_o_coords : np.ndarray
            Coordinates of water oxygen
        pa_key, pb_key : tuple
            Partner keys: (resname, chain, resnum, atom_name, atom_idx)
        pa_info, pb_info : dict
            Partner info containing 'hbond' and 'side'
        """
        # Extract partner atom coordinates
        pa_atom = self.atom_container[pa_key[4]]
        pb_atom = self.atom_container[pb_key[4]]
        pa_coords = pa_atom.coords
        pb_coords = pb_atom.coords

        # Calculate distances from partners to water oxygen
        dist_aw = euclidean3d(pa_coords, water_o_coords)
        dist_bw = euclidean3d(pb_coords, water_o_coords)

        # Calculate water bridge distance (sum of both distances)
        bridge_distance = dist_aw + dist_bw

        # Calculate water angle (A-O-B angle)
        w_angle = vecangle(vector(water_o_coords, pa_coords), vector(water_o_coords, pb_coords))

        # Create interaction
        interaction = Interaction(
            type='water_bridge',
            res_a_name=pa_key[0],
            res_a_chain=pa_key[1],
            res_a_num=pa_key[2],
            res_b_name=pb_key[0],
            res_b_chain=pb_key[1],
            res_b_num=pb_key[2],
            atom_a_name=pa_key[3],
            atom_a_idx=pa_key[4],
            atom_b_name=pb_key[3],
            atom_b_idx=pb_key[4],
            distance=bridge_distance,
            angle=w_angle,
            details={
                'water_residue': water_res.resid,
                'water_atom_idx': water_o.idx,
                'distance_aw': dist_aw,
                'distance_bw': dist_bw,
            }
        )
        self.interactions['water_bridge'].append(interaction)

    def _detect_water_bridges(self, verbose=False):
        """Detect water bridges (water mediating between two molecules)"""
        # Find water residues (sorted for deterministic ordering)
        # Skip water residues marked as distant (is_skip=True) for performance
        water_residues = {r._hash: r for r in self.residues if r.is_water and not r.is_skip}
        if not water_residues:
            return
        # Combine standard H-bonds and heavy atom H-bonds involving water for water bridge detection
        all_hbonds = self.interactions['hbond'] + self.interactions['hbond_heavy_atom']
        all_hbonds = list(filter(lambda hb: hb.objs['donor'].residue_obj.is_water or hb.objs['acceptor'].residue_obj.is_water, all_hbonds))
        res2hbonds = defaultdict(list)
        for hb in all_hbonds:
            if hb.objs['acceptor'].residue_obj._hash in water_residues:
                res2hbonds[hb.objs['acceptor'].residue_obj._hash].append(hb)
            if hb.objs['donor'].residue_obj._hash in water_residues:
                res2hbonds[hb.objs['donor'].residue_obj._hash].append(hb)
        for water_res in water_residues.values():
            # all H-bonds involving this water residue
            water_hbonds = res2hbonds[water_res._hash]            
            # Check if water bridges two different residues
            if len(water_hbonds) >= 2:
                # Build partner info with atom-level keys to handle ligand internal water bridges
                partner_info = {}
                for hb in water_hbonds:
                    # res_a side (if not water)
                    if hb.res_a_name != water_res.resname:
                        key = (hb.res_a_name, hb.res_a_chain, hb.res_a_num, hb.atom_a_name, hb.atom_a_idx)
                        if key not in partner_info:
                            partner_info[key] = {'hbond': hb, 'side': 'a'}
                    # res_b side (if not water)
                    if hb.res_b_name != water_res.resname:
                        key = (hb.res_b_name, hb.res_b_chain, hb.res_b_num, hb.atom_b_name, hb.atom_b_idx)
                        if key not in partner_info:
                            partner_info[key] = {'hbond': hb, 'side': 'b'}

                partner_keys = list(partner_info.keys())
                if len(partner_keys) >= 2:
                    # Get water oxygen atom and its coordinates
                    water_o = None
                    for atom in water_res.atoms:
                        if atom.atomic_num == 8:  # Oxygen
                            water_o = atom
                            break

                    if water_o is None:
                        continue

                    water_o_coords = water_o.coords

                    # Build water bridge interactions for each pair of partners
                    if len(partner_keys) == 2:
                        # Common case: exactly 2 partners
                        pa_key, pb_key = partner_keys
                        self._create_water_bridge_interaction(
                            water_res, water_o, water_o_coords,
                            pa_key, partner_info[pa_key],
                            pb_key, partner_info[pb_key]
                        )
                    else:
                        # Rare case: more than 2 partners, use combinations
                        import itertools
                        for pa_key, pb_key in itertools.combinations(partner_keys, 2):
                            self._create_water_bridge_interaction(
                                water_res, water_o, water_o_coords,
                                pa_key, partner_info[pa_key],
                                pb_key, partner_info[pb_key]
                            )

    def _detect_water_bridges_plip_style(self, verbose: bool = False):
        """Detect water bridges using PLIP-style distance+angle criteria

        This method implements the PLIP water bridge detection approach:
        - Acceptor-water: distance check only (2.5-4.1 Å)
        - Donor-water: distance + angle check
        - Water bridge: same water molecule mediates between two residues
        - Water angle: calculated using donor hydrogen (A-O-H_donor)

        Results are stored in 'water_bridge_possible' to distinguish from
        the stricter H-bond-based water_bridge detection.

        Fully vectorized implementation:
        1. Pre-extract all coordinates (waters, acceptors, donors+Hs)
        2. Batch distance calculations using cdist
        3. Batch angle calculations using vectorized dot product
        4. Mask filtering for efficient pair identification

        Note: All arrays are sorted by atom indices for deterministic behavior.
        """
        # Find water residues and extract oxygen atoms (sorted for deterministic ordering)
        # Skip water residues marked as distant (is_skip=True) for performance
        water_residues = sorted([r for r in self.residues if r.is_water and not r.is_skip],
                                key=lambda r: (r.chain, r.resnum))

        if not water_residues:
            return

        # Extract water oxygen atoms and coordinates
        water_objects = []  # List of (water_res, water_o_atom)
        water_o_coords_list = []
        for water_res in water_residues:
            for atom in water_res.atoms:
                if atom.atomic_num == 8:  # Oxygen
                    water_objects.append((water_res, atom))
                    water_o_coords_list.append(atom.coords)
                    break

        if not water_objects:
            return

        water_o_coords = np.array(water_o_coords_list)  # [n_water, 3]

        # Get all H-bond acceptors
        all_hba = self._all_hba  # Already cached in _precompute_cached_data
        if not all_hba:
            return

        acc_coords = self._all_hba_coords  # [n_acc, 3], cached

        # Get all H-bond donors with hydrogens
        # Use cached _all_hbd_pairs which contains (donor, h_atom) tuples
        if not self._all_hbd_pairs:
            # Fallback: build from atom_props if not cached
            donor_h_pairs = []
            for donor_idx, h_indices in self.atom_props.hbond_donors.items():
                if h_indices:
                    first_h_idx = sorted(h_indices)[0]
                    donor = self.atom_container[donor_idx]
                    h_atom = self.atom_container[first_h_idx]
                    donor_h_pairs.append((donor, h_atom))
        else:
            donor_h_pairs = self._all_hbd_pairs

        if not donor_h_pairs:
            return

        don_coords = self._all_hbd_don_coords  # [n_don, 3], cached
        don_h_coords = self._all_hbd_h_coords   # [n_don, 3], cached

        # Build residue identity lookup for self-filtering
        # water_res_idx -> set of (resname, chain, resnum)
        water_res_identities = []
        for water_res, _ in water_objects:
            water_res_identities.append((water_res.resname, water_res.chain, water_res.resnum))

        # Batch distance calculation: water oxygens vs all acceptors
        # dist_aw[water_i, acc_j] = distance between water_i oxygen and acceptor_j
        dist_aw_matrix = cdist(water_o_coords, acc_coords)

        # Batch distance calculation: water oxygens vs all donors
        # dist_dw[water_i, don_j] = distance between water_i oxygen and donor_j
        dist_dw_matrix = cdist(water_o_coords, don_coords)

        # Filter by distance criteria for acceptors
        acc_dist_mask = (dist_aw_matrix >= config.WATER_BRIDGE_MINDIST) & \
                        (dist_aw_matrix <= config.WATER_BRIDGE_MAXDIST)

        # Filter by distance criteria for donors
        don_dist_mask = (dist_dw_matrix >= config.WATER_BRIDGE_MINDIST) & \
                        (dist_dw_matrix <= config.WATER_BRIDGE_MAXDIST)

        # Sparse donor angle calculation (D-H···O)
        # Only calculate angles for pairs that passed distance filter
        vec_dh = don_coords - don_h_coords  # [n_don, 3]

        # Get indices of pairs that passed distance filter
        water_indices, don_indices = np.where(don_dist_mask)

        if len(water_indices) == 0:
            return

        # Extract only the vectors needed for angle calculation
        vec_dh_sparse = vec_dh[don_indices]  # [N, 3]
        vec_do_sparse = water_o_coords[water_indices] - don_coords[don_indices]  # [N, 3]

        # Compute donor angles using dot product (only for sparse pairs)
        norm_dh_sparse = np.linalg.norm(vec_dh_sparse, axis=1)  # [N]
        norm_do_sparse = np.linalg.norm(vec_do_sparse, axis=1)  # [N]

        # Avoid division by zero
        norm_dh_safe = np.where(norm_dh_sparse == 0, 1, norm_dh_sparse)
        norm_do_safe = np.where(norm_do_sparse == 0, 1, norm_do_sparse)

        # Dot product
        dot_dh_do_sparse = np.sum(vec_do_sparse * vec_dh_sparse, axis=1)  # [N]

        cos_d_angle_sparse = dot_dh_do_sparse / (norm_dh_safe * norm_do_safe)
        cos_d_angle_sparse = np.clip(cos_d_angle_sparse, -1.0, 1.0)
        d_angle_sparse = np.degrees(np.arccos(cos_d_angle_sparse))  # [N]

        # Filter by donor angle
        don_angle_mask_sparse = d_angle_sparse > config.WATER_BRIDGE_THETA_MIN

        # Create full angle matrix for compatibility with downstream code
        # Initialize with zeros and fill in calculated values
        d_angle_matrix = np.zeros((len(water_objects), len(donor_h_pairs)))
        valid_indices = np.where(don_angle_mask_sparse)[0]
        for idx in valid_indices:
            w_idx = water_indices[idx]
            d_idx = don_indices[idx]
            d_angle_matrix[w_idx, d_idx] = d_angle_sparse[idx]

        # Combined donor mask (full matrix for compatibility)
        don_valid_mask = np.zeros((len(water_objects), len(donor_h_pairs)), dtype=bool)
        for idx in valid_indices:
            w_idx = water_indices[idx]
            d_idx = don_indices[idx]
            don_valid_mask[w_idx, d_idx] = True

        # Find valid water-acc-don combinations using broadcasting
        # For each water, get valid acceptors and donors
        for water_idx in tqdm(range(len(water_objects)), desc="Detecting PLIP style water bridges",
                              disable=not verbose, leave=False):
            water_res, water_o = water_objects[water_idx]
            water_o_coord = water_o_coords[water_idx]
            water_identity = water_res_identities[water_idx]

            # Get valid acceptors for this water
            valid_acc_indices = np.where(acc_dist_mask[water_idx])[0]

            # Get valid donors for this water
            valid_don_indices = np.where(don_valid_mask[water_idx])[0]

            if len(valid_acc_indices) == 0 or len(valid_don_indices) == 0:
                continue

            # Pre-compute acceptor coords for this water
            acc_coords_water = acc_coords[valid_acc_indices]  # [n_valid_acc, 3]
            dist_aw_valid = dist_aw_matrix[water_idx, valid_acc_indices]  # [n_valid_acc]

            # Pre-compute donor data for this water (only need hydrogen coords for water angle)
            don_h_coords_water = don_h_coords[valid_don_indices]  # [n_valid_don, 3]
            dist_dw_valid = dist_dw_matrix[water_idx, valid_don_indices]  # [n_valid_don]
            d_angle_valid = d_angle_matrix[water_idx, valid_don_indices]  # [n_valid_don]

            # Sparse water angle calculation (A-O-H_donor)
            # For each acc-don pair, calculate angle between A-O and H_donor-O vectors
            # Only compute angles for pairs that will be checked
            vec_ao = acc_coords_water - water_o_coord  # [n_valid_acc, 3]
            vec_ho = don_h_coords_water - water_o_coord  # [n_valid_don, 3]

            # Compute norms for all vectors first (needed for angle calculation)
            norm_ao = np.linalg.norm(vec_ao, axis=1)  # [n_valid_acc]
            norm_ho = np.linalg.norm(vec_ho, axis=1)  # [n_valid_don]

            # Create all combinations of acc-don pairs using broadcasting
            # For sparse calculation, we compute the full angle matrix but it's now
            # [n_valid_acc, n_valid_don] which is much smaller than original [n_acc, n_don]
            # where n_acc/n_don are the global counts
            vec_ao_expanded = vec_ao[:, np.newaxis, :]  # [n_valid_acc, 1, 3]
            vec_ho_expanded = vec_ho[np.newaxis, :, :]  # [1, n_valid_don, 3]

            norm_ao_expanded = norm_ao[:, np.newaxis]  # [n_valid_acc, 1]
            norm_ho_expanded = norm_ho[np.newaxis, :]  # [1, n_valid_don]

            norm_ao_safe = np.where(norm_ao_expanded == 0, 1, norm_ao_expanded)
            norm_ho_safe = np.where(norm_ho_expanded == 0, 1, norm_ho_expanded)

            dot_ao_ho = np.sum(vec_ao_expanded * vec_ho_expanded, axis=2)  # [n_valid_acc, n_valid_don]

            cos_w_angle = dot_ao_ho / (norm_ao_safe * norm_ho_safe)
            cos_w_angle = np.clip(cos_w_angle, -1.0, 1.0)
            w_angle_matrix = np.degrees(np.arccos(cos_w_angle))  # [n_valid_acc, n_valid_don]

            # Filter by water angle criteria
            w_angle_mask = (w_angle_matrix > config.WATER_BRIDGE_OMEGA_MIN) & \
                           (w_angle_matrix < config.WATER_BRIDGE_OMEGA_MAX)

            # Get all valid combinations
            valid_combinations = np.argwhere(w_angle_mask)

            for acc_offset, don_offset in valid_combinations:
                acc_idx = valid_acc_indices[acc_offset]
                don_idx = valid_don_indices[don_offset]

                acc = all_hba[acc_idx]
                don, _ = donor_h_pairs[don_idx]

                # Skip if same residue (water's own atoms already filtered, but check partners)
                acc_identity = (acc.resname, acc.chain, acc.resnum)
                don_identity = (don.resname, don.chain, don.resnum)

                if acc_identity == water_identity or don_identity == water_identity:
                    continue

                dist_aw = float(dist_aw_valid[acc_offset])
                dist_dw = float(dist_dw_valid[don_offset])
                d_angle = float(d_angle_valid[don_offset])
                w_angle = float(w_angle_matrix[acc_offset, don_offset])

                # is_donor_a = False since acc is acceptor, don is donor
                is_donor_a = False

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
                    distance=dist_aw + dist_dw,
                    angle=w_angle,
                    details={
                        'distance_aw': dist_aw,
                        'distance_dw': dist_dw,
                        'd_angle': d_angle,
                        'w_angle': w_angle,
                        'water_residue': water_res.resid,
                        'water_atom_idx': water_o.idx,  # Add water oxygen atom index for visualization
                        'is_donor_a': is_donor_a,
                        'protisdon': True
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

    def update_coords(self, mda_coords: np.ndarray):
        """Update coordinates from MDAnalysis coordinates array.

        This method updates atom coordinates and refreshes all cached coordinate
        arrays for rapid trajectory analysis.

        Parameters
        ----------
        mda_coords : np.ndarray
            MDA atoms positions array, shape (n_atoms, 3)
        """
        self.atom_container.update_coords_from_mda(mda_coords, aligned_only=True)
        self.atom_container.rebuild_coords_array()
        self.all_coords = self.atom_container.coords_array