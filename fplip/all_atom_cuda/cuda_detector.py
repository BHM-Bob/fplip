"""
CUDA-Accelerated Interaction Detector

Extends UnifiedInteractionDetector with GPU-accelerated computation.
Uses the ComputeBackend abstraction to transparently switch between
NumPy (CPU), CuPy (GPU), and PyTorch (GPU) backends.

Design principles:
1. Maximum reuse: inherits all logic from UnifiedInteractionDetector
2. Only overrides compute-intensive methods (distance, angle calculations)
3. Backend-agnostic: same code works with any ComputeBackend
4. Graceful fallback: if no backend is specified, behaves identically to parent
"""

from typing import Dict, List, Optional

import numpy as np

from fplip.all_atom.atom_container import AtomContainer
from fplip.all_atom.atom_properties import AtomProperties
from fplip.all_atom.interaction_detector import (Interaction,
                                                 UnifiedInteractionDetector)
from fplip.all_atom.residue import Residue
from fplip.all_atom_cuda.backend import ComputeBackend
from fplip.all_atom_cuda.torch_backend import TorchBackend
from fplip.basic import config
from fplip.basic.logger import logger


class CudaInteractionDetector(UnifiedInteractionDetector):
    """CUDA-accelerated interaction detector.

    Inherits from UnifiedInteractionDetector and replaces compute-intensive
    operations with backend-accelerated versions. All non-compute logic
    (property aggregation, deduplication, refinement, water bridges)
    is directly reused from the parent class.

    Parameters
    ----------
    atom_container : AtomContainer
        Container with all atom information
    atom_props : AtomProperties
        Pre-computed atom properties
    residues : List[Residue]
        List of residues to analyze
    backend : ComputeBackend, optional
        Compute backend to use. If None, defaults to TorchBackend().
        Options: CuPyBackend(), TorchBackend()
    keep_on_gpu : bool, optional
        If True, keep Interaction coordinates and distances as GPU tensors.
        This avoids CPU-GPU transfer overhead but requires GPU-aware processing
        of results. Default is False (convert to CPU numpy arrays).

    Examples
    --------
    Using CuPy backend (GPU):

    >>> from plip.all_atom_cuda import CudaInteractionDetector, CuPyBackend
    >>> detector = CudaInteractionDetector(container, props, residues, backend=CuPyBackend())

    Using PyTorch backend (GPU):

    >>> from plip.all_atom_cuda import CudaInteractionDetector, TorchBackend
    >>> detector = CudaInteractionDetector(container, props, residues, backend=TorchBackend())
    """

    def __init__(self, atom_container: AtomContainer, atom_props: AtomProperties,
                 residues: List[Residue], backend: Optional[ComputeBackend] = None,
                 keep_on_gpu: bool = False):
        super().__init__(atom_container, atom_props, residues)
        self.backend = backend if backend is not None else TorchBackend()
        self.keep_on_gpu = keep_on_gpu and self.backend.is_gpu

    def _to_result_value(self, arr):
        """Convert array/scalar to result value based on keep_on_gpu setting.

        GPU-Centric Mode:
        - If keep_on_gpu=True: returns device array/tensor (stays on GPU)
        - If keep_on_gpu=False: transfers to CPU and returns Python float/numpy array
        """
        # if arr is None:
        return None

        # if hasattr(arr, 'shape') and arr.shape == ():
        #     # Scalar tensor
        #     if self.keep_on_gpu:
        #         return arr
        #     return float(self.backend.to_numpy(arr))
        # elif hasattr(arr, '__len__') and len(arr) == 1:
        #     # Single-element array
        #     if self.keep_on_gpu:
        #         return arr
        #     return float(self.backend.to_numpy(arr))
        # else:
        #     # Multi-element array or already a scalar
        #     if self.keep_on_gpu:
        #         return arr
        #     return float(self.backend.to_numpy(arr))

    def detect_all(self, verbose: bool = False) -> Dict[str, List[Interaction]]:
        """Main entry point for detection"""
        logger.info(f'Starting unified interaction detection for {len(self.residues)} residues...')
        
        # First, aggregate properties to residues
        self._aggregate_properties_to_residues()
        
        # Pre-compute charge groups for all residues (for salt bridge detection)
        self._precompute_residue_charge_groups()
        
        # Pre-compute cached data for performance
        self._precompute_cached_data()

        # Detect interactions for each interaction-type
        ## Hydrophobic interactions
        self._detect_hydrophobic()
        ## Hydrogen bonds
        if self._has_explicit_h:
            self._detect_hbonds_case1_vectorized()
        elif config.ALLOW_HEAVY_ATOM_HBOND:
            # Use distance-only detection for heavy atom H-bonds (optional, less reliable)
            logger.info("Using heavy atom H-bond detection (distance-only). "
                       "Note: Results may be less reliable without explicit hydrogens.")
            self._detect_hbonds_without_h()
        ## Salt bridges
        self._detect_saltbridges()
        ## Pistack interactions
        self._detect_pistacking()
        ## Pication interactions
        self._detect_pication()
        ## Halogen bonds
        self._detect_halogen()
        ## Metal interactions
        self._detect_metal()

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

    def _precompute_cached_data(self):
        """Extend parent precomputation with GPU data transfer."""
        super()._precompute_cached_data()
        # move data to gpu
        self.atom_container.coords_array = self.backend.to_device(self.atom_container.coords_array)  # pyright: ignore[reportAttributeAccessIssue]
        self._all_ring_centers = self.backend.to_device(self._all_ring_centers) \
            if len(self._all_ring_centers) > 0 else None
        self._all_ring_normals = self.backend.to_device(self._all_ring_normals) \
            if len(self._all_ring_normals) > 0 else None
        self._all_aromatic_ring_centers = self.backend.to_device(self._all_ring_centers[self._aromatic_ring_mask]) \
            if len(self._aromatic_ring_mask) > 0 else None
        self._pos_coords = self.backend.to_device(self._pos_coords) \
            if len(self._pos_coords) > 0 else None
        self._neg_grouped_centers = self.backend.to_device(self._neg_grouped_centers) \
            if len(self._neg_grouped_centers) > 0 else None
        self._metals_coords = self.backend.to_device(self._metals_coords) \
            if len(self._metals_coords) > 0 else None
        self._binding_coords = self.backend.to_device(self._binding_coords) \
            if len(self._binding_coords) > 0 else None
        self._all_halogen_donor_coords = self.backend.to_device(self._all_halogen_donor_coords) \
            if len(self._all_halogen_donor_coords) > 0 else None
        self._all_halogen_acceptor_coords = self.backend.to_device(self._all_halogen_acceptor_coords) \
            if isinstance(self._all_halogen_acceptor_coords, np.ndarray) and len(self._all_halogen_acceptor_coords) > 0 else None
        # calcu all-atom distance matrix
        if self.atom_container.remain_atom_idxs is None:
            self._all_atom_dist_matrix = self.backend.cdist(self.atom_container.coords_array, self.atom_container.coords_array)
        else:
            coords = self.atom_container.coords_array[self.atom_container.remain_atom_idxs]
            self._all_atom_dist_matrix = self.backend.cdist(coords, coords)
        # special atoms idxs
        self._hydrophobic_atoms_idxs = self.atom_container.get_atom_coords_idxs_from_atoms(self._hydrophobic_atoms_list)
        self._hbond_acc_idxs = self.atom_container.get_atom_coords_idxs_from_atoms(self._all_hba)
        self._hbond_don_idxs, self._hbond_donh_idxs = [], []
        for don, donhs in self._all_hbd:
            for h in donhs:
                self._hbond_don_idxs.append(don.idx)
                self._hbond_donh_idxs.append(h.idx)
        self._hbond_don_idxs = self.atom_container.idx_to_array_pos_array[self._hbond_don_idxs]
        self._hbond_donh_idxs = self.atom_container.idx_to_array_pos_array[self._hbond_donh_idxs]
        # special atoms skip-attr vector
        if self.atom_container.remain_atom_mask is not None:
            sorted_indices = np.array(self.atom_container.sorted_indices)[self.atom_container.remain_atom_mask]
            self._n_to_k_map = self.backend.full((self.atom_container.coords_array.shape[0],), -1, dtype=self.backend.long)
            self._n_to_k_map[self.atom_container.remain_atom_idxs] = self.backend.arange(self._all_atom_dist_matrix.shape[0])
        else:
            sorted_indices = self.atom_container.sorted_indices
        self._is_std_aa = self.backend.to_device([self.atom_container.atoms[i].residue_obj.should_filter_self() for i in sorted_indices])
        self._res_uids = self.backend.to_device([self.atom_container.atoms[i].residue_obj._hash for i in sorted_indices])
        # use direct calculation to avoid useless self._skip_mask to save cuda memory
        # self._skip_mask = self._res_uids[None, :] == self._res_uids[:, None]
        # self._skip_mask &= (self._is_std_aa[None, :] & self._is_std_aa[:, None])
        # self._remain_mask = ~self._skip_mask
        self._remain_mask = ~((self._res_uids[None, :] == self._res_uids[:, None]) & (self._is_std_aa[None, :] & self._is_std_aa[:, None]))
        
    def _index_dist_matrix(self, i_idxs, j_idxs, matrix=None):
        if matrix is None:
            matrix = self._all_atom_dist_matrix
        if self.atom_container.remain_atom_mask is None:
            return matrix[i_idxs][:, j_idxs]
        mapped_k_i = self._n_to_k_map[i_idxs]
        mapped_k_j = self._n_to_k_map[j_idxs]
        assert (mapped_k_i != -1).all(), "n_i_idxs including skipped atoms"
        assert (mapped_k_j != -1).all(), "n_j_idxs including skipped atoms"
        return matrix[mapped_k_i][:, mapped_k_j]
        
    def _detect_hydrophobic(self):
        """Detect hydrophobic interactions with backend-accelerated distance calculation."""
        dist_matrix = self._index_dist_matrix(self._hydrophobic_atoms_idxs, self._hydrophobic_atoms_idxs)
        valid_mask = (dist_matrix < config.HYDROPH_DIST_MAX) & \
                     (dist_matrix > config.MIN_DIST)
        remain_mask = self._index_dist_matrix(self._hydrophobic_atoms_idxs, self._hydrophobic_atoms_idxs, self._remain_mask)
        final_mask = valid_mask & remain_mask
        if not final_mask.any():
            return 
        
        idxs_i, idxs_j = self.backend.argwhere(final_mask)
        for i, j in zip(idxs_i, idxs_j):
            atom_a = self._hydrophobic_atoms_list[i]
            atom_b = self._hydrophobic_atoms_list[j]

            # Skip if both atoms belong to the same 3~6 ring
            rings_a = self._small_ring_atom_sets.get(atom_a.idx, set())
            rings_b = self._small_ring_atom_sets.get(atom_b.idx, set())
            if rings_a & rings_b:  # intersection - same 3~6 ring
                continue

            # Skip if one atom is connected to the other's aromatic ring via chemical bond
            # (e.g., S attached to benzene ring - S cannot form hydrophobic interaction with ring atoms)
            # A connected to ring R means: A is not in R, but A has a neighbor atom that is in R
            # This filters out 1-3 and 1-4 invalid interaction
            if rings_b:
                neighbors_a = self.atom_props.atom_neighbors.get(atom_a.idx, set())
                ring_atoms_b = set()
                for ring_idx in rings_b:
                    ring_atoms_b.update(self._small_rings[ring_idx]['indices'])
                if neighbors_a & ring_atoms_b:
                    continue
            if rings_a:
                neighbors_b = self.atom_props.atom_neighbors.get(atom_b.idx, set())
                ring_atoms_a = set()
                for ring_idx in rings_a:
                    ring_atoms_a.update(self._small_rings[ring_idx]['indices'])
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
                distance=self._to_result_value(dist_matrix[i, j]),
                angle=None,
                details={}
            )
            self.interactions['hydrophobic'].append(interaction)

    def _detect_hbonds_case1_vectorized(self):
        """Case 1: Residue is donor, other is acceptor (backend-accelerated)."""
        # Convert all coordinates to backend arrays for consistency
        don_coords = self.atom_container.coords_array[self._hbond_don_idxs]  # [n_pairs, 3]
        h_coords = self.atom_container.coords_array[self._hbond_donh_idxs]    # [n_pairs, 3]
        acc_coords = self._all_hba_coords  # [n_hba, 3]

        dist_ad_matrix = self._index_dist_matrix(self._hbond_don_idxs, self._hbond_acc_idxs)
        dist_ah_matrix = self._index_dist_matrix(self._hbond_donh_idxs, self._hbond_acc_idxs)

        dist_mask = (dist_ad_matrix > config.MIN_DIST) & (dist_ad_matrix < config.HBOND_DIST_MAX)
        remain_mask = self._index_dist_matrix(self._hbond_don_idxs, self._hbond_acc_idxs, self._remain_mask)
        final_mask = dist_mask & remain_mask

        vec_hd = don_coords - h_coords
        vec_ha = self.backend.expand_dims(acc_coords, 0) - self.backend.expand_dims(h_coords, 1)

        norm_hd = self.backend.norm(vec_hd, axis=1)
        norm_ha = self.backend.norm(vec_ha, axis=2)

        norm_hd_safe = self.backend.where(norm_hd == 0, 1, norm_hd)
        norm_ha_safe = self.backend.where(norm_ha == 0, 1, norm_ha)

        dot_product = self.backend.sum(vec_ha * self.backend.expand_dims(vec_hd, 1), axis=2)

        cos_angle = dot_product / (self.backend.expand_dims(norm_hd_safe, 1) * norm_ha_safe)
        cos_angle = self.backend.clip(cos_angle, -1.0, 1.0)
        angle_matrix = self.backend.degrees(self.backend.arccos(cos_angle))

        # Add small tolerance for floating-point precision differences across backends
        angle_mask = angle_matrix > (config.HBOND_DON_ANGLE_MIN - 1e-10)
        valid_mask = final_mask & angle_mask

        if not valid_mask.any():
            return 
        
        pair_idxs, acc_idxs = self.backend.argwhere(valid_mask)
        dists_ad, dists_ah, angles = dist_ad_matrix[pair_idxs, acc_idxs], dist_ah_matrix[pair_idxs, acc_idxs], angle_matrix[pair_idxs, acc_idxs]
        dists_ad, dists_ah, angles = self.backend.to_numpy(dists_ad), self.backend.to_numpy(dists_ah), self.backend.to_numpy(angles)
        hbond_types = np.where((dists_ad < 3.2) & (angles > 140), 'strong', 'weak')
        for pair_idx, acc_idx, dist_ad, dist_ah, angle, htype in zip(pair_idxs, acc_idxs, dists_ad, dists_ah, angles, hbond_types):
            pair_idx, acc_idx = int(pair_idx), int(acc_idx)
            donor, h_atom = self._hbond_don_idxs[pair_idx], self._hbond_donh_idxs[pair_idx]
            donor, h_atom = self.atom_container.atoms[self.atom_container.array_pos_to_idx_array[donor]], self.atom_container.atoms[self.atom_container.array_pos_to_idx_array[h_atom]]
            hba = self._all_hba[acc_idx]

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
                    'type': htype,
                    'donor_idx': donor.idx,
                    'acceptor_idx': hba.idx
                },
                objs={'donor': donor, 'h_atom': h_atom, 'acceptor': hba}
            )
            self.interactions['hbond'].append(interaction)

    def _detect_hbonds_without_h(self):
        """Detect H-bonds without explicit hydrogens (backend-accelerated)."""
        # Convert all coordinates to backend arrays for consistency
        dist_ad_matrix = self._index_dist_matrix(self._hbond_don_idxs, self._hbond_acc_idxs)

        dist_mask = (dist_ad_matrix >= 2.5) & (dist_ad_matrix <= 3.5)
        remain_mask = self._index_dist_matrix(self._hbond_don_idxs, self._hbond_acc_idxs, self._remain_mask)
        final_mask = dist_mask & remain_mask

        if not final_mask.any():
            return 
        
        pair_idxs, acc_idxs = self.backend.argwhere(final_mask)
        dists_ad = dist_ad_matrix[pair_idxs, acc_idxs]
        dists_ad = self.backend.to_numpy(dists_ad)
        for pair_idx, acc_idx, dist_ad in zip(pair_idxs, acc_idxs, dists_ad):
            pair_idx, acc_idx = int(pair_idx), int(acc_idx)
            donor = self._hbond_don_idxs[pair_idx]
            hba = self._all_hba[acc_idx]

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

    def _detect_saltbridges(self):
        """Detect salt bridges with backend-accelerated distance calculation."""
        if len(self._neg_grouped_centers) > 0 and len(self._pos_grouped_atoms) > 0:
            dist_matrix = self.backend.cdist(self._pos_grouped_centers, self._neg_grouped_centers)
            dist_mask = dist_matrix < config.SALTBRIDGE_DIST_MAX
            
            if not dist_mask.any():
                return
            pos_idx, neg_idx = self.backend.argwhere(dist_mask)
            dists = self.backend.to_numpy(dist_matrix[pos_idx, neg_idx])
            for pos_idx, neg_idx, distance in zip(pos_idx, neg_idx, dists):
                pos_atoms = self._pos_grouped_atoms[pos_idx]
                neg_atoms = self._neg_grouped_atoms[neg_idx]
                pos_key = self._pos_grouped_keys[pos_idx]
                neg_key = self._neg_grouped_keys[neg_idx]

                pos_atom = pos_atoms[0]
                neg_atom = neg_atoms[0]

                if self._should_skip_interaction(pos_atom.residue_obj, pos_atom, neg_atom):
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
                    distance=distance,
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

    def _detect_pistacking(self):
        """Detect pi-stacking interactions with backend-accelerated calculations."""
        all_rings = self.atom_props.rings
        if not all_rings:
            return

        dist_matrix = self.backend.cdist(self._all_ring_centers, self._all_ring_centers)

        valid_dist_mask = dist_matrix <= config.PISTACK_DIST_MAX

        dot_products = self.backend.sum(
            self._all_ring_normals[:, np.newaxis, :] * self._all_ring_normals[np.newaxis, :, :], axis=2
        )
        angles = self.backend.arccos(self.backend.clip(dot_products, -1.0, 1.0)) * 180.0 / np.pi

        parallel_mask = (angles < config.PISTACK_ANG_DEV) | (angles > (180 - config.PISTACK_ANG_DEV))
        perp_mask = ((90 - config.PISTACK_ANG_DEV) < angles) & (angles < (90 + config.PISTACK_ANG_DEV))
        valid_angle_mask = parallel_mask | perp_mask
        valid_mask = valid_dist_mask & valid_angle_mask
        if not valid_mask.any():
            return 
        
        a_idxs, b_idxs = self.backend.argwhere(valid_mask)
        dists = self.backend.to_numpy(dist_matrix[a_idxs, b_idxs])
        angles = self.backend.to_numpy(angles[a_idxs, b_idxs])
        for i, j, distance, angle in zip(a_idxs, b_idxs, dists, angles):
            ring_a = all_rings[i]
            ring_b = all_rings[j]

            if ring_a['indices'][0] == ring_b['indices'][0]:
                continue

            atom_a = self.atom_container[ring_a['indices'][0]]
            atom_b = self.atom_container[ring_b['indices'][0]]

            if self._should_skip_interaction(atom_a.residue_obj, atom_a, atom_b):
                continue

            if angle < config.PISTACK_ANG_DEV or angle > (180 - config.PISTACK_ANG_DEV):
                ptype = 'parallel'
            else:
                ptype = 'perpendicular'

            vec_ab = ring_a['center'] - ring_b['center']
            proj_dist = np.dot(vec_ab, ring_b['normal'])
            proj1 = ring_a['center'] - proj_dist * ring_b['normal']
            offset1 = np.linalg.norm(proj1 - ring_b['center'])

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

    def _detect_pication(self):
        """Detect pi-cation interactions with backend-accelerated calculations."""
        all_pos = self.atom_props.get_pos_charged()

        if not all_pos or not self._aromatic_rings:
            return

        dist_matrix = self.backend.cdist(self._all_aromatic_ring_centers, self._pos_coords)

        valid_dist_mask = dist_matrix < config.PICATION_DIST_MAX
        if valid_dist_mask.any():
            ring_idxs, pos_idxs = self.backend.argwhere(valid_dist_mask)
            dists = self.backend.to_numpy(dist_matrix[ring_idxs, pos_idxs])
            for i, j, distance in zip(ring_idxs, pos_idxs, dists):
                ring = self._aromatic_rings[i]
                pos_atom = all_pos[j]

                atom_a = self.atom_container[ring['indices'][0]]
                if self._should_skip_interaction(atom_a.residue_obj, atom_a, pos_atom):
                    continue

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

    def _detect_halogen(self):
        """Detect halogen bonds with backend-accelerated distance calculation."""
        if not (self._all_halogen_donors or self._all_halogen_acceptors):
            return
        don_atoms = list(map(lambda x: x[0], self._all_halogen_donors))
        don_idxs = self.atom_container.get_atom_coords_idxs_from_atoms(don_atoms)
        acc_idxs = self.atom_container.get_atom_coords_idxs_from_atoms(self._all_halogen_acceptors)
        dist_matrix = self._index_dist_matrix(don_idxs, acc_idxs)
        valid_mask = (dist_matrix > config.MIN_DIST) & (dist_matrix < config.HALOGEN_DIST_MAX)
        remain_mask = self._index_dist_matrix(don_idxs, acc_idxs, self._remain_mask)
        final_mask = valid_mask & remain_mask
        if not final_mask.any():
            return 
        don_idxs, acc_idxs = self.backend.argwhere(final_mask)
        dists = self.backend.to_numpy(dist_matrix[don_idxs, acc_idxs])
        for i, j, distance in zip(don_idxs, acc_idxs, dists):
            donor, htype = self._all_halogen_donors[i]
            acc = self._all_halogen_acceptors[j]

            if self._should_skip_interaction(donor.residue_obj, donor, acc):
                continue
            
            don_angle, acc_angle = self._get_halogen_bond_angles(donor, acc)

            if don_angle is None:
                continue

            if not (config.HALOGEN_DON_ANGLE - config.HALOGEN_ANGLE_DEV < don_angle):
                continue

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

    def _detect_metal(self):
        """Detect metal complexation with backend-accelerated distance calculation."""
        if not (self.all_metal_binding_atoms or self.all_metals_atoms):
            return
        
        metal_idxs = self.atom_container.get_atom_coords_idxs_from_atoms(self.all_metals_atoms)
        binding_idxs = self.atom_container.get_atom_coords_idxs_from_atoms(self.all_metal_binding_atoms)

        dist_matrix = self._index_dist_matrix(metal_idxs, binding_idxs)
        dist_mask = (dist_matrix < config.METAL_DIST_MAX)
        remain_mask = self._index_dist_matrix(metal_idxs, binding_idxs, self._remain_mask)
        final_mask = dist_mask & remain_mask
        if not final_mask.any():
            return 
        
        metal_idxs, binding_idxs = self.backend.argwhere(final_mask)
        dists = self.backend.to_numpy(dist_matrix[metal_idxs, binding_idxs])
        for i, j, distance in zip(metal_idxs, binding_idxs, dists):
            metal = self.all_metals_atoms[i]
            binding = self.all_metal_binding_atoms[j]

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
                distance=distance,
                angle=None,
                details={}
            )
            self.interactions['metal'].append(interaction)

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
        self.atom_container.coords_array = self.backend.to_device(self.atom_container.coords_array) # type: ignore
        self.all_coords = self.atom_container.coords_array