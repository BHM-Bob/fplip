"""
Molecule Complex Module

Provides unified loading and management of all molecular components
in a PDB structure, treating all atoms equally regardless of their
role (protein, ligand, DNA/RNA, etc.)
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import scipy.spatial.distance
from mbapy import opts_file
from openbabel import pybel

from fplip.all_atom.atom_container import AtomContainer, AtomInfo
from fplip.all_atom.residue import Residue
from fplip.basic import config
from fplip.basic.logger import logger
from fplip.basic.supplemental import extract_pdbid, read_pdb, tilde_expansion
from fplip.structure.pdb import PDBParser


class MoleculeComplex:
    """
    Unified representation of a molecular complex.
    
    Loads all atoms from a PDB file without distinguishing between
    ligands and receptors. All atoms are treated equally and stored
    with their residue and chain information.
    """
    
    def __init__(self):
        # Source information
        self.pdb_path: Optional[str] = None
        self.pdb_id: str = "Unknown"
        self.filetype: str = "PDB"
        self.corrected_pdb: Optional[str] = None
        self.hydro_pdb: Optional[str] = None

        # OpenBabel molecule
        self.ob_molecule: Optional[pybel.Molecule] = None

        # Atom container
        self.atom_container = AtomContainer()

        # Polars DataFrame for fast atom queries (built after initialization)
        self.atoms_df: Optional[pl.DataFrame] = None

        # Distance matrix
        self.distance_matrix: Optional[np.ndarray] = None

        # Component tracking
        self.chains: set = set()
        self.residues_info: Dict[Tuple[str, str, int], dict] = {}
        self.residues: List[Residue] = []  # List of Residue objects
        ## Group atoms by residue
        self.residue_groups = defaultdict(list) # Dict[Tuple[str, str, int], List[AtomInfo]]

        # Source files
        self.sourcefiles: Dict[str, str] = {}
        
    def load_pdb(self, pdbpath: str, as_string: bool = False, 
                 fix_pdb: bool = True) -> None:
        """
        Load a PDB file and initialize all atom information.
        
        Parameters
        ----------
        pdbpath : str
            Path to PDB file or PDB content as string
        as_string : bool
            If True, pdbpath is treated as PDB content string
        fix_pdb : bool, optional
            If True, fix PDB structure before loading
        """
        self.pdb_path = pdbpath if not as_string else "from_string"
        
        # Parse and fix PDB
        if fix_pdb:
            pdbparser = PDBParser(pdbpath, as_string=as_string)
            self.corrected_pdb = pdbparser.corrected_pdb
        else:
            if as_string:
                self.corrected_pdb = pdbpath
            else:
                self.corrected_pdb = opts_file(pdbpath, 'r', way='str') # type: ignore
        
        # Read the structure
        if not as_string and os.path.exists(pdbpath):
            self.sourcefiles['pdbcomplex'] = os.path.abspath(pdbpath)
            self.sourcefiles['filename'] = os.path.basename(pdbpath)
        else:
            self.sourcefiles['pdbcomplex'] = 'from_string'
            self.sourcefiles['filename'] = None
        
        # Load with OpenBabel
        if as_string:
            self.ob_molecule, self.filetype = read_pdb(self.corrected_pdb, as_string=True)
        else:
            self.ob_molecule, self.filetype = read_pdb(
                self.corrected_pdb, 
                as_string=(self.corrected_pdb != pdbpath)
            )
        
        # Set PDB ID
        self._set_pdb_id()
        
        logger.info(f'PDB structure successfully read: {self.pdb_id}')
        
        # Add polar hydrogens if needed
        if not config.NOHYDRO:
            self.ob_molecule.OBMol.AddPolarHydrogens()
            logger.info('Added polar hydrogens')
            # Save the PDB with hydrogens for visualization
            self.hydro_pdb = self.ob_molecule.write("pdb")
            logger.info(f'Saved corrected PDB with hydrogens ({len(self.hydro_pdb)} chars)')
        
        # Initialize all atoms
        self._initialize_atoms()

        # Build coordinate array
        self.atom_container.build_coordinate_array()

        # Build residues
        self._build_residues()

        logger.info(f'Loaded {len(self.atom_container)} atoms, {len(self.residues)} residues')
        logger.info(f'Components: {self._get_component_summary()}')
        
    def _set_pdb_id(self):
        """Determine PDB ID from header or filename"""
        if 'HEADER' in self.ob_molecule.data:
            potential_name = self.ob_molecule.data['HEADER'][56:60].lower()
            if extract_pdbid(potential_name) != 'UnknownProtein':
                self.pdb_id = potential_name
            else:
                self.pdb_id = extract_pdbid(self.pdb_path)
        else:
            self.pdb_id = extract_pdbid(self.pdb_path)
    
    def _initialize_atoms(self):
        """Initialize all atoms from the OpenBabel molecule"""
        obmol = self.ob_molecule.OBMol
        
        for obatom in pybel.ob.OBMolAtomIter(obmol):
            # Get original index (1-based in PDB)
            orig_idx = obatom.GetIdx()
            
            # Create atom info
            atom_info = AtomInfo(
                idx=orig_idx,
                obatom=obatom,
                orig_idx=orig_idx
            )
            
            # Add to container
            self.atom_container.add_atom(atom_info)

            # Track chains and residues
            self.chains.add(atom_info.chain)

            res_key = (atom_info.resname, atom_info.chain, atom_info.resnum)
            if res_key not in self.residues_info:
                self.residues_info[res_key] = {
                    'name': atom_info.resname,
                    'chain': atom_info.chain,
                    'num': atom_info.resnum,
                    'component_type': atom_info.component_type
                }

        # # Build Polars DataFrame for fast atom queries
        # self._build_atoms_dataframe()

    def _build_atoms_dataframe(self):
        """Build Polars DataFrame with atom information for fast queries.

        Creates a DataFrame containing essential atom properties:
        - idx: Atom index
        - x, y, z: Coordinates
        - atom_name: Atom name
        - resname: Residue name
        - chain: Chain identifier
        - resnum: Residue number
        - atomic_num: Atomic number
        - component_type: Type of molecular component
        """
        if len(self.atom_container) == 0:
            self.atoms_df = pl.DataFrame()
            return

        # Extract data from atom container
        data = {
            'idx': [],
            'x': [],
            'y': [],
            'z': [],
            'atom_name': [],
            'resname': [],
            'chain': [],
            'resnum': [],
            'atomic_num': [],
            'component_type': [],
        }

        for atom in self.atom_container:
            data['idx'].append(atom.idx)
            data['x'].append(atom.coords[0])
            data['y'].append(atom.coords[1])
            data['z'].append(atom.coords[2])
            data['atom_name'].append(atom.atom_name)
            data['resname'].append(atom.resname)
            data['chain'].append(atom.chain)
            data['resnum'].append(atom.resnum)
            data['atomic_num'].append(atom.atomic_num)
            data['component_type'].append(atom.component_type)

        self.atoms_df = pl.DataFrame(data)
        logger.debug(f'Built atoms DataFrame with {len(self.atoms_df)} rows')

    def _build_residues(self):
        """Build Residue objects from atoms"""

        # Group atoms by residue
        for atom in self.atom_container:
            res_key = (atom.resname, atom.chain, atom.resnum)
            self.residue_groups[res_key].append(atom)

        # Create Residue objects (sorted by residue key for determinism)
        self.residues = []
        for res_key in sorted(self.residue_groups.keys()):
            atoms = self.residue_groups[res_key]
            resname, chain, resnum = res_key
            residue = Residue(resname, chain, resnum)

            for atom in atoms:
                residue.add_atom(atom)
                # Set back-reference from atom to residue for efficient lookup
                atom.residue_obj = residue

            residue.finalize()
            self.residues.append(residue)

        logger.info(f'Built {len(self.residues)} residues')

    def _get_component_summary(self) -> Dict[str, int]:
        """Get summary of component types"""
        summary = {}
        for comp_type in sorted(self.atom_container.component_atoms.keys()):
            atom_set = self.atom_container.component_atoms[comp_type]
            if len(atom_set) > 0:
                summary[comp_type] = len(atom_set)
        return summary
    
    def build_distance_matrix(self, max_distance: float = 10.0) -> np.ndarray:
        """
        Build pairwise distance matrix for all atoms.
        
        Parameters
        ----------
        max_distance : float
            Maximum distance to store (to save memory). Distances > max_distance
            are set to infinity.
            
        Returns
        -------
        np.ndarray
            Distance matrix [n_atoms x n_atoms]
        """
        logger.info(f'Building distance matrix for {len(self.atom_container)} atoms...')
        
        coords = self.atom_container.coords_array
        
        # Compute full distance matrix
        self.distance_matrix = scipy.spatial.distance.cdist(coords, coords)
        
        # Mask distances > max_distance to save memory
        if max_distance > 0:
            self.distance_matrix[self.distance_matrix > max_distance] = np.inf
        
        logger.info(f'Distance matrix built: {self.distance_matrix.shape}')
        
        return self.distance_matrix
    
    def get_atoms_within_distance(self, atom_idx: int, max_dist: float, 
                                   min_dist: float = 0.0) -> List[Tuple[int, float]]:
        """
        Get all atoms within a distance range of a given atom.
        
        Parameters
        ----------
        atom_idx : int
            Index of the reference atom
        max_dist : float
            Maximum distance
        min_dist : float
            Minimum distance (default 0.0)
            
        Returns
        -------
        List[Tuple[int, float]]
            List of (atom_idx, distance) tuples
        """
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not built. Call build_distance_matrix() first.")
        
        if atom_idx not in self.atom_container.idx_to_array_pos:
            return []
        
        array_pos = self.atom_container.idx_to_array_pos[atom_idx]
        distances = self.distance_matrix[array_pos]
        
        # Find atoms within range
        mask = (distances >= min_dist) & (distances <= max_dist)
        valid_positions = np.where(mask)[0]
        
        # Convert back to atom indices
        array_pos_to_idx = {v: k for k, v in self.atom_container.idx_to_array_pos.items()}
        
        result = []
        for pos in valid_positions:
            idx = array_pos_to_idx[pos]
            if idx != atom_idx:  # Exclude self
                result.append((idx, distances[pos]))
        
        return result
    
    def get_component_type(self, atom_idx: int) -> str:
        """Get the component type of an atom"""
        return self.atom_container[atom_idx].component_type
    
    def get_residue_info(self, atom_idx: int) -> Tuple[str, str, int]:
        """Get residue information for an atom"""
        atom = self.atom_container[atom_idx]
        return (atom.resname, atom.chain, atom.resnum)
    
    def __len__(self):
        return len(self.atom_container)
    
    def __iter__(self):
        return iter(self.atom_container)
