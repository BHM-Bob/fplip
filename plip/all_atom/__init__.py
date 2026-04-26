"""
All-Atom Interaction Detection Module for PLIP

This module provides comprehensive interaction detection capabilities
for all atoms in a molecular structure, including:
- Intra-molecular interactions (within ligands, proteins, etc.)
- Inter-molecular interactions (ligand-protein, protein-protein, etc.)

Author: PLIP Optimization Team
"""

from plip.all_atom.atom_container import AtomContainer, AtomInfo
from plip.all_atom.atom_properties import AtomProperties
from plip.all_atom.interaction_catalog import InteractionCatalog
from plip.all_atom.interaction_detector import (Interaction,
                                                UnifiedInteractionDetector)
from plip.all_atom.molecule_complex import MoleculeComplex
from plip.all_atom.residue import Residue
from plip.all_atom.simple_report import SimpleReport

__all__ = [
    'AtomContainer',
    'AtomInfo',
    'AtomProperties',
    'MoleculeComplex',
    'Residue',
    'UnifiedInteractionDetector',
    'Interaction',
    'InteractionCatalog',
    'SimpleReport',
]

def _analyze_complex(pdb_file: str):
    """Helper method to analyze a PDB file."""
    mol = MoleculeComplex()
    mol.load_pdb(pdb_file)
    props = AtomProperties(mol.atom_container)
    detector = UnifiedInteractionDetector(mol.atom_container, props, mol.residues)
    interactions = detector.detect_all()
    return interactions, mol, props


if __name__ == '__main__':
    interactions, mol, props = _analyze_complex('test_data/GPCR_pep.pdb')
    print(interactions)