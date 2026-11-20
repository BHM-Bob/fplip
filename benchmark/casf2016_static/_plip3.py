from plip.structure.preparation import PDBComplex


def analyze_from_pdbstr(pdbstr: str):
    """Helper method to analyze a PDB string."""
    mol = PDBComplex()
    mol.load_pdb(pdbstr, as_string=True)
    mol.analyze()
    return mol.interaction_sets