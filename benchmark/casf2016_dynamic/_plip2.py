import os
from collections import namedtuple
from typing import Union
from gc import collect

import numpy as np
from lazydock.gmx.mda.convert import FakeAtomGroup, PDBConverter
from lazydock.gmx.mda.utils import filter_atoms_by_chains
from mbapy import TaskPool
from MDAnalysis import AtomGroup, Universe
from plip.structure.preparation import PDBComplex
from tqdm import tqdm

from _fplip import set_chainIDs

Analyzer = namedtuple('Analyzer', ['u'])


def _load_traj(root: str):
    tpr = os.path.join(root, "step3_input.pdb")
    xtc = os.path.join(root, "step5_production.dcd")
    u = Universe(tpr, xtc)
    analyzer = Analyzer(u)
    u = set_chainIDs(analyzer).u
    return u


def _perform_analyze_from_ag(ag: Union[AtomGroup, FakeAtomGroup]):
    converter = PDBConverter(ag, reindex=False)
    pdbstr = converter.fast_convert()
    mol = PDBComplex()
    mol.load_pdb(pdbstr, as_string=True)
    mol.analyze()
    pass


def analyze_from_traj(root: str):
    try:
        u = _load_traj(root)
        ag = filter_atoms_by_chains(u.atoms, ['R', 'L'])
        for _ in tqdm(u.trajectory, leave=False):
            _perform_analyze_from_ag(ag)
        return True
    except MemoryError:
        return None

def analyze_from_traj_with_16_parallel(root: str, pool: TaskPool):
    try:
        u = _load_traj(root)
        ag = filter_atoms_by_chains(u.atoms, ['R', 'L'])
        for _ in tqdm(u.trajectory, leave=False):
            _ag = FakeAtomGroup(ag)
            pool.add_task(None, _perform_analyze_from_ag, _ag)
            pool.wait_till_free(0.001)
        pool.wait_till_free(0.001)
        pool.tasks.clear()
        collect()
        return True
    except MemoryError:
        return None


def analyze_from_traj_with_SOL(root: str):
    try:
        u = _load_traj(root)
        # reset segids from such PROA, HETA, SOLV, IONS to PRO, HET, SOL, ION
        # if there are multiple protein chains, this will analyze all of them
        segids = u.atoms.segids
        segids = np.array([s[:3] for s in u.atoms.segments.segids])
        u.atoms.segments.segids = segids
        
        for _ in tqdm(u.trajectory, leave=False):
            _ag = u.select_atoms("(segid PRO or segid HET) or ((around 5 (segid PRO or segid HET)) and segid SOL)")
            _perform_analyze_from_ag(_ag)
        return True
    except MemoryError:
        return None


def analyze_from_traj_with_SOL_16_parallel(root: str, pool: TaskPool):
    try:
        u = _load_traj(root)
        # reset segids from such PROA, HETA, SOLV, IONS to PRO, HET, SOL, ION
        # if there are multiple protein chains, this will analyze all of them
        segids = u.atoms.segids
        segids = np.array([s[:3] for s in u.atoms.segments.segids])
        u.atoms.segments.segids = segids
        
        for _ in tqdm(u.trajectory, leave=False):
            _ag = u.select_atoms("(segid PRO or segid HET) or ((around 5 (segid PRO or segid HET)) and segid SOL)")
            _ag = FakeAtomGroup(_ag)
            pool.add_task(None, _perform_analyze_from_ag, _ag)
            pool.wait_till_free(0.001)
        pool.wait_till_free(0.001)
        pool.tasks.clear()
        collect()
        return True
    except MemoryError:
        return None
