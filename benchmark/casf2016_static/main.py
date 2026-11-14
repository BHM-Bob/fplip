import math
import os
import time
from typing import Any, Callable

import pandas as pd
from pymol import cmd
from tqdm import tqdm

from _fplip import analyze_from_pdbstr as fplip_analyze_from_pdbstr
from _plip2 import analyze_from_pdbstr as plip2_analyze_from_pdbstr
from _plip3 import analyze_from_pdbstr as plip3_analyze_from_pdbstr
from _fplip import analyze_from_pdbstr_numpy as fplip_numpy_analyze_from_pdbstr
from _fplip import analyze_from_pdbstr_torch as fplip_torch_analyze_from_pdbstr
from _fplip import analyze_from_pdbstr_cupy as fplip_cupy_analyze_from_pdbstr

DATA_ROOT = 'benchmark/data/CASF-2016'
METHOD_TO_FN = {
    'fplip': fplip_analyze_from_pdbstr,
    'fplip_numpy': fplip_numpy_analyze_from_pdbstr,
    'fplip_torch': fplip_torch_analyze_from_pdbstr,
    'fplip_cupy': fplip_cupy_analyze_from_pdbstr,
    'plip2': plip2_analyze_from_pdbstr,
    'plip3': plip3_analyze_from_pdbstr,
}

def load_complex(cid, receptor: str = 'protein'):
    cmd.reinitialize()
    cmd.load(os.path.join(DATA_ROOT, cid, cid + f'_{receptor}.pdb'), 'rec')
    cmd.alter('rec', 'chain = "R"')
    cmd.load(os.path.join(DATA_ROOT, cid, cid + '_ligand.mol2'), 'lig')
    cmd.alter('lig', 'chain = "L"')
    cmd.alter('lig', 'resn = "UNK"')
    cmd.alter('lig', 'record = "HETATM"')
    return cmd.get_pdbstr('rec or lig')


def benchmark_performance_once(method_name: str, fn: Callable[[str], Any], protein: str = 'protein'):
    df = pd.read_csv(os.path.join(DATA_ROOT, 'test2016.csv'))
    times = []
    for cid in tqdm(df['pdbid'], desc=method_name, leave=False, unit='complex'):
        pdbstr = load_complex(cid, protein)
        start_time = time.time()
        ret = fn(pdbstr)
        if ret is None:
            times.append(math.inf)
            continue
        end_time = time.time()
        times.append(end_time - start_time)
    print(f"{method_name}: {sum(times) / len(times)} s")
    return times


def benchmark_performance(repeat: int = 10, protein: str = 'protein'):
    df = pd.read_csv(os.path.join(DATA_ROOT, 'test2016.csv'))
    perf_df = pd.DataFrame(columns=['pdbid', 'time', 'method', 'repeat'])
    for i in range(repeat):
        print(f"Repeat {i}")
        for method, fn in METHOD_TO_FN.items():
            times = benchmark_performance_once(method, fn, protein)
            perf_df = pd.concat([perf_df, pd.DataFrame({
                'pdbid': df['pdbid'],
                'time': [t for t in times],
                'method': [method] * len(times),
                'repeat': [i] * len(times),
            })], axis=0)
    perf_df.to_csv(f'benchmark/casf2016_static/benchmark_performance_{protein}.csv', index=False)
    print(perf_df)

if __name__ == '__main__':
    # benchmark_performance(3, 'pocket')
    benchmark_performance(3, 'protein')
