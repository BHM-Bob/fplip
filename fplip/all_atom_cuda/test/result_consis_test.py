"""
All-Atom Module Consistency and Performance Testing Framework

This module provides comprehensive testing for all-atom module analysis results:
1. Consistency Testing: Verify results match expected baseline
2. Performance Testing: Measure execution time and identify bottlenecks

Usage:
    # Run all tests with default NumPy backend
    python -m fplip.all_atom_cuda.test.result_consis_test
    
    # Run with specific backend
    python -m fplip.all_atom_cuda.test.result_consis_test --backend numpy
    python -m fplip.all_atom_cuda.test.result_consis_test --backend cupy
    python -m fplip.all_atom_cuda.test.result_consis_test --backend torch
    python -m fplip.all_atom_cuda.test.result_consis_test --backend original
    
    # Run specific test types
    python -m fplip.all_atom_cuda.test.result_consis_test --test-type consistency
    python -m fplip.all_atom_cuda.test.result_consis_test --test-type performance
    python -m fplip.all_atom_cuda.test.result_consis_test --test-type all --verbose
    
    # Run specific test cases with specific backend
    python -m fplip.all_atom_cuda.test.result_consis_test --backend cupy --test-cases 2w0s 4kya
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

sys.path.insert(0, str((Path(__file__).parent / '../../..').resolve()))

from fplip.all_atom.atom_properties import AtomProperties
from fplip.all_atom.interaction_detector import UnifiedInteractionDetector
from fplip.all_atom.molecule_complex import MoleculeComplex
from fplip.basic import config

# =============================================================================
# Test Configuration
# =============================================================================

# Test data directories
# Standard PLIP test PDB files
PLIP_TEST_DATA_DIR = Path(__file__).parent.parent.parent / 'test/pdb'
# Custom test data (if available)
CUSTOM_TEST_DATA_DIR = Path(__file__).parent.parent.parent.parent / 'test_data'

RESULTS_DIR = CUSTOM_TEST_DATA_DIR / "all_atom_cuda_test_results"
BASELINE_DIR = CUSTOM_TEST_DATA_DIR / "all_atom_cuda_baselines"


def get_pdb_path(filename: str) -> Path:
    """Get the full path to a PDB test file.
    
    First checks custom test data directory, then falls back to PLIP test data.
    """
    # Check custom directory first
    custom_path = CUSTOM_TEST_DATA_DIR / filename
    if custom_path.exists():
        return custom_path
    
    # Fall back to PLIP test data directory
    plip_path = PLIP_TEST_DATA_DIR / filename
    if plip_path.exists():
        return plip_path
    
    # Return the path anyway (will be checked later for existence)
    return plip_path

# Test cases with expected characteristics
# Selected to cover various interaction types for comprehensive testing
TEST_CASES = {
    "GPCR_pep": {
        "file": "GPCR_pep.pdb",
        "description": "GPCR peptide complex - medium size",
        "expected_interactions": {
            "hbond": ">=3",
            "hydrophobic": ">=2",
            "saltbridge": ">=1"
        },
        "categories": ["peptide", "medium"]
    },
    "2w0s": {
        "file": "2w0s.pdb",
        "description": "Vacc-TK to TDP complex - halogen bond, pi-stacking, salt bridge",
        "expected_interactions": {
            "hbond": ">=2",
            "halogen": ">=1",
            "pistacking": ">=1",
            "saltbridge": ">=2"
        },
        "categories": ["halogen", "pistacking", "saltbridge", "small"]
    },
    # "4kya": {
    #     "file": "4kya.pdb",
    #     "description": "TS inhibitor with hydrophobic, pi-stacking, salt bridge",
    #     "expected_interactions": {
    #         "hbond": ">=1",
    #         "saltbridge": ">=1",
    #         "hydrophobic": ">=4",
    #         "pistacking": ">=2"
    #     },
    #     "categories": ["hydrophobic", "pistacking", "saltbridge", "medium"]
    # },
    "1rmd": {
        "file": "1rmd.pdb",
        "description": "Zinc coordination in RAG1 dimerization domain",
        "expected_interactions": {
            "metal": ">=4"
        },
        "categories": ["metal", "small"]
    },
    "2zoz": {
        "file": "2zoz.pdb",
        "description": "Cyclophilin A complex - extensive water bridges (357 waters, 100+ water bridges)",
        "expected_interactions": {
            "hbond": ">=5",
            "water_bridge": ">=50",
            "water_bridge_possible": ">=300"
        },
        "categories": ["water_bridge", "large"]
    },
    "GPCR_MD": {
        "file": "",
        "description": "GPCR peptide MD trajectory - full iteration benchmark",
        "type": "trajectory",
        "tpr": "pull/pull.tpr",
        "xtc": "pull/pull_center.xtc",
        "gro": "pull/pull.gro",
        "expected_interactions": {},
        "categories": ["trajectory", "large"]
    }
}

# Performance benchmarks (seconds)
PERFORMANCE_THRESHOLDS = {
    "small": {"max_time": 2.0, "atoms": "<500"},      # < 500 atoms
    "medium": {"max_time": 10.0, "atoms": "500-2000"}, # 500-2000 atoms
    "large": {"max_time": 60.0, "atoms": ">2000"},    # > 2000 atoms
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class InteractionSummary:
    """Summary of interactions for a single residue pair"""
    interaction_type: str
    res_a_name: str
    res_a_num: int
    res_a_chain: str
    res_b_name: str
    res_b_num: int
    res_b_chain: str
    distance: float


@dataclass
class AnalysisResult:
    """Complete analysis result for a PDB structure"""
    pdb_id: str
    timestamp: str
    num_residues: int
    num_atoms: int
    interactions: Dict[str, List[InteractionSummary]]
    execution_time: float


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single run"""
    pdb_id: str
    total_time: float
    load_time: float
    analysis_time: float
    num_atoms: int
    num_residues: int
    num_interactions: int


# =============================================================================
# Core Analysis Function
# =============================================================================

def process_pdb_detailed(pdbfile: str, backend: str = 'numpy') -> Tuple[AnalysisResult, PerformanceMetrics]:
    """
    Process a PDB file with detailed timing and result collection.
    
    Uses NOHYDRO=True to ensure deterministic results by using hydrogen atoms
    already present in the PDB file, avoiding non-deterministic behavior from
    OpenBabel's AddPolarHydrogens() function.
    
    Args:
        pdbfile: Path to PDB file
        backend: Compute backend to use ('numpy', 'cupy', 'torch', or None for original)
    
    Returns:
        Tuple of (AnalysisResult, PerformanceMetrics]
    """
    # Use hydrogen atoms from PDB file for deterministic results
    # OpenBabel's AddPolarHydrogens() produces non-deterministic hydrogen positions
    config.NOHYDRO = True
    
    start_time = time.perf_counter()
    
    pdb_file_name = pdbfile.split('/')[-1]
    print(f'  Starting analysis of {pdb_file_name} (backend: {backend})')
    
    # Load PDB
    load_start = time.perf_counter()
    mol = MoleculeComplex()
    mol.load_pdb(pdbfile)
    load_end = time.perf_counter()
    load_time = load_end - load_start
    
    # Count atoms and residues
    num_atoms = len(mol.atom_container.atoms)
    num_residues = len(mol.residues)
    
    # Analyze interactions
    analysis_start = time.perf_counter()
    props = AtomProperties(mol.atom_container)
    
    # Create detector based on backend choice
    if backend == 'cupy':
        from fplip.all_atom_cuda import CudaInteractionDetector, CuPyBackend
        detector = CudaInteractionDetector(mol.atom_container, props, mol.residues,
                                           backend=CuPyBackend())
    elif backend == 'torch':
        from fplip.all_atom_cuda import CudaInteractionDetector, TorchBackend
        detector = CudaInteractionDetector(mol.atom_container, props, mol.residues,
                                           backend=TorchBackend())
    elif backend == 'numpy':
        from fplip.all_atom_cuda import CudaInteractionDetector
        from fplip.all_atom_cuda.numpy_backend import NumPyBackend
        detector = CudaInteractionDetector(mol.atom_container, props, mol.residues,
                                           backend=NumPyBackend())
    else:  # 'original' or any other value
        detector = UnifiedInteractionDetector(mol.atom_container, props, mol.residues)
    
    interactions = detector.detect_all()
    analysis_end = time.perf_counter()
    analysis_time = analysis_end - analysis_start
    
    total_time = time.perf_counter() - start_time
    
    # Convert interactions to summary format
    interaction_summaries = {}
    total_interactions = 0
    
    for itype, inters in interactions.items():
        interaction_summaries[itype] = []
        for inter in inters:
            summary = InteractionSummary(
                interaction_type=itype,
                res_a_name=inter.res_a_name,
                res_a_num=inter.res_a_num,
                res_a_chain=inter.res_a_chain,
                res_b_name=inter.res_b_name,
                res_b_num=inter.res_b_num,
                res_b_chain=inter.res_b_chain,
                distance=inter.distance
            )
            interaction_summaries[itype].append(summary)
        total_interactions += len(inters)
    
    # Create result objects
    analysis_result = AnalysisResult(
        pdb_id=Path(pdbfile).stem,
        timestamp=datetime.now().isoformat(),
        num_residues=num_residues,
        num_atoms=num_atoms,
        interactions=interaction_summaries,
        execution_time=total_time
    )
    
    performance_metrics = PerformanceMetrics(
        pdb_id=Path(pdbfile).stem,
        total_time=total_time,
        load_time=load_time,
        analysis_time=analysis_time,
        num_atoms=num_atoms,
        num_residues=num_residues,
        num_interactions=total_interactions
    )
    
    return analysis_result, performance_metrics


# =============================================================================
# Consistency Testing
# =============================================================================

class ConsistencyTester:
    """Test consistency of all-atom module results against baselines"""
    
    def __init__(self, baseline_dir: Path = BASELINE_DIR, backend: str = 'numpy'):
        self.baseline_dir = baseline_dir
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.backend = backend
    
    def generate_baseline(self, test_case: str, pdb_path: Path) -> Dict:
        """Generate a new baseline for a test case"""
        print(f"Generating baseline for {test_case}...")
        
        result, metrics = process_pdb_detailed(str(pdb_path), backend=self.backend)
        
        # Convert to serializable format
        baseline = {
            "test_case": test_case,
            "pdb_file": os.path.basename(pdb_path),
            "created_at": result.timestamp,
            "num_residues": result.num_residues,
            "num_atoms": result.num_atoms,
            "interactions": {
                itype: [asdict(inter) for inter in inters]
                for itype, inters in result.interactions.items()
            },
            "metrics": asdict(metrics)
        }
        
        # Save baseline
        baseline_file = self.baseline_dir / f"{test_case}_baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        print(f"  Baseline saved to {baseline_file}")
        return baseline
    
    def load_baseline(self, test_case: str) -> Optional[Dict]:
        """Load existing baseline for a test case"""
        baseline_file = self.baseline_dir / f"{test_case}_baseline.json"
        if not baseline_file.exists():
            return None
        
        with open(baseline_file, 'r') as f:
            return json.load(f)
    
    def compare_results(self, current: AnalysisResult, baseline: Dict) -> Dict:
        """Compare current results with baseline"""
        differences = {
            "match": True,
            "residue_count_match": current.num_residues == baseline["num_residues"],
            "atom_count_match": current.num_atoms == baseline["num_atoms"],
            "interaction_counts": {},
            "details": []
        }
        
        # Check residue count
        if not differences["residue_count_match"]:
            differences["match"] = False
            differences["details"].append(
                f"Residue count mismatch: current={current.num_residues}, "
                f"baseline={baseline['num_residues']}"
            )
        
        # Check atom count
        if not differences["atom_count_match"]:
            differences["match"] = False
            differences["details"].append(
                f"Atom count mismatch: current={current.num_atoms}, "
                f"baseline={baseline['num_atoms']}"
            )
        
        # Compare interaction counts
        baseline_interactions = baseline.get("interactions", {})
        current_interactions = {
            itype: len(inters) for itype, inters in current.interactions.items()
        }
        
        all_types = set(baseline_interactions.keys()) | set(current_interactions.keys())
        
        for itype in all_types:
            b_count = len(baseline_interactions.get(itype, []))
            c_count = current_interactions.get(itype, 0)
            if b_count != c_count:
                differences["match"] = False
                differences["details"].append(
                    f"{itype}: baseline={b_count}, current={c_count}"
                )
        
        return differences
    
    def run_test(self, test_case: str, pdb_path: Path, 
                 generate_baseline: bool = False) -> bool:
        """Run a single consistency test"""
        print(f"\n{'='*60}")
        print(f"Testing: {test_case}")
        print(f"PDB: {pdb_path}")
        print(f"{'='*60}")
        
        if generate_baseline:
            baseline = self.generate_baseline(test_case, pdb_path)
            print(f"  Status: BASELINE GENERATED")
            return True
        
        # Load existing baseline
        baseline = self.load_baseline(test_case)
        if baseline is None:
            print(f"  Status: NO BASELINE - Generating...")
            baseline = self.generate_baseline(test_case, pdb_path)
            print(f"  Status: BASELINE GENERATED")
            return True
        
        # Run analysis
        print(f"  Running analysis...")
        current_result, _ = process_pdb_detailed(str(pdb_path), backend=self.backend)
        
        # Compare with baseline
        print(f"  Comparing with baseline...")
        differences = self.compare_results(current_result, baseline)
        
        if differences["match"]:
            print(f"  Status: PASS")
            return True
        else:
            print(f"  Status: FAIL")
            for detail in differences["details"]:
                print(f"    - {detail}")
            return False
    
    def run_all_tests(self, generate_baselines: bool = False, 
                      test_cases: Optional[List[str]] = None) -> bool:
        """Run all consistency tests
        
        Args:
            generate_baselines: Whether to generate new baselines
            test_cases: Optional list of specific test case names to run. 
                       If None, runs all test cases.
        """
        print("\n" + "="*60)
        print("ALL-ATOM MODULE CONSISTENCY TESTS")
        print("="*60)
        
        all_passed = True
        
        # Filter test cases if specified
        cases_to_run = TEST_CASES
        if test_cases is not None:
            cases_to_run = {k: v for k, v in TEST_CASES.items() if k in test_cases}
            if not cases_to_run:
                print(f"\nWARNING: No matching test cases found for: {test_cases}")
                return False
        
        for test_name, test_info in cases_to_run.items():
            pdb_file = get_pdb_path(test_info["file"])
            if not pdb_file.exists():
                print(f"\nWARNING: Test file not found: {pdb_file}")
                continue
            
            passed = self.run_test(test_name, pdb_file, generate_baselines)
            all_passed = all_passed and passed
        
        print("\n" + "="*60)
        if all_passed:
            print("ALL CONSISTENCY TESTS PASSED")
        else:
            print("SOME CONSISTENCY TESTS FAILED")
        print("="*60)
        
        return all_passed


# =============================================================================
# Performance Testing
# =============================================================================

class PerformanceTester:
    """Test performance of all-atom module analysis"""
    
    def __init__(self, results_dir: Path = RESULTS_DIR, backend: str = 'numpy'):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
        self.backend = backend
    
    def run_single_benchmark(self, test_case: str, pdb_path: Path, 
                            num_runs: int = 3) -> Dict:
        """Run performance benchmark multiple times"""
        print(f"\nBenchmarking {test_case} ({num_runs} runs, backend: {self.backend})...")
        
        metrics_list = []
        
        for i in range(num_runs):
            print(f"  Run {i+1}/{num_runs}...")
            _, metrics = process_pdb_detailed(str(pdb_path), backend=self.backend)
            metrics_list.append(metrics)
            print(f"    {metrics.total_time:.2f}s")
        
        # Calculate statistics
        total_times = [m.total_time for m in metrics_list]
        load_times = [m.load_time for m in metrics_list]
        analysis_times = [m.analysis_time for m in metrics_list]
        
        benchmark = {
            "test_case": test_case,
            "pdb_file": str(pdb_path),
            "num_runs": num_runs,
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "total_time": {
                    "mean": statistics.mean(total_times),
                    "stdev": statistics.stdev(total_times) if len(total_times) > 1 else 0,
                    "min": min(total_times),
                    "max": max(total_times)
                },
                "load_time": {
                    "mean": statistics.mean(load_times),
                    "stdev": statistics.stdev(load_times) if len(load_times) > 1 else 0,
                },
                "analysis_time": {
                    "mean": statistics.mean(analysis_times),
                    "stdev": statistics.stdev(analysis_times) if len(analysis_times) > 1 else 0,
                }
            },
            "structure_info": {
                "num_atoms": metrics_list[0].num_atoms,
                "num_residues": metrics_list[0].num_residues,
                "num_interactions": metrics_list[0].num_interactions
            },
            "all_runs": [asdict(m) for m in metrics_list]
        }
        
        # Save benchmark results
        result_file = self.results_dir / f"{test_case}_performance.json"
        with open(result_file, 'w') as f:
            json.dump(benchmark, f, indent=2)
        
        return benchmark
    
    def print_benchmark_summary(self, benchmark: Dict):
        """Print formatted benchmark summary"""
        stats = benchmark["statistics"]
        info = benchmark["structure_info"]
        
        print(f"\n  Summary for {benchmark['test_case']}:")
        print(f"    Structure: {info['num_atoms']} atoms, {info['num_residues']} residues")
        print(f"    Interactions: {info['num_interactions']}")
        print(f"    Total Time: {stats['total_time']['mean']:.3f} ± "
              f"{stats['total_time']['stdev']:.3f}s "
              f"(range: {stats['total_time']['min']:.3f} - {stats['total_time']['max']:.3f}s)")
        print(f"    Load Time: {stats['load_time']['mean']:.3f}s")
        print(f"    Analysis Time: {stats['analysis_time']['mean']:.3f}s")
        
        # Performance assessment
        mean_time = stats['total_time']['mean']
        atoms = info['num_atoms']
        
        if atoms < 500:
            category = "small"
        elif atoms < 2000:
            category = "medium"
        else:
            category = "large"
        
        threshold = PERFORMANCE_THRESHOLDS[category]['max_time']
        
        if mean_time < threshold:
            print(f"    Performance: PASS (< {threshold}s for {category} structure)")
        else:
            print(f"    Performance: SLOW (> {threshold}s for {category} structure)")
    
    def run_all_benchmarks(self, num_runs: int = 3,
                           test_cases: Optional[List[str]] = None) -> List[Dict]:
        """Run all performance benchmarks
        
        Args:
            num_runs: Number of runs for each benchmark
            test_cases: Optional list of specific test case names to run.
                       If None, runs all test cases.
        """
        print("\n" + "="*60)
        print("ALL-ATOM MODULE PERFORMANCE BENCHMARKS")
        print("="*60)
        
        results = []
        
        # Filter test cases if specified
        cases_to_run = TEST_CASES
        if test_cases is not None:
            cases_to_run = {k: v for k, v in TEST_CASES.items() if k in test_cases}
            if not cases_to_run:
                print(f"\nWARNING: No matching test cases found for: {test_cases}")
                return []
        
        for test_name, test_info in cases_to_run.items():
            pdb_file = get_pdb_path(test_info["file"])
            if not pdb_file.exists():
                print(f"\nWARNING: Test file not found: {pdb_file}")
                continue
            
            benchmark = self.run_single_benchmark(test_name, pdb_file, num_runs)
            self.print_benchmark_summary(benchmark)
            results.append(benchmark)
        
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARKS COMPLETE")
        print(f"Results saved to: {self.results_dir}")
        print("="*60)
        
        return results


# =============================================================================
# Trajectory Performance Testing
# =============================================================================

class TrajectoryPerformanceTester:
    """Trajectory-based performance benchmark for GPCR_MD test case."""

    def __init__(self, results_dir: Path = RESULTS_DIR, backend: str = 'numpy'):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend

    def _create_backend(self):
        """Create compute backend instance based on backend name."""
        if self.backend == 'cupy':
            from fplip.all_atom_cuda.cupy_backend import CuPyBackend
            return CuPyBackend()
        elif self.backend == 'torch':
            from fplip.all_atom_cuda.torch_backend import TorchBackend
            return TorchBackend()
        else:
            from fplip.all_atom_cuda.numpy_backend import NumPyBackend
            return NumPyBackend()

    def run_benchmark(self, test_case: str, test_info: Dict,
                      n_frame: int = 10) -> Dict:
        """Run trajectory iteration benchmark.

        Args:
            test_case: Name of the test case
            test_info: Test case configuration dict
            n_frame: Number of frames to iterate (will cycle 0~10 if > 10)
        """
        from lazydock.gmx.mda.convert import PDBConverter
        from lazydock.gmx.mda.utils import filter_atoms_by_chains

        from fplip.all_atom_cuda.trajectory_analyzer import TrajectoryAnalyzer

        tpr = str(CUSTOM_TEST_DATA_DIR / test_info["tpr"])
        xtc = str(CUSTOM_TEST_DATA_DIR / test_info["xtc"])
        gro = str(CUSTOM_TEST_DATA_DIR / test_info["gro"])

        print(f"\nBenchmarking {test_case} ({n_frame} frames, backend: {self.backend})...")

        analyzer = TrajectoryAnalyzer(tpr, xtc, gro, tolerance=1e-4)
        analyzer.load_universe()
        analyzer.u.trajectory[0]
        converter = PDBConverter(filter_atoms_by_chains(analyzer.u.atoms, ['A', 'B', 'CL']))
        pdb_str = converter.fast_convert()
        analyzer.load_molecule(pdb_str, as_string=True)
        analyzer.align_with_mda(frame=0)
        analyzer.load_waters('SOL')
        backend_instance = self._create_backend()
        analyzer.setup_detector(backend=backend_instance)
        analyzer.precompute_detector_once()

        n_frames_total = len(analyzer.u.trajectory)
        actual_n_frame = min(n_frame, n_frames_total)

        frame_times = []
        detect_times = []

        for frame_idx in tqdm(range(actual_n_frame), desc=self.backend):
            frame_start = time.perf_counter()
            analyzer.update_frame(frame_idx)
            frame_elapsed = time.perf_counter() - frame_start

            detect_start = time.perf_counter()
            interactions = analyzer.detect_all()
            detect_elapsed = time.perf_counter() - detect_start

            frame_times.append(frame_elapsed)
            detect_times.append(detect_elapsed)

        total_interactions = sum(len(v) for v in interactions.values())

        benchmark = {
            "test_case": test_case,
            "backend": self.backend,
            "n_frame": actual_n_frame,
            "n_frames_total": n_frames_total,
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "coordinate_update": {
                    "total_ms": sum(frame_times) * 1000,
                    "mean_ms": statistics.mean(frame_times) * 1000,
                    "stdev_ms": statistics.stdev(frame_times) * 1000 if len(frame_times) > 1 else 0,
                },
                "detect_all": {
                    "total_ms": sum(detect_times) * 1000,
                    "mean_ms": statistics.mean(detect_times) * 1000,
                    "stdev_ms": statistics.stdev(detect_times) * 1000 if len(detect_times) > 1 else 0,
                },
                "total_per_frame": {
                    "mean_ms": (statistics.mean(frame_times) + statistics.mean(detect_times)) * 1000,
                }
            },
            "structure_info": {
                "num_atoms": len(analyzer.mol.atom_container),
                "num_residues": len(analyzer.mol.residues),
                "total_interactions": total_interactions
            }
        }

        # Save benchmark results
        result_file = self.results_dir / f"{test_case}_performance.json"
        with open(result_file, 'w') as f:
            json.dump(benchmark, f, indent=2)

        self._print_summary(benchmark)
        return benchmark

    def _print_summary(self, benchmark: Dict):
        stats = benchmark["statistics"]
        info = benchmark["structure_info"]

        print(f"\n  Summary for {benchmark['test_case']} (backend: {benchmark['backend']}):")
        print(f"    Structure: {info['num_atoms']} atoms, {info['num_residues']} residues")
        print(f"    Frames: {benchmark['n_frame']} (total available: {benchmark['n_frames_total']})")
        print(f"    Interactions (final frame): {info['total_interactions']}")
        print(f"    Coordinate update:")
        print(f"      Total: {stats['coordinate_update']['total_ms']:.2f} ms")
        print(f"      Average per frame: {stats['coordinate_update']['mean_ms']:.2f} ms")
        print(f"    detect_all:")
        print(f"      Total: {stats['detect_all']['total_ms']:.2f} ms")
        print(f"      Average per frame: {stats['detect_all']['mean_ms']:.2f} ms")
        print(f"    Total per frame (update + detect):")
        print(f"      Average: {stats['total_per_frame']['mean_ms']:.2f} ms")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='All-Atom Module Consistency and Performance Testing Framework'
    )
    parser.add_argument(
        '--test-type',
        choices=['consistency', 'performance', 'all'],
        default='all',
        help='Type of tests to run (default: all)'
    )
    parser.add_argument(
        '--generate-baselines',
        action='store_true',
        help='Generate new baselines for consistency tests'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=3,
        help='Number of runs for performance benchmarks (default: 3)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--test-cases',
        nargs='+',
        choices=list(TEST_CASES.keys()),
        default=None,
        help='Specific test case(s) to run. If not specified, runs all test cases.'
    )
    parser.add_argument(
        '--list-cases',
        action='store_true',
        help='List all available test cases and exit'
    )
    parser.add_argument(
        '--n-frame',
        type=int,
        default=10,
        help='Number of frames for trajectory benchmark (default: 10, max: 10, will cycle 0~10 if > 10)'
    )
    parser.add_argument(
        '--backend',
        choices=['numpy', 'cupy', 'torch'],
        default='numpy',
        help='Compute backend to use (default: numpy). '
             'numpy: CudaInteractionDetector with NumPyBackend (CPU), '
             'cupy: CudaInteractionDetector with CuPyBackend (GPU), '
             'torch: CudaInteractionDetector with TorchBackend (GPU), '
    )
    
    args = parser.parse_args()
    
    # List test cases if requested
    if args.list_cases:
        print("\n" + "="*60)
        print("AVAILABLE TEST CASES")
        print("="*60)
        for name, info in TEST_CASES.items():
            print(f"\n{name}:")
            if info.get("type") == "trajectory":
                print(f"  Type: trajectory")
                print(f"  TPR: {info['tpr']}")
                print(f"  XTC: {info['xtc']}")
                print(f"  GRO: {info['gro']}")
            else:
                print(f"  File: {info['file']}")
            print(f"  Description: {info['description']}")
            print(f"  Categories: {', '.join(info.get('categories', []))}")
            print(f"  Expected Interactions: {info['expected_interactions']}")
        print("\n" + "="*60)
        return 0
    
    print("\n" + "="*60)
    print("ALL-ATOM MODULE TESTING FRAMEWORK")
    print("="*60)
    print(f"PLIP Test Data: {PLIP_TEST_DATA_DIR}")
    print(f"Custom Test Data: {CUSTOM_TEST_DATA_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Baseline Directory: {BASELINE_DIR}")
    print(f"Backend: {args.backend}")
    
    # Show which test cases will be run
    if args.test_cases:
        print(f"\nSelected Test Cases: {', '.join(args.test_cases)}")
    else:
        print(f"\nRunning All Test Cases: {', '.join(TEST_CASES.keys())}")

    # Separate trajectory test cases from regular ones
    regular_test_cases = None
    trajectory_test_cases = None
    if args.test_cases:
        regular_test_cases = [c for c in args.test_cases if TEST_CASES[c].get("type") != "trajectory"]
        trajectory_test_cases = [c for c in args.test_cases if TEST_CASES[c].get("type") == "trajectory"]
    else:
        regular_test_cases = [c for c in TEST_CASES if TEST_CASES[c].get("type") != "trajectory"]
        trajectory_test_cases = [c for c in TEST_CASES if TEST_CASES[c].get("type") == "trajectory"]

    success = True

    # Run consistency tests (skip trajectory test cases)
    if args.test_type in ['consistency', 'all'] and regular_test_cases:
        consistency_tester = ConsistencyTester(backend=args.backend)
        consistency_passed = consistency_tester.run_all_tests(
            args.generate_baselines,
            test_cases=regular_test_cases
        )
        success = success and consistency_passed

    # Run performance tests (skip trajectory test cases)
    if args.test_type in ['performance', 'all'] and regular_test_cases:
        performance_tester = PerformanceTester(backend=args.backend)
        performance_tester.run_all_benchmarks(
            args.num_runs,
            test_cases=regular_test_cases
        )

    # Run trajectory benchmarks
    if args.test_type in ['performance', 'all'] and trajectory_test_cases:
        print("\n" + "="*60)
        print("TRAJECTORY PERFORMANCE BENCHMARKS")
        print("="*60)
        trajectory_tester = TrajectoryPerformanceTester(backend=args.backend)
        for tc_name in trajectory_test_cases:
            tc_info = TEST_CASES[tc_name]
            trajectory_tester.run_benchmark(tc_name, tc_info, n_frame=args.n_frame)
        print("\n" + "="*60)
        print("TRAJECTORY BENCHMARKS COMPLETE")
        print(f"Results saved to: {RESULTS_DIR}")
        print("="*60)
    
    print("\n" + "="*60)
    if success:
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
