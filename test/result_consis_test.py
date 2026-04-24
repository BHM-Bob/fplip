"""
PLIP Consistency and Performance Testing Framework

This module provides comprehensive testing for PLIP analysis results:
1. Consistency Testing: Verify results match expected baseline
2. Performance Testing: Measure execution time and identify bottlenecks

Usage:
    python -m test.result_consis_test
    python -m test.result_consis_test --test-type consistency
    python -m test.result_consis_test --test-type performance
    python -m test.result_consis_test --test-type all --verbose
"""

import argparse
import hashlib
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

from plip.basic import config, logger
from plip.exchange.report import StructureReport
from plip.structure.preparation import PDBComplex, create_folder_if_not_exists

logger = logger.get_logger()


# =============================================================================
# Test Configuration
# =============================================================================

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
RESULTS_DIR = TEST_DATA_DIR / "test_results"
BASELINE_DIR = TEST_DATA_DIR / "baselines"

# Test cases with expected characteristics
TEST_CASES = {
    "GPCR_pep": {
        "file": "GPCR_pep.pdb",
        "description": "GPCR peptide complex - medium size",
        "expected_ligands": 5,
        "expected_interactions": {
            "hydrogen_bonds": ">=3",
            "hydrophobic": ">=2",
            "salt_bridges": ">=1"
        }
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
    """Summary of interactions for a single ligand"""
    ligand_id: str
    ligand_type: str
    num_hydrogen_bonds: int
    num_hydrophobic: int
    num_salt_bridges: int
    num_pi_stacking: int
    num_pi_cation: int
    num_halogen_bonds: int
    num_water_bridges: int
    num_metal_complexes: int
    interacting_chains: List[str]
    interacting_residues: int


@dataclass
class AnalysisResult:
    """Complete analysis result for a PDB structure"""
    pdb_id: str
    timestamp: str
    plip_version: str
    num_ligands: int
    ligands: List[InteractionSummary]
    execution_time: float
    memory_usage: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single run"""
    pdb_id: str
    total_time: float
    load_time: float
    analysis_time: float
    report_time: float
    num_atoms: int
    num_ligands: int
    num_interactions: int


# =============================================================================
# Core Analysis Function
# =============================================================================

def process_pdb_detailed(pdbfile: str, outpath: str, as_string: bool = False, 
                         outputprefix: str = 'report') -> Tuple[AnalysisResult, PerformanceMetrics]:
    """
    Process a PDB file with detailed timing and result collection.
    
    Returns:
        Tuple of (AnalysisResult, PerformanceMetrics)
    """
    start_time = time.perf_counter()
    load_start = start_time
    
    if not as_string:
        pdb_file_name = pdbfile.split('/')[-1]
        logger.info(f'Starting analysis of {pdb_file_name}')
    else:
        logger.info('Starting analysis from STDIN')
    
    # Initialize PDBComplex
    mol = PDBComplex()
    mol.output_path = outpath
    
    # Load PDB
    mol.load_pdb(pdbfile, as_string=as_string)
    load_end = time.perf_counter()
    load_time = load_end - load_start
    
    # Count atoms
    num_atoms = len(mol.atoms)
    
    # Analyze ligands
    analysis_start = time.perf_counter()
    ligand_summaries = []
    total_interactions = 0
    
    for ligand in mol.ligands:
        mol.characterize_complex(ligand)
    
    analysis_end = time.perf_counter()
    analysis_time = analysis_end - analysis_start
    
    # Extract results from interaction_sets
    for site_id, pli_obj in mol.interaction_sets.items():
        summary = InteractionSummary(
            ligand_id=site_id,
            ligand_type=getattr(pli_obj.ligand, 'type', 'UNKNOWN'),
            num_hydrogen_bonds=len(pli_obj.hbonds_ldon) + len(pli_obj.hbonds_pdon),
            num_hydrophobic=len(pli_obj.hydrophobic_contacts),
            num_salt_bridges=len(pli_obj.saltbridge_lneg) + len(pli_obj.saltbridge_pneg),
            num_pi_stacking=len(pli_obj.pistacking),
            num_pi_cation=len(pli_obj.pication_laro) + len(pli_obj.pication_paro),
            num_halogen_bonds=len(pli_obj.halogen_bonds),
            num_water_bridges=len(pli_obj.water_bridges),
            num_metal_complexes=len(pli_obj.metal_complexes),
            interacting_chains=pli_obj.interacting_chains,
            interacting_residues=len(pli_obj.interacting_res)
        )
        ligand_summaries.append(summary)
        total_interactions += (
            summary.num_hydrogen_bonds + summary.num_hydrophobic +
            summary.num_salt_bridges + summary.num_pi_stacking +
            summary.num_pi_cation + summary.num_halogen_bonds +
            summary.num_water_bridges + summary.num_metal_complexes
        )
    
    # Generate report
    report_start = time.perf_counter()
    create_folder_if_not_exists(outpath)
    streport = StructureReport(mol, outputprefix=outputprefix)
    streport.write_txt(as_string=config.STDOUT)
    report_end = time.perf_counter()
    report_time = report_end - report_start
    
    total_time = time.perf_counter() - start_time
    
    # Create result objects
    from plip.basic.config import __version__
    analysis_result = AnalysisResult(
        pdb_id=Path(pdbfile).stem,
        timestamp=datetime.now().isoformat(),
        plip_version=__version__,
        num_ligands=len(mol.ligands),
        ligands=ligand_summaries,
        execution_time=total_time
    )
    
    performance_metrics = PerformanceMetrics(
        pdb_id=Path(pdbfile).stem,
        total_time=total_time,
        load_time=load_time,
        analysis_time=analysis_time,
        report_time=report_time,
        num_atoms=num_atoms,
        num_ligands=len(mol.ligands),
        num_interactions=total_interactions
    )
    
    return analysis_result, performance_metrics


# =============================================================================
# Consistency Testing
# =============================================================================

class ConsistencyTester:
    """Test consistency of PLIP results against baselines"""
    
    def __init__(self, baseline_dir: Path = BASELINE_DIR):
        self.baseline_dir = baseline_dir
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def generate_baseline(self, test_case: str, pdb_path: Path) -> Dict:
        """Generate a new baseline for a test case"""
        print(f"Generating baseline for {test_case}...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result, metrics = process_pdb_detailed(
                str(pdb_path), 
                tmpdir, 
                outputprefix='baseline'
            )
        
        # Convert to serializable format
        baseline = {
            "test_case": test_case,
            "pdb_file": str(pdb_path),
            "created_at": result.timestamp,
            "plip_version": result.plip_version,
            "num_ligands": result.num_ligands,
            "ligands": [asdict(lig) for lig in result.ligands],
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
            "ligand_count_match": current.num_ligands == baseline["num_ligands"],
            "interaction_counts": {},
            "missing_ligands": [],
            "new_ligands": [],
            "details": []
        }
        
        # Check ligand count
        if not differences["ligand_count_match"]:
            differences["match"] = False
            differences["details"].append(
                f"Ligand count mismatch: current={current.num_ligands}, "
                f"baseline={baseline['num_ligands']}"
            )
        
        # Compare each ligand
        baseline_ligands = {lig["ligand_id"]: lig for lig in baseline["ligands"]}
        current_ligands = {lig.ligand_id: asdict(lig) for lig in current.ligands}
        
        # Check for missing ligands
        for lig_id in baseline_ligands:
            if lig_id not in current_ligands:
                differences["match"] = False
                differences["missing_ligands"].append(lig_id)
                differences["details"].append(f"Missing ligand: {lig_id}")
        
        # Check for new ligands
        for lig_id in current_ligands:
            if lig_id not in baseline_ligands:
                differences["new_ligands"].append(lig_id)
                differences["details"].append(f"New ligand: {lig_id}")
        
        # Compare interaction counts for matching ligands
        for lig_id in set(baseline_ligands.keys()) & set(current_ligands.keys()):
            bl = baseline_ligands[lig_id]
            cl = current_ligands[lig_id]
            
            interaction_types = [
                "num_hydrogen_bonds", "num_hydrophobic", "num_salt_bridges",
                "num_pi_stacking", "num_pi_cation", "num_halogen_bonds",
                "num_water_bridges", "num_metal_complexes"
            ]
            
            for itype in interaction_types:
                b_count = bl.get(itype, 0)
                c_count = cl.get(itype, 0)
                if b_count != c_count:
                    differences["match"] = False
                    differences["details"].append(
                        f"{lig_id}.{itype}: baseline={b_count}, current={c_count}"
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
        with tempfile.TemporaryDirectory() as tmpdir:
            current_result, metrics = process_pdb_detailed(
                str(pdb_path),
                tmpdir,
                outputprefix='test'
            )
        
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
    
    def run_all_tests(self, generate_baselines: bool = False) -> bool:
        """Run all consistency tests"""
        print("\n" + "="*60)
        print("CONSISTENCY TESTS")
        print("="*60)
        
        all_passed = True
        
        for test_name, test_info in TEST_CASES.items():
            pdb_file = TEST_DATA_DIR / test_info["file"]
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
    """Test performance of PLIP analysis"""
    
    def __init__(self, results_dir: Path = RESULTS_DIR):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
    
    def run_single_benchmark(self, test_case: str, pdb_path: Path, 
                            num_runs: int = 3) -> Dict:
        """Run performance benchmark multiple times"""
        print(f"\nBenchmarking {test_case} ({num_runs} runs)...")
        
        metrics_list = []
        
        for i in range(num_runs):
            print(f"  Run {i+1}/{num_runs}...", end=' ')
            with tempfile.TemporaryDirectory() as tmpdir:
                result, metrics = process_pdb_detailed(
                    str(pdb_path),
                    tmpdir,
                    outputprefix=f'perf_{i}'
                )
            metrics_list.append(metrics)
            print(f"{metrics.total_time:.2f}s")
        
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
                "num_ligands": metrics_list[0].num_ligands,
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
        print(f"    Structure: {info['num_atoms']} atoms, {info['num_ligands']} ligands")
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
    
    def run_all_benchmarks(self, num_runs: int = 3) -> List[Dict]:
        """Run all performance benchmarks"""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARKS")
        print("="*60)
        
        results = []
        
        for test_name, test_info in TEST_CASES.items():
            pdb_file = TEST_DATA_DIR / test_info["file"]
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
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PLIP Consistency and Performance Testing Framework'
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
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(10)  # DEBUG
    
    print("\n" + "="*60)
    print("PLIP TESTING FRAMEWORK")
    print("="*60)
    print(f"Test Data Directory: {TEST_DATA_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Baseline Directory: {BASELINE_DIR}")
    
    success = True
    
    # Run consistency tests
    if args.test_type in ['consistency', 'all']:
        consistency_tester = ConsistencyTester()
        consistency_passed = consistency_tester.run_all_tests(args.generate_baselines)
        success = success and consistency_passed
    
    # Run performance tests
    if args.test_type in ['performance', 'all']:
        performance_tester = PerformanceTester()
        performance_tester.run_all_benchmarks(args.num_runs)
    
    print("\n" + "="*60)
    if success:
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
