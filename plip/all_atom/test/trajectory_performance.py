"""
All-Atom Module Trajectory Analyzer Performance Tests

Performance benchmarks for trajectory analysis:
- Alignment performance
- Coordinate update performance
- Frame iteration performance
"""

import time
import unittest

import numpy as np
from lazydock.gmx.mda.convert import PDBConverter
from tqdm import tqdm

from plip.all_atom.trajectory_analyzer import TrajectoryAnalyzer


class TrajectoryAnalyzerPerformanceTest(unittest.TestCase):
    """Performance test for trajectory analyzer."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures with GPCR-peptide trajectory."""
        cls.tpr = "/home/pcmd36/Desktop/BHM/My_Progs/fplip/test_data/pull/pull.tpr"
        cls.xtc = "/home/pcmd36/Desktop/BHM/My_Progs/fplip/test_data/pull/pull_center.xtc"
        cls.gro = "/home/pcmd36/Desktop/BHM/My_Progs/fplip/test_data/pull/pull.gro"

        cls.analyzer = TrajectoryAnalyzer(cls.tpr, cls.xtc, cls.gro, tolerance=1e-4)

    def test_universe_loading_performance(self):
        """Benchmark MDA universe loading."""
        times = []
        for _ in range(3):
            analyzer = TrajectoryAnalyzer(self.tpr, self.xtc, self.gro, tolerance=1e-4)
            start = time.perf_counter()
            analyzer.load_universe()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        print(f"\n[PERF] Universe loading:")
        print(f"  Average: {np.mean(times)*1000:.2f} ms")
        print(f"  Min: {np.min(times)*1000:.2f} ms")
        print(f"  Max: {np.max(times)*1000:.2f} ms")

    def test_pdb_conversion_performance(self):
        """Benchmark PDB conversion from MDA to OpenBabel format."""
        self.analyzer.load_universe()
        self.analyzer.u.trajectory[0]

        times = []
        for _ in range(3):
            start = time.perf_counter()
            converter = PDBConverter(self.analyzer.u.atoms, reindex=False)
            pdb_str = converter.fast_convert()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        print(f"\n[PERF] PDB conversion (frame 0):")
        print(f"  Average: {np.mean(times)*1000:.2f} ms")
        print(f"  Min: {np.min(times)*1000:.2f} ms")
        print(f"  Max: {np.max(times)*1000:.2f} ms")
        print(f"  PDB string length: {len(pdb_str)} chars")

    def test_molecule_loading_performance(self):
        """Benchmark OpenBabel molecule loading."""
        self.analyzer.load_universe()
        self.analyzer.u.trajectory[0]
        converter = PDBConverter(self.analyzer.u.atoms, reindex=False)
        pdb_str = converter.fast_convert()

        times = []
        for _ in range(3):
            start = time.perf_counter()
            self.analyzer.load_molecule(pdb_str, as_string=True)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        print(f"\n[PERF] Molecule loading (OpenBabel):")
        print(f"  Average: {np.mean(times)*1000:.2f} ms")
        print(f"  Min: {np.min(times)*1000:.2f} ms")
        print(f"  Max: {np.max(times)*1000:.2f} ms")
        print(f"  Atoms: {len(self.analyzer.mol.atom_container)}")

    def test_kdtree_alignment_performance(self):
        """Benchmark KD-tree based alignment."""
        self.analyzer.load_universe()
        self.analyzer.u.trajectory[0]
        converter = PDBConverter(self.analyzer.u.atoms, reindex=False)
        pdb_str = converter.fast_convert()
        self.analyzer.load_molecule(pdb_str, as_string=True)

        build_times = []
        align_times = []
        total_times = []

        for _ in range(3):
            start = time.perf_counter()
            self.analyzer.u.trajectory[0]
            mda_coords = self.analyzer.u.atoms.positions

            from scipy.spatial import cKDTree
            build_start = time.perf_counter()
            tree = cKDTree(mda_coords)
            build_elapsed = time.perf_counter() - build_start

            align_start = time.perf_counter()
            matched, unmatched = 0, 0
            for atom in self.analyzer.mol.atom_container:
                dist, idx = tree.query(atom.coords)
                if dist < 1e-4:
                    atom.mda_idx = idx
                    matched += 1
                else:
                    atom.mda_idx = None
                    unmatched += 1
            align_elapsed = time.perf_counter() - align_start
            total_elapsed = time.perf_counter() - start

            build_times.append(build_elapsed)
            align_times.append(align_elapsed)
            total_times.append(total_elapsed)

        print(f"\n[PERF] KD-tree alignment (73644 atoms):")
        print(f"  Tree building:")
        print(f"    Average: {np.mean(build_times)*1000:.2f} ms")
        print(f"    Min: {np.min(build_times)*1000:.2f} ms")
        print(f"  Atom alignment:")
        print(f"    Average: {np.mean(align_times)*1000:.2f} ms")
        print(f"    Min: {np.min(align_times)*1000:.2f} ms")
        print(f"  Total (tree + alignment):")
        print(f"    Average: {np.mean(total_times)*1000:.2f} ms")
        print(f"    Min: {np.min(total_times)*1000:.2f} ms")
        print(f"  Matched atoms: {matched}")
        print(f"  Unmatched atoms: {unmatched}")

    def test_coordinate_update_performance(self):
        """Benchmark coordinate update for a single frame."""
        self.analyzer.load_universe()
        self.analyzer.u.trajectory[0]
        converter = PDBConverter(self.analyzer.u.atoms, reindex=False)
        pdb_str = converter.fast_convert()
        self.analyzer.load_molecule(pdb_str, as_string=True)
        self.analyzer.align_with_mda(frame=0)
        self.analyzer.setup_detector()

        times = []
        for frame_idx in range(20):
            self.analyzer.u.trajectory[frame_idx]
            mda_coords = self.analyzer.u.atoms.positions

            start = time.perf_counter()
            self.analyzer.detector.update_coords(mda_coords)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        print(f"\n[PERF] Coordinate update per frame (73644 atoms):")
        print(f"  Average: {np.mean(times)*1000:.2f} ms")
        print(f"  Min: {np.min(times)*1000:.2f} ms")
        print(f"  Max: {np.max(times)*1000:.2f} ms")
        print(f"  Std: {np.std(times)*1000:.2f} ms")
        print(f"  Per atom: {np.mean(times)/73644*1000000:.3f} µs")

    def test_detect_all_performance(self):
        """Benchmark detect_all for a single frame."""
        self.analyzer.load_universe()
        self.analyzer.u.trajectory[0]
        converter = PDBConverter(self.analyzer.u.atoms, reindex=False)
        pdb_str = converter.fast_convert()
        self.analyzer.load_molecule(pdb_str, as_string=True)
        self.analyzer.align_with_mda(frame=0)
        self.analyzer.setup_detector()
        self.analyzer.update_frame(0)

        times = []
        for _ in tqdm(range(3), desc="test_detect_all_performance"):
            start = time.perf_counter()
            interactions = self.analyzer.detector.detect_all(verbose=True)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        total_interactions = sum(len(v) for v in interactions.values())

        print(f"\n[PERF] detect_all per frame (73644 atoms):")
        print(f"  Average: {np.mean(times)*1000:.2f} ms")
        print(f"  Min: {np.min(times)*1000:.2f} ms")
        print(f"  Max: {np.max(times)*1000:.2f} ms")
        print(f"  Interactions detected: {total_interactions}")

    def test_full_iteration_performance(self):
        """Benchmark full iteration over multiple frames."""
        self.analyzer.load_universe()
        self.analyzer.u.trajectory[0]
        converter = PDBConverter(self.analyzer.u.atoms, reindex=False)
        pdb_str = converter.fast_convert()
        self.analyzer.load_molecule(pdb_str, as_string=True)
        self.analyzer.align_with_mda(frame=0)
        self.analyzer.setup_detector()
        self.analyzer.precompute_detector_once()

        n_frames = 10
        frame_times = []
        detect_times = []

        for frame_idx in range(n_frames):
            frame_start = time.perf_counter()
            self.analyzer.update_frame(frame_idx)
            frame_elapsed = time.perf_counter() - frame_start

            detect_start = time.perf_counter()
            interactions = self.analyzer.detector.detect_all(verbose=True)
            detect_elapsed = time.perf_counter() - detect_start

            frame_times.append(frame_elapsed)
            detect_times.append(detect_elapsed)

        total_interactions = sum(len(v) for v in interactions.values())

        print(f"\n[PERF] Full iteration ({n_frames} frames, 73644 atoms):")
        print(f"  Coordinate update:")
        print(f"    Total: {sum(frame_times)*1000:.2f} ms")
        print(f"    Average per frame: {np.mean(frame_times)*1000:.2f} ms")
        print(f"  detect_all:")
        print(f"    Total: {sum(detect_times)*1000:.2f} ms")
        print(f"    Average per frame: {np.mean(detect_times)*1000:.2f} ms")
        print(f"  Total per frame (update + detect):")
        print(f"    Average: {(np.mean(frame_times) + np.mean(detect_times))*1000:.2f} ms")
        print(f"  Final frame interactions: {total_interactions}")


    def precompute_detector_once(self):
        """Benchmark the overhead of precompute_detector_once."""
        self.analyzer.load_universe()
        self.analyzer.u.trajectory[0]
        converter = PDBConverter(self.analyzer.u.atoms, reindex=False)
        pdb_str = converter.fast_convert()
        self.analyzer.load_molecule(pdb_str, as_string=True)
        self.analyzer.align_with_mda(frame=0)
        self.analyzer.setup_detector()

        start = time.perf_counter()
        self.analyzer.precompute_detector_once()
        elapsed = time.perf_counter() - start
        self.analyzer._detector_precomputed = False

        print(f"\n[PERF] setup_detector_once overhead:")
        print(f"  : {elapsed*1000:.2f} ms")


if __name__ == '__main__':
    unittest.main(verbosity=2)
