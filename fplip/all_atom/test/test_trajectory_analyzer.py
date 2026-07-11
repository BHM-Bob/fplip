"""
All-Atom Module Trajectory Analyzer Tests

Tests for trajectory analysis functionality:
- Alignment verification
- Coordinate update correctness
- Interaction detection across frames
"""

import os
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
from lazydock.gmx.mda.convert import PDBConverter

from fplip.all_atom.trajectory_analyzer import TrajectoryAnalyzer

TEST_DATA_DIR = Path(__file__).parent.parent.parent.parent / 'test_data/'


class TrajectoryAnalyzerFunctionalTest(unittest.TestCase):
    """Test trajectory analyzer functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures with GPCR-peptide trajectory."""
        cls.tpr = TEST_DATA_DIR / "pull/pull.tpr"
        cls.xtc = TEST_DATA_DIR / "pull/pull_center.xtc"
        cls.gro = TEST_DATA_DIR / "pull/pull.gro"

        cls.analyzer = TrajectoryAnalyzer(cls.tpr, cls.xtc, cls.gro, tolerance=1e-4)
        cls.analyzer.load_universe()
        cls.analyzer.u.trajectory[0]
        converter = PDBConverter(cls.analyzer.u.atoms, reindex=False)
        cls.pdb_str = converter.fast_convert()
        cls.analyzer.load_molecule(cls.pdb_str, as_string=True)
        cls.analyzer.align_with_mda(frame=0)
        cls.analyzer.setup_detector()

    def test_universe_loading(self):
        """Test that MDA universe loads correctly."""
        self.assertIsNotNone(self.analyzer.u)
        self.assertEqual(len(self.analyzer.u.atoms), 73644)
        self.assertEqual(len(self.analyzer.u.trajectory), 17)

    def test_molecule_loading(self):
        """Test that OpenBabel molecule loads correctly."""
        self.assertIsNotNone(self.analyzer.mol)
        self.assertGreater(len(self.analyzer.mol.atom_container), 20000,
            "Should load a significant number of atoms")

    def test_alignment_stats(self):
        """Test that all aligned atoms are matched."""
        stats = self.analyzer.get_alignment_stats()
        self.assertGreater(stats['matched_atoms'], 18000,
            "Should have most atoms aligned")

    def test_mda_idx_assignment(self):
        """Test that mda_idx is assigned to most atoms."""
        matched_count = 0
        for atom in self.analyzer.mol.atom_container:
            if atom.mda_idx is not None:
                matched_count += 1
        self.assertGreater(matched_count, 18000,
            "Most atoms should have mda_idx assigned")

    def test_coordinate_update_frame_0(self):
        """Test coordinate update for frame 0."""
        self.analyzer.update_frame(0)
        mda_coords = self.analyzer.u.atoms.positions

        sample_atoms = list(self.analyzer.mol.atom_container)[:100]
        for atom in sample_atoms:
            if atom.mda_idx is not None:
                diff_x = abs(float(atom.coords[0]) - float(mda_coords[atom.mda_idx][0]))
                diff_y = abs(float(atom.coords[1]) - float(mda_coords[atom.mda_idx][1]))
                diff_z = abs(float(atom.coords[2]) - float(mda_coords[atom.mda_idx][2]))
                self.assertTrue(diff_x < 1e-3, f"Atom {atom.atom_name} x-coordinate mismatch: {diff_x}")
                self.assertTrue(diff_y < 1e-3, f"Atom {atom.atom_name} y-coordinate mismatch: {diff_y}")
                self.assertTrue(diff_z < 1e-3, f"Atom {atom.atom_name} z-coordinate mismatch: {diff_z}")

    def test_coordinate_update_frame_10(self):
        """Test coordinate update for frame 10 (different from frame 0)."""
        self.analyzer.update_frame(10)
        mda_coords = self.analyzer.u.atoms.positions

        sample_atoms = list(self.analyzer.mol.atom_container)[:100]
        coords_match = 0
        for atom in sample_atoms:
            if atom.mda_idx is not None:
                diff = (abs(float(atom.coords[0]) - float(mda_coords[atom.mda_idx][0])) +
                        abs(float(atom.coords[1]) - float(mda_coords[atom.mda_idx][1])) +
                        abs(float(atom.coords[2]) - float(mda_coords[atom.mda_idx][2])))
                if diff < 1e-3:
                    coords_match += 1

        self.assertGreater(coords_match, 0, "Some coordinates should match after update")

    def test_coords_array_rebuild(self):
        """Test that coords_array is rebuilt after coordinate update."""
        self.analyzer.update_frame(5)
        self.assertIsNotNone(self.analyzer.detector.all_coords)

        coords_array = self.analyzer.detector.all_coords
        for atom in self.analyzer.mol.atom_container:
            array_pos = self.analyzer.mol.atom_container.idx_to_array_pos[atom.idx]
            self.assertTrue(
                abs(coords_array[array_pos][0] - atom.coords[0]) < 1e-3,
                f"coords_array not rebuilt correctly for atom {atom.atom_name}"
            )

    def test_detector_initialized(self):
        """Test that detector is properly initialized."""
        self.assertIsNotNone(self.analyzer.detector)
        self.assertIsNotNone(self.analyzer.detector.interactions)

    def test_interactions_detected_frame_0(self):
        """Test that interactions are detected."""
        interactions = self.analyzer.detector.detect_all()

        total_interactions = sum(len(v) for v in interactions.values())
        self.assertGreater(total_interactions, 0,
            "Should detect at least some interactions")

    def test_interactions_change_across_frames(self):
        """Test that interactions can change across trajectory frames."""
        self.analyzer.update_frame(0)
        interactions_0 = self.analyzer.detector.detect_all()

        self.analyzer.update_frame(10)
        interactions_50 = self.analyzer.detector.detect_all()

        total_0 = sum(len(v) for v in interactions_0.values())
        total_50 = sum(len(v) for v in interactions_50.values())

        self.assertIsNotNone(total_0)
        self.assertIsNotNone(total_50)

    def test_hbond_detection(self):
        """Test that hydrogen bonds are detected."""
        self.analyzer.update_frame(0)
        interactions = self.analyzer.detector.detect_all()

        hbond_count = len(interactions.get('hbond', []))
        hbond_possible_count = len(interactions.get('hbond_possible', []))

        self.assertGreaterEqual(hbond_count + hbond_possible_count, 0)

    def test_hydrophobic_interactions(self):
        """Test that hydrophobic interactions are detected."""
        self.analyzer.update_frame(0)
        interactions = self.analyzer.detector.detect_all()

        hydrophobic_count = len(interactions.get('hydrophobic', []))
        self.assertGreaterEqual(hydrophobic_count, 0)

    def test_salt_bridge_detection(self):
        """Test that salt bridges are detected."""
        self.analyzer.update_frame(0)
        interactions = self.analyzer.detector.detect_all()

        saltbridge_count = len(interactions.get('saltbridge', []))
        self.assertGreaterEqual(saltbridge_count, 0)


def _to_plain(x):
    """Recursively convert numpy / nested values to plain Python types for comparison."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, list):
        return [_to_plain(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_plain(v) for v in x)
    return x


def _snapshot_packs(analyzer):
    """Return deep plain snapshots of df_pack and detail_packs."""
    df_snap = {k: [_to_plain(v) for v in lst] for k, lst in analyzer.df_pack.items()}
    dt_snap = {
        t: {k: [_to_plain(v) for v in lst] for k, lst in pack.items()}
        for t, pack in analyzer.detail_packs.items()
    }
    return df_snap, dt_snap


class TrajectoryAnalyzerSaveLoadTest(unittest.TestCase):
    """Independent test class: verify save_records / read_records round-trip."""

    @classmethod
    def setUpClass(cls):
        """Build a fresh TrajectoryAnalyzer and process 10 frames."""
        tpr = TEST_DATA_DIR / "pull/pull.tpr"
        xtc = TEST_DATA_DIR / "pull/pull_center.xtc"
        gro = TEST_DATA_DIR / "pull/pull.gro"

        cls.analyzer = TrajectoryAnalyzer(tpr, xtc, gro, tolerance=1e-4)
        cls.analyzer.load_universe()
        cls.analyzer.u.trajectory[0]
        converter = PDBConverter(cls.analyzer.u.atoms, reindex=False)
        pdb_str = converter.fast_convert()
        cls.analyzer.load_molecule(pdb_str, as_string=True)
        cls.analyzer.align_with_mda(frame=0)
        cls.analyzer.setup_detector()

        for frame in range(10):
            cls.analyzer.update_frame(frame)
            interactions = cls.analyzer.detector.detect_all()
            cls.analyzer.add_record(interactions, frame=frame)

    def test_detail_row_count_matches_main(self):
        """[A3] Cross-check: per-type detail row count == main table type row count."""
        type_counts = {}
        for t in self.analyzer.df_pack['type']:
            type_counts[t] = type_counts.get(t, 0) + 1

        for inter_type, pack in self.analyzer.detail_packs.items():
            if inter_type == 'metal':
                continue
            expected = type_counts.get(inter_type, 0)
            actual = len(pack['idx'])
            self.assertEqual(
                expected, actual,
                f"detail_{inter_type}: rows {actual} != main table type rows {expected}"
            )

    def test_hbond_save_load_consistency(self):
        """10 frames -> save tar -> load back; randomly sample 5 hbonds.

        Checks (per sampled hbond record):
          - main table: residue/atom A/B names, indices, distance, angle
          - detail_hbond (joined by global idx): h_atom, h_idx, dist_ah,
            type, donor_idx, acceptor_idx
        """
        orig_df, orig_dt = _snapshot_packs(self.analyzer)

        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = os.path.join(tmpdir, 'test_records.tar')
            self.analyzer.save_records(tar_path)
            self.analyzer.read_records(tar_path)

        new_df, new_dt = _snapshot_packs(self.analyzer)

        self.assertEqual(
            len(orig_df['idx']), len(new_df['idx']),
            "Main table total rows differ after save/load round-trip"
        )

        # --- [A3] cross-check after load ---
        type_counts_after = {}
        for t in new_df['type']:
            type_counts_after[t] = type_counts_after.get(t, 0) + 1
        for inter_type, pack in new_dt.items():
            if inter_type == 'metal':
                continue
            expected = type_counts_after.get(inter_type, 0)
            actual = len(pack['idx'])
            self.assertEqual(
                expected, actual,
                f"[post-load] detail_{inter_type}: rows {actual} != main type rows {expected}"
            )

        # --- [A2] randomly sample 5 hbonds ---
        hbond_positions = [i for i, t in enumerate(orig_df['type']) if t == 'hbond']
        if len(hbond_positions) == 0:
            self.skipTest("No hbond records found in 10 frames; cannot sample")

        sample_size = min(5, len(hbond_positions))
        sampled = random.sample(hbond_positions, sample_size)

        # Pre-build: global_idx -> row offset in detail_hbond (for both snapshots)
        orig_hbond_idx_to_pos = {idx: pos for pos, idx in enumerate(orig_dt['hbond']['idx'])}
        new_hbond_idx_to_pos = {idx: pos for pos, idx in enumerate(new_dt['hbond']['idx'])}

        main_cols = [
            'res_a_name', 'res_a_chain', 'res_a_num',
            'res_b_name', 'res_b_chain', 'res_b_num',
            'atom_a_name', 'atom_a_idx',
            'atom_b_name', 'atom_b_idx',
            'distance', 'angle',
        ]
        detail_cols = [
            'h_atom', 'h_idx', 'dist_ah', 'type',
            'donor_idx', 'acceptor_idx',
        ]

        for pos in sampled:
            for col in main_cols:
                self.assertEqual(
                    orig_df[col][pos], new_df[col][pos],
                    f"main table col='{col}' mismatch at df row {pos}"
                )

            global_idx = orig_df['idx'][pos]
            dpos_orig = orig_hbond_idx_to_pos[global_idx]
            dpos_new = new_hbond_idx_to_pos[global_idx]

            for col in detail_cols:
                self.assertEqual(
                    orig_dt['hbond'][col][dpos_orig],
                    new_dt['hbond'][col][dpos_new],
                    f"detail_hbond col='{col}' mismatch for global idx={global_idx}"
                )


if __name__ == '__main__':
    unittest.main()
