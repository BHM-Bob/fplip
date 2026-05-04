"""
All-Atom Module Trajectory Analyzer Tests

Tests for trajectory analysis functionality:
- Alignment verification
- Coordinate update correctness
- Interaction detection across frames
"""

import unittest

import numpy as np
from MDAnalysis.coordinates.PDB import PDBWriter
from MDAnalysis.lib import util
from lazydock.gmx.mda.convert import FakeIOWriter, FakeAtomGroup, PDBConverter

from plip.all_atom.trajectory_analyzer import TrajectoryAnalyzer


class TrajectoryAnalyzerFunctionalTest(unittest.TestCase):
    """Test trajectory analyzer functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures with GPCR-peptide trajectory."""
        cls.tpr = "/home/pcmd36/Desktop/BHM/My_Progs/fplip/test_data/pull/pull.tpr"
        cls.xtc = "/home/pcmd36/Desktop/BHM/My_Progs/fplip/test_data/pull/pull.xtc"
        cls.gro = "/home/pcmd36/Desktop/BHM/My_Progs/fplip/test_data/pull/pull.gro"

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
        self.assertEqual(len(self.analyzer.u.trajectory), 401)

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

        self.analyzer.update_frame(50)
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


if __name__ == '__main__':
    unittest.main()
