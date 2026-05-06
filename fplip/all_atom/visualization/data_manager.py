"""
Data Manager for All-Atom Visualization

Manages interaction data loading, filtering, and export functionality.
"""

import json
import pickle
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


@dataclass
class Interaction:
    """Represents a single interaction."""
    type: str
    res_a_name: str
    res_a_chain: str
    res_a_num: int
    res_b_name: str
    res_b_chain: str
    res_b_num: int
    atom_a_name: str
    atom_a_idx: int
    atom_b_name: str
    atom_b_idx: int
    distance: float
    angle: Optional[float] = None
    details: Optional[Dict[str, Any]] = None

    @property
    def res_a_key(self) -> Tuple[str, str, int]:
        """Get residue A key as tuple."""
        return (self.res_a_name, self.res_a_chain, self.res_a_num)

    @property
    def res_b_key(self) -> Tuple[str, str, int]:
        """Get residue B key as tuple."""
        return (self.res_b_name, self.res_b_chain, self.res_b_num)

    @property
    def atom_a_key(self) -> Tuple[str, str, int, str]:
        """Get atom A key as tuple."""
        return (self.res_a_name, self.res_a_chain, self.res_a_num, self.atom_a_name)

    @property
    def atom_b_key(self) -> Tuple[str, str, int, str]:
        """Get atom B key as tuple."""
        return (self.res_b_name, self.res_b_chain, self.res_b_num, self.atom_b_name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Interaction':
        """Create from dictionary."""
        return cls(**data)


class InteractionGroup:
    """Represents a group of interactions."""

    def __init__(self, name: str, interactions: List[Interaction] = None):
        self.name = name
        self.interactions = interactions or []
        self.visible = True

    def add(self, interaction: Interaction):
        """Add an interaction to the group."""
        self.interactions.append(interaction)

    def remove(self, interaction: Interaction):
        """Remove an interaction from the group."""
        self.interactions.remove(interaction)

    def toggle_visibility(self):
        """Toggle group visibility."""
        self.visible = not self.visible

    def __len__(self) -> int:
        return len(self.interactions)


class InteractionDataManager:
    """
    Manages interaction data for visualization.

    Provides functionality for:
    - Loading interaction data from JSON
    - Filtering interactions by various criteria
    - Creating and managing interaction groups
    - Exporting data to various formats (JSON, PyMOL)

    Example:
        >>> dm = InteractionDataManager('results.json')
        >>> hbonds = dm.filter(type='hbond', distance_max=3.5)
        >>> dm.create_group('Strong H-bonds', hbonds)
        >>> dm.export_to_pymol('output.pse', dm.groups['Strong H-bonds'])
    """

    def __init__(self, data_source: Union[str, Path, Dict[str, Any]]):
        """
        Initialize data manager.

        Args:
            data_source: Path to JSON file or dictionary containing interaction data
        """
        self.groups: Dict[str, InteractionGroup] = {}
        self._interactions: List[Interaction] = []
        self.metadata: Dict[str, Any] = {}
        self.residues: Set[Tuple[str, str, int]] = set()
        self.atom_coords: Dict[int, Tuple[float, float, float]] = {}  # atom_idx -> (x, y, z)

        if isinstance(data_source, (str, Path)):
            self._load_from_file(data_source)
        else:
            self._load_from_dict(data_source)

        self._build_residue_index()

    def _load_from_file(self, filepath: Union[str, Path]):
        """Load data from JSON file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        self._load_from_dict(data)

    def _load_from_dict(self, data: Dict[str, Any]):
        """Load data from dictionary."""
        self.metadata = data.get('metadata', {})

        # Load atom coordinates if available
        coords_data = data.get('atom_coords', {})
        self.atom_coords = {}
        for idx_str, coords in coords_data.items():
            try:
                idx = int(idx_str)
                if isinstance(coords, (list, tuple)) and len(coords) == 3:
                    self.atom_coords[idx] = tuple(float(c) for c in coords)
            except (ValueError, TypeError):
                continue

        # Load interactions from all types
        interactions_data = data.get('interactions', {})
        self._interactions = []

        for interaction_type, interactions_list in interactions_data.items():
            for interaction_data in interactions_list:
                # Handle both dict and namedtuple-like objects
                if hasattr(interaction_data, '_asdict'):
                    interaction_dict = interaction_data._asdict()
                elif hasattr(interaction_data, '__dict__'):
                    interaction_dict = interaction_data.__dict__
                else:
                    interaction_dict = interaction_data

                interaction_dict['type'] = interaction_type
                self._interactions.append(Interaction.from_dict(interaction_dict))

    def _build_residue_index(self):
        """Build index of all residues."""
        for interaction in self._interactions:
            self.residues.add(interaction.res_a_key)
            self.residues.add(interaction.res_b_key)

    @property
    def all_interactions(self) -> List[Interaction]:
        """Get all interactions."""
        return self._interactions.copy()

    def get_interaction_types(self) -> List[str]:
        """Get list of all interaction types."""
        types = set()
        for interaction in self._interactions:
            types.add(interaction.type)
        return sorted(list(types))

    def get_residue_list(self) -> List[Tuple[str, str, int]]:
        """Get sorted list of all residues."""
        return sorted(list(self.residues), key=lambda x: (x[1], x[2]))

    def get_chains(self) -> List[str]:
        """Get list of all chains."""
        chains = set()
        for res in self.residues:
            chains.add(res[1])
        return sorted(list(chains))

    def filter(self,
               interaction_types: Optional[List[str]] = None,
               residues: Optional[List[Tuple[str, str, int]]] = None,
               distance_min: Optional[float] = None,
               distance_max: Optional[float] = None,
               angle_min: Optional[float] = None,
               angle_max: Optional[float] = None) -> List[Interaction]:
        """
        Filter interactions by criteria.

        Args:
            interaction_types: List of interaction types to include (e.g., ['hbond', 'saltbridge'])
            residues: List of residues to include (as tuples of (name, chain, num))
            distance_min: Minimum distance threshold
            distance_max: Maximum distance threshold
            angle_min: Minimum angle threshold
            angle_max: Maximum angle threshold

        Returns:
            List of interactions matching all criteria
        """
        filtered = self._interactions.copy()

        # Filter by interaction type
        if interaction_types:
            filtered = [i for i in filtered if i.type in interaction_types]

        # Filter by residues
        if residues:
            residue_set = set(residues)
            filtered = [
                i for i in filtered
                if i.res_a_key in residue_set or i.res_b_key in residue_set
            ]

        # Filter by distance
        if distance_min is not None:
            filtered = [i for i in filtered if i.distance >= distance_min]
        if distance_max is not None:
            filtered = [i for i in filtered if i.distance <= distance_max]

        # Filter by angle
        if angle_min is not None:
            filtered = [i for i in filtered if i.angle is not None and i.angle >= angle_min]
        if angle_max is not None:
            filtered = [i for i in filtered if i.angle is not None and i.angle <= angle_max]

        return filtered

    def create_group(self, name: str, interactions: List[Interaction]) -> InteractionGroup:
        """
        Create a new interaction group.

        Args:
            name: Group name
            interactions: List of interactions to include in the group

        Returns:
            Created group
        """
        group = InteractionGroup(name, interactions)
        self.groups[name] = group
        return group

    def delete_group(self, name: str):
        """Delete a group."""
        if name in self.groups:
            del self.groups[name]

    def get_group(self, name: str) -> Optional[InteractionGroup]:
        """Get a group by name."""
        return self.groups.get(name)

    def toggle_group_visibility(self, name: str):
        """Toggle visibility of a group."""
        if name in self.groups:
            self.groups[name].toggle_visibility()

    def get_interaction_summary(self) -> Dict[str, Any]:
        """Get summary statistics of interactions."""
        summary = {
            'total_interactions': len(self._interactions),
            'by_type': defaultdict(int),
            'by_chain': defaultdict(int),
            'distance_range': {'min': float('inf'), 'max': 0},
            'angle_range': {'min': float('inf'), 'max': 0}
        }

        for interaction in self._interactions:
            # Count by type
            summary['by_type'][interaction.type] += 1

            # Count by chain
            summary['by_chain'][interaction.res_a_chain] += 1
            summary['by_chain'][interaction.res_b_chain] += 1

            # Distance range
            summary['distance_range']['min'] = min(summary['distance_range']['min'], interaction.distance)
            summary['distance_range']['max'] = max(summary['distance_range']['max'], interaction.distance)

            # Angle range
            if interaction.angle is not None:
                summary['angle_range']['min'] = min(summary['angle_range']['min'], interaction.angle)
                summary['angle_range']['max'] = max(summary['angle_range']['max'], interaction.angle)

        # Convert defaultdict to dict
        summary['by_type'] = dict(summary['by_type'])
        summary['by_chain'] = dict(summary['by_chain'])

        return summary

    def get_residue_interaction_matrix(self) -> Dict[Tuple[str, str, int], Dict[Tuple[str, str, int], int]]:
        """
        Get interaction count matrix between residues.

        Returns:
            Dictionary mapping (res_a_key, res_b_key) -> interaction count
        """
        matrix = defaultdict(lambda: defaultdict(int))

        for interaction in self._interactions:
            matrix[interaction.res_a_key][interaction.res_b_key] += 1
            matrix[interaction.res_b_key][interaction.res_a_key] += 1

        return matrix

    def export_to_json(self, filepath: Union[str, Path], interactions: Optional[List[Interaction]] = None):
        """
        Export interactions to JSON file.

        Args:
            filepath: Output file path
            interactions: Interactions to export (default: all)
        """
        if interactions is None:
            interactions = self._interactions

        data = {
            'metadata': self.metadata,
            'interactions': [i.to_dict() for i in interactions]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def export_to_pymol(self, filepath: Union[str, Path], interactions: Optional[List[Interaction]] = None):
        """
        Export interactions to PyMOL Python script.

        Args:
            filepath: Output file path
            interactions: Interactions to export (default: all)
        """
        if interactions is None:
            interactions = self._interactions

        lines = [
            "# PyMOL script generated by All-Atom Visualization",
            "from pymol import cmd",
            "",
            "# Define colors",
            "cmd.set_color('myorange', '[253, 174, 97]')",
            "cmd.set_color('myblue', '[43, 131, 186]')",
            "cmd.set_color('mygreen', '[171, 221, 164]')",
            "",
        ]

        # Group interactions by type
        by_type = defaultdict(list)
        for interaction in interactions:
            by_type[interaction.type].append(interaction)

        # Color mapping
        colors = {
            'hbond': 'blue',
            'hbond_possible': 'lightblue',
            'saltbridge': 'yellow',
            'hydrophobic': 'grey50',
            'pistacking': 'green',
            'pication': 'orange',
            'halogen': 'greencyan',
            'water_bridge': 'lightblue',
            'metal': 'violetpurple'
        }

        # Generate PyMOL commands for each interaction type
        for interaction_type, type_interactions in by_type.items():
            color = colors.get(interaction_type, 'white')
            group_name = interaction_type.replace('_', '').capitalize()

            lines.append(f"# {interaction_type}")

            for i, interaction in enumerate(type_interactions):
                # Create pseudoatoms for interaction endpoints
                lines.append(f"cmd.pseudoatom('ps_{interaction_type}_{i}_a', "
                           f"resi={interaction.res_a_num}, chain='{interaction.res_a_chain}')")
                lines.append(f"cmd.pseudoatom('ps_{interaction_type}_{i}_b', "
                           f"resi={interaction.res_b_num}, chain='{interaction.res_b_chain}')")

                # Create distance object
                lines.append(f"cmd.distance('{group_name}', "
                           f"'ps_{interaction_type}_{i}_a', 'ps_{interaction_type}_{i}_b')")

            lines.append(f"cmd.set('dash_color', '{color}', '{group_name}')")
            lines.append("")

        # Cleanup
        lines.append("# Cleanup")
        lines.append("cmd.delete('ps_*')")
        lines.append("cmd.deselect()")

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

    def save(self, filepath: Union[str, Path]):
        """
        Save data manager state to file (using pickle).

        Args:
            filepath: Output file path
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'interactions': self._interactions,
                'groups': self.groups,
                'metadata': self.metadata
            }, f)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'InteractionDataManager':
        """
        Load data manager state from file.

        Args:
            filepath: Input file path

        Returns:
            Loaded InteractionDataManager instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance._interactions = data['interactions']
        instance.groups = data['groups']
        instance.metadata = data['metadata']
        instance.residues = set()
        instance._build_residue_index()

        return instance
