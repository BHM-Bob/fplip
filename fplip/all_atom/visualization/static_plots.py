"""
Static Visualization for All-Atom Interactions

Provides static plot generation using matplotlib and seaborn.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .data_manager import InteractionDataManager


class StaticVisualizer:
    """
    Generates static visualizations for All-Atom interaction data.

    Example:
        >>> dm = InteractionDataManager('results.json')
        >>> viz = StaticVisualizer(dm)
        >>> viz.plot_residue_matrix('matrix.png', figsize=(12, 10))
        >>> viz.plot_interaction_type_distribution('distribution.png')
    """

    # Color scheme matching main PLIP
    INTERACTION_COLORS = {
        'hbond': '#0066CC',           # Blue
        'hbond_possible': '#99CCFF',  # Light blue
        'saltbridge': '#FFCC00',      # Yellow
        'hydrophobic': '#808080',     # Grey
        'pistacking': '#009900',      # Green
        'pication': '#FF6600',        # Orange
        'halogen': '#00CCCC',         # Cyan
        'water_bridge': '#99CCFF',    # Light blue
        'metal': '#9900CC',           # Purple
    }

    def __init__(self, data_manager: InteractionDataManager):
        """
        Initialize visualizer.

        Args:
            data_manager: InteractionDataManager instance
        """
        self.dm = data_manager
        self._setup_style()

    def _setup_style(self):
        """Setup matplotlib style."""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

    def plot_residue_matrix(self,
                           output_path: Union[str, Path],
                           figsize: Tuple[int, int] = (12, 10),
                           interaction_types: Optional[List[str]] = None,
                           cmap: str = 'YlOrRd',
                           title: Optional[str] = None):
        """
        Plot residue interaction matrix heatmap.

        Args:
            output_path: Output file path
            figsize: Figure size (width, height)
            interaction_types: List of interaction types to include (default: all)
            cmap: Colormap name
            title: Plot title (default: auto-generated)
        """
        # Get interactions
        if interaction_types:
            interactions = self.dm.filter(interaction_types=interaction_types)
        else:
            interactions = self.dm.all_interactions

        # Get residue list
        residues = self.dm.get_residue_list()
        if not residues:
            print("No residues found in data")
            return

        # Create matrix
        n = len(residues)
        matrix = np.zeros((n, n))
        residue_to_idx = {res: i for i, res in enumerate(residues)}

        for interaction in interactions:
            idx_a = residue_to_idx.get(interaction.res_a_key)
            idx_b = residue_to_idx.get(interaction.res_b_key)
            if idx_a is not None and idx_b is not None:
                matrix[idx_a][idx_b] += 1
                matrix[idx_b][idx_a] += 1

        # Create labels
        labels = [f"{res[0]}:{res[1]}:{res[2]}" for res in residues]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        sns.heatmap(matrix,
                   xticklabels=labels,
                   yticklabels=labels,
                   cmap=cmap,
                   annot=True,
                   fmt='g',
                   square=True,
                   linewidths=0.5,
                   cbar_kws={'label': 'Interaction Count'},
                   ax=ax)

        # Rotate labels
        plt.xticks(rotation=90, ha='center', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)

        # Title
        if title is None:
            type_str = ', '.join(interaction_types) if interaction_types else 'All'
            title = f'Residue Interaction Matrix ({type_str})'
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved residue matrix to {output_path}")

    def plot_interaction_type_distribution(self,
                                          output_path: Union[str, Path],
                                          figsize: Tuple[int, int] = (10, 6),
                                          title: Optional[str] = None):
        """
        Plot interaction type distribution bar chart.

        Args:
            output_path: Output file path
            figsize: Figure size
            title: Plot title
        """
        summary = self.dm.get_interaction_summary()
        by_type = summary['by_type']

        if not by_type:
            print("No interactions found")
            return

        # Sort by count
        types = sorted(by_type.keys(), key=lambda x: by_type[x], reverse=True)
        counts = [by_type[t] for t in types]

        # Get colors
        colors = [self.INTERACTION_COLORS.get(t, '#999999') for t in types]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        bars = ax.bar(types, counts, color=colors, edgecolor='black', linewidth=0.5)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Interaction Type', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)

        if title is None:
            title = 'Interaction Type Distribution'
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved type distribution to {output_path}")

    def plot_distance_distribution(self,
                                  interaction_type: str,
                                  output_path: Union[str, Path],
                                  figsize: Tuple[int, int] = (10, 6),
                                  bins: int = 30,
                                  title: Optional[str] = None):
        """
        Plot distance distribution for a specific interaction type.

        Args:
            interaction_type: Type of interaction to plot
            output_path: Output file path
            figsize: Figure size
            bins: Number of histogram bins
            title: Plot title
        """
        interactions = self.dm.filter(interaction_types=[interaction_type])

        if not interactions:
            print(f"No {interaction_type} interactions found")
            return

        distances = [i.distance for i in interactions]

        fig, ax = plt.subplots(figsize=figsize)

        color = self.INTERACTION_COLORS.get(interaction_type, '#999999')

        ax.hist(distances, bins=bins, color=color, edgecolor='black',
               alpha=0.7, linewidth=0.5)

        ax.set_xlabel('Distance (Å)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)

        if title is None:
            title = f'{interaction_type.replace("_", " ").title()} Distance Distribution'
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add statistics
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        ax.axvline(mean_dist, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_dist:.2f} Å')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved distance distribution to {output_path}")

    def plot_angle_distribution(self,
                               interaction_type: str,
                               output_path: Union[str, Path],
                               figsize: Tuple[int, int] = (10, 6),
                               bins: int = 30,
                               title: Optional[str] = None):
        """
        Plot angle distribution for a specific interaction type.

        Args:
            interaction_type: Type of interaction to plot
            output_path: Output file path
            figsize: Figure size
            bins: Number of histogram bins
            title: Plot title
        """
        interactions = self.dm.filter(interaction_types=[interaction_type])

        # Filter interactions with angles
        angles = [i.angle for i in interactions if i.angle is not None]

        if not angles:
            print(f"No angle data for {interaction_type}")
            return

        fig, ax = plt.subplots(figsize=figsize)

        color = self.INTERACTION_COLORS.get(interaction_type, '#999999')

        ax.hist(angles, bins=bins, color=color, edgecolor='black',
               alpha=0.7, linewidth=0.5)

        ax.set_xlabel('Angle (degrees)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)

        if title is None:
            title = f'{interaction_type.replace("_", " ").title()} Angle Distribution'
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add statistics
        mean_angle = np.mean(angles)
        ax.axvline(mean_angle, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_angle:.2f}°')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved angle distribution to {output_path}")

    def plot_residue_network(self,
                            output_path: Union[str, Path],
                            figsize: Tuple[int, int] = (12, 12),
                            min_interactions: int = 1,
                            interaction_types: Optional[List[str]] = None,
                            title: Optional[str] = None):
        """
        Plot residue interaction network using networkx.

        Args:
            output_path: Output file path
            figsize: Figure size
            min_interactions: Minimum number of interactions to include edge
            interaction_types: List of interaction types to include
            title: Plot title
        """
        try:
            import networkx as nx
        except ImportError:
            print("networkx is required for network plots. Install with: pip install networkx")
            return

        # Get interactions
        if interaction_types:
            interactions = self.dm.filter(interaction_types=interaction_types)
        else:
            interactions = self.dm.all_interactions

        # Build graph
        G = nx.Graph()

        # Add edges with weights
        edge_weights = defaultdict(int)
        for interaction in interactions:
            res_a = f"{interaction.res_a_name}:{interaction.res_a_chain}:{interaction.res_a_num}"
            res_b = f"{interaction.res_b_name}:{interaction.res_b_chain}:{interaction.res_b_num}"

            if res_a != res_b:  # Skip self-interactions
                edge_key = tuple(sorted([res_a, res_b]))
                edge_weights[edge_key] += 1

        # Add edges meeting threshold
        for (res_a, res_b), weight in edge_weights.items():
            if weight >= min_interactions:
                G.add_edge(res_a, res_b, weight=weight)

        if not G.edges():
            print("No edges meet the minimum interaction threshold")
            return

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                              node_size=500, ax=ax)

        # Draw edges with width proportional to weight
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1

        nx.draw_networkx_edges(G, pos, width=[w/max_weight*5 for w in weights],
                              alpha=0.6, ax=ax)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

        if title is None:
            title = f'Residue Interaction Network (min {min_interactions} interactions)'
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved network plot to {output_path}")

    def plot_chain_interaction_summary(self,
                                      output_path: Union[str, Path],
                                      figsize: Tuple[int, int] = (10, 6)):
        """
        Plot summary of interactions between chains.

        Args:
            output_path: Output file path
            figsize: Figure size
        """
        chains = self.dm.get_chains()

        if len(chains) < 2:
            print("Need at least 2 chains for chain interaction summary")
            return

        # Count interactions between chains
        chain_counts = defaultdict(lambda: defaultdict(int))

        for interaction in self.dm.all_interactions:
            chain_a = interaction.res_a_chain
            chain_b = interaction.res_b_chain

            if chain_a != chain_b:
                chain_counts[chain_a][chain_b] += 1
                chain_counts[chain_b][chain_a] += 1

        # Create matrix
        n = len(chains)
        matrix = np.zeros((n, n))

        for i, chain_a in enumerate(chains):
            for j, chain_b in enumerate(chains):
                if i != j:
                    matrix[i][j] = chain_counts[chain_a][chain_b]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(matrix,
                   xticklabels=chains,
                   yticklabels=chains,
                   annot=True,
                   fmt='g',
                   cmap='Blues',
                   square=True,
                   linewidths=0.5,
                   cbar_kws={'label': 'Interaction Count'},
                   ax=ax)

        ax.set_title('Inter-Chain Interaction Summary', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved chain summary to {output_path}")

    def plot_residue_interaction_summary(self,
                                        output_path: Union[str, Path],
                                        top_n: int = 20,
                                        figsize: Tuple[int, int] = (12, 8),
                                        title: Optional[str] = None):
        """
        Plot summary of top interacting residues.

        Args:
            output_path: Output file path
            top_n: Number of top residues to show
            figsize: Figure size
            title: Plot title
        """
        # Count interactions per residue
        residue_counts = defaultdict(int)
        for interaction in self.dm.all_interactions:
            residue_counts[interaction.res_a_key] += 1
            residue_counts[interaction.res_b_key] += 1

        # Get top N
        top_residues = sorted(residue_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

        if not top_residues:
            print("No interactions found")
            return

        # Prepare data
        labels = [f"{res[0]}:{res[1]}:{res[2]}" for res, _ in top_residues]
        counts = [count for _, count in top_residues]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot horizontal bar chart
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, counts, color='steelblue', edgecolor='black', linewidth=0.5)

        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()  # Top to bottom
        ax.set_xlabel('Number of Interactions', fontsize=12)

        if title is None:
            title = f'Top {top_n} Interacting Residues'
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add value labels
        for i, v in enumerate(counts):
            ax.text(v + 0.5, i, str(v), va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved residue summary to {output_path}")

    def plot_interaction_heatmap_by_type(self,
                                        output_path: Union[str, Path],
                                        figsize: Tuple[int, int] = (14, 10)):
        """
        Plot separate heatmaps for each interaction type.

        Args:
            output_path: Output file path
            figsize: Figure size
        """
        interaction_types = self.dm.get_interaction_types()

        # Filter out types with no interactions or too many (like water_bridge_possible)
        significant_types = [
            t for t in interaction_types
            if len(self.dm.filter(interaction_types=[t])) > 0
            and 'possible' not in t  # Skip "possible" variants for clarity
        ]

        if not significant_types:
            print("No significant interaction types found")
            return

        n_types = len(significant_types)
        n_cols = 3
        n_rows = (n_types + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_types == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        residues = self.dm.get_residue_list()
        residue_to_idx = {res: i for i, res in enumerate(residues)}
        labels = [f"{res[0]}:{res[2]}" for res in residues]  # Short labels

        for idx, interaction_type in enumerate(significant_types):
            ax = axes[idx]

            # Create matrix for this type
            n = len(residues)
            matrix = np.zeros((n, n))

            interactions = self.dm.filter(interaction_types=[interaction_type])
            for interaction in interactions:
                i = residue_to_idx.get(interaction.res_a_key)
                j = residue_to_idx.get(interaction.res_b_key)
                if i is not None and j is not None:
                    matrix[i][j] += 1
                    matrix[j][i] += 1

            # Plot
            color = self.INTERACTION_COLORS.get(interaction_type, 'Blues')
            sns.heatmap(matrix, cmap='YlOrRd', square=True,
                       xticklabels=False, yticklabels=False,
                       cbar_kws={'label': 'Count', 'shrink': 0.5},
                       ax=ax, vmin=0, vmax=max(1, matrix.max()))

            ax.set_title(interaction_type.replace('_', ' ').title(),
                        fontsize=10, fontweight='bold')

        # Hide unused subplots
        for idx in range(n_types, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Interaction Heatmaps by Type', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved type heatmaps to {output_path}")

    def plot_all(self,
                output_dir: Union[str, Path],
                prefix: str = ""):
        """
        Generate all standard plots.

        Args:
            output_dir: Output directory
            prefix: Prefix for output filenames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Residue matrix
        self.plot_residue_matrix(output_dir / f"{prefix}residue_matrix.png")

        # Type distribution
        self.plot_interaction_type_distribution(output_dir / f"{prefix}type_distribution.png")

        # Distance distributions for each type
        for interaction_type in self.dm.get_interaction_types():
            safe_name = interaction_type.replace('_', '_')
            self.plot_distance_distribution(
                interaction_type,
                output_dir / f"{prefix}distance_dist_{safe_name}.png"
            )

        # Chain summary (if multiple chains)
        chains = self.dm.get_chains()
        if len(chains) > 1:
            self.plot_chain_interaction_summary(output_dir / f"{prefix}chain_summary.png")

        # Network plot
        self.plot_residue_network(output_dir / f"{prefix}network.png")

        # Residue summary (top interacting)
        self.plot_residue_interaction_summary(output_dir / f"{prefix}top_residues.png")

        # Type-specific heatmaps
        self.plot_interaction_heatmap_by_type(output_dir / f"{prefix}type_heatmaps.png")

        print(f"All plots saved to {output_dir}")
