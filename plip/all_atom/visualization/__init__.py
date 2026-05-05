"""
All-Atom Visualization Module

Provides visualization capabilities for All-Atom interaction detection results:
- Static visualizations (residue matrix heatmaps, distribution plots)
- Interactive web-based visualization with NGL Viewer
- PyMOL export functionality

Example:
    >>> from plip.all_atom.visualization import InteractionDataManager, StaticVisualizer
    >>> dm = InteractionDataManager('results.json')
    >>> viz = StaticVisualizer(dm)
    >>> viz.plot_residue_matrix('matrix.png')
"""

from .data_manager import InteractionDataManager
from .static_plots import StaticVisualizer
from .web_app import create_app

__all__ = [
    'InteractionDataManager',
    'StaticVisualizer',
    'create_app',
]
