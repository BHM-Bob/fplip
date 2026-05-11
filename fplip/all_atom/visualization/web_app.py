"""
Interactive Web Visualization for All-Atom Interactions

Flask-based web application with NGL Viewer integration.
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import (Flask, jsonify, render_template, request, send_file,
                   send_from_directory)

from .data_manager import InteractionDataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class WebAppState:
    """Global state management for the web application."""

    def __init__(self):
        # For single conformation (standard analysis)
        self.data_manager: Optional[InteractionDataManager] = None

        # For multi-conformation (docking analysis)
        self.is_docking: bool = False
        self.conformations: List[Dict] = []  # List of conformation data
        self.current_conformation_idx: int = 0  # Currently selected conformation
        self._conformation_data_managers: Dict[int, InteractionDataManager] = {}  # Cached data managers

        self.pdb_path: Optional[str] = None
        self.pdb_content: Optional[str] = None  # Store PDB content for serving
        self.groups: Dict[str, Dict] = {}  # User-defined interaction groups
        self.visible_interactions: set = set()  # Currently visible interaction IDs
        self.current_filters: Dict[str, Any] = {}
        self._interaction_id_map: Dict[int, int] = {}  # id(interaction) -> sequential_id
        self._next_id: int = 1  # Next sequential ID
        self._raw_data: Dict[str, Any] = {}  # Store raw JSON data for conformation switching
        logger.info("WebAppState initialized")

    def load_data(self, json_path: str, pdb_path: Optional[str] = None):
        """Load interaction data from JSON file."""
        logger.info(f"Loading data from JSON: {json_path}")

        # Load raw JSON data
        with open(json_path, 'r') as f:
            self._raw_data = json.load(f)

        # Check if this is docking data (multi-conformation)
        metadata = self._raw_data.get('metadata', {})
        self.is_docking = metadata.get('analysis_type') == 'docking'

        if self.is_docking:
            logger.info("Detected docking data with multiple conformations")
            self._load_docking_data(json_path, pdb_path)
        else:
            logger.info("Detected standard single-structure data")
            self._load_standard_data(json_path, pdb_path)

    def _load_standard_data(self, json_path: str, pdb_path: Optional[str] = None):
        """Load standard single-conformation data."""
        self.data_manager = InteractionDataManager(self._raw_data)
        logger.info(f"Loaded {len(self.data_manager.all_interactions)} interactions")

        # Try to find PDB file
        self._load_pdb_file(json_path, pdb_path)

        # Build interaction ID map
        self._build_interaction_id_map()

    def _load_docking_data(self, json_path: str, pdb_path: Optional[str] = None):
        """Load docking multi-conformation data."""
        self.conformations = self._raw_data.get('conformations', [])
        logger.info(f"Loaded {len(self.conformations)} conformations")

        if not self.conformations:
            raise ValueError("No conformations found in docking data")

        # Load first conformation by default
        self.current_conformation_idx = 0
        self._load_conformation(0)

        # Try to find PDB file (for receptor structure)
        self._load_pdb_file(json_path, pdb_path)

    def _load_conformation(self, idx: int):
        """Load a specific conformation by index."""
        if idx < 0 or idx >= len(self.conformations):
            raise ValueError(f"Invalid conformation index: {idx}")

        # Check if we have a cached data manager for this conformation
        if idx in self._conformation_data_managers:
            self.data_manager = self._conformation_data_managers[idx]
            logger.info(f"Using cached data manager for conformation {idx + 1}")
        else:
            # Create a data manager for this conformation
            conf_data = self.conformations[idx]
            conformation_dict = {
                'metadata': {
                    **self._raw_data.get('metadata', {}),
                    'model_num': conf_data.get('model_num'),
                    'vina_result': conf_data.get('vina_result')
                },
                'atom_coords': conf_data.get('atom_coords', {}),
                'interactions': conf_data.get('interactions', {})
            }
            self.data_manager = InteractionDataManager(conformation_dict)
            self._conformation_data_managers[idx] = self.data_manager
            logger.info(f"Created data manager for conformation {idx + 1} with "
                       f"{len(self.data_manager.all_interactions)} interactions")

        # Load PDB content for this conformation (for NGL Viewer)
        conf_data = self.conformations[idx]
        self.pdb_content = conf_data.get('pdb_content')
        if self.pdb_content:
            logger.info(f"Loaded PDB content for conformation {idx + 1}: {len(self.pdb_content)} characters")
        else:
            logger.warning(f"No PDB content available for conformation {idx + 1}")

        self.current_conformation_idx = idx
        self._build_interaction_id_map()

    def switch_conformation(self, idx: int) -> bool:
        """Switch to a different conformation.

        Args:
            idx: Conformation index (0-based)

        Returns:
            True if switch was successful
        """
        if not self.is_docking:
            logger.warning("Cannot switch conformation: not docking data")
            return False

        if idx < 0 or idx >= len(self.conformations):
            logger.error(f"Invalid conformation index: {idx}")
            return False

        if idx == self.current_conformation_idx:
            return True

        logger.info(f"Switching from conformation {self.current_conformation_idx + 1} to {idx + 1}")
        self._load_conformation(idx)

        # Reset groups when switching conformations
        self.groups = {}
        self.visible_interactions = set(self._interaction_id_map.values())

        return True

    def get_current_conformation_info(self) -> Optional[Dict]:
        """Get information about the current conformation."""
        if not self.is_docking or not self.conformations:
            return None

        conf = self.conformations[self.current_conformation_idx]
        return {
            'index': self.current_conformation_idx,
            'model_num': conf.get('model_num'),
            'vina_result': conf.get('vina_result'),
            'atom_count': conf.get('atom_count'),
            'residue_count': conf.get('residue_count')
        }

    def get_all_conformations_info(self) -> List[Dict]:
        """Get information about all conformations."""
        if not self.is_docking:
            return []

        return [
            {
                'index': i,
                'model_num': c.get('model_num'),
                'vina_result': c.get('vina_result')
            }
            for i, c in enumerate(self.conformations)
        ]

    def _load_pdb_file(self, json_path: str, pdb_path: Optional[str] = None):
        """Load PDB file for 3D visualization."""
        self.pdb_path = pdb_path

        # Priority 1: Use embedded PDB content from JSON (with hydrogens if NOHYDRO=False)
        if 'pdb_content' in self._raw_data:
            self.pdb_content = self._raw_data['pdb_content']
            logger.info(f"Using embedded PDB content from JSON ({len(self.pdb_content)} chars)")
            return

        # Priority 2: Use provided or inferred PDB file path
        if self.pdb_path is None and self.data_manager:
            # Try to infer from metadata
            metadata = self.data_manager.metadata
            if 'pdb_file' in metadata:
                inferred_path = metadata['pdb_file']
                logger.info(f"Trying to infer PDB path from metadata: {inferred_path}")
                if Path(inferred_path).exists():
                    self.pdb_path = inferred_path
                    logger.info(f"PDB file found at: {self.pdb_path}")
                else:
                    logger.warning(f"PDB file not found at inferred path: {inferred_path}")
                    # Try relative to JSON file
                    json_dir = Path(json_path).parent
                    pdb_name = Path(inferred_path).name
                    alternative_path = json_dir / pdb_name
                    if alternative_path.exists():
                        self.pdb_path = str(alternative_path)
                        logger.info(f"PDB file found at alternative path: {self.pdb_path}")

        # Load PDB content if available (for NGL Viewer)
        if self.pdb_path and Path(self.pdb_path).exists():
            try:
                with open(self.pdb_path, 'r') as f:
                    self.pdb_content = f.read()
                logger.info(f"Loaded PDB content from file: {len(self.pdb_content)} characters")
            except Exception as e:
                logger.error(f"Error loading PDB content: {e}")
        else:
            logger.warning("No PDB file available for 3D visualization")

    def _build_interaction_id_map(self):
        """Build interaction ID map for stable IDs."""
        self._interaction_id_map = {}
        self._next_id = 1

        if self.data_manager:
            for interaction in self.data_manager.all_interactions:
                self._interaction_id_map[id(interaction)] = self._next_id
                self._next_id += 1

        logger.info(f"Built ID map with {len(self._interaction_id_map)} interactions")

        # Auto-show all interactions initially
        self.visible_interactions = set(self._interaction_id_map.values())
        logger.info(f"Initialized {len(self.visible_interactions)} visible interactions")

    def create_group(self, name: str, interaction_ids: List[int]) -> Dict:
        """Create a new interaction group."""
        logger.info(f"Creating group '{name}' with {len(interaction_ids)} interactions")
        self.groups[name] = {
            'name': name,
            'interaction_ids': set(interaction_ids),
            'visible': True,
            'created_at': datetime.now().isoformat(),
        }
        return self.groups[name]

    def delete_group(self, name: str):
        """Delete an interaction group."""
        logger.info(f"Deleting group '{name}'")
        if name in self.groups:
            del self.groups[name]

    def toggle_group_visibility(self, name: str, visible: bool):
        """Toggle group visibility."""
        logger.info(f"Toggling group '{name}' visibility to {visible}")
        if name in self.groups:
            self.groups[name]['visible'] = visible

    def get_visible_interactions(self) -> List[Dict]:
        """Get list of currently visible interactions."""
        if not self.data_manager:
            return []

        visible = []
        for interaction in self.data_manager.all_interactions:
            inter_id = id(interaction)
            if inter_id in self.visible_interactions:
                visible.append(self._interaction_to_dict(interaction))

        return visible

    def _interaction_to_dict(self, interaction) -> Dict:
        """Convert interaction to dictionary for JSON serialization."""
        # Use sequential ID for JavaScript compatibility
        sequential_id = self._interaction_id_map.get(id(interaction), 0)

        # Get coordinates for atom A
        # For RING pseudo-atoms, use ring_center from details
        if interaction.atom_a_name == 'RING' and interaction.details:
            coords_a = interaction.details.get('ring_center')
        else:
            coords_a = self.data_manager.atom_coords.get(interaction.atom_a_idx)

        # Get coordinates for atom B
        if interaction.atom_b_name == 'RING' and interaction.details:
            coords_b = interaction.details.get('ring_center')
        else:
            coords_b = self.data_manager.atom_coords.get(interaction.atom_b_idx)

        result = {
            'id': sequential_id,
            'type': interaction.type,
            'res_a_name': interaction.res_a_name,
            'res_a_chain': interaction.res_a_chain,
            'res_a_num': interaction.res_a_num,
            'res_b_name': interaction.res_b_name,
            'res_b_chain': interaction.res_b_chain,
            'res_b_num': interaction.res_b_num,
            'atom_a_name': interaction.atom_a_name,
            'atom_a_idx': interaction.atom_a_idx,
            'atom_b_name': interaction.atom_b_name,
            'atom_b_idx': interaction.atom_b_idx,
            'distance': interaction.distance,
            'angle': interaction.angle,
            'res_a_key': interaction.res_a_key,
            'res_b_key': interaction.res_b_key,
            'coords_a': coords_a,
            'coords_b': coords_b,
        }

        # For water bridges, add water coordinates and additional info
        if interaction.type in ('water_bridge', 'water_bridge_possible') and interaction.details:
            # Get water oxygen coordinates
            water_atom_idx = interaction.details.get('water_atom_idx')
            if water_atom_idx:
                result['coords_water'] = self.data_manager.atom_coords.get(water_atom_idx)
                result['water_atom_idx'] = water_atom_idx

            # Add water residue info
            result['water_residue'] = interaction.details.get('water_residue')

            # Add segment distances
            if 'distance_aw' in interaction.details:
                result['distance_aw'] = interaction.details['distance_aw']
            if 'distance_bw' in interaction.details:
                result['distance_bw'] = interaction.details['distance_bw']
            if 'distance_dw' in interaction.details:
                result['distance_dw'] = interaction.details['distance_dw']

            # Add angles
            if 'd_angle' in interaction.details:
                result['d_angle'] = interaction.details['d_angle']
            if 'w_angle' in interaction.details:
                result['w_angle'] = interaction.details['w_angle']

        return result


# Global state instance
app_state = WebAppState()


def create_app(json_path: Optional[str] = None, pdb_path: Optional[str] = None) -> Flask:
    """
    Create and configure Flask application.

    Args:
        json_path: Path to interaction JSON file
        pdb_path: Path to PDB file (optional, will try to infer from JSON)

    Returns:
        Configured Flask application
    """
    app = Flask(__name__,
                template_folder='templates',
                static_folder='static')

    # Load data if provided
    if json_path:
        logger.info(f"Creating app with json_path={json_path}, pdb_path={pdb_path}")
        app_state.load_data(json_path, pdb_path)
    else:
        logger.info("Creating app without initial data")

    # Check for local NGL dev mode (set NGL_DEV_MODE=true to use local ngl.js)
    app.config['NGL_DEV_MODE'] = os.environ.get('NGL_DEV_MODE', '').lower() == 'true'
    if app.config['NGL_DEV_MODE']:
        logger.info("NGL_DEV_MODE enabled: using local ngl.js")

    # Register routes
    _register_routes(app)

    return app


def _register_routes(app: Flask):
    """Register Flask routes."""

    @app.route('/')
    def index():
        """Main page."""
        logger.info("Serving index page")
        return render_template('index.html', ngl_dev_mode=app.config['NGL_DEV_MODE'])

    @app.route('/api/interactions')
    def get_interactions():
        """Get interactions with optional filtering."""
        logger.info(f"API: get_interactions called with params: {dict(request.args)}")

        if not app_state.data_manager:
            logger.error("No data loaded")
            return jsonify({'error': 'No data loaded'}), 400

        # Parse filter parameters
        interaction_types = request.args.getlist('types')
        residues = request.args.getlist('residues')
        distance_min = request.args.get('distance_min', type=float)
        distance_max = request.args.get('distance_max', type=float)

        logger.info(f"Filters - types: {interaction_types}, residues: {residues}, "
                   f"distance_min: {distance_min}, distance_max: {distance_max}")

        # Build filter criteria
        filters = {}
        if interaction_types:
            filters['interaction_types'] = interaction_types
        if residues:
            # Parse residue strings like "LYS:A:193"
            parsed_residues = []
            for res in residues:
                parts = res.split(':')
                if len(parts) == 3:
                    parsed_residues.append((parts[0], parts[1], int(parts[2])))
            filters['residues'] = parsed_residues
        if distance_min is not None:
            filters['distance_min'] = distance_min
        if distance_max is not None:
            filters['distance_max'] = distance_max

        # Apply filters
        if filters:
            interactions = app_state.data_manager.filter(**filters)
        else:
            interactions = app_state.data_manager.all_interactions

        logger.info(f"Returning {len(interactions)} interactions")

        # Convert to dictionaries
        result = [app_state._interaction_to_dict(i) for i in interactions]

        return jsonify({
            'interactions': result,
            'count': len(result),
            'filters': filters
        })

    @app.route('/api/interactions/summary')
    def get_summary():
        """Get interaction summary statistics."""
        logger.info("API: get_summary called")

        if not app_state.data_manager:
            logger.error("No data loaded")
            return jsonify({'error': 'No data loaded'}), 400

        summary = app_state.data_manager.get_interaction_summary()
        logger.info(f"Summary: {summary}")
        return jsonify(summary)

    @app.route('/api/interactions/types')
    def get_types():
        """Get available interaction types."""
        logger.info("API: get_types called")

        if not app_state.data_manager:
            logger.error("No data loaded")
            return jsonify({'error': 'No data loaded'}), 400

        types = app_state.data_manager.get_interaction_types()
        logger.info(f"Returning types: {types}")
        return jsonify({'types': types})

    @app.route('/api/residues')
    def get_residues():
        """Get list of all residues."""
        logger.info("API: get_residues called")

        if not app_state.data_manager:
            logger.error("No data loaded")
            return jsonify({'error': 'No data loaded'}), 400

        residues = app_state.data_manager.get_residue_list()
        logger.info(f"Returning {len(residues)} residues")
        return jsonify({
            'residues': [
                {'name': r[0], 'chain': r[1], 'num': r[2]}
                for r in residues
            ]
        })

    @app.route('/api/chains')
    def get_chains():
        """Get list of all chains."""
        logger.info("API: get_chains called")

        if not app_state.data_manager:
            logger.error("No data loaded")
            return jsonify({'error': 'No data loaded'}), 400

        chains = app_state.data_manager.get_chains()
        logger.info(f"Returning chains: {chains}")
        return jsonify({'chains': chains})

    @app.route('/api/conformations')
    def get_conformations():
        """Get list of all conformations (for docking data)."""
        logger.info("API: get_conformations called")

        if not app_state.is_docking:
            return jsonify({
                'is_docking': False,
                'conformations': [],
                'current': None
            })

        conformations = app_state.get_all_conformations_info()
        current = app_state.get_current_conformation_info()

        logger.info(f"Returning {len(conformations)} conformations, current: {current['model_num'] if current else None}")
        return jsonify({
            'is_docking': True,
            'conformations': conformations,
            'current': current
        })

    @app.route('/api/conformations/<int:idx>', methods=['POST'])
    def switch_conformation(idx):
        """Switch to a specific conformation."""
        logger.info(f"API: switch_conformation called for index {idx}")

        if not app_state.is_docking:
            logger.error("Not docking data")
            return jsonify({'error': 'Not docking data'}), 400

        success = app_state.switch_conformation(idx)
        if not success:
            logger.error(f"Failed to switch to conformation {idx}")
            return jsonify({'error': f'Failed to switch to conformation {idx}'}), 400

        current = app_state.get_current_conformation_info()
        logger.info(f"Switched to conformation {current['model_num'] if current else 'unknown'}")
        return jsonify({
            'status': 'success',
            'current': current
        })

    @app.route('/api/export/conformation', methods=['POST'])
    def export_conformation_pdb():
        """Export current conformation as PDB file."""
        logger.info("API: export_conformation_pdb called")

        if not app_state.data_manager:
            logger.error("No data loaded")
            return jsonify({'error': 'No data loaded'}), 400

        # Get current conformation info
        conf_info = app_state.get_current_conformation_info()
        model_num = conf_info['model_num'] if conf_info else app_state.current_conformation_idx + 1

        # Generate PDB content from atom coordinates
        pdb_lines = []
        pdb_lines.append(f"REMARK   4 Exported from All-Atom Visualization")
        pdb_lines.append(f"REMARK   4 Conformation: {model_num}")

        if conf_info and conf_info.get('vina_result'):
            vina = conf_info['vina_result']
            pdb_lines.append(f"REMARK   4 VINA_AFFINITY: {vina.get('affinity', 'N/A')}")
            pdb_lines.append(f"REMARK   4 VINA_RMSD_LB: {vina.get('rmsd_lb', 'N/A')}")
            pdb_lines.append(f"REMARK   4 VINA_RMSD_UB: {vina.get('rmsd_ub', 'N/A')}")

        # Add atom coordinates
        for idx_str, coords in app_state.data_manager.atom_coords.items():
            # Simple PDB ATOM record format
            # This is a simplified version - real implementation would need proper formatting
            x, y, z = coords
            pdb_lines.append(
                f"ATOM  {int(idx_str):5d}  X   RES A{int(idx_str):4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           X"
            )

        pdb_lines.append("END")
        pdb_content = '\n'.join(pdb_lines)

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_content)
            temp_path = f.name

        logger.info(f"Exported conformation {model_num} to {temp_path}")

        # Send file
        response = send_file(
            temp_path,
            as_attachment=True,
            download_name=f'conformation_{model_num}.pdb',
            mimetype='chemical/x-pdb'
        )

        # Schedule cleanup
        @response.call_on_close
        def cleanup():
            try:
                os.unlink(temp_path)
                logger.info(f"Temporary file cleaned up: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")

        return response

    @app.route('/api/groups', methods=['GET'])
    def get_groups():
        """Get all interaction groups."""
        logger.info("API: get_groups called")
        return jsonify({
            'groups': {
                name: {
                    **info,
                    'interaction_ids': list(info['interaction_ids'])
                }
                for name, info in app_state.groups.items()
            }
        })

    @app.route('/api/groups', methods=['POST'])
    def create_group():
        """Create a new interaction group."""
        data = request.get_json()
        logger.info(f"API: create_group called with data: {data}")

        if not data or 'name' not in data:
            logger.error("Group name required")
            return jsonify({'error': 'Group name required'}), 400

        name = data['name']
        interaction_ids = data.get('interaction_ids', [])

        if name in app_state.groups:
            logger.error(f"Group '{name}' already exists")
            return jsonify({'error': f'Group "{name}" already exists'}), 400

        group = app_state.create_group(name, interaction_ids)
        logger.info(f"Group '{name}' created successfully")
        return jsonify({
            'status': 'success',
            'group': {
                **group,
                'interaction_ids': list(group['interaction_ids'])
            }
        })

    @app.route('/api/groups/<name>', methods=['DELETE'])
    def delete_group(name: str):
        """Delete an interaction group."""
        logger.info(f"API: delete_group called for '{name}'")

        if name not in app_state.groups:
            logger.error(f"Group '{name}' not found")
            return jsonify({'error': f'Group "{name}" not found'}), 404

        app_state.delete_group(name)
        logger.info(f"Group '{name}' deleted")
        return jsonify({'status': 'success'})

    @app.route('/api/groups/<name>/visibility', methods=['POST'])
    def toggle_group_visibility(name: str):
        """Toggle group visibility."""
        logger.info(f"API: toggle_group_visibility called for '{name}'")

        if name not in app_state.groups:
            logger.error(f"Group '{name}' not found")
            return jsonify({'error': f'Group "{name}" not found'}), 404

        data = request.get_json()
        visible = data.get('visible', True)

        app_state.toggle_group_visibility(name, visible)
        logger.info(f"Group '{name}' visibility set to {visible}")
        return jsonify({'status': 'success', 'visible': visible})

    @app.route('/api/pdb')
    def get_pdb_info():
        """Get PDB file information."""
        logger.info("API: get_pdb_info called")

        if not app_state.data_manager:
            logger.error("No data loaded")
            return jsonify({'error': 'No data loaded'}), 400

        metadata = app_state.data_manager.metadata
        pdb_info = {
            'pdb_file': app_state.pdb_path,
            'atom_count': metadata.get('atom_count'),
            'residue_count': metadata.get('residue_count'),
            'has_pdb_content': app_state.pdb_content is not None
        }
        logger.info(f"Returning PDB info: {pdb_info}")
        return jsonify(pdb_info)

    @app.route('/api/pdb/content')
    def get_pdb_content():
        """Get PDB file content for NGL Viewer."""
        logger.info("API: get_pdb_content called")

        if app_state.pdb_content:
            logger.info(f"Returning PDB content: {len(app_state.pdb_content)} characters")
            return app_state.pdb_content, 200, {'Content-Type': 'text/plain'}
        else:
            logger.error("No PDB content available")
            return jsonify({'error': 'No PDB content available'}), 404

    @app.route('/api/export/pymol', methods=['POST'])
    def export_pymol():
        """Export interactions to PyMOL script."""
        logger.info("API: export_pymol called")

        if not app_state.data_manager:
            logger.error("No data loaded")
            return jsonify({'error': 'No data loaded'}), 400

        data = request.get_json()
        interaction_ids = data.get('interaction_ids', [])
        residue_colors = data.get('residue_colors', {})
        logger.info(f"Exporting {len(interaction_ids)} interactions to PyMOL")

        # Get interactions by sequential ID (frontend uses sequential IDs,
        # not Python memory addresses)
        seq_id_to_interaction = {}
        for interaction in app_state.data_manager.all_interactions:
            seq_id = app_state._interaction_id_map.get(id(interaction))
            if seq_id is not None:
                seq_id_to_interaction[seq_id] = interaction

        interactions = [
            seq_id_to_interaction[seq_id]
            for seq_id in interaction_ids
            if seq_id in seq_id_to_interaction
        ]

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            temp_path = f.name
            app_state.data_manager.export_to_pymol(
                temp_path,
                interactions,
                pdb_path=app_state.pdb_path,
                pdb_content=app_state.pdb_content,
                residue_colors=residue_colors
            )

        logger.info(f"PyMOL script saved to: {temp_path}")

        # Send file
        response = send_file(
            temp_path,
            as_attachment=True,
            download_name='interactions.py',
            mimetype='text/x-python'
        )

        # Schedule cleanup
        @response.call_on_close
        def cleanup():
            try:
                os.unlink(temp_path)
                logger.info(f"Temporary file cleaned up: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")

        return response

    @app.route('/api/export/json', methods=['POST'])
    def export_json():
        """Export interactions to JSON."""
        logger.info("API: export_json called")

        if not app_state.data_manager:
            logger.error("No data loaded")
            return jsonify({'error': 'No data loaded'}), 400

        data = request.get_json()
        interaction_ids = data.get('interaction_ids', [])
        logger.info(f"Exporting {len(interaction_ids)} interactions to JSON")

        # Get interactions by sequential ID (frontend uses sequential IDs,
        # not Python memory addresses)
        seq_id_to_interaction = {}
        for interaction in app_state.data_manager.all_interactions:
            seq_id = app_state._interaction_id_map.get(id(interaction))
            if seq_id is not None:
                seq_id_to_interaction[seq_id] = interaction

        interactions = [
            app_state._interaction_to_dict(seq_id_to_interaction[seq_id])
            for seq_id in interaction_ids
            if seq_id in seq_id_to_interaction
        ]

        return jsonify({
            'interactions': interactions,
            'metadata': app_state.data_manager.metadata,
            'exported_at': datetime.now().isoformat()
        })


if __name__ == '__main__':
    # Development server
    app = create_app()
    app.run(debug=True, host='127.0.0.1', port=5000)
