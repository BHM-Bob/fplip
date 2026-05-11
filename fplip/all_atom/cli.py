"""
All-Atom Command Line Interface

Provides command-line tools for:
1. Analyzing PDB files and generating interaction data
2. Analyzing AutoDock Vina docking results
3. Generating static visualizations
4. Launching interactive web visualization

Usage:
    python -m plip.all_atom.cli analyze input.pdb -o results.json
    python -m plip.all_atom.cli dock --ligand dock.pdbqt --receptor receptor.pdbqt -o dock_results.json
    python -m plip.all_atom.cli static results.json --plot-matrix -o matrix.png
    python -m plip.all_atom.cli interactive results.json --port 8080
"""

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fplip.all_atom.atom_properties import AtomProperties
from fplip.all_atom.interaction_detector import UnifiedInteractionDetector
from fplip.all_atom.molecule_complex import MoleculeComplex
from fplip.basic import config
from fplip.basic.logger import logger
from fplip.structure.pdb import PDBParser


class AllAtomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for All-Atom data."""

    def default(self, obj: Any) -> Any:
        """Convert non-serializable objects."""
        # Handle namedtuples
        if hasattr(obj, '_asdict'):
            return dict(obj._asdict())

        # Handle objects with __dict__
        if hasattr(obj, '__dict__'):
            return dict(obj.__dict__)

        # Handle sets
        if isinstance(obj, set):
            return list(obj)

        # Handle other non-serializable types
        try:
            return str(obj)
        except Exception:
            return f"<{obj.__class__.__name__}>"


def _parse_dock_pdbqt(filepath: Path) -> List[Dict]:
    """Parse AutoDock Vina PDBQT result file and extract all conformations.

    Args:
        filepath: Path to the PDBQT file containing docking results

    Returns:
        List of dictionaries containing conformation data
    """
    with open(filepath, 'r') as f:
        content = f.read()

    models = []
    # Pattern to match MODEL records and their content
    model_pattern = r'MODEL\s+(\d+)\n(.*?)(?=ENDMDL|MODEL|$)'

    for match in re.finditer(model_pattern, content, re.DOTALL):
        model_num = int(match.group(1))
        model_content = match.group(2)

        # Extract VINA RESULT information
        vina_result = None
        vina_match = re.search(
            r'REMARK VINA RESULT:\s+([\-\d.]+)\s+([\-\d.]+)\s+([\-\d.]+)',
            model_content
        )
        if vina_match:
            vina_result = {
                'affinity': float(vina_match.group(1)),
                'rmsd_lb': float(vina_match.group(2)),
                'rmsd_ub': float(vina_match.group(3))
            }

        # Extract ATOM/HETATM lines
        atoms = []
        for line in model_content.split('\n'):
            if line.startswith(('ATOM', 'HETATM')):
                atoms.append(line)

        models.append({
            'model_num': model_num,
            'vina_result': vina_result,
            'atoms': atoms,
            'atom_count': len(atoms)
        })

    return models


def _merge_receptor_ligand(receptor_path: Path, ligand_models: List[Dict]) -> List[Dict]:
    """Merge receptor PDBQT with ligand conformations.

    Args:
        receptor_path: Path to receptor PDBQT file
        ligand_models: List of ligand conformation dictionaries

    Returns:
        List of merged complex dictionaries
    """
    with open(receptor_path, 'r') as f:
        receptor_content = f.read()

    # Extract receptor ATOM lines
    receptor_atoms = []
    for line in receptor_content.split('\n'):
        if line.startswith(('ATOM', 'HETATM')):
            receptor_atoms.append(line)

    # Create merged complexes for each ligand conformation
    complexes = []
    for ligand_model in ligand_models:
        # Merge receptor and ligand atoms
        complex_atoms = receptor_atoms + ligand_model['atoms']

        # Create PDBQT content with REMARKs for metadata
        pdbqt_content = f"REMARK   4 Model {ligand_model['model_num']}\n"
        if ligand_model['vina_result']:
            pdbqt_content += f"REMARK   4 VINA_AFFINITY: {ligand_model['vina_result']['affinity']}\n"

        # Renumber atoms sequentially
        atom_num = 1
        for atom_line in complex_atoms:
            new_line = f"{atom_line[:6]}{atom_num:5d}{atom_line[11:]}"
            pdbqt_content += new_line + "\n"
            atom_num += 1

        complexes.append({
            'model_num': ligand_model['model_num'],
            'vina_result': ligand_model['vina_result'],
            'pdbqt_content': pdbqt_content,
            'atom_count': len(complex_atoms)
        })

    return complexes


def _analyze_conformation(complex_data: Dict, nohydro: bool = True) -> Dict:
    """Analyze a single docking conformation.

    Args:
        complex_data: Dictionary containing merged complex data
        nohydro: Whether to use existing hydrogens

    Returns:
        Dictionary containing analysis results
    """
    # Set NOHYDRO config (PDBQT already has hydrogens)
    config.NOHYDRO = nohydro

    # Use PDBParser to convert PDBQT to standard PDB
    pdb_parser = PDBParser(complex_data['pdbqt_content'], as_string=True)
    corrected_pdb = pdb_parser.corrected_pdb

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(corrected_pdb)
        temp_path = f.name

    try:
        # Load molecule
        mol = MoleculeComplex()
        mol.load_pdb(temp_path, as_string=False)
        mol.build_distance_matrix()

        # Detect atom properties
        props = AtomProperties(mol.atom_container)

        # Detect interactions
        detector = UnifiedInteractionDetector(mol.atom_container, props, mol.residues)
        interactions = detector.detect_all()

        # Prepare atom coordinates
        atom_coords = {}
        for atom in mol.atom_container:
            atom_coords[str(atom.idx)] = atom.coords.tolist()

        # Convert interactions to serializable format
        interactions_dict = {}
        for interaction_type, interaction_list in interactions.items():
            interactions_dict[interaction_type] = []
            for interaction in interaction_list:
                interaction_dict = {
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
                    'details': _serialize_details(interaction.details) if hasattr(interaction, 'details') else None
                }
                interactions_dict[interaction_type].append(interaction_dict)

        return {
            'atom_coords': atom_coords,
            'interactions': interactions_dict,
            'atom_count': len(mol.atom_container),
            'residue_count': len(mol.residues),
            'pdb_content': corrected_pdb  # Save PDB content for visualization
        }

    finally:
        # Clean up temporary file
        import os
        os.unlink(temp_path)


def cmd_dock(args):
    """Analyze AutoDock Vina docking results."""
    ligand_path = Path(args.ligand)
    receptor_path = Path(args.receptor)
    output_path = Path(args.output)
    nohydro = args.nohydro

    # Validate input files
    if not ligand_path.exists():
        print(f"Error: Ligand file not found: {ligand_path}")
        sys.exit(1)

    if not receptor_path.exists():
        print(f"Error: Receptor file not found: {receptor_path}")
        sys.exit(1)

    print(f"Analyzing AutoDock Vina results...")
    print(f"  Ligand: {ligand_path}")
    print(f"  Receptor: {receptor_path}")

    # Parse ligand PDBQT file
    print("\nParsing ligand conformations...")
    ligand_models = _parse_dock_pdbqt(ligand_path)
    print(f"  Found {len(ligand_models)} conformations")

    # Merge receptor with ligand conformations
    print("\nMerging receptor with ligand conformations...")
    complexes = _merge_receptor_ligand(receptor_path, ligand_models)
    print(f"  Created {len(complexes)} receptor-ligand complexes")

    # Analyze each conformation
    print("\nAnalyzing conformations...")

    # Try to import tqdm for progress bar
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    conformations = []
    iterator = tqdm(complexes, desc="Analyzing") if use_tqdm else complexes

    for complex_data in iterator:
        if not use_tqdm:
            print(f"  Conformation {complex_data['model_num']}/{len(complexes)}...", end=" ")

        # Analyze this conformation
        analysis = _analyze_conformation(complex_data, nohydro)

        # Count interactions
        total_interactions = sum(len(v) for v in analysis['interactions'].values())

        if not use_tqdm:
            print(f"{analysis['atom_count']} atoms, {total_interactions} interactions")

        # Build conformation result
        conf_result = {
            'model_num': complex_data['model_num'],
            'vina_result': complex_data['vina_result'],
            'atom_count': analysis['atom_count'],
            'residue_count': analysis['residue_count'],
            'interactions': analysis['interactions'],
            'atom_coords': analysis['atom_coords'],
            'pdb_content': analysis['pdb_content']  # Include PDB content for visualization
        }
        conformations.append(conf_result)

    # Prepare output data
    output_data = {
        'metadata': {
            'analysis_type': 'docking',
            'ligand_file': str(ligand_path),
            'receptor_file': str(receptor_path),
            'conformation_count': len(conformations),
            'nohydro': nohydro
        },
        'conformations': conformations
    }

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, cls=AllAtomJSONEncoder)

    print(f"\nResults saved to: {output_path}")
    print(f"  Total conformations: {len(conformations)}")

    # Print summary of top conformations
    print("\nTop 5 conformations by affinity:")
    for conf in conformations[:5]:
        vina = conf['vina_result']
        if vina:
            total = sum(len(v) for v in conf['interactions'].values())
            print(f"  Model {conf['model_num']}: "
                  f"Affinity = {vina['affinity']:.3f} kcal/mol, "
                  f"Interactions = {total}")


def _serialize_details(details) -> Optional[dict]:
    """Serialize details dict, handling non-serializable objects.

    Only keeps primitive types and simple structures to avoid
    circular references and non-serializable objects.
    """
    if details is None:
        return None

    serialized = {}
    for key, value in details.items():
        # Keep only primitive types
        if isinstance(value, (str, int, float, bool, type(None))):
            serialized[key] = value
        elif isinstance(value, (list, tuple)):
            # Only keep primitive items from lists
            serialized[key] = [
                item if isinstance(item, (str, int, float, bool, type(None))) else str(item)
                for item in value
            ]
        elif isinstance(value, dict):
            # Recursively serialize nested dicts, but limit depth
            nested = _serialize_details(value)
            if nested is not None:
                serialized[key] = nested
        # Skip all other types (objects, OBResidue, etc.)

    return serialized


def cmd_analyze(args):
    """Analyze PDB file and generate interaction data."""
    pdb_path = args.pdb_file
    output_path = args.output
    nohydro = args.nohydro

    print(f"Analyzing PDB file: {pdb_path}")

    # Set NOHYDRO config
    config.NOHYDRO = nohydro

    # Load molecule
    mol = MoleculeComplex()
    mol.load_pdb(pdb_path)
    mol.build_distance_matrix()

    print(f"  Loaded {len(mol.atom_container)} atoms")
    print(f"  Found {len(mol.residues)} residues")

    # Detect atom properties
    props = AtomProperties(mol.atom_container)

    # Detect interactions
    detector = UnifiedInteractionDetector(mol.atom_container, props, mol.residues)
    interactions = detector.detect_all()

    # Count interactions
    total = 0
    print("\nInteractions detected:")
    for interaction_type, interaction_list in interactions.items():
        count = len(interaction_list)
        total += count
        print(f"  {interaction_type}: {count}")
    print(f"  Total: {total}")

    # Prepare output data
    output_data = {
        'metadata': {
            'pdb_file': str(pdb_path),
            'nohydro': nohydro,
            'atom_count': len(mol.atom_container),
            'residue_count': len(mol.residues)
        },
        'atom_coords': {},  # Store atom coordinates for visualization
        'interactions': {}
    }

    # Save corrected PDB content (with hydrogens if NOHYDRO=False) for visualization
    if not nohydro and mol.hydro_pdb:
        output_data['pdb_content'] = mol.hydro_pdb
        logger.info(f"Embedded corrected PDB content in JSON ({len(mol.hydro_pdb)} chars)")

    # Store atom coordinates (needed for 3D visualization)
    # Atom indices in interactions are 1-based and correspond to processed molecule
    for atom in mol.atom_container:
        idx = atom.idx
        coords = atom.coords.tolist()  # Convert numpy array to list for JSON
        output_data['atom_coords'][str(idx)] = coords  # Use string keys for JSON

    # Convert interactions to serializable format
    for interaction_type, interaction_list in interactions.items():
        output_data['interactions'][interaction_type] = []
        for interaction in interaction_list:
            # Extract only serializable fields from Interaction namedtuple
            interaction_dict = {
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
                'details': _serialize_details(interaction.details) if hasattr(interaction, 'details') else None
            }
            output_data['interactions'][interaction_type].append(interaction_dict)

    # Save to JSON using custom encoder
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, cls=AllAtomJSONEncoder)

    print(f"\nResults saved to: {output_path}")

    # Generate plots if requested
    if args.plot_matrix or args.plot_all:
        from fplip.all_atom.visualization import (InteractionDataManager,
                                                  StaticVisualizer)

        dm = InteractionDataManager(output_data)
        viz = StaticVisualizer(dm)

        output_dir = Path(output_path).parent
        prefix = Path(output_path).stem + '_'

        if args.plot_matrix:
            viz.plot_residue_matrix(output_dir / f"{prefix}matrix.png")

        if args.plot_all:
            viz.plot_all(output_dir, prefix)


def cmd_static(args):
    """Generate static visualizations from JSON data."""
    input_path = args.input

    # Validate input file
    if not Path(input_path).exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    try:
        from fplip.all_atom.visualization import (InteractionDataManager,
                                                  StaticVisualizer)

        # Load data
        dm = InteractionDataManager(input_path)
        viz = StaticVisualizer(dm)

        figsize = tuple(args.figsize)

        # Generate requested plots
        if args.plot_matrix:
            viz.plot_residue_matrix(args.output, figsize=figsize)

        elif args.plot_distribution:
            viz.plot_interaction_type_distribution(args.output, figsize=figsize)

        elif args.plot_dist:
            viz.plot_distance_distribution(args.plot_dist, args.output, figsize=figsize)

        elif args.plot_angle:
            viz.plot_angle_distribution(args.plot_angle, args.output, figsize=figsize)

        elif args.plot_network:
            viz.plot_residue_network(args.output, figsize=figsize,
                                    min_interactions=args.min_interactions)

        elif args.plot_chains:
            viz.plot_chain_interaction_summary(args.output, figsize=figsize)

        elif args.plot_summary:
            viz.plot_residue_interaction_summary(args.output, figsize=figsize, top_n=args.top_n)

        elif args.plot_type_heatmaps:
            viz.plot_interaction_heatmap_by_type(args.output, figsize=figsize)

        elif args.plot_all:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            viz.plot_all(output_dir)

        else:
            print("Error: No plot type specified. Use --help for options.")
            sys.exit(1)

    except Exception as e:
        print(f"Error generating plot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_interactive(args):
    """Launch interactive web visualization."""
    input_path = args.input
    pdb_path = args.pdb

    print(f"Starting interactive visualization...")
    print(f"  JSON file: {input_path}")
    if pdb_path:
        print(f"  PDB file: {pdb_path}")
    else:
        print(f"  PDB file: (will try to infer from JSON metadata)")

    # Validate input file
    if not Path(input_path).exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Validate PDB file if provided
    if pdb_path and not Path(pdb_path).exists():
        print(f"Error: PDB file not found: {pdb_path}")
        sys.exit(1)

    try:
        from fplip.all_atom.visualization.web_app import create_app

        app = create_app(input_path, pdb_path)

        if args.open_browser:
            import webbrowser
            url = f"http://{args.host}:{args.port}"
            webbrowser.open(url)

        print(f"Starting interactive visualization server...")
        print(f"Open http://{args.host}:{args.port} in your browser")
        print("Press Ctrl+C to stop")

        app.run(host=args.host, port=args.port, debug=False)

    except ImportError as e:
        print(f"Error: Required dependencies not installed: {e}")
        print("Install with: pip install flask")
        sys.exit(1)

    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description='All-Atom Visualization Tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze PDB file
  %(prog)s analyze input.pdb -o results.json

  # Generate static visualization
  %(prog)s static results.json --plot-matrix -o matrix.png

  # Launch interactive web visualization
  %(prog)s interactive results.json --port 8080
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze PDB file and generate interaction data',
        description='Analyze a PDB file to detect molecular interactions and save results to JSON.'
    )
    analyze_parser.add_argument('pdb_file', help='Input PDB file path')
    analyze_parser.add_argument('-o', '--output', required=True,
                               help='Output JSON file path')
    analyze_parser.add_argument('--nohydro', action='store_true',
                               help='Use existing hydrogens (do not add with OpenBabel)')
    analyze_parser.add_argument('--plot-matrix', action='store_true',
                               help='Also generate residue matrix heatmap')
    analyze_parser.add_argument('--plot-all', action='store_true',
                               help='Generate all standard plots')
    analyze_parser.set_defaults(func=cmd_analyze)

    # Dock command (AutoDock Vina analysis)
    dock_parser = subparsers.add_parser(
        'dock',
        help='Analyze AutoDock Vina docking results',
        description='Analyze docking results from AutoDock Vina PDBQT files.'
    )
    dock_parser.add_argument('--ligand', required=True,
                            help='Ligand PDBQT file with multiple conformations')
    dock_parser.add_argument('--receptor', required=True,
                            help='Receptor PDBQT file')
    dock_parser.add_argument('-o', '--output', required=True,
                            help='Output JSON file path')
    dock_parser.add_argument('--nohydro', action='store_true',
                            help='Use existing hydrogens (do not add with OpenBabel)')
    dock_parser.set_defaults(func=cmd_dock)

    # Static command
    static_parser = subparsers.add_parser(
        'static',
        help='Generate static visualizations from JSON data',
        description='Generate various static plots from interaction data.'
    )
    static_parser.add_argument('input', help='Input JSON file from all-atom-analyze')
    static_parser.add_argument('-o', '--output', required=True,
                              help='Output file or directory path')

    # Plot options
    plot_group = static_parser.add_argument_group('plot options')
    plot_group.add_argument('--plot-matrix', action='store_true',
                           help='Generate residue matrix heatmap')
    plot_group.add_argument('--plot-distribution', action='store_true',
                           help='Generate interaction type distribution')
    plot_group.add_argument('--plot-dist', metavar='TYPE',
                           help='Generate distance distribution for interaction type')
    plot_group.add_argument('--plot-angle', metavar='TYPE',
                           help='Generate angle distribution for interaction type')
    plot_group.add_argument('--plot-network', action='store_true',
                           help='Generate residue network plot')
    plot_group.add_argument('--plot-chains', action='store_true',
                           help='Generate chain interaction summary')
    plot_group.add_argument('--plot-summary', action='store_true',
                           help='Generate top residues summary')
    plot_group.add_argument('--plot-type-heatmaps', action='store_true',
                           help='Generate heatmaps by interaction type')
    plot_group.add_argument('--plot-all', action='store_true',
                           help='Generate all standard plots')

    # Output options
    static_parser.add_argument('--figsize', nargs=2, type=int, default=[10, 8],
                              help='Figure size (width height)')
    static_parser.add_argument('--min-interactions', type=int, default=1,
                              help='Minimum interactions for network plot')
    static_parser.add_argument('--top-n', type=int, default=20,
                              help='Number of top residues for summary plot')
    static_parser.set_defaults(func=cmd_static)

    # Interactive command
    interactive_parser = subparsers.add_parser(
        'interactive',
        help='Launch interactive web visualization',
        description='Start a web server for interactive exploration of interaction data.'
    )
    interactive_parser.add_argument('input', help='Input JSON file from all-atom-analyze')
    interactive_parser.add_argument('--pdb', help='PDB file path (optional, will try to infer from JSON metadata)')
    interactive_parser.add_argument('--port', type=int, default=5000,
                                   help='Port number (default: 5000)')
    interactive_parser.add_argument('--host', default='127.0.0.1',
                                   help='Host address (default: 127.0.0.1)')
    interactive_parser.add_argument('--open-browser', action='store_true',
                                   help='Open browser automatically')
    interactive_parser.set_defaults(func=cmd_interactive)

    # Parse arguments
    args = parser.parse_args()

    # If no command specified, show help
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute the selected command
    args.func(args)


def main_analyze():
    """Entry point for all-atom-analyze command."""
    # Create argument parser for analyze only
    parser = argparse.ArgumentParser(
        description='Analyze PDB file and generate interaction data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  all-atom-analyze input.pdb -o results.json
  all-atom-analyze input.pdb -o results.json --nohydro
  all-atom-analyze input.pdb -o results.json --plot-all
        """
    )
    parser.add_argument('pdb_file', help='Input PDB file path')
    parser.add_argument('-o', '--output', required=True,
                       help='Output JSON file path')
    parser.add_argument('--nohydro', action='store_true',
                       help='Use existing hydrogens (do not add with OpenBabel)')
    parser.add_argument('--plot-matrix', action='store_true',
                       help='Also generate residue matrix heatmap')
    parser.add_argument('--plot-all', action='store_true',
                       help='Generate all standard plots')

    args = parser.parse_args()
    cmd_analyze(args)


def main_static():
    """Entry point for all-atom-static command."""
    # Create argument parser for static only
    parser = argparse.ArgumentParser(
        description='Generate static visualizations from JSON data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  all-atom-static results.json --plot-matrix -o matrix.png
  all-atom-static results.json --plot-all -o ./plots/
  all-atom-static results.json --plot-dist hbond -o hbond_dist.png
        """
    )
    parser.add_argument('input', help='Input JSON file from all-atom-analyze')
    parser.add_argument('-o', '--output', required=True,
                       help='Output file or directory path')

    # Plot options
    plot_group = parser.add_argument_group('plot options')
    plot_group.add_argument('--plot-matrix', action='store_true',
                           help='Generate residue matrix heatmap')
    plot_group.add_argument('--plot-distribution', action='store_true',
                           help='Generate interaction type distribution')
    plot_group.add_argument('--plot-dist', metavar='TYPE',
                           help='Generate distance distribution for interaction type')
    plot_group.add_argument('--plot-angle', metavar='TYPE',
                           help='Generate angle distribution for interaction type')
    plot_group.add_argument('--plot-network', action='store_true',
                           help='Generate residue network plot')
    plot_group.add_argument('--plot-chains', action='store_true',
                           help='Generate chain interaction summary')
    plot_group.add_argument('--plot-summary', action='store_true',
                           help='Generate top residues summary')
    plot_group.add_argument('--plot-type-heatmaps', action='store_true',
                           help='Generate heatmaps by interaction type')
    plot_group.add_argument('--plot-all', action='store_true',
                           help='Generate all standard plots')

    # Output options
    parser.add_argument('--figsize', nargs=2, type=int, default=[10, 8],
                       help='Figure size (width height)')
    parser.add_argument('--min-interactions', type=int, default=1,
                       help='Minimum interactions for network plot')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of top residues for summary plot')

    args = parser.parse_args()
    cmd_static(args)


def main_interactive():
    """Entry point for all-atom-interactive command."""
    # Create argument parser for interactive only
    parser = argparse.ArgumentParser(
        description='Launch interactive web visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  all-atom-interactive results.json
  all-atom-interactive results.json --port 8080
  all-atom-interactive results.json --open-browser
        """
    )
    parser.add_argument('input', help='Input JSON file from all-atom-analyze')
    parser.add_argument('--pdb', help='PDB file path (optional, will try to infer from JSON metadata)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port number (default: 5000)')
    parser.add_argument('--host', default='127.0.0.1',
                       help='Host address (default: 127.0.0.1)')
    parser.add_argument('--open-browser', action='store_true',
                       help='Open browser automatically')

    args = parser.parse_args()
    cmd_interactive(args)


if __name__ == '__main__':
    main()
