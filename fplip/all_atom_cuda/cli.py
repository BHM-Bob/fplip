"""
All-Atom CUDA Command Line Interface

Provides GPU-accelerated command-line tools for:
1. Analyzing PDB files and generating interaction data
2. Analyzing AutoDock Vina docking results
3. Analyzing MD trajectories
4. Analyzing MD trajectories with multiple worker processes

Usage:
    python -m fplip.all_atom_cuda.cli analyze input.pdb -o results.json
    python -m fplip.all_atom_cuda.cli dock --ligand dock.pdbqt --receptor receptor.pdbqt -o dock_results.json
    python -m fplip.all_atom_cuda.cli analyze-traj pull.tpr pull_center.xtc -g pull.gro -o traj.tar
    python -m fplip.all_atom_cuda.cli analyze-traj-parallel pull.tpr pull_center.xtc -g pull.gro -j 4 -o traj.tar
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from fplip.all_atom.atom_properties import AtomProperties
from fplip.all_atom.cli import (AllAtomJSONEncoder, _merge_receptor_ligand,
                                _parse_dock_pdbqt, _serialize_details)
from fplip.all_atom.molecule_complex import MoleculeComplex
from fplip.all_atom_cuda.backend import ComputeBackend
from fplip.all_atom_cuda.cuda_detector import CudaInteractionDetector
from fplip.basic import config
from fplip.structure.pdb import PDBParser


def _build_backend(name: str, device: Optional[str] = None) -> ComputeBackend:
    """Create a compute backend from the CLI arguments.

    Args:
        name: Backend name, one of 'numpy', 'torch', 'cupy'.
        device: Backend-specific device. For torch this is a device string
            (e.g. 'cuda:0', 'cpu'); for cupy it is an integer device id.

    Returns:
        Instantiated compute backend.
    """
    if name == 'numpy':
        from fplip.all_atom_cuda.numpy_backend import NumPyBackend
        backend: ComputeBackend = NumPyBackend()
    elif name == 'torch':
        from fplip.all_atom_cuda.torch_backend import TorchBackend
        backend = TorchBackend(device or 'cuda:0')
    else:
        from fplip.all_atom_cuda.cupy_backend import CuPyBackend
        backend = CuPyBackend(int(device) if device is not None else 0)

    if backend.is_gpu:
        print(f'Compute backend: {backend.name} (GPU)')
    else:
        print(f'Compute backend: {backend.name} (CPU)')
    return backend


def _interactions_to_json(interactions: Dict[str, List]) -> Dict[str, List[Dict[str, Any]]]:
    """Convert detected interactions into a JSON-serializable dictionary."""
    payload: Dict[str, List[Dict[str, Any]]] = {}
    for interaction_type, interaction_list in interactions.items():
        payload[interaction_type] = []
        for interaction in interaction_list:
            payload[interaction_type].append({
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
                'details': _serialize_details(interaction.details)
            })
    return payload


def _atom_coords(mol: MoleculeComplex) -> Dict[str, List[float]]:
    """Collect atom coordinates keyed by 1-based atom index."""
    coords = {}
    for atom in mol.atom_container:
        coords[str(atom.idx)] = atom.coords.tolist()
    return coords


def _analyze_conformation(complex_data: Dict, nohydro: bool,
                           backend: ComputeBackend, verbose: bool = False) -> Dict:
    """Analyze a single docking conformation.

    Args:
        complex_data: Dictionary containing merged complex data.
        nohydro: Whether to use existing hydrogens.
        backend: Compute backend shared by all conformations.
        verbose: Whether to print detector progress.

    Returns:
        Dictionary containing analysis results.
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
        detector = CudaInteractionDetector(mol.atom_container, props,
                                           mol.residues, backend=backend)
        interactions = detector.detect_all(verbose=verbose)

        return {
            'atom_coords': _atom_coords(mol),
            'interactions': _interactions_to_json(interactions),
            'atom_count': len(mol.atom_container),
            'residue_count': len(mol.residues),
            'pdb_content': corrected_pdb
        }

    finally:
        # Clean up temporary file
        os.unlink(temp_path)


def cmd_analyze(args):
    """Analyze PDB file and generate interaction data."""
    pdb_path = Path(args.pdb_file)
    output_path = Path(args.output)
    nohydro = args.nohydro

    if not pdb_path.exists():
        print(f"Error: PDB file not found: {pdb_path}")
        sys.exit(1)

    backend = _build_backend(args.backend, args.device)
    try:
        # Set NOHYDRO config
        config.NOHYDRO = nohydro

        print(f"Analyzing PDB file: {pdb_path}")

        # Load molecule
        mol = MoleculeComplex()
        mol.load_pdb(pdb_path)
        mol.build_distance_matrix()

        print(f"  Loaded {len(mol.atom_container)} atoms")
        print(f"  Found {len(mol.residues)} residues")

        # Detect atom properties
        props = AtomProperties(mol.atom_container)

        # Detect interactions
        detector = CudaInteractionDetector(mol.atom_container, props,
                                           mol.residues, backend=backend)
        interactions = detector.detect_all(verbose=args.verbose)

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
                'backend': backend.name,
                'device': args.device,
                'atom_count': len(mol.atom_container),
                'residue_count': len(mol.residues)
            },
            'atom_coords': _atom_coords(mol),
            'interactions': _interactions_to_json(interactions)
        }

        # Save corrected PDB content for visualization
        if not nohydro and mol.hydro_pdb:
            output_data['pdb_content'] = mol.hydro_pdb
            print(f"  Embedded corrected PDB content in JSON ({len(mol.hydro_pdb)} chars)")

        # Save to JSON using custom encoder
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, cls=AllAtomJSONEncoder)

        print(f"\nResults saved to: {output_path}")

    finally:
        backend.free_mem()


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

    backend = _build_backend(args.backend, args.device)
    try:
        print("Analyzing AutoDock Vina results...")
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

        conformations = []
        for complex_data in tqdm(complexes, desc="Analyzing"):
            # Analyze this conformation
            analysis = _analyze_conformation(complex_data, nohydro, backend,
                                             verbose=args.verbose)

            # Build conformation result
            conf_result = {
                'model_num': complex_data['model_num'],
                'vina_result': complex_data['vina_result'],
                'atom_count': analysis['atom_count'],
                'residue_count': analysis['residue_count'],
                'interactions': analysis['interactions'],
                'atom_coords': analysis['atom_coords'],
                'pdb_content': analysis['pdb_content']
            }
            conformations.append(conf_result)

        # Prepare output data
        output_data = {
            'metadata': {
                'analysis_type': 'docking',
                'ligand_file': str(ligand_path),
                'receptor_file': str(receptor_path),
                'conformation_count': len(conformations),
                'nohydro': nohydro,
                'backend': backend.name,
                'device': args.device
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

    finally:
        backend.free_mem()


def cmd_analyze_traj(args):
    """Analyze an MD trajectory and export interaction records."""
    from lazydock.gmx.mda.convert import PDBConverter
    from lazydock.gmx.mda.utils import filter_atoms_by_chains

    from fplip.all_atom_cuda.trajectory_analyzer import TrajectoryAnalyzer

    tpr_path = Path(args.tpr)
    xtc_path = Path(args.xtc)

    for path, label in ((tpr_path, 'TPR'), (xtc_path, 'XTC')):
        if not path.exists():
            print(f"Error: {label} file not found: {path}")
            sys.exit(1)
    if args.gro is not None and not Path(args.gro).exists():
        print(f"Error: GRO file not found: {args.gro}")
        sys.exit(1)

    backend = _build_backend(args.backend, args.device)
    try:
        print("Analyzing trajectory...")
        print(f"  TPR: {tpr_path}")
        print(f"  XTC: {xtc_path}")
        print(f"  GRO: {args.gro if args.gro else '(none, using TPR residue ids)'}")

        analyzer = TrajectoryAnalyzer(str(tpr_path), str(xtc_path), args.gro,
                                      tolerance=args.tolerance)
        u = analyzer.load_universe()
        n_frames = len(u.trajectory)
        print(f"  Frames in trajectory: {n_frames}")

        # Resolve which chains go into the OpenBabel side of the complex
        available_chains = list(dict.fromkeys(u.atoms.chainIDs))
        if args.chains:
            chains = args.chains
        else:
            excluded = {args.water_chain} if args.water_chain else set()
            chains = [c for c in available_chains if c not in excluded]
        chains = [c for c in chains if c in available_chains]
        if not chains:
            print(f"Error: no valid chains selected. Available: {available_chains}")
            sys.exit(1)

        print(f"  Chains: {' '.join(chains)}")
        converter = PDBConverter(filter_atoms_by_chains(u.atoms, chains), reindex=False)
        analyzer.load_molecule(converter.fast_convert(), as_string=True)
        print(f"  Loaded {len(analyzer.mol.atom_container)} atoms")

        analyzer.align_with_mda(frame=args.align_frame)
        stats = analyzer.get_alignment_stats()
        print(f"  Alignment: {stats['matched_atoms']}/{stats['total_atoms']} atoms matched")
        if stats['unmatched_atoms'] > 0:
            print(f"  Warning: {stats['unmatched_atoms']} atoms could not be matched "
                  f"at frame {args.align_frame} (tolerance={args.tolerance})")

        if args.water_chain:
            analyzer.load_waters(args.water_chain)
            print(f"  Loaded {len(analyzer.water_atoms)} water atoms")

        analyzer.setup_detector(backend=backend)
        analyzer.precompute_detector_once()

        if args.dry_run:
            print("\nDry run: skipping frame iteration.")
            return

        start = args.start
        stop = n_frames if args.stop is None else min(args.stop, n_frames)
        if stop <= start:
            print(f"Error: empty frame range (start={start}, stop={stop}).")
            sys.exit(1)

        filter_waters = None if args.no_filter_waters else args.filter_waters
        frame_indices = range(start, stop, args.step)
        n_selected = len(frame_indices)
        print(f"\nProcessing frames {start}..{stop} step {args.step} ({n_selected} frames)")

        total_interactions = 0
        for frame_idx in tqdm(frame_indices, desc="Processing frames"):
            interactions = analyzer.detect_frame_fast(
                frame_idx,
                filter_waters=filter_waters,
                verbose=args.verbose,
                detect_water_bridges_plip_style=args.plip_style_water_bridges
            )
            total_interactions += sum(len(v) for v in interactions.values())
            analyzer.add_record(interactions, frame_idx)

        analyzer.save_records(args.output)
        print(f"\nResults saved to: {args.output}")
        print(f"  Frames processed: {n_selected}")
        print(f"  Interactions detected: {total_interactions}")
        print(f"  Records written: {len(analyzer.df)}")
        for inter_type, detail_pack in analyzer.detail_packs.items():
            print(f"    detail_{inter_type}: {len(detail_pack['idx'])}")

    finally:
        backend.free_mem()


def cmd_analyze_traj_parallel(args):
    """Analyze an MD trajectory with multiple worker processes."""
    from lazydock.gmx.mda.convert import PDBConverter
    from lazydock.gmx.mda.utils import filter_atoms_by_chains

    from fplip.all_atom_cuda.trajectory_analyzer import TrajectoryParallelAnalyzer

    tpr_path = Path(args.tpr)
    xtc_path = Path(args.xtc)

    for path, label in ((tpr_path, 'TPR'), (xtc_path, 'XTC')):
        if not path.exists():
            print(f"Error: {label} file not found: {path}")
            sys.exit(1)
    if args.gro is not None and not Path(args.gro).exists():
        print(f"Error: GRO file not found: {args.gro}")
        sys.exit(1)

    backend = _build_backend(args.backend, args.device)
    try:
        print("Analyzing trajectory in parallel...")
        print(f"  TPR: {tpr_path}")
        print(f"  XTC: {xtc_path}")
        print(f"  GRO: {args.gro if args.gro else '(none, using TPR residue ids)'}")
        print(f"  Workers: {args.workers}")

        analyzer = TrajectoryParallelAnalyzer(args.workers, str(tpr_path), str(xtc_path),
                                              args.gro, tolerance=args.tolerance)
        analyzer.load_universe()
        u = analyzer.u
        n_frames = len(u.trajectory)
        print(f"  Frames in trajectory: {n_frames}")

        # Resolve which chains go into the OpenBabel side of the complex
        available_chains = list(dict.fromkeys(u.atoms.chainIDs))
        if args.chains:
            chains = args.chains
        else:
            excluded = {args.water_chain} if args.water_chain else set()
            chains = [c for c in available_chains if c not in excluded]
        chains = [c for c in chains if c in available_chains]
        if not chains:
            print(f"Error: no valid chains selected. Available: {available_chains}")
            sys.exit(1)

        print(f"  Chains: {' '.join(chains)}")
        converter = PDBConverter(filter_atoms_by_chains(u.atoms, chains), reindex=False)
        analyzer.load_molecule(converter.fast_convert(), as_string=True)
        analyzer.align_with_mda(frame=args.align_frame)
        if args.water_chain:
            analyzer.load_waters(args.water_chain)
        analyzer.setup_detector(backend=backend)
        analyzer.precompute_detector_once()

        if args.dry_run:
            print("\nDry run: inputs validated, skipping frame iteration.")
            return

        start = args.start
        stop = n_frames if args.stop is None else min(args.stop, n_frames)
        if stop <= start:
            print(f"Error: empty frame range (start={start}, stop={stop}).")
            sys.exit(1)

        # Snapshot the alignment frame, then replace worker universes with
        # picklable FakeUniverse objects before spawning processes.
        analyzer.u.trajectory[args.align_frame]
        analyzer.transfer_unverse()

        filter_waters = None if args.no_filter_waters else args.filter_waters
        n_selected = len(range(start, stop, args.step))
        print(f"\nProcessing frames {start}..{stop} step {args.step} ({n_selected} frames)")

        total_interactions = 0
        for frame_idx, interactions in analyzer.iterate_frames_parallel(
                start=start, stop=stop, step=args.step,
                filter_waters=filter_waters,
                detect_water_bridges_plip_style=args.plip_style_water_bridges,
                verbose=args.verbose):
            total_interactions += sum(len(v) for v in interactions.values())
            analyzer.add_record(interactions, frame_idx)

        analyzer.save_records(args.output)
        print(f"\nResults saved to: {args.output}")
        print(f"  Frames processed: {n_selected}")
        print(f"  Interactions detected: {total_interactions}")
        print(f"  Records written: {len(analyzer.df)}")
        for inter_type, detail_pack in analyzer.detail_packs.items():
            print(f"    detail_{inter_type}: {len(detail_pack['idx'])}")

    finally:
        backend.free_mem()


def _add_backend_args(parser: argparse.ArgumentParser):
    """Add shared compute-backend arguments."""
    parser.add_argument('--backend', default='numpy',
                        choices=['numpy', 'torch', 'cupy'],
                        help='Compute backend (default: numpy)')
    parser.add_argument('--device', default=None,
                        help='Compute device: torch device string (e.g. cuda:0, cpu) '
                             'or integer CuPy device id (e.g. 0)')


def _add_traj_args(parser: argparse.ArgumentParser):
    """Add shared trajectory-analysis arguments."""
    parser.add_argument('tpr', help='GROMACS TPR file (topology)')
    parser.add_argument('xtc', help='GROMACS XTC file (trajectory)')
    parser.add_argument('-g', '--gro', default=None,
                        help='GROMACS GRO file (preferred residue ids)')
    parser.add_argument('-o', '--output', default='fplip.tar',
                        help='Output TAR archive path (default: fplip.tar)')
    parser.add_argument('--chains', nargs='+', default=None,
                        help='Chains included in detection. Default: all '
                             'chains except --water-chain.')
    parser.add_argument('--water-chain', default=None,
                        help='Chain ID of water residues (e.g. SOL). '
                             'Waters are only loaded when this is given.')
    parser.add_argument('--start', type=int, default=0,
                        help='First frame index (default: 0)')
    parser.add_argument('--stop', type=int, default=None,
                        help='Stop frame index, exclusive (default: all frames)')
    parser.add_argument('--step', type=int, default=1,
                        help='Frame step (default: 1)')
    parser.add_argument('--align-frame', type=int, default=0,
                        help='Frame used to align OpenBabel atoms to MDA '
                             '(default: 0)')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='KD-tree matching tolerance in Angstrom '
                             '(default: 1e-4)')
    parser.add_argument('--filter-waters', type=float, default=5.0,
                        help='Distance threshold for filtering distant '
                             'waters (default: 5.0)')
    parser.add_argument('--no-filter-waters', action='store_true',
                        help='Disable water filtering entirely')
    parser.add_argument('--plip-style-water-bridges', action='store_true',
                        help='Also detect PLIP-style water bridges '
                             '(stored as water_bridge_possible)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show per-frame and per-residue progress bars')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only set up the system, do not process frames')


def main():
    """Main entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description='All-Atom CUDA Visualization Tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze PDB file
  %(prog)s analyze input.pdb -o results.json

  # Analyze AutoDock Vina docking results
  %(prog)s dock --ligand dock.pdbqt --receptor receptor.pdbqt -o dock_results.json

  # Analyze MD trajectory
  %(prog)s analyze-traj pull.tpr pull_center.xtc -g pull.gro --water-chain SOL -o traj.tar

  # Analyze MD trajectory in parallel
  %(prog)s analyze-traj-parallel pull.tpr pull_center.xtc -g pull.gro -j 4 -o traj.tar
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
    analyze_parser.add_argument('-v', '--verbose', action='store_true',
                                help='Show detector progress')
    _add_backend_args(analyze_parser)
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
    dock_parser.add_argument('-v', '--verbose', action='store_true',
                             help='Show detector progress')
    _add_backend_args(dock_parser)
    dock_parser.set_defaults(func=cmd_dock)

    # Analyze-traj command (MD trajectory)
    traj_parser = subparsers.add_parser(
        'analyze-traj',
        help='Analyze MD trajectory and export interaction records',
        description='Run interaction detection over a GROMACS trajectory and '
                    'export the records to a Snappy-compressed Parquet TAR archive.'
    )
    _add_traj_args(traj_parser)
    _add_backend_args(traj_parser)
    traj_parser.set_defaults(func=cmd_analyze_traj)

    # Analyze-traj-parallel command (MD trajectory, multiple processes)
    parallel_parser = subparsers.add_parser(
        'analyze-traj-parallel',
        help='Analyze MD trajectory with multiple worker processes',
        description='Run interaction detection over a GROMACS trajectory using '
                    'multiple worker processes and export the records to a '
                    'Snappy-compressed Parquet TAR archive.'
    )
    _add_traj_args(parallel_parser)
    parallel_parser.add_argument('-j', '--workers', type=int,
                                 default=min(4, os.cpu_count() or 1),
                                 help='Number of worker processes '
                                      '(default: min(4, CPU count))')
    _add_backend_args(parallel_parser)
    parallel_parser.set_defaults(func=cmd_analyze_traj_parallel)

    # Parse arguments
    args = parser.parse_args()

    # If no command specified, show help
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute the selected command
    args.func(args)


if __name__ == '__main__':
    main()
