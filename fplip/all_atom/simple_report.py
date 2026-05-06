"""
Simple Report Module

Generates simplified text reports for all-atom interaction detection.
Format: CHAIN:RESNAME:RESNUM:ATOM_NAME
"""

from typing import Dict, List
from collections import defaultdict

from .interaction_detector import Interaction
from .residue import Residue


class SimpleReport:
    """
    Simple text report generator for interactions.
    """

    def __init__(self, pdb_id: str, residues: List[Residue]):
        self.pdb_id = pdb_id
        self.residues = residues

        # Count residue types
        self.type_counts = defaultdict(int)
        for res in residues:
            if res.is_protein:
                self.type_counts['protein'] += 1
            elif res.is_ligand:
                self.type_counts['ligand'] += 1
            elif res.is_water:
                self.type_counts['water'] += 1
            elif res.is_ion:
                self.type_counts['ion'] += 1
            elif res.is_dna:
                self.type_counts['dna'] += 1
            elif res.is_rna:
                self.type_counts['rna'] += 1

    def generate(self, interactions: Dict[str, List[Interaction]]) -> str:
        """Generate full report"""
        lines = []
        lines.append("=" * 70)
        lines.append(f"All-Atom Interaction Report: {self.pdb_id}")
        lines.append("=" * 70)
        lines.append("")

        # Summary section
        lines.append("STRUCTURE SUMMARY")
        lines.append("-" * 70)
        lines.append(f"Total residues: {len(self.residues)}")
        for comp_type, count in sorted(self.type_counts.items()):
            lines.append(f"  {comp_type}: {count}")
        lines.append("")

        # Interaction summary
        lines.append("INTERACTION SUMMARY")
        lines.append("-" * 70)
        total = 0
        for itype, inters in sorted(interactions.items()):
            if inters:
                lines.append(f"  {itype:15s}: {len(inters):4d}")
                total += len(inters)
        lines.append(f"  {'TOTAL':15s}: {total:4d}")
        lines.append("")

        # Detailed interactions
        for itype, inters in sorted(interactions.items()):
            if inters:
                lines.append(self._format_interaction_section(itype, inters))
                lines.append("")

        lines.append("=" * 70)
        lines.append("End of Report")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _format_interaction_section(self, itype: str, interactions: List[Interaction]) -> str:
        """Format a section for one interaction type"""
        lines = []
        lines.append(f"{itype.upper()} INTERACTIONS ({len(interactions)})")
        lines.append("-" * 70)

        # Group by residue pair for cleaner output
        grouped = defaultdict(list)
        for inter in interactions:
            res_a = f"{inter.res_a_chain}:{inter.res_a_name}:{inter.res_a_num}"
            res_b = f"{inter.res_b_chain}:{inter.res_b_name}:{inter.res_b_num}"
            pair = tuple(sorted([res_a, res_b]))
            grouped[pair].append(inter)

        for pair, inters in sorted(grouped.items()):
            res_a, res_b = pair
            lines.append(f"\n{res_a} <-> {res_b}")

            for inter in inters:
                atom_a = f"{inter.atom_a_name}({inter.atom_a_idx})"
                atom_b = f"{inter.atom_b_name}({inter.atom_b_idx})"

                base_info = f"  {atom_a:12s} <-> {atom_b:12s}  d={inter.distance:.2f}A"

                if inter.angle is not None:
                    base_info += f"  angle={inter.angle:.1f}°"

                lines.append(base_info)

                # Add details if present
                if inter.details:
                    detail_str = self._format_details(inter.details)
                    if detail_str:
                        lines.append(f"    [{detail_str}]")

        return "\n".join(lines)

    def _format_details(self, details: Dict) -> str:
        """Format interaction details"""
        parts = []

        # Common detail fields
        if 'type' in details:
            parts.append(f"type={details['type']}")
        if 'h_atom' in details:
            parts.append(f"H={details['h_atom']}")
        if 'dist_ah' in details:
            parts.append(f"d_AH={details['dist_ah']:.2f}A")
        if 'charge_type' in details:
            parts.append(f"charge={details['charge_type']}")
        if 'halogen_type' in details:
            parts.append(f"halogen={details['halogen_type']}")
        if 'water_residue' in details:
            parts.append(f"via={details['water_residue']}")

        return ", ".join(parts)

    def generate_csv(self, interactions: Dict[str, List[Interaction]]) -> str:
        """Generate CSV format report"""
        lines = []

        # Header
        header = [
            "type",
            "res_a_name", "res_a_chain", "res_a_num", "atom_a_name", "atom_a_idx",
            "res_b_name", "res_b_chain", "res_b_num", "atom_b_name", "atom_b_idx",
            "distance", "angle"
        ]
        lines.append(",".join(header))

        # Data rows
        for itype, inters in interactions.items():
            for inter in inters:
                row = [
                    itype,
                    inter.res_a_name, inter.res_a_chain, str(inter.res_a_num),
                    inter.atom_a_name, str(inter.atom_a_idx),
                    inter.res_b_name, inter.res_b_chain, str(inter.res_b_num),
                    inter.atom_b_name, str(inter.atom_b_idx),
                    f"{inter.distance:.3f}",
                    f"{inter.angle:.2f}" if inter.angle else ""
                ]
                lines.append(",".join(row))

        return "\n".join(lines)

    def save(self, filepath: str, interactions: Dict[str, List[Interaction]], format: str = "txt"):
        """Save report to file"""
        if format.lower() == "txt":
            content = self.generate(interactions)
        elif format.lower() == "csv":
            content = self.generate_csv(interactions)
        else:
            raise ValueError(f"Unknown format: {format}")

        with open(filepath, 'w') as f:
            f.write(content)
