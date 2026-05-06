# FPLIP - Fast and Full atom Protein-Ligand Interaction Profiler

> A comprehensively restructured and enhanced protein-ligand interaction analysis tool based on PLIP, providing a performance-optimized PLIP-compatible mode and a brand new All-Atom full-atom detection mode.

[中文文档](README_CN.md)

---

## Overview

FPLIP builds upon PLIP's core functionality with the following major improvements:

### 1. Performance Optimization
- **OpenBabel C-object property pre-extraction optimization**: Eliminates repeated Python-C boundary crossing by caching OBAtom properties in Python data structures
- **Vectorized distance calculation**: Uses scipy distance matrices and numpy array operations instead of pairwise loops
- **Cython-accelerated PDB parser**: Direct byte array manipulation with pre-compiled lookup tables
- **Cuda-accelerated interaction detection**: in-development

### 2. All-Atom Module (New)
- **Full-atom interaction detection**: No distinction between ligand/receptor; detects all interactions in molecular complexes
- **Expanded detection scope**:
  - Ligand-protein interactions
  - Protein-protein interactions
  - Intra-protein interactions
  - Intra-ligand interactions (intramolecular H-bonds, etc.)
  - Protein-water, ligand-water, water-water interactions
  - Reamin competitive with DNA/RNA detection
- **Chemistry-topology-based charge detection**: Unified handling of chemical groups in both proteins and ligands instead of residue-type-based detection in PLIP
- **Hydrogen bond-based water bridge detection**: Detects water-water interactions based on hydrogen bonds
- **Smart self-filtering**: Protein residues filter internal interactions; ligands retain intramolecular interactions
- **Information preservation strategy**: H-bonds filtered by salt bridges are marked as `hbond_possible` instead of being deleted

---

## FPLIP vs PLIP

### Detection Scope Comparison

| Interaction Type | PLIP | FPLIP All-Atom |
|-----------------|------|----------------|
| Ligand-Protein | ✓ | ✓ |
| Intra-Protein | fragement based | residue-residue based |
| Intra-Ligand | ✗ | ✓ |

### Chemical Detection Improvements

FPLIP All-Atom uses chemistry-topology-based detection, which is more chemically sound than PLIP's residue-type-based detection:

1. **Charge Group Detection**
   - **PLIP**: Based on residue names (e.g., `arginine_guanidinium`)
   - **FPLIP**: Based on chemical structure (e.g., `guanidinium`), applicable to both proteins and ligands

2. **Smart Self-Filtering**
   - **Protein residues**: Filter self-interactions within the same residue (no biological meaning)
   - **Ligand residues**: Retain intramolecular interactions (intramolecular H-bonds are important for ligand conformation)

3. **Hydrogen Bond Refinement Strategy**
   - **PLIP**: Directly deletes filtered H-bonds (e.g., H-bonds involved in salt bridges, weaker H-bonds from the same donor)
   - **FPLIP**: Moves filtered H-bonds to `hbond_possible`, providing a complete interaction landscape

---

## Recommended Use Cases

| Scenario | Recommended Mode | Description |
|----------|-----------------|-------------|
| Standard drug discovery (ligand-protein) | PLIP-compatible | Compatible with literature results |
| Comprehensive interaction analysis | All-Atom | Detect all interactions in molecular complexes |
| Protein-protein interaction research | All-Atom | Analyze protein complex interfaces |
| Water network research | All-Atom | Direct use of water from MD for water bridge detection |
| MD trajectory analysis | All-Atom + TrajectoryAnalyzer | Track interaction changes over time |
| Ligand conformation analysis | All-Atom | Detect intramolecular interactions (intramolecular H-bonds) |

### Importance in CADD

**Intra-Protein Interactions**:
- Critical for understanding protein stability and allosteric regulation
- Important for analyzing protein-protein interaction interfaces
- Essential for studying conformational changes upon ligand binding

**Intra-Ligand Interactions**:
- Intramolecular H-bonds significantly affect ligand conformation and binding affinity
- Understanding internal strain and conformational preferences
- Important for ligand design and optimization

---

## Installation

```bash
pip install fplip
```

---

## Quick Start

### PLIP-Compatible Mode (Ligand-Protein Interactions)

```python
from fplip.structure.preparation import PDBComplex

mol = PDBComplex()
mol.load_pdb('structure.pdb')
mol.analyze()
interactions = mol.interaction_sets[ligand_id].all_itypes
```

### All-Atom Mode (Full-Atom Interaction Detection)

```python
from fplip.all_atom import MoleculeComplex, AtomProperties, UnifiedInteractionDetector

# Load molecule
mol = MoleculeComplex()
mol.load_pdb('structure.pdb')

# Detect all interactions
props = AtomProperties(mol.atom_container)
detector = UnifiedInteractionDetector(mol.atom_container, props, mol.residues)
interactions = detector.detect_all(verbose=True)

# Output includes:
# - 'hbond': Confirmed H-bonds
# - 'hbond_possible': Filtered H-bonds (weaker H-bonds from same donor, H-bonds involved in salt bridges)
# - 'hbond_heavy_atom': Heavy-atom H-bonds when no explicit hydrogens
# - 'hydrophobic', 'saltbridge', 'pistacking', 'pication', 'halogen', 'metal'
# - 'water_bridge': Strict H-bond-based water bridges
# - 'water_bridge_possible': PLIP-style distance+angle water bridges
```

### Convenience Function

```python
from fplip.all_atom import _analyze_complex

interactions, mol, props = _analyze_complex('structure.pdb')
```

### MD Trajectory Analysis

FPLIP provides specialized performance optimizations for MD trajectory analysis:

```python
from fplip.all_atom.trajectory_analyzer import TrajectoryAnalyzer

analyzer = TrajectoryAnalyzer(
    tpr_file='topology.tpr',
    xtc_file='trajectory.xtc',
    gro_file='structure.gro'  # Optional, for residue ID alignment
)

# Load and align
analyzer.load_universe()
analyzer.load_molecule(pdb_str, as_string=True)
analyzer.align_with_mda(frame=0)
analyzer.setup_detector()
analyzer.precompute_detector_once()

# Analyze frame by frame
for frame_idx, interactions in analyzer.iterate_frames(start=0, stop=100, step=1):
    # Process interactions for each frame
    pass
```

**Key Features for MD Analysis**:
- **One-time initialization**: Atom properties and detector caches are computed once
- **Fast coordinate updates**: Uses KD-tree alignment for rapid coordinate transfer from MDAnalysis
- **Water filtering**: Automatically filters distant water molecules to focus on relevant hydration shells
- **High-order water bridges**: Detects water networks connecting two residues through multiple water molecules

### Command Line Tools

```bash
# Analyze PDB file
python -m fplip.all_atom.cli analyze input.pdb -o results.json

# Generate visualizations
python -m fplip.all_atom.cli static results.json --plot-matrix -o matrix.png

# Launch interactive web visualization
python -m fplip.all_atom.cli interactive results.json --port 8080
```

---

## Dependencies

- Python >= 3.8
- OpenBabel >= 3.0.0 with Python bindings
- NumPy, SciPy
- MDAnalysis (optional, for trajectory analysis)
- Cython (optional, for performance optimization)

---

## License

GPLv2

---

## Citation

If you use FPLIP in your work, please cite the original PLIP publications:

> Adasme,M. et al. PLIP 2021: expanding the scope of the protein-ligand interaction profiler to DNA and RNA.
> Nucl. Acids Res. (05 May 2021), gkab294. doi: 10.1093/nar/gkab294

> Salentin,S. et al. PLIP: fully automated protein-ligand interaction profiler.
> Nucl. Acids Res. (1 July 2015) 43 (W1): W443-W447. doi: 10.1093/nar/gkv315

---

## Maintainer

BHM-Bob_G (bhmfly@foxmail.com)

GitHub: https://github.com/BHM-Bob/fplip
