# FPLIP - Fast and Full atom Protein-Ligand Interaction Profiler

> 一个基于PLIP进行全面重构和增强的蛋白质-配体相互作用分析工具，提供性能优化的PLIP兼容模式和全新的All-Atom全原子检测模式。

[English Documentation](README.md)

---

## 概述

FPLIP在保留PLIP核心功能的基础上，进行了以下重大改进：

### 1. 性能优化
- **OpenBabel C对象属性预提取优化**：通过将OBAtom属性缓存到Python数据结构中，消除重复的Python-C边界跨越
- **向量化距离计算**：使用scipy距离矩阵替代逐对循环
- **Cython加速的PDB解析器**：直接字节数组操作，使用预编译查找表

### 2. All-Atom模块（全新）
- **全原子相互作用检测**：不区分配体/受体，检测分子复合物中所有相互作用
- **检测范围扩展**：
  - 配体-蛋白质相互作用
  - 蛋白质-蛋白质相互作用
  - 蛋白质内部相互作用
  - 配体内部相互作用（分子内氢键等）
  - 蛋白质-水、配体-水、水-水相互作用
- **基于化学拓扑的电荷检测**：统一处理蛋白质和配体中的化学基团
- **智能自过滤**：蛋白质残基过滤内部相互作用，配体保留分子内相互作用
- **信息保留策略**：被精炼过滤的氢键标记为`hbond_possible`而非删除

---

## FPLIP vs PLIP

### 检测范围对比

| 相互作用类型 | PLIP | FPLIP All-Atom |
|-------------|------|----------------|
| 配体-蛋白质 | ✓ | ✓ |
| 蛋白质内部 | ✗ | ✓ |
| 配体内部 | ✗ | ✓ |

### 化学检测改进

FPLIP All-Atom采用基于化学拓扑的检测方法，相比PLIP基于残基类型的检测更加化学合理：

1. **电荷基团检测**
   - **PLIP**：基于残基名称（如`arginine_guanidinium`）
   - **FPLIP**：基于化学结构（如`guanidinium`），同时适用于蛋白质和配体中的相同化学基团

2. **智能自过滤**
   - **蛋白质残基**：过滤同一残基内的自相互作用（无生物学意义）
   - **配体残基**：保留分子内相互作用（分子内氢键对配体构象很重要）

3. **氢键精炼策略**
   - **PLIP**：直接删除被过滤的氢键（如涉及盐桥的氢键、同一供体的较弱氢键）
   - **FPLIP**：将被盐桥过滤的氢键移至`hbond_possible`，提供完整的相互作用图景

---

## 推荐使用场景

| 场景 | 推荐模式 | 说明 |
|------|---------|------|
| 标准药物发现（配体-蛋白质） | PLIP兼容模式 | 与文献结果兼容 |
| 全面相互作用分析 | All-Atom | 检测分子复合物中所有相互作用 |
| 蛋白质-蛋白质相互作用研究 | All-Atom | 分析蛋白复合物界面 |
| 水分子网络研究 | All-Atom | 直接使用MD中的水进行水桥检测 |
| 分子动力学轨迹分析 | All-Atom + TrajectoryAnalyzer | 追踪相互作用随时间的变化 |
| 配体构象分析 | All-Atom | 检测配体内部相互作用（分子内氢键） |

### CADD中的重要性

**蛋白质内部相互作用**：
- 对理解蛋白质稳定性和变构调控至关重要
- 对分析蛋白质-蛋白质相互作用界面很重要
- 对研究配体结合时的构象变化必不可少

**配体内部相互作用**：
- 分子内氢键显著影响配体构象和结合亲和力
- 理解内部张力和构象偏好
- 对配体设计和优化很重要

---

## 安装

```bash
pip install fplip
```

---

## 快速开始

### PLIP兼容模式（配体-蛋白质相互作用）

```python
from fplip.structure.preparation import PDBComplex

mol = PDBComplex()
mol.load_pdb('structure.pdb')
mol.analyze()
interactions = mol.interaction_sets[ligand_id].all_itypes
```

### All-Atom模式（全原子相互作用检测）

```python
from fplip.all_atom import MoleculeComplex, AtomProperties, UnifiedInteractionDetector

# 加载分子
mol = MoleculeComplex()
mol.load_pdb('structure.pdb')

# 检测所有相互作用
props = AtomProperties(mol.atom_container)
detector = UnifiedInteractionDetector(mol.atom_container, props, mol.residues)
interactions = detector.detect_all(verbose=True)

# 输出包含：
# - 'hbond': 确认的氢键
# - 'hbond_possible': 被过滤的氢键（同一供体的较弱氢键、涉及盐桥的氢键）
# - 'hbond_heavy_atom': 无显式氢时的重原子氢键
# - 'hydrophobic', 'saltbridge', 'pistacking', 'pication', 'halogen', 'metal'
# - 'water_bridge': 基于氢键的严格水桥
# - 'water_bridge_possible': PLIP-style距离+角度水桥
```

### 便捷函数

```python
from fplip.all_atom import _analyze_complex

interactions, mol, props = _analyze_complex('structure.pdb')
```

### 分子动力学轨迹分析

FPLIP为MD轨迹分析提供了专门的性能优化：

```python
from fplip.all_atom.trajectory_analyzer import TrajectoryAnalyzer

analyzer = TrajectoryAnalyzer(
    tpr_file='topology.tpr',
    xtc_file='trajectory.xtc',
    gro_file='structure.gro'  # 可选，用于残基ID对齐
)

# 加载，对齐，预计算属性
analyzer.load_universe()
analyzer.load_molecule(pdb_str, as_string=True)
analyzer.align_with_mda(frame=0)
analyzer.setup_detector()
analyzer.precompute_detector_once()

# 逐帧分析
for frame_idx, interactions in analyzer.iterate_frames(start=0, stop=100, step=1):
    # 处理每帧的相互作用
    pass
```

**MD分析的关键特性**：
- **一次性初始化**：原子属性和检测器缓存只计算一次
- **快速坐标更新**：使用KD树对齐实现从MDAnalysis的快速坐标传输
- **水分子过滤**：自动过滤远距离水分子，聚焦于相关的水化层
- **高阶水桥**：检测通过多个水分子连接两个残基的水网络

### 命令行工具

```bash
# 分析PDB文件
python -m fplip.all_atom.cli analyze input.pdb -o results.json

# 生成可视化
python -m fplip.all_atom.cli static results.json --plot-matrix -o matrix.png

# 启动交互式Web可视化
python -m fplip.all_atom.cli interactive results.json --port 8080
```

---

## 依赖

- Python >= 3.8
- OpenBabel >= 3.0.0 with Python bindings
- NumPy, SciPy
- MDAnalysis（可选，用于轨迹分析）
- Cython（可选，用于性能优化）

---

## 许可证

GPLv2

---

## 引用

如果您在工作中使用FPLIP，请引用原始PLIP文献：

> Adasme,M. et al. PLIP 2021: expanding the scope of the protein-ligand interaction profiler to DNA and RNA.
> Nucl. Acids Res. (05 May 2021), gkab294. doi: 10.1093/nar/gkab294

> Salentin,S. et al. PLIP: fully automated protein-ligand interaction profiler.
> Nucl. Acids Res. (1 July 2015) 43 (W1): W443-W447. doi: 10.1093/nar/gkv315

---

## 维护者

BHM-Bob_G (bhmfly@foxmail.com)

GitHub: https://github.com/BHM-Bob/fplip
