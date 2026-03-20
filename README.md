# Multi-Material 3D Printing Toolkit

A comprehensive Python toolkit for advanced multi-material FDM/FFF 3D printing post-processing, structural analysis, and optimization. Designed primarily for **Bambu Studio** and **Bambu Lab printers** (X1, P1, A1 series with AMS).

## Overview

This toolkit enables researchers and engineers to push beyond what slicers offer out of the box for multi-material printing. It provides:

- **Layer-by-layer material assignment** with custom repeating patterns
- **Classical Laminate Theory (CLT)** analysis for predicting composite properties
- **Interlocking perimeter generation** for improved Z-direction strength
- **Topology-optimized variable-density infill** based on stress analysis
- **Wall-infill mechanical interlocking** via complementary sinusoidal geometries
- **Thermal stress and warping prediction** for multi-material interfaces
- **Material distribution optimization** for minimum weight/cost
- **Print time and cost estimation**

## Installation

```bash
# Basic installation (numpy only)
pip install -e .

# With analysis features (scipy, shapely)
pip install -e ".[analysis]"

# With visualization (matplotlib)
pip install -e ".[visualization]"

# Everything
pip install -e ".[all]"
```

## Quick Start

### 1. Layer Pattern Assignment

Assign different filaments to each layer with custom patterns:

```bash
# Alternating PLA/TPU layers
python -m multimaterial_3d.postprocessors.layer_pattern input.3mf output.3mf \
    --pattern 1,2 --layer-height 0.2 --total-height 10

# Sandwich structure: stiff skin, soft core
python -m multimaterial_3d.postprocessors.layer_pattern input.3mf output.3mf \
    --pattern F1:5,F2:10,F1:5 --analyze --materials "1:PLA-CF,2:TPU"

# Pattern notation: F1:3,F2:2 = 3 layers of filament 1, then 2 layers of filament 2
```

### 2. Mechanical Analysis (No Printing Required)

Predict composite properties without modifying any files:

```python
from multimaterial_3d.analysis.mechanical import analyze_layup
from multimaterial_3d.analysis.thermal import thermal_stress_analysis

# Sandwich composite: PLA-CF skin / TPU core / PLA-CF skin
pattern = [1]*5 + [2]*10 + [1]*5
material_map = {1: 'PLA-CF', 2: 'TPU'}

results = analyze_layup(pattern, total_height=4.0, layer_height=0.2,
                        material_map=material_map)
# -> E_voigt, E_reuss, ABD matrix, neutral axis, flexural rigidity, etc.

thermal = thermal_stress_analysis(pattern, layer_height=0.2, total_height=4.0,
                                   material_map=material_map)
# -> interfacial stresses, warping prediction, risk assessment
```

### 3. Interlocking Perimeters

Improve Z-strength by offsetting alternating wall loops:

```bash
python -m multimaterial_3d.postprocessors.interlocking_perimeters input.3mf output.3mf
python -m multimaterial_3d.postprocessors.interlocking_perimeters input.3mf output.3mf --offset 0.3 --walls 1,3
```

### 4. Topology-Optimized Infill

Automatically increase infill density in high-stress regions:

```bash
python -m multimaterial_3d.postprocessors.topology_infill model.3mf optimized.3mf
python -m multimaterial_3d.postprocessors.topology_infill model.3mf optimized.3mf \
    --min-density 10 --max-density 100 --sensitivity 0.8 --visualize
```

### 5. Wall-Infill Interlocking

Create complementary teeth patterns between walls and infill:

```bash
python -m multimaterial_3d.postprocessors.wall_infill_interlock model.3mf output.3mf
python -m multimaterial_3d.postprocessors.wall_infill_interlock model.3mf output.3mf \
    --teeth-depth 0.5 --teeth-pitch 1.5
```

## Architecture

```
src/multimaterial_3d/
    core/
        materials.py         # Material database (12 filaments, full property sets)
        file_io.py           # 3MF/G-code read/write/repack utilities
    analysis/
        mechanical.py        # CLT, ABD matrix, Hashin-Shtrikman, Tsai-Wu failure
        thermal.py           # Timoshenko/Suhir thermal stress, CLT warping
        optimizer.py         # Weight/cost optimization, gradient transitions
        print_estimator.py   # Print time/cost estimation from G-code
        adaptive_layers.py   # Geometry-based variable layer height
    postprocessors/
        layer_pattern.py     # Per-layer filament assignment in 3MF
        interlocking_perimeters.py  # Z-offset wall interlock
        topology_infill.py   # Stress-based variable infill density
        wall_infill_interlock.py    # Sinusoidal teeth interlocking
    utils/
        gcode_parser.py      # G-code parsing and feature detection
samples/
    inputs/                  # Source 3MF models for testing
    outputs/                 # Example post-processed results
archive/                     # Original standalone scripts (pre-refactor)
```

## Material Database

The toolkit includes a comprehensive material database with 12 common FDM filaments:

| Material | E (MPa) | sigma_t (MPa) | Density (g/cm3) | CTE (1/K) | Category |
|----------|---------|----------------|-----------------|-----------|----------|
| PLA      | 3500    | 50             | 1.24            | 68e-6     | rigid |
| PETG     | 2100    | 45             | 1.27            | 60e-6     | rigid |
| ABS      | 2300    | 40             | 1.04            | 90e-6     | rigid |
| TPU 95A  | 26      | 30             | 1.21            | 150e-6    | flexible |
| ASA      | 2200    | 42             | 1.07            | 95e-6     | rigid |
| PA/Nylon | 1800    | 70             | 1.14            | 80e-6     | engineering |
| PC       | 2400    | 60             | 1.20            | 65e-6     | engineering |
| PLA-CF   | 8000    | 65             | 1.30            | 30e-6     | composite |
| PETG-CF  | 6000    | 55             | 1.35            | 35e-6     | composite |
| PA-CF    | 9500    | 80             | 1.25            | 25e-6     | composite |
| PVA      | 1500    | 30             | 1.23            | 85e-6     | support |
| HIPS     | 2000    | 25             | 1.04            | 80e-6     | support |

Each material includes: Young's modulus (XY and Z), tensile and compressive strength, inter-layer bond strength, shear modulus, Poisson's ratio, CTE, glass transition temperature, print/bed temperatures, thermal conductivity, and cost per kg.

The database also includes an **inter-material adhesion compatibility matrix** for common material combinations.

## Analysis Features

### Classical Laminate Theory (CLT)

Computes the full **ABD stiffness matrix** for the multi-material layup:

- **A matrix**: Extensional stiffness (in-plane loads)
- **B matrix**: Extension-bending coupling (nonzero for asymmetric layups)
- **D matrix**: Bending stiffness (flexural rigidity)

Symmetric layups (e.g., PLA/TPU/PLA) have B=0, meaning in-plane loads don't cause bending. Asymmetric layups (e.g., PLA on bottom, TPU on top) have B≠0 and will warp under load or thermal changes.

### Thermal Stress Analysis

Predicts residual stresses at multi-material interfaces using:

- **Timoshenko bimetallic strip theory** for warping curvature
- **CTE mismatch analysis** at each material interface
- **Interfacial shear and normal stress** calculation
- **Delamination risk assessment** (comparing thermal stress to bond strength)

### Material Distribution Optimization

Finds the optimal layer pattern to minimize weight or cost while meeting stiffness constraints:

- Continuous relaxation for optimal volume fractions
- Dithering algorithm for discrete layer assignment
- Gradient transition profiles (linear, sigmoid, quadratic, sqrt)

## How It Works (Technical Details)

### 3MF File Format

A .3mf file is a ZIP archive containing the 3D model, slicer settings, and (after slicing) G-code. This toolkit modifies:

- `Metadata/layer_config_ranges.xml` — for per-layer extruder assignment
- `Metadata/plate_N.gcode` — for G-code post-processing
- `Metadata/plate_N.gcode.md5` — checksum update after modification

### G-code Post-Processing

The post-processors parse Bambu Studio's annotated G-code comments (`;FEATURE:`, `;CHANGE_LAYER`, `;Z_HEIGHT:`) to identify wall loops, infill sections, and layer boundaries. Modifications are applied to specific sections while preserving all other G-code unchanged.

## Examples

Run the example scripts to see the analysis in action:

```bash
# Sandwich composite analysis
python examples/analyze_sandwich.py

# Compare different layup configurations
python examples/compare_layups.py
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Requirements

- Python >= 3.9
- numpy >= 1.21
- scipy >= 1.7 (optional, for advanced stress analysis)
- shapely >= 2.0 (optional, for geometry analysis)
- matplotlib >= 3.5 (optional, for visualization)
- trimesh >= 3.20 (optional, for 3D mesh processing)

## Contributing

Contributions are welcome. Areas of particular interest:

- Additional material property data (especially from experimental testing)
- Support for other slicers (PrusaSlicer, Cura, OrcaSlicer)
- FEA integration for more accurate stress prediction
- Experimental validation of interlocking effectiveness

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*This project was developed with assistance from [Claude](https://claude.ai) by Anthropic.*
