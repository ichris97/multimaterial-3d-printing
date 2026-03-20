#!/usr/bin/env python3
"""
Example: Analyze a PLA-CF / TPU / PLA-CF sandwich composite layup.

This demonstrates how to use the analysis module to predict mechanical
properties, thermal stresses, and warping for a multi-material layup
without modifying any files.

A sandwich structure uses stiff skins (PLA-CF) with a flexible core (TPU)
to achieve high bending stiffness at low weight — the same principle used
in aerospace honeycomb panels.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multimaterial_3d.analysis.mechanical import analyze_layup
from multimaterial_3d.analysis.thermal import thermal_stress_analysis, predict_warping
from multimaterial_3d.analysis.optimizer import optimize_material_distribution
from multimaterial_3d.core.materials import list_materials


def main():
    print("=" * 70)
    print("  Multi-Material Sandwich Composite Analysis")
    print("=" * 70)

    # Print available materials
    print("\nAvailable materials:")
    list_materials()

    # Define sandwich layup: 5 layers PLA-CF skin, 10 layers TPU core, 5 layers PLA-CF skin
    pattern = [1]*5 + [2]*10 + [1]*5  # Symmetric sandwich
    material_map = {1: 'PLA-CF', 2: 'TPU'}
    layer_height = 0.2  # mm
    total_height = len(pattern) * layer_height  # 4mm total

    print(f"\n{'='*70}")
    print(f"  Sandwich Layup: PLA-CF skin (5 layers) / TPU core (10 layers) / PLA-CF skin (5 layers)")
    print(f"  Layer height: {layer_height}mm, Total height: {total_height}mm")
    print(f"{'='*70}")

    # 1. Mechanical analysis
    results = analyze_layup(pattern, total_height, layer_height, material_map, verbose=True)

    # 2. Thermal stress analysis
    thermal_stress_analysis(pattern, layer_height, total_height, material_map, verbose=True)

    # 3. Warping prediction for different part sizes
    print("\n  Warping Prediction for Different Part Sizes:")
    for length in [50, 100, 150, 200]:
        warp = predict_warping(pattern, layer_height, total_height, material_map,
                               part_length=length)
        print(f"    {length}mm part: deflection = {warp['deflection']:.3f}mm "
              f"(L/{length/warp['deflection']:.0f})" if warp['deflection'] > 0.001
              else f"    {length}mm part: negligible warping")

    # 4. Optimization: find minimum weight that achieves E >= 2000 MPa
    print("\n")
    optimize_material_distribution(
        available_materials={1: 'PLA-CF', 2: 'TPU'},
        total_height=total_height,
        layer_height=layer_height,
        objective='weight',
        E_min=2000,
        verbose=True,
    )


if __name__ == '__main__':
    main()
