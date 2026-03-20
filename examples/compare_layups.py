#!/usr/bin/env python3
"""
Example: Compare different multi-material layup configurations.

Analyzes several common layup patterns and compares their mechanical
properties side by side. Useful for selecting the best configuration
for a specific application.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multimaterial_3d.analysis.mechanical import analyze_layup
from multimaterial_3d.analysis.thermal import predict_warping


def main():
    layer_height = 0.2
    total_height = 4.0  # 20 layers
    material_map = {1: 'PLA', 2: 'TPU'}

    configs = {
        'Pure PLA':           ([1], {1: 'PLA'}),
        'Pure TPU':           ([2], {2: 'TPU'}),
        'Alternating 1:1':    ([1, 2], material_map),
        'PLA-heavy 3:1':      ([1, 1, 1, 2], material_map),
        'Sandwich 5/10/5':    ([1]*5 + [2]*10 + [1]*5, material_map),
        'Symmetric 1-2-2-1':  ([1, 2, 2, 1], material_map),
        'Asymmetric 1-2':     ([1, 1, 1, 2, 2], material_map),
    }

    print(f"{'Configuration':<25} {'E_voigt':<10} {'E_reuss':<10} {'sigma_t':<10} "
          f"{'rho':<8} {'E/rho':<10} {'Warp(mm)':<10} {'Symmetric'}")
    print("-" * 103)

    for name, (pattern, mat_map) in configs.items():
        result = analyze_layup(pattern, total_height, layer_height, mat_map, verbose=False)
        warp = predict_warping(pattern, layer_height, total_height, mat_map)

        sym = "Yes" if result['abd']['symmetric'] else "No"
        print(f"{name:<25} {result['E_voigt']:<10.0f} {result['E_reuss']:<10.0f} "
              f"{result['sigma_t']:<10.1f} {result['density']:<8.2f} "
              f"{result['specific_stiffness']:<10.0f} {warp['deflection']:<10.3f} {sym}")


if __name__ == '__main__':
    main()
