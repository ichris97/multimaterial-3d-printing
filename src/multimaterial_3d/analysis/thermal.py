"""
Thermal stress and warping analysis for multi-material FDM prints.

When two materials with different coefficients of thermal expansion (CTE) are
bonded together and the assembly cools from print temperature to room temperature,
thermal stresses develop at the interface. These stresses can cause:

1. **Interfacial shear stress** — Drives delamination at material boundaries
2. **Residual bending** — Asymmetric layups warp like a bimetallic strip
3. **Microcracking** — Excessive tensile stress in the brittle material

Physics
-------
During printing, each layer is deposited at its print temperature and bonds
to the previous layer. As the assembly cools through the glass transition
temperature (Tg), thermal stresses begin to accumulate. The effective
temperature drop for stress calculation is:

    ΔT = T_stress_free - T_room

where T_stress_free ≈ min(Tg_material_1, Tg_material_2), because above Tg
the polymer can relax and no stress accumulates.

Timoshenko Bimetallic Strip Model
----------------------------------
For a two-material beam (or layup), Timoshenko (1925) derived the curvature
due to differential thermal expansion:

    κ = 6 * (α₂ - α₁) * ΔT * (1 + m)² /
        (h * (3*(1+m)² + (1+m*n)*(m² + 1/(m*n))))

where:
    m = h₁/h₂ (thickness ratio)
    n = E₁/E₂ (modulus ratio)
    h = h₁ + h₂ (total thickness)
    α₁, α₂ = CTE of materials 1 and 2
    ΔT = temperature change

The maximum interfacial shear stress occurs at the free edges and is:

    τ_max ≈ (α₂ - α₁) * ΔT * E₁ * E₂ * h₁ * h₂ /
             ((E₁*h₁ + E₂*h₂) * (h₁ + h₂)) * L

where L is a characteristic length related to edge effects.

References
----------
- Timoshenko, S. "Analysis of Bi-Metal Thermostats" (1925)
- Suhir, E. "Interfacial stresses in bimetal thermostats" (1989)
- Comminal et al. "Numerical simulation of FDM residual stresses" (2018)
"""

import numpy as np
from typing import Dict, List, Tuple
from ..core.materials import get_material


def thermal_stress_analysis(pattern: List[int], layer_height: float,
                            total_height: float, material_map: Dict[int, str],
                            T_room: float = 23.0,
                            verbose: bool = True) -> dict:
    """Analyze residual thermal stresses in a multi-material layup.

    Computes the thermal stress state at each material interface caused by
    differential thermal expansion during cooling from print temperature
    to room temperature.

    Parameters
    ----------
    pattern : list of int
        Repeating filament pattern.
    layer_height : float
        Layer height in mm.
    total_height : float
        Total part height in mm.
    material_map : dict
        Filament number to material key mapping.
    T_room : float
        Room temperature in degC (default: 23).
    verbose : bool
        If True, print detailed thermal analysis report.

    Returns
    -------
    dict
        Dictionary containing:
        - 'interface_stresses': list of dicts with stress at each interface
        - 'max_shear_stress': maximum interfacial shear stress (MPa)
        - 'max_normal_stress': maximum interfacial normal stress (MPa)
        - 'curvature': predicted thermal curvature (1/mm)
        - 'warp_deflection': predicted warping at part edges (mm)
        - 'risk_level': 'low', 'moderate', or 'high'
    """
    num_layers = int(total_height / layer_height)
    pattern_len = len(pattern)

    # Build the full layer stack
    layer_materials = []
    for i in range(num_layers):
        filament = pattern[i % pattern_len]
        mat_key = material_map.get(filament, 'PLA')
        layer_materials.append(get_material(mat_key))

    # ── Analyze each material interface ──────────────────────────────────
    interface_stresses = []
    max_shear = 0.0
    max_normal = 0.0

    for i in range(num_layers - 1):
        mat_bot = layer_materials[i]
        mat_top = layer_materials[i + 1]

        # Skip same-material interfaces (no CTE mismatch)
        if mat_bot.name == mat_top.name:
            continue

        # Stress-free temperature: the lower Tg, because above Tg the polymer
        # relaxes and cannot sustain elastic stress
        T_stress_free = min(mat_bot.T_glass, mat_top.T_glass)
        delta_T = T_stress_free - T_room

        if delta_T <= 0:
            continue  # No thermal stress if Tg is below room temperature

        # CTE mismatch
        delta_alpha = abs(mat_top.CTE - mat_bot.CTE)

        # Thermal strain mismatch
        thermal_strain = delta_alpha * delta_T

        # ── Interfacial normal stress (peel stress) ──────────────────
        # From force balance on a constrained bilayer:
        # Each material wants to shrink differently. The stiffer material
        # constrains the more expansive one, creating tension in one and
        # compression in the other.
        #
        # σ_thermal ≈ E_eff * Δα * ΔT / (1 + E_top*h_top/(E_bot*h_bot))
        # For a single-layer interface, we use the harmonic mean modulus

        E_harm = 2 * mat_bot.E * mat_top.E / (mat_bot.E + mat_top.E)
        sigma_normal = E_harm * thermal_strain

        # ── Interfacial shear stress ─────────────────────────────────
        # The shear stress is highest at the free edges and decays
        # exponentially toward the center. For a simple bilayer:
        #
        # τ_max ≈ Δα * ΔT * E1 * E2 * h1 * h2 /
        #         ((E1*h1 + E2*h2) * √(h1*h2)) * correction_factor
        #
        # The correction factor accounts for the characteristic shear
        # transfer length. For thin layers, shear stress is relatively
        # higher because there's less material to distribute the load.

        h1 = layer_height  # bottom layer thickness
        h2 = layer_height  # top layer thickness
        E1, E2 = mat_bot.E, mat_top.E

        tau_max = (delta_alpha * delta_T * E1 * E2 * h1 * h2 /
                   ((E1 * h1 + E2 * h2) * np.sqrt(h1 * h2 + 1e-10)))

        interface_stresses.append({
            'layer_index': i,
            'z_position': (i + 1) * layer_height,
            'mat_bottom': mat_bot.name,
            'mat_top': mat_top.name,
            'delta_T': delta_T,
            'delta_alpha': delta_alpha,
            'thermal_strain': thermal_strain,
            'sigma_normal': sigma_normal,
            'tau_max': tau_max,
        })

        max_shear = max(max_shear, tau_max)
        max_normal = max(max_normal, sigma_normal)

    # ── Overall warping prediction (Timoshenko bimetallic strip) ─────────
    curvature, warp_deflection = _predict_bilayer_warping(
        layer_materials, layer_height, num_layers, T_room
    )

    # ── Risk assessment ──────────────────────────────────────────────────
    # Compare thermal stresses to inter-layer bond strength
    risk_level = 'low'
    for iface in interface_stresses:
        mat_top = get_material(material_map.get(
            pattern[(iface['layer_index'] + 1) % pattern_len], 'PLA'))
        mat_bot = get_material(material_map.get(
            pattern[iface['layer_index'] % pattern_len], 'PLA'))
        bond_strength = min(mat_top.sigma_t_z, mat_bot.sigma_t_z)

        if iface['sigma_normal'] > bond_strength * 0.8:
            risk_level = 'high'
            break
        elif iface['sigma_normal'] > bond_strength * 0.5:
            risk_level = 'moderate'

    results = {
        'interface_stresses': interface_stresses,
        'max_shear_stress': max_shear,
        'max_normal_stress': max_normal,
        'curvature': curvature,
        'warp_deflection': warp_deflection,
        'risk_level': risk_level,
    }

    if verbose:
        _print_thermal_report(results, total_height)

    return results


def _predict_bilayer_warping(layer_materials: list, layer_height: float,
                              num_layers: int, T_room: float) -> Tuple[float, float]:
    """Predict warping using generalized Timoshenko bilayer model.

    For a multi-material stack, we compute the effective CTE-weighted
    curvature by integrating the thermal strain mismatch through the
    thickness, weighted by the distance from the neutral axis.

    Returns
    -------
    tuple of (curvature, max_deflection)
        curvature in 1/mm, deflection in mm (assuming 100mm part length).
    """
    h_total = num_layers * layer_height

    # Compute weighted neutral axis
    sum_EA_z = 0.0
    sum_EA = 0.0
    for k, mat in enumerate(layer_materials):
        z_mid = (k + 0.5) * layer_height
        sum_EA_z += mat.E * layer_height * z_mid
        sum_EA += mat.E * layer_height
    z_neutral = sum_EA_z / sum_EA if sum_EA > 0 else h_total / 2.0

    # Compute thermal curvature
    # κ_thermal = Σ (E_k * α_k * ΔT_k * (z_k - z_neutral) * h_k) / EI_total
    #
    # This generalizes the Timoshenko formula to N layers.
    # Positive curvature = concave up (bottom shrinks more).

    # EI about neutral axis
    EI = 0.0
    sum_E_alpha_dT_z = 0.0

    for k, mat in enumerate(layer_materials):
        z_mid = (k + 0.5) * layer_height
        d = z_mid - z_neutral

        # EI contribution (parallel axis theorem)
        I_layer = layer_height**3 / 12.0
        EI += mat.E * (I_layer + layer_height * d**2)

        # Thermal moment contribution
        T_stress_free = mat.T_glass
        delta_T = T_stress_free - T_room
        if delta_T > 0:
            sum_E_alpha_dT_z += mat.E * mat.CTE * delta_T * d * layer_height

    curvature = sum_E_alpha_dT_z / EI if EI > 0 else 0.0

    # Max deflection for a simply supported beam of length L
    # δ = κ * L² / 8 (for uniform curvature)
    L = 100.0  # mm, assume 100mm part length for reference
    warp_deflection = abs(curvature) * L**2 / 8.0

    return curvature, warp_deflection


def predict_warping(pattern: List[int], layer_height: float,
                    total_height: float, material_map: Dict[int, str],
                    part_length: float = 100.0,
                    T_room: float = 23.0) -> dict:
    """Predict warping deflection for a given part length.

    This is a convenience function that wraps the thermal analysis
    and returns warping-specific results.

    Parameters
    ----------
    pattern : list of int
        Repeating filament pattern.
    layer_height : float
        Layer height in mm.
    total_height : float
        Total part height in mm.
    material_map : dict
        Filament number to material key mapping.
    part_length : float
        Part length/width in mm (the dimension along which warping occurs).
    T_room : float
        Room/ambient temperature in degC.

    Returns
    -------
    dict
        'curvature' (1/mm), 'deflection' (mm), 'radius' (mm),
        'deflection_ratio' (deflection/length).
    """
    result = thermal_stress_analysis(
        pattern, layer_height, total_height, material_map,
        T_room=T_room, verbose=False
    )

    curvature = result['curvature']
    deflection = abs(curvature) * part_length**2 / 8.0
    radius = 1.0 / abs(curvature) if abs(curvature) > 1e-12 else float('inf')

    return {
        'curvature': curvature,
        'deflection': deflection,
        'radius': radius,
        'deflection_ratio': deflection / part_length if part_length > 0 else 0,
    }


def _print_thermal_report(results: dict, total_height: float) -> None:
    """Print formatted thermal stress analysis report."""
    print("\n" + "=" * 70)
    print("  THERMAL STRESS ANALYSIS")
    print("=" * 70)

    if not results['interface_stresses']:
        print("\n  No multi-material interfaces found (single material).")
        print("=" * 70)
        return

    print(f"\n  Interface Stresses:")
    print(f"    {'Z(mm)':<8} {'Bottom':<20} {'Top':<20} {'DT(C)':<8} "
          f"{'sigma(MPa)':<12} {'tau(MPa)':<10}")
    print("    " + "-" * 78)

    for iface in results['interface_stresses']:
        print(f"    {iface['z_position']:<8.2f} {iface['mat_bottom'][:18]:<20} "
              f"{iface['mat_top'][:18]:<20} {iface['delta_T']:<8.0f} "
              f"{iface['sigma_normal']:<12.2f} {iface['tau_max']:<10.2f}")

    print(f"\n  Summary:")
    print(f"    Maximum normal stress: {results['max_normal_stress']:.2f} MPa")
    print(f"    Maximum shear stress: {results['max_shear_stress']:.2f} MPa")
    print(f"    Thermal curvature: {results['curvature']:.6f} 1/mm")
    print(f"    Warp deflection (100mm part): {results['warp_deflection']:.3f} mm")
    print(f"    Risk level: {results['risk_level'].upper()}")

    if results['risk_level'] == 'high':
        print(f"\n    RECOMMENDATION: Thermal stresses approach inter-layer bond strength.")
        print(f"    Consider: heated bed, enclosure, slower cooling, or symmetric layup.")
    elif results['risk_level'] == 'moderate':
        print(f"\n    RECOMMENDATION: Monitor for warping. Use heated bed and enclosure.")

    print("\n" + "=" * 70)
