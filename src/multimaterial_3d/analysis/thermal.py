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
    num_layers = round(total_height / layer_height)
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

        # Stress-free temperature: the lower Tg of the pair.
        # Above Tg, the polymer chain mobility allows stress relaxation,
        # so thermal stresses only accumulate below Tg.
        T_stress_free = min(mat_bot.T_glass, mat_top.T_glass)
        delta_T = T_stress_free - T_room

        if delta_T <= 0:
            continue  # No thermal stress if Tg below room temperature

        # CTE mismatch (signed: positive means top expands more than bottom)
        delta_alpha = mat_top.CTE - mat_bot.CTE
        thermal_strain = delta_alpha * delta_T

        h1 = layer_height  # bottom layer
        h2 = layer_height  # top layer
        E1, E2 = mat_bot.E, mat_top.E
        nu1, nu2 = mat_bot.nu, mat_top.nu

        # ── Biaxial plane-stress modulus ─────────────────────────────
        # In a bonded bilayer, the interface constrains both in-plane
        # directions. The biaxial modulus E_bar = E / (1 - nu) is the
        # appropriate stiffness for plane-stress thermal problems.
        E1_bar = E1 / (1.0 - nu1)
        E2_bar = E2 / (1.0 - nu2)

        # ── Membrane (axial) stress at interface ─────────────────────
        # From force equilibrium of a constrained bilayer (Timoshenko):
        # The total thermal force mismatch is distributed between layers
        # inversely proportional to their axial rigidity.
        #
        # Using signed delta_alpha: if alpha_top > alpha_bot, then on
        # cooling (dT > 0 from Tg to room), the top material wants to
        # shrink more and is put in tension; the bottom in compression.
        # sigma_1 (bottom) is compressive (negative), sigma_2 (top) is tensile.
        #
        # Force balance: sigma_1*h1 + sigma_2*h2 = 0
        # Compatibility: sigma_1/E1_bar - sigma_2/E2_bar = delta_alpha*delta_T
        #
        # Solving:
        #   sigma_1 = -delta_alpha*dT * E1_bar*E2_bar*h2 / (E1_bar*h1 + E2_bar*h2)
        #   sigma_2 = +delta_alpha*dT * E1_bar*E2_bar*h1 / (E1_bar*h1 + E2_bar*h2)
        sigma_1 = -thermal_strain * E1_bar * E2_bar * h2 / (E1_bar * h1 + E2_bar * h2)
        sigma_2 = thermal_strain * E1_bar * E2_bar * h1 / (E1_bar * h1 + E2_bar * h2)
        sigma_normal = max(abs(sigma_1), abs(sigma_2))

        # ── Interfacial shear stress (Suhir model) ───────────────────
        # Suhir (1989) showed that the interfacial shear stress in a
        # bonded bilayer decays exponentially from the free edge:
        #
        #   tau(x) = tau_max * exp(-beta * x)
        #
        # where x is distance from the free edge and beta is the
        # characteristic shear transfer parameter:
        #
        #   beta = sqrt(K_shear * (1/(E1_bar*h1) + 1/(E2_bar*h2)))
        #
        # K_shear accounts for the shear compliance of the interface
        # region. For FDM inter-layer bonds, the bonding zone thickness
        # is approximately 0.1 * layer_height (the remelted region).
        #
        # The maximum shear stress at the free edge is:
        #   tau_max = delta_alpha * delta_T * beta /
        #             (1/(E1_bar*h1) + 1/(E2_bar*h2))

        # Effective shear compliance of the interface zone
        # Bond zone thickness ~ 10% of layer height for typical FDM
        h_bond = 0.1 * layer_height
        G_avg = 0.5 * (mat_bot.G_xy + mat_top.G_xy)
        K_shear = G_avg / h_bond  # Shear stiffness per unit area (MPa/mm)

        compliance_sum = 1.0 / (E1_bar * h1) + 1.0 / (E2_bar * h2)
        beta = np.sqrt(K_shear * compliance_sum)

        # Note: tau_max is the magnitude of the peak shear stress at the
        # free edge. The Suhir formula assumes an infinite strip (L >> 1/beta),
        # i.e., coth(beta*L/2) -> 1. This is reasonable for most 3D printed parts
        # where part dimensions are much larger than the shear transfer length 1/beta.
        tau_max = abs(thermal_strain) * beta / compliance_sum

        # ── Timoshenko curvature for this bilayer ────────────────────
        # Classic Timoshenko (1925) formula for bilayer curvature:
        #   kappa = 6*(a2-a1)*dT*(1+m)^2 /
        #           (h*(3*(1+m)^2 + (1+m*n)*(m^2 + 1/(m*n))))
        # where m = h1/h2, n = E1/E2, h = h1 + h2
        m = h1 / h2
        n = E1 / E2
        h = h1 + h2
        kappa_local = (6.0 * delta_alpha * delta_T * (1 + m)**2 /
                       (h * (3*(1+m)**2 + (1+m*n)*(m**2 + 1.0/(m*n + 1e-20)))))

        interface_stresses.append({
            'layer_index': i,
            'z_position': (i + 1) * layer_height,
            'mat_bottom': mat_bot.name,
            'mat_top': mat_top.name,
            'delta_T': delta_T,
            'delta_alpha': delta_alpha,  # signed: top CTE - bottom CTE
            'thermal_strain': thermal_strain,  # signed
            'sigma_bot': sigma_1,  # signed stress in bottom layer (MPa)
            'sigma_top': sigma_2,  # signed stress in top layer (MPa)
            'sigma_normal': sigma_normal,  # max absolute value for risk assessment
            'tau_max': tau_max,
            'kappa_local': kappa_local,
            'beta': beta,
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
    """Predict warping using CLT thermal force and moment resultants.

    This uses the proper Classical Laminate Theory approach:
    1. Compute thermal force resultant N_T and moment resultant M_T
       by integrating the thermal stress through the thickness.
    2. Compute the ABD matrix (A, B, D) for the laminate.
    3. Solve the coupled system [A,B;B,D] * [eps0;kappa] = [N_T;M_T]
       for the midplane strains eps0 and curvatures kappa.

    For symmetric layups (B=0), N_T produces only membrane strain (no
    curvature), and M_T=0, so no warping occurs. For asymmetric layups
    (B!=0), the coupling between extension and bending causes the thermal
    loads to produce curvature (warping).

    The key physical insight: warping is driven by the B matrix (asymmetry),
    not just by CTE differences. A symmetric layup with large CTE mismatch
    will NOT warp (it will have internal stresses but no curvature).

    Returns
    -------
    tuple of (curvature, max_deflection)
        curvature in 1/mm, deflection in mm (assuming 100mm part length).
    """
    h_total = num_layers * layer_height
    z_ref = h_total / 2.0

    # Build ABD matrix and thermal resultants simultaneously
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    N_T = np.zeros(3)  # Thermal force resultant
    M_T = np.zeros(3)  # Thermal moment resultant

    for k, mat in enumerate(layer_materials):
        # Biaxial plane-stress stiffness for thermal analysis
        E_bar = mat.E / (1.0 - mat.nu)
        nu = mat.nu

        # Simplified Q matrix (quasi-isotropic for thermal)
        Q11 = mat.E / (1.0 - nu**2)
        Q12 = nu * mat.E / (1.0 - nu**2)
        Q = np.array([
            [Q11, Q12, 0],
            [Q12, Q11, 0],
            [0,   0,   mat.G_xy]
        ])

        z_bot = k * layer_height - z_ref
        z_top = (k + 1) * layer_height - z_ref

        # ABD contributions
        A += Q * (z_top - z_bot)
        B += 0.5 * Q * (z_top**2 - z_bot**2)
        D += (1.0 / 3.0) * Q * (z_top**3 - z_bot**3)

        # Thermal loads: stress-free temp is each material's Tg
        T_stress_free = mat.T_glass
        delta_T = T_stress_free - T_room
        if delta_T <= 0:
            continue

        # Thermal strain vector (isotropic CTE: equal in x and y, zero shear)
        alpha_vec = np.array([mat.CTE, mat.CTE, 0.0])

        # Thermal stress contribution from this layer: Q * alpha * dT
        thermal_stress = Q @ alpha_vec * delta_T

        # Integrate through layer thickness for N_T and M_T
        N_T += thermal_stress * (z_top - z_bot)
        M_T += 0.5 * thermal_stress * (z_top**2 - z_bot**2)

    # Assemble and solve the 6x6 system
    ABD = np.zeros((6, 6))
    ABD[:3, :3] = A
    ABD[:3, 3:] = B
    ABD[3:, :3] = B
    ABD[3:, 3:] = D

    load = np.concatenate([N_T, M_T])

    try:
        response = np.linalg.solve(ABD, load)
    except np.linalg.LinAlgError:
        return 0.0, 0.0

    # eps0 = response[:3]  # midplane strains
    kappa = response[3:]  # curvatures [kappa_x, kappa_y, kappa_xy]

    # Primary warping curvature (take the larger of kx, ky)
    curvature = kappa[0]  # kappa_x

    # Neutral axis (for reporting, not used in CLT warping calc)
    sum_EA_z = 0.0
    sum_EA = 0.0
    for k, mat in enumerate(layer_materials):
        z_mid = (k + 0.5) * layer_height
        sum_EA_z += mat.E * layer_height * z_mid
        sum_EA += mat.E * layer_height

    # Max deflection for uniform curvature: delta = kappa * L^2 / 8
    L = 100.0  # mm reference part length
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
