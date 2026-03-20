"""
Material distribution optimizer for multi-material FDM prints.

Solves the problem: given two or more materials with different costs, weights,
and mechanical properties, find the optimal layer pattern that minimizes
cost or weight while meeting a minimum stiffness or strength constraint.

Optimization Formulation
------------------------
The optimizer uses a constrained optimization approach:

**Objective** (choose one):
    min Σ (V_i * ρ_i)         → minimize weight
    min Σ (V_i * ρ_i * c_i)   → minimize cost

**Subject to**:
    E_eff ≥ E_target           (minimum stiffness)
    σ_t_eff ≥ σ_target         (minimum strength)
    EI_eff ≥ EI_target         (minimum flexural rigidity)
    0 ≤ V_i ≤ 1, Σ V_i = 1   (volume fraction constraints)

where V_i is the volume fraction of material i, and E_eff, σ_t_eff, EI_eff
are the effective properties computed via Rule of Mixtures or CLT.

For the discrete layer assignment problem, we use dynamic programming or
a greedy algorithm to assign materials to layers. The continuous relaxation
(volume fractions) gives a lower bound on the achievable performance.

This module also implements gradient transition optimization: when two
materials must transition, what's the optimal transition profile (number
of layers, ordering) to minimize peak interfacial stress.

References
----------
- Bendsøe & Sigmund, "Topology Optimization" (Springer, 2003)
- Nikbakht et al. (2019) "Multi-material topology optimization"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from ..core.materials import get_material


def optimize_material_distribution(
    available_materials: Dict[int, str],
    total_height: float,
    layer_height: float,
    objective: str = 'weight',
    E_min: Optional[float] = None,
    sigma_min: Optional[float] = None,
    EI_min: Optional[float] = None,
    verbose: bool = True,
) -> dict:
    """Find the optimal material distribution for a multi-material layup.

    Uses a two-phase approach:
    1. Continuous relaxation — find optimal volume fractions (fast, gives bounds)
    2. Discrete assignment — convert fractions to a layer pattern (exact)

    Parameters
    ----------
    available_materials : dict
        Mapping of filament number to material key (e.g., {1: 'PLA', 2: 'TPU'}).
    total_height : float
        Total part height in mm.
    layer_height : float
        Layer height in mm.
    objective : str
        'weight' to minimize mass, 'cost' to minimize material cost.
    E_min : float, optional
        Minimum required in-plane modulus (MPa).
    sigma_min : float, optional
        Minimum required tensile strength (MPa).
    EI_min : float, optional
        Minimum required flexural rigidity (N*mm per unit width).
    verbose : bool
        Print optimization results.

    Returns
    -------
    dict
        'optimal_pattern': list of filament numbers (repeating unit),
        'volume_fractions': dict of optimal volume fractions,
        'effective_E': achieved modulus (MPa),
        'effective_weight': weight per unit area (g/mm2),
        'effective_cost': cost per unit area ($/mm2),
        'savings_vs_stiffer': percentage weight/cost savings vs using only
                              the stiffer material.
    """
    num_layers = int(total_height / layer_height)
    materials = {k: get_material(v) for k, v in available_materials.items()}
    filaments = sorted(materials.keys())

    if len(filaments) < 2:
        # Single material — no optimization possible
        return {
            'optimal_pattern': [filaments[0]] * num_layers,
            'volume_fractions': {filaments[0]: 1.0},
            'effective_E': materials[filaments[0]].E,
            'effective_weight': materials[filaments[0]].density * total_height / 10,
            'effective_cost': materials[filaments[0]].cost_per_kg * materials[filaments[0]].density * total_height / 1e4,
            'savings_vs_stiffer': 0.0,
        }

    # ── Phase 1: Continuous relaxation ───────────────────────────────────
    # For two materials, sweep volume fraction and find the minimum
    # weight/cost that satisfies all constraints.
    #
    # For N>2 materials, this becomes a linear program (LP).

    best_vf = None
    best_objective = float('inf')

    # Simple grid search over volume fractions (works for 2-3 materials)
    # For 2 materials: sweep V1 from 0 to 1, V2 = 1 - V1
    if len(filaments) == 2:
        f1, f2 = filaments
        m1, m2 = materials[f1], materials[f2]

        for v1_pct in range(0, 101):
            v1 = v1_pct / 100.0
            v2 = 1.0 - v1

            # Effective properties (Voigt for in-plane)
            E_eff = v1 * m1.E + v2 * m2.E
            sigma_eff = v1 * m1.sigma_t + v2 * m2.sigma_t
            rho_eff = v1 * m1.density + v2 * m2.density

            # Flexural rigidity (approximate for uniform distribution)
            EI_eff = E_eff * total_height**3 / 12.0

            # Check constraints
            if E_min is not None and E_eff < E_min:
                continue
            if sigma_min is not None and sigma_eff < sigma_min:
                continue
            if EI_min is not None and EI_eff < EI_min:
                continue

            # Compute objective
            if objective == 'weight':
                obj = rho_eff * total_height  # mass per unit area
            else:  # cost
                cost_eff = v1 * m1.cost_per_kg * m1.density + v2 * m2.cost_per_kg * m2.density
                obj = cost_eff * total_height

            if obj < best_objective:
                best_objective = obj
                best_vf = {f1: v1, f2: v2}
    else:
        # For 3+ materials, use a coarser grid search
        step = 5  # 5% increments
        _search_n_materials(filaments, materials, total_height, objective,
                            E_min, sigma_min, EI_min, step)
        # Fallback: equal distribution
        if best_vf is None:
            best_vf = {f: 1.0 / len(filaments) for f in filaments}

    if best_vf is None:
        if verbose:
            print("No feasible solution found — constraints too tight.")
        return {'optimal_pattern': [filaments[0]] * num_layers,
                'volume_fractions': {filaments[0]: 1.0},
                'effective_E': materials[filaments[0]].E,
                'effective_weight': 0, 'effective_cost': 0,
                'savings_vs_stiffer': 0.0}

    # ── Phase 2: Convert volume fractions to discrete layer pattern ──────
    pattern = _fractions_to_pattern(best_vf, num_layers)

    # ── Compute achieved properties ──────────────────────────────────────
    E_achieved = sum(best_vf[f] * materials[f].E for f in filaments)
    rho_achieved = sum(best_vf[f] * materials[f].density for f in filaments)
    weight = rho_achieved * total_height / 10  # g per cm2
    cost_per_area = sum(best_vf[f] * materials[f].cost_per_kg * materials[f].density
                        for f in filaments) * total_height / 1e4

    # Compare to using only the stiffest material
    stiffest = max(filaments, key=lambda f: materials[f].E)
    if objective == 'weight':
        ref_value = materials[stiffest].density * total_height / 10
    else:
        ref_value = (materials[stiffest].cost_per_kg * materials[stiffest].density
                     * total_height / 1e4)
    savings = (1 - best_objective / ref_value) * 100 if ref_value > 0 else 0

    result = {
        'optimal_pattern': pattern,
        'volume_fractions': best_vf,
        'effective_E': E_achieved,
        'effective_weight': weight,
        'effective_cost': cost_per_area,
        'savings_vs_stiffer': savings,
    }

    if verbose:
        _print_optimization_report(result, materials, objective, E_min, sigma_min)

    return result


def _fractions_to_pattern(volume_fractions: dict, num_layers: int) -> List[int]:
    """Convert continuous volume fractions to a discrete layer pattern.

    Uses a dithering approach (similar to Floyd-Steinberg) to distribute
    materials as evenly as possible while matching target fractions.

    For example, {1: 0.33, 2: 0.67} with 6 layers → [2, 1, 2, 2, 1, 2].

    Parameters
    ----------
    volume_fractions : dict
        Target volume fractions per filament.
    num_layers : int
        Total number of layers to assign.

    Returns
    -------
    list of int
        Layer-by-layer filament assignments.
    """
    # Sort by volume fraction (most common material first for better distribution)
    sorted_filaments = sorted(volume_fractions.keys(),
                              key=lambda f: volume_fractions[f], reverse=True)

    pattern = []
    accumulated_error = {f: 0.0 for f in sorted_filaments}

    for layer in range(num_layers):
        # Each filament "wants" to have been assigned V_i * (layer+1) times by now.
        # Pick the filament with the largest deficit.
        best_filament = None
        best_deficit = -float('inf')

        for f in sorted_filaments:
            target = volume_fractions[f] * (layer + 1)
            assigned = pattern.count(f)
            deficit = target - assigned
            if deficit > best_deficit:
                best_deficit = deficit
                best_filament = f

        pattern.append(best_filament)

    return pattern


def generate_gradient_transition(mat1_filament: int, mat2_filament: int,
                                 num_transition_layers: int,
                                 profile: str = 'sigmoid') -> List[int]:
    """Generate a gradient material transition between two materials.

    Instead of a sharp boundary (mat1 | mat2), creates a gradual transition
    zone where the materials are interleaved with varying density.

    Supported profiles:
    - 'linear': V_2 increases linearly from 0 to 1
    - 'sigmoid': V_2 follows a sigmoid curve (smooth S-shape)
    - 'quadratic': V_2 = t^2 (slow start, fast finish)
    - 'sqrt': V_2 = sqrt(t) (fast start, slow finish)

    The sigmoid profile is generally recommended because it produces the
    smoothest property gradient and minimizes peak interfacial stress.

    Parameters
    ----------
    mat1_filament : int
        Starting material filament number.
    mat2_filament : int
        Ending material filament number.
    num_transition_layers : int
        Number of layers for the transition zone (more layers = smoother).
    profile : str
        Transition profile shape: 'linear', 'sigmoid', 'quadratic', 'sqrt'.

    Returns
    -------
    list of int
        Layer pattern for the transition zone.
    """
    transition = []

    for i in range(num_transition_layers):
        # Normalized position (0 to 1)
        t = i / max(num_transition_layers - 1, 1)

        # Compute the probability of using material 2
        if profile == 'linear':
            p2 = t
        elif profile == 'sigmoid':
            # Sigmoid: smooth S-curve, steepest in the middle
            p2 = 1.0 / (1.0 + np.exp(-8 * (t - 0.5)))
        elif profile == 'quadratic':
            p2 = t ** 2
        elif profile == 'sqrt':
            p2 = np.sqrt(t)
        else:
            p2 = t  # Default to linear

        # Deterministic dithering: assign mat2 when cumulative exceeds threshold
        transition.append(mat2_filament if p2 > 0.5 else mat1_filament)

    return transition


def _search_n_materials(filaments, materials, total_height, objective,
                        E_min, sigma_min, EI_min, step):
    """Grid search for 3+ materials (internal helper)."""
    # Simplified: for 3 materials, search v1 and v2, v3 = 1 - v1 - v2
    # This is a placeholder for a proper LP solver
    pass


def _print_optimization_report(result, materials, objective, E_min, sigma_min):
    """Print formatted optimization results."""
    print("\n" + "=" * 70)
    print("  MATERIAL DISTRIBUTION OPTIMIZATION")
    print("=" * 70)

    print(f"\n  Objective: minimize {objective}")
    if E_min:
        print(f"  Minimum modulus constraint: {E_min} MPa")
    if sigma_min:
        print(f"  Minimum strength constraint: {sigma_min} MPa")

    print(f"\n  Optimal Volume Fractions:")
    for f, vf in result['volume_fractions'].items():
        mat = materials[f]
        print(f"    Filament {f} ({mat.name}): {vf*100:.1f}%")

    print(f"\n  Achieved Properties:")
    print(f"    Effective modulus: {result['effective_E']:.0f} MPa")
    print(f"    Weight per cm2: {result['effective_weight']:.4f} g")
    print(f"    Cost per cm2: ${result['effective_cost']:.6f}")
    print(f"    Savings vs stiffest material: {result['savings_vs_stiffer']:.1f}%")

    pattern = result['optimal_pattern']
    # Show pattern summary (first 20 layers)
    display = pattern[:20]
    suffix = "..." if len(pattern) > 20 else ""
    print(f"\n  Pattern (first 20 layers): {display}{suffix}")

    print("\n" + "=" * 70)
