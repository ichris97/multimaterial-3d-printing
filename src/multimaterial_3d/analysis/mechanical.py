"""
Mechanical property analysis for multi-material layered FDM prints.

Implements Classical Laminate Theory (CLT) adapted for FDM printing, including:
- Rule of Mixtures (Voigt/Reuss bounds) for effective properties
- Full ABD stiffness matrix for laminate bending/extension coupling
- Neutral axis calculation for asymmetric layups
- Flexural rigidity prediction
- Interface stress concentration analysis
- Delamination risk assessment

Background
----------
Classical Laminate Theory treats the multi-material print as a stack of thin
layers (laminae), each with its own stiffness. The theory computes the
laminate stiffness matrix [A, B, D] where:

    [A] = Extensional stiffness matrix (in-plane loads)
    [B] = Coupling matrix (extension-bending coupling; nonzero for asymmetric layups)
    [D] = Bending stiffness matrix (out-of-plane bending)

For symmetric layups (e.g., PLA-TPU-PLA), [B] = 0 and there is no
extension-bending coupling. For asymmetric layups (e.g., PLA bottom, TPU top),
[B] ≠ 0, meaning that in-plane loads will cause bending (warping) and vice
versa. This is a critical consideration for multi-material prints.

References
----------
- Jones, R.M. "Mechanics of Composite Materials" (2nd ed.)
- Gibson, R.F. "Principles of Composite Material Mechanics"
- Cantrell et al. (2017) "Experimental characterization of the mechanical
  properties of 3D-printed ABS and polycarbonate parts"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from ..core.materials import MaterialProperties, get_material, get_adhesion, MATERIAL_DB


def compute_layer_stiffness_matrix(mat: MaterialProperties) -> np.ndarray:
    """Compute the reduced stiffness matrix [Q] for a single FDM layer.

    For FDM printing, each layer is treated as a transversely isotropic
    lamina with the fiber direction (raster lines) in the XY plane.
    The reduced stiffness matrix relates in-plane stresses to in-plane
    strains:

        [σ_x ]   [Q11 Q12  0 ] [ε_x ]
        [σ_y ] = [Q12 Q22  0 ] [ε_y ]
        [τ_xy]   [ 0   0  Q66] [γ_xy]

    For quasi-isotropic in-plane behavior (typical of alternating raster):
        Q11 = Q22 = E / (1 - ν²)
        Q12 = ν * E / (1 - ν²)
        Q66 = G_xy

    Parameters
    ----------
    mat : MaterialProperties
        Material property set for this layer.

    Returns
    -------
    np.ndarray
        3x3 reduced stiffness matrix [Q] in MPa.
    """
    E = mat.E
    nu = mat.nu
    G = mat.G_xy

    # Quasi-isotropic in-plane assumption (common for cross-raster FDM)
    Q11 = E / (1 - nu**2)
    Q22 = Q11  # Isotropic in-plane
    Q12 = nu * E / (1 - nu**2)
    Q66 = G

    Q = np.array([
        [Q11, Q12, 0],
        [Q12, Q22, 0],
        [0,   0,  Q66]
    ])
    return Q


def compute_abd_matrix(pattern: List[int], layer_height: float,
                       total_height: float,
                       material_map: Dict[int, str]) -> Dict[str, np.ndarray]:
    """Compute the full ABD stiffness matrix for a multi-material laminate.

    The ABD matrix is the fundamental result of Classical Laminate Theory.
    It relates the force and moment resultants to the mid-plane strains
    and curvatures:

        [N]   [A  B] [ε°]
        [M] = [B  D] [κ ]

    where:
        N = in-plane force resultants (N/mm) [Nx, Ny, Nxy]
        M = moment resultants (N) [Mx, My, Mxy]
        ε° = mid-plane strains [-] [εx°, εy°, γxy°]
        κ = curvatures (1/mm) [κx, κy, κxy]

    Matrix components:
        A_ij = Σ Q_ij^k * (z_k - z_{k-1})              [extensional stiffness]
        B_ij = (1/2) Σ Q_ij^k * (z_k² - z_{k-1}²)     [coupling stiffness]
        D_ij = (1/3) Σ Q_ij^k * (z_k³ - z_{k-1}³)     [bending stiffness]

    Parameters
    ----------
    pattern : list of int
        Repeating filament pattern (e.g., [1, 2, 2, 1]).
    layer_height : float
        Layer height in mm.
    total_height : float
        Total part height in mm.
    material_map : dict
        Mapping of filament number to material key (e.g., {1: 'PLA', 2: 'TPU'}).

    Returns
    -------
    dict
        Dictionary containing:
        - 'A': 3x3 extensional stiffness matrix (N/mm)
        - 'B': 3x3 coupling matrix (N)
        - 'D': 3x3 bending stiffness matrix (N·mm)
        - 'ABD': 6x6 combined matrix
        - 'abd_inv': 6x6 compliance matrix (inverse of ABD)
        - 'z_neutral': neutral axis position from bottom (mm)
        - 'symmetric': whether the layup is symmetric
        - 'Ex_eff': effective in-plane modulus in x (MPa)
        - 'Ey_eff': effective in-plane modulus in y (MPa)
        - 'EI_eff': effective flexural rigidity per unit width (N·mm)
    """
    num_layers = int(total_height / layer_height)
    pattern_len = len(pattern)

    # Reference plane at geometric mid-plane
    h_total = num_layers * layer_height
    z_mid = h_total / 2.0

    # Initialize ABD matrices
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))

    # Build layer stack and accumulate ABD
    for k in range(num_layers):
        filament = pattern[k % pattern_len]
        mat_key = material_map.get(filament, 'PLA')
        mat = get_material(mat_key)
        Q = compute_layer_stiffness_matrix(mat)

        # Layer boundaries relative to mid-plane
        z_bot = k * layer_height - z_mid
        z_top = (k + 1) * layer_height - z_mid

        # ABD integration (exact for constant Q within layer)
        A += Q * (z_top - z_bot)
        B += 0.5 * Q * (z_top**2 - z_bot**2)
        D += (1.0 / 3.0) * Q * (z_top**3 - z_bot**3)

    # Assemble 6x6 ABD matrix
    ABD = np.zeros((6, 6))
    ABD[:3, :3] = A
    ABD[:3, 3:] = B
    ABD[3:, :3] = B
    ABD[3:, 3:] = D

    # Compute compliance matrix (inverse of ABD)
    try:
        abd_inv = np.linalg.inv(ABD)
    except np.linalg.LinAlgError:
        abd_inv = np.full((6, 6), np.nan)

    # Check symmetry (B ≈ 0 means symmetric layup)
    is_symmetric = np.allclose(B, 0, atol=1e-6 * np.max(np.abs(A)))

    # Effective in-plane moduli from A matrix
    # A = Q * h for a homogeneous material, so E_eff = A11 * (1-nu^2) / h
    # More precisely, from compliance: Ex = 1 / (h * a11) where a = inv(A)
    a_inv = np.linalg.inv(A)
    Ex_eff = 1.0 / (h_total * a_inv[0, 0])
    Ey_eff = 1.0 / (h_total * a_inv[1, 1])

    # Effective flexural rigidity: EI_eff = D11 per unit width
    EI_eff = D[0, 0]

    # Neutral axis calculation using first moment of weighted stiffness
    sum_EA_z = 0.0
    sum_EA = 0.0
    for k in range(num_layers):
        filament = pattern[k % pattern_len]
        mat_key = material_map.get(filament, 'PLA')
        mat = get_material(mat_key)
        z_mid_layer = (k + 0.5) * layer_height
        sum_EA_z += mat.E * layer_height * z_mid_layer
        sum_EA += mat.E * layer_height
    z_neutral = sum_EA_z / sum_EA if sum_EA > 0 else h_total / 2.0

    return {
        'A': A,
        'B': B,
        'D': D,
        'ABD': ABD,
        'abd_inv': abd_inv,
        'z_neutral': z_neutral,
        'symmetric': is_symmetric,
        'Ex_eff': Ex_eff,
        'Ey_eff': Ey_eff,
        'EI_eff': EI_eff,
    }


def analyze_layup(pattern: List[int], total_height: float,
                  layer_height: float, material_map: Dict[int, str],
                  verbose: bool = True) -> dict:
    """Comprehensive mechanical analysis of a multi-material layup.

    Performs Rule of Mixtures calculations, CLT analysis, interface stress
    assessment, and adhesion compatibility checks. This is the main entry
    point for mechanical analysis.

    Parameters
    ----------
    pattern : list of int
        Repeating filament pattern (e.g., [1, 2] for alternating).
    total_height : float
        Total part height in mm.
    layer_height : float
        Layer height in mm.
    material_map : dict
        Mapping of filament numbers to material keys.
    verbose : bool
        If True, print detailed analysis report.

    Returns
    -------
    dict
        Analysis results including effective properties, ABD matrix data,
        volume fractions, interface counts, and compatibility warnings.
    """
    num_layers = int(total_height / layer_height)
    pattern_len = len(pattern)

    # ── Volume fractions ─────────────────────────────────────────────────
    layer_counts = {}
    for i in range(num_layers):
        filament = pattern[i % pattern_len]
        layer_counts[filament] = layer_counts.get(filament, 0) + 1

    total_layers = sum(layer_counts.values())
    volume_fractions = {f: count / total_layers for f, count in layer_counts.items()}

    # Resolve materials
    materials = {}
    for filament in layer_counts:
        mat_key = material_map.get(filament, 'PLA')
        materials[filament] = get_material(mat_key)

    # ── Rule of Mixtures (Voigt/Reuss bounds) ────────────────────────────
    # Voigt (iso-strain, upper bound): E_V = Σ V_i * E_i
    # Physically: all layers stretch the same amount (in-plane loading)
    E_voigt = sum(volume_fractions[f] * materials[f].E for f in layer_counts)
    sigma_t_voigt = sum(volume_fractions[f] * materials[f].sigma_t for f in layer_counts)

    # Reuss (iso-stress, lower bound): 1/E_R = Σ V_i / E_i
    # Physically: all layers carry the same stress (through-thickness loading)
    E_reuss = 1.0 / sum(volume_fractions[f] / materials[f].E for f in layer_counts)

    # Weighted density
    density_avg = sum(volume_fractions[f] * materials[f].density for f in layer_counts)

    # ── CLT analysis ────────────────────────────────────────────────────
    abd_results = compute_abd_matrix(pattern, layer_height, total_height, material_map)

    # ── Interface analysis ───────────────────────────────────────────────
    # Count material transitions (where delamination risk is highest)
    transitions_per_pattern = 0
    for i in range(pattern_len):
        if pattern[i] != pattern[(i + 1) % pattern_len]:
            transitions_per_pattern += 1

    complete_cycles = num_layers // pattern_len
    remaining = num_layers % pattern_len
    total_interfaces = transitions_per_pattern * complete_cycles
    for i in range(remaining - 1):
        if pattern[i] != pattern[i + 1]:
            total_interfaces += 1

    # ── Adhesion compatibility ───────────────────────────────────────────
    filament_list = list(layer_counts.keys())
    adhesion_warnings = []
    for i in range(len(filament_list)):
        for j in range(i + 1, len(filament_list)):
            mat1_key = material_map.get(filament_list[i], 'PLA')
            mat2_key = material_map.get(filament_list[j], 'PLA')
            adhesion = get_adhesion(mat1_key, mat2_key)
            adhesion_warnings.append({
                'mat1': mat1_key, 'mat2': mat2_key,
                'score': adhesion['score'], 'note': adhesion['note']
            })

    # ── Modulus mismatch (stress concentration factor) ───────────────────
    E_values = [materials[f].E for f in layer_counts]
    E_ratio = max(E_values) / min(E_values) if min(E_values) > 0 else float('inf')

    # ── Assemble results ────────────────────────────────────────────────
    results = {
        'E_voigt': E_voigt,
        'E_reuss': E_reuss,
        'sigma_t': sigma_t_voigt,
        'density': density_avg,
        'specific_stiffness': E_voigt / density_avg,
        'volume_fractions': volume_fractions,
        'layer_counts': layer_counts,
        'total_layers': total_layers,
        'abd': abd_results,
        'interfaces': total_interfaces,
        'transitions_per_pattern': transitions_per_pattern,
        'E_ratio': E_ratio,
        'adhesion_warnings': adhesion_warnings,
    }

    if verbose:
        _print_analysis_report(results, pattern, materials, material_map, layer_height, total_height)

    return results


def _print_analysis_report(results: dict, pattern: list,
                           materials: dict, material_map: dict,
                           layer_height: float, total_height: float) -> None:
    """Print a formatted mechanical analysis report."""
    print("\n" + "=" * 70)
    print("  MECHANICAL PROPERTY ANALYSIS")
    print("=" * 70)

    # Layer distribution
    print(f"\n  Layer Distribution:")
    print(f"    Total layers: {results['total_layers']}")
    print(f"    Pattern length: {len(pattern)} layers")
    print(f"    Pattern repeats: {results['total_layers'] / len(pattern):.1f}x")
    for f, count in sorted(results['layer_counts'].items()):
        mat = materials[f]
        vf = results['volume_fractions'][f]
        print(f"    Filament {f} ({mat.name}): {count} layers ({vf*100:.1f}%)")

    # Material properties
    print(f"\n  Material Properties:")
    for f, mat in materials.items():
        print(f"    F{f} ({mat.name}): E={mat.E} MPa, sigma_t={mat.sigma_t} MPa, "
              f"CTE={mat.CTE:.1e} 1/K")

    # Rule of Mixtures
    print(f"\n  Rule of Mixtures Bounds:")
    print(f"    In-plane modulus (Voigt upper bound): {results['E_voigt']:.0f} MPa")
    print(f"    Through-thickness modulus (Reuss lower bound): {results['E_reuss']:.0f} MPa")
    print(f"    In-plane tensile strength: {results['sigma_t']:.1f} MPa")
    print(f"    Average density: {results['density']:.2f} g/cm3")
    print(f"    Specific stiffness (E/rho): {results['specific_stiffness']:.0f} MPa*cm3/g")

    # CLT results
    abd = results['abd']
    print(f"\n  Classical Laminate Theory:")
    print(f"    Effective in-plane Ex: {abd['Ex_eff']:.0f} MPa")
    print(f"    Effective in-plane Ey: {abd['Ey_eff']:.0f} MPa")
    print(f"    Effective flexural rigidity D11: {abd['EI_eff']:.1f} N*mm")
    print(f"    Neutral axis: {abd['z_neutral']:.3f} mm from bottom")
    print(f"    Symmetric layup: {'Yes' if abd['symmetric'] else 'No (extension-bending coupling present)'}")

    # Interface analysis
    print(f"\n  Interface Analysis:")
    print(f"    Material transitions per pattern: {results['transitions_per_pattern']}")
    print(f"    Total material interfaces: {results['interfaces']}")
    print(f"    Interface density: {results['interfaces'] / total_height:.1f} interfaces/mm")
    print(f"    Modulus mismatch ratio: {results['E_ratio']:.1f}x")

    if results['E_ratio'] > 10:
        print(f"    HIGH stress concentration risk at interfaces")
    elif results['E_ratio'] > 3:
        print(f"    MODERATE stress concentration at interfaces")
    else:
        print(f"    LOW stress concentration - smooth load transfer")

    # Adhesion compatibility
    for warn in results['adhesion_warnings']:
        score_str = f"{warn['score']}/5" if warn['score'] is not None else "unknown"
        print(f"    {warn['mat1']} + {warn['mat2']}: adhesion {score_str} - {warn['note']}")

    # Comparison to homogeneous
    print(f"\n  Comparison to Homogeneous Materials:")
    h = total_height
    I_homo = h**3 / 12
    for f, mat in materials.items():
        EI_homo = mat.E * I_homo
        ratio = abd['EI_eff'] / EI_homo if EI_homo > 0 else 0
        print(f"    vs pure {mat.name}: {ratio:.2f}x flexural rigidity")

    print("\n" + "=" * 70)
