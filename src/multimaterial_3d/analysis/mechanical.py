"""
Mechanical property analysis for multi-material layered FDM prints.

Implements Classical Laminate Theory (CLT) adapted for FDM printing, including:
- Orthotropic reduced stiffness matrix with raster angle rotation
- Full ABD stiffness matrix for laminate bending/extension coupling
- Voigt/Reuss/Hashin-Shtrikman bounds for effective properties
- Through-thickness properties using E_z and Reuss averaging
- Neutral axis calculation for asymmetric layups
- Interlaminar shear stress distribution
- First-ply failure analysis using Tsai-Wu criterion
- Interface stress concentration analysis

FDM Anisotropy Model
---------------------
FDM parts are orthotropic at the layer level due to raster (bead) orientation:
- E1 = modulus along rasters (longitudinal), typically the reported "E"
- E2 = modulus transverse to rasters, typically 60-85% of E1 for FDM
- G12 = in-plane shear modulus
- nu12 = major Poisson's ratio

When rasters alternate at 0/90 or +45/-45 between layers, the laminate
becomes quasi-isotropic in-plane. This module handles both cases: if a
raster angle is specified per layer, it uses the full transformation; if
not, it defaults to quasi-isotropic (E1 = E2 = E).

The reduced stiffness matrix Q relates in-plane stresses to strains for
a single orthotropic lamina in its material coordinate system:

    [s1]   [Q11 Q12  0 ] [e1]
    [s2] = [Q12 Q22  0 ] [e2]
    [t12]  [ 0   0  Q66] [g12]

For an off-axis lamina rotated by angle theta, the transformed stiffness
Q_bar is computed using the rotation transformation matrix T.

Classical Laminate Theory (CLT)
-------------------------------
The ABD matrix relates force/moment resultants to midplane strains/curvatures:

    [N]   [A  B] [eps0]
    [M] = [B  D] [kappa]

Components are integrated through the thickness:
    A_ij = Sum_k Q_ij^k * (z_k - z_{k-1})
    B_ij = (1/2) Sum_k Q_ij^k * (z_k^2 - z_{k-1}^2)
    D_ij = (1/3) Sum_k Q_ij^k * (z_k^3 - z_{k-1}^3)

Hashin-Shtrikman Bounds
-----------------------
Tighter than Voigt/Reuss for two-phase composites. For bulk modulus K
and shear modulus G of phases 1 and 2 with volume fractions V1 and V2:

    K_HS+ = K2 + V1 / (1/(K1-K2) + V2/(K2 + G2))
    K_HS- = K1 + V2 / (1/(K2-K1) + V1/(K1 + G1))

These give the tightest possible bounds for an isotropic composite
of two isotropic phases with known volume fractions but unknown
microstructure.

Tsai-Wu Failure Criterion
--------------------------
A quadratic interaction failure criterion for orthotropic materials:

    F1*s1 + F2*s2 + F11*s1^2 + F22*s2^2 + F66*t12^2 + 2*F12*s1*s2 = 1

where F_i and F_ij are strength parameters. The failure index (FI) for
each layer is computed; FI >= 1 indicates failure. This is more accurate
than maximum stress/strain criteria because it accounts for stress
interaction effects.

References
----------
- Jones, R.M. "Mechanics of Composite Materials" (2nd ed., Taylor & Francis)
- Gibson, R.F. "Principles of Composite Material Mechanics" (4th ed., CRC)
- Tsai, S.W. & Wu, E.M. "A General Theory of Strength for Anisotropic
  Materials" J. Composite Materials, 5:58-80 (1971)
- Hashin, Z. & Shtrikman, S. "A variational approach to the theory of the
  elastic behaviour of multiphase materials" (1963)
- Cantrell et al. "Experimental characterization of the mechanical
  properties of 3D-printed ABS and polycarbonate parts" (2017)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from ..core.materials import MaterialProperties, get_material, get_adhesion, MATERIAL_DB


# ─────────────────────────────────────────────────────────────────────────────
# Orthotropic FDM anisotropy ratio: E_transverse / E_raster
# Literature values for FDM parts printed at 0.2mm layer height.
# This captures the weak direction perpendicular to raster lines.
# ─────────────────────────────────────────────────────────────────────────────
FDM_ANISOTROPY_RATIO = {
    'rigid':       0.85,  # PLA, PETG, ABS etc. — moderate anisotropy
    'flexible':    0.95,  # TPU — nearly isotropic due to high elongation
    'engineering': 0.80,  # PA, PC — crystalline/semi-crystalline, more aniso
    'composite':   0.60,  # CF-filled — fibers align along rasters, strong aniso
    'support':     0.85,  # PVA, HIPS — similar to rigid
}


def compute_orthotropic_stiffness(mat: MaterialProperties,
                                   aniso_ratio: Optional[float] = None
                                   ) -> np.ndarray:
    """Compute orthotropic reduced stiffness matrix [Q] for a single FDM layer.

    For FDM printing, each layer is orthotropic with the principal material
    direction aligned with the raster (bead) direction. The transverse
    modulus E2 is lower than E1 due to inter-bead bonding being weaker
    than the continuous polymer along the bead.

    The reduced stiffness matrix relates in-plane stresses to in-plane
    strains in the material (1-2) coordinate system:

        [s1 ]   [Q11 Q12  0 ] [e1 ]
        [s2 ] = [Q12 Q22  0 ] [e2 ]
        [t12]   [ 0   0  Q66] [g12]

    where:
        Q11 = E1 / (1 - nu12 * nu21)
        Q22 = E2 / (1 - nu12 * nu21)
        Q12 = nu12 * E2 / (1 - nu12 * nu21) = nu21 * E1 / (1 - nu12 * nu21)
        Q66 = G12

    The reciprocal relation nu21 = nu12 * E2 / E1 is enforced for
    thermodynamic consistency (symmetric compliance matrix).

    Parameters
    ----------
    mat : MaterialProperties
        Material property set for this layer.
    aniso_ratio : float, optional
        Ratio E2/E1 for FDM anisotropy. If None, uses the default for the
        material category. Set to 1.0 for quasi-isotropic behavior.

    Returns
    -------
    np.ndarray
        3x3 reduced stiffness matrix [Q] in MPa.
    """
    E1 = mat.E
    if aniso_ratio is None:
        aniso_ratio = FDM_ANISOTROPY_RATIO.get(mat.category, 0.85)
    E2 = E1 * aniso_ratio

    nu12 = mat.nu
    # Reciprocal relation ensures compliance matrix symmetry
    nu21 = nu12 * E2 / E1
    G12 = mat.G_xy

    denom = 1.0 - nu12 * nu21
    Q11 = E1 / denom
    Q22 = E2 / denom
    Q12 = nu12 * E2 / denom  # = nu21 * E1 / denom (by reciprocal relation)
    Q66 = G12

    return np.array([
        [Q11, Q12, 0.0],
        [Q12, Q22, 0.0],
        [0.0, 0.0, Q66]
    ])


def rotate_stiffness_matrix(Q: np.ndarray, theta_deg: float) -> np.ndarray:
    """Transform the reduced stiffness matrix from material to laminate axes.

    When raster lines are oriented at angle theta from the laminate x-axis,
    the stiffness matrix must be rotated. The transformation uses:

        Q_bar = T_inv @ Q @ R @ T @ R_inv

    where T is the stress transformation matrix and R is the Reuter matrix
    that converts engineering shear strain to tensorial shear strain.

    For the common FDM case of alternating 0/90 rasters, average the
    Q matrices at 0 and 90 degrees for a balanced laminate approximation.

    Parameters
    ----------
    Q : np.ndarray
        3x3 reduced stiffness matrix in material coordinates.
    theta_deg : float
        Raster angle in degrees (0 = along x-axis).

    Returns
    -------
    np.ndarray
        3x3 transformed stiffness matrix in laminate coordinates.
    """
    theta = np.radians(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)

    # Stress transformation matrix (Voigt notation)
    T = np.array([
        [c**2,    s**2,    2*c*s    ],
        [s**2,    c**2,   -2*c*s    ],
        [-c*s,    c*s,     c**2 - s**2]
    ])

    T_inv = np.linalg.inv(T)

    # Reuter matrix: converts engineering shear strain to tensorial
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0]
    ])
    R_inv = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.5]
    ])

    Q_bar = T_inv @ Q @ R @ T @ R_inv
    return Q_bar


def compute_layer_stiffness_matrix(mat: MaterialProperties,
                                    raster_angle: float = None,
                                    quasi_isotropic: bool = True
                                    ) -> np.ndarray:
    """Compute the effective in-plane stiffness matrix for an FDM layer.

    Two modes of operation:
    1. quasi_isotropic=True (default): Assumes alternating 0/90 rasters within
       the layer, averaging the Q matrices. This gives E_x = E_y, appropriate
       for Bambu Studio's default cross-raster pattern.
    2. quasi_isotropic=False with raster_angle: Uses the specified angle for
       single-direction rasters.

    Parameters
    ----------
    mat : MaterialProperties
        Material property set.
    raster_angle : float, optional
        Raster angle in degrees. Only used when quasi_isotropic=False.
    quasi_isotropic : bool
        If True, average 0 and 90 degree Q matrices (default Bambu behavior).

    Returns
    -------
    np.ndarray
        3x3 reduced stiffness matrix [Q] in MPa.
    """
    Q_mat = compute_orthotropic_stiffness(mat)

    if quasi_isotropic:
        # Average 0 and 90 degree orientations for balanced laminate
        Q_0 = Q_mat
        Q_90 = rotate_stiffness_matrix(Q_mat, 90.0)
        return 0.5 * (Q_0 + Q_90)
    elif raster_angle is not None:
        return rotate_stiffness_matrix(Q_mat, raster_angle)
    else:
        return Q_mat


def compute_abd_matrix(pattern: List[int], layer_height: float,
                       total_height: float,
                       material_map: Dict[int, str],
                       raster_angles: Optional[List[float]] = None
                       ) -> Dict[str, np.ndarray]:
    """Compute the full ABD stiffness matrix for a multi-material laminate.

    The ABD matrix is the fundamental result of Classical Laminate Theory.
    It relates the force and moment resultants to the mid-plane strains
    and curvatures:

        [N]   [A  B] [eps0]
        [M] = [B  D] [kappa]

    where:
        N = in-plane force resultants (N/mm) [Nx, Ny, Nxy]
        M = moment resultants (N*mm/mm) [Mx, My, Mxy]
        eps0 = mid-plane strains [-] [ex0, ey0, gxy0]
        kappa = curvatures (1/mm) [kx, ky, kxy]

    Parameters
    ----------
    pattern : list of int
        Repeating filament pattern (e.g., [1, 2, 2, 1]).
    layer_height : float
        Layer height in mm.
    total_height : float
        Total part height in mm.
    material_map : dict
        Filament number to material key (e.g., {1: 'PLA', 2: 'TPU'}).
    raster_angles : list of float, optional
        Per-layer raster angles in degrees. If None, quasi-isotropic assumed.

    Returns
    -------
    dict
        'A': 3x3 extensional stiffness matrix (N/mm)
        'B': 3x3 coupling matrix (N)
        'D': 3x3 bending stiffness matrix (N*mm)
        'ABD': 6x6 combined matrix
        'abd_inv': 6x6 compliance matrix (inverse of ABD)
        'z_neutral': neutral axis position from bottom (mm)
        'symmetric': bool, whether the layup is symmetric (B ~= 0)
        'Ex_eff': effective in-plane modulus in x (MPa)
        'Ey_eff': effective in-plane modulus in y (MPa)
        'Gxy_eff': effective in-plane shear modulus (MPa)
        'nuxy_eff': effective in-plane Poisson's ratio
        'EI_eff': effective flexural rigidity D11 per unit width (N*mm)
        'Ef_x': effective flexural modulus in x (MPa)
    """
    num_layers = int(total_height / layer_height)
    pattern_len = len(pattern)

    # Reference plane at geometric mid-plane
    h_total = num_layers * layer_height
    z_ref = h_total / 2.0

    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))

    for k in range(num_layers):
        filament = pattern[k % pattern_len]
        mat_key = material_map.get(filament, 'PLA')
        mat = get_material(mat_key)

        if raster_angles is not None and k < len(raster_angles):
            Q = compute_layer_stiffness_matrix(mat, raster_angle=raster_angles[k],
                                                quasi_isotropic=False)
        else:
            Q = compute_layer_stiffness_matrix(mat, quasi_isotropic=True)

        # Layer boundaries relative to mid-plane (z=0 at midplane)
        z_bot = k * layer_height - z_ref
        z_top = (k + 1) * layer_height - z_ref

        # CLT integration — exact for constant Q within each layer
        A += Q * (z_top - z_bot)
        B += 0.5 * Q * (z_top**2 - z_bot**2)
        D += (1.0 / 3.0) * Q * (z_top**3 - z_bot**3)

    # Assemble 6x6 ABD matrix
    ABD = np.zeros((6, 6))
    ABD[:3, :3] = A
    ABD[:3, 3:] = B
    ABD[3:, :3] = B
    ABD[3:, 3:] = D

    # Compliance matrix
    try:
        abd_inv = np.linalg.inv(ABD)
    except np.linalg.LinAlgError:
        abd_inv = np.full((6, 6), np.nan)

    # Check symmetry: B ~= 0 means symmetric layup (no extension-bending coupling)
    A_norm = np.max(np.abs(A)) if np.max(np.abs(A)) > 0 else 1.0
    is_symmetric = np.allclose(B, 0, atol=1e-6 * A_norm)

    # Effective in-plane properties from A-matrix compliance
    # For a laminate: a = inv(A), then Ex = 1/(h*a11), Ey = 1/(h*a22), etc.
    a = np.linalg.inv(A)
    Ex_eff = 1.0 / (h_total * a[0, 0])
    Ey_eff = 1.0 / (h_total * a[1, 1])
    Gxy_eff = 1.0 / (h_total * a[2, 2])
    nuxy_eff = -a[0, 1] / a[0, 0]

    # Effective flexural properties from D-matrix compliance
    # Ef_x = 12 / (h^3 * d11) where d = inv(D)
    d = np.linalg.inv(D)
    Ef_x = 12.0 / (h_total**3 * d[0, 0])
    EI_eff = D[0, 0]  # Flexural rigidity per unit width

    # Neutral axis from first moment of axial stiffness
    sum_EA_z = 0.0
    sum_EA = 0.0
    for k in range(num_layers):
        filament = pattern[k % pattern_len]
        mat = get_material(material_map.get(filament, 'PLA'))
        z_mid_layer = (k + 0.5) * layer_height
        sum_EA_z += mat.E * layer_height * z_mid_layer
        sum_EA += mat.E * layer_height
    z_neutral = sum_EA_z / sum_EA if sum_EA > 0 else h_total / 2.0

    return {
        'A': A, 'B': B, 'D': D,
        'ABD': ABD, 'abd_inv': abd_inv,
        'z_neutral': z_neutral,
        'symmetric': is_symmetric,
        'Ex_eff': Ex_eff, 'Ey_eff': Ey_eff,
        'Gxy_eff': Gxy_eff, 'nuxy_eff': nuxy_eff,
        'EI_eff': EI_eff, 'Ef_x': Ef_x,
    }


def compute_hashin_shtrikman_bounds(mat1: MaterialProperties, mat2: MaterialProperties,
                                     v1: float) -> dict:
    """Compute Hashin-Shtrikman bounds for a two-phase composite.

    These are the tightest possible bounds on the elastic moduli of an
    isotropic composite of two isotropic phases, given only volume fractions.
    They are tighter than Voigt/Reuss because they incorporate additional
    physical constraints (positive definiteness of strain energy).

    For the special case of FDM multi-material laminates, the HS bounds
    bracket the true effective properties more tightly than simple
    Rule of Mixtures.

    The bounds are derived from the variational principle:
        K_HS+ uses the stiffer material as the "matrix" phase
        K_HS- uses the softer material as the "matrix" phase

    From K_HS and G_HS, the effective E and nu are computed using
    standard isotropic relations.

    Parameters
    ----------
    mat1, mat2 : MaterialProperties
        The two constituent materials.
    v1 : float
        Volume fraction of material 1 (0 to 1).

    Returns
    -------
    dict
        'E_HS_upper': Upper HS bound on Young's modulus (MPa)
        'E_HS_lower': Lower HS bound on Young's modulus (MPa)
        'K_HS_upper': Upper bound on bulk modulus (MPa)
        'K_HS_lower': Lower bound on bulk modulus (MPa)
        'G_HS_upper': Upper bound on shear modulus (MPa)
        'G_HS_lower': Lower bound on shear modulus (MPa)
    """
    v2 = 1.0 - v1

    # Convert E, nu to K, G for each phase (plane stress bulk modulus)
    K1 = mat1.E / (2.0 * (1.0 - mat1.nu))
    G1 = mat1.G_xy
    K2 = mat2.E / (2.0 * (1.0 - mat2.nu))
    G2 = mat2.G_xy

    # Ensure K1 >= K2 for upper/lower bound assignment
    if K1 < K2:
        K1, K2 = K2, K1
        G1, G2 = G2, G1
        v1, v2 = v2, v1

    # HS upper bounds (stiffer material as matrix)
    if abs(K1 - K2) > 1e-10:
        K_upper = K1 + v2 / (1.0 / (K2 - K1) + v1 / (K1 + G1))
    else:
        K_upper = K1

    if abs(G1 - G2) > 1e-10:
        G_upper = G1 + v2 / (1.0 / (G2 - G1) + v1 * (K1 + 2*G1) / (2*G1 * (K1 + G1)))
    else:
        G_upper = G1

    # HS lower bounds (softer material as matrix)
    if abs(K2 - K1) > 1e-10:
        K_lower = K2 + v1 / (1.0 / (K1 - K2) + v2 / (K2 + G2))
    else:
        K_lower = K2

    if abs(G2 - G1) > 1e-10:
        G_lower = G2 + v1 / (1.0 / (G1 - G2) + v2 * (K2 + 2*G2) / (2*G2 * (K2 + G2)))
    else:
        G_lower = G2

    # Convert back to E, nu
    E_upper = 9 * K_upper * G_upper / (3 * K_upper + G_upper)
    E_lower = 9 * K_lower * G_lower / (3 * K_lower + G_lower)

    return {
        'E_HS_upper': E_upper, 'E_HS_lower': E_lower,
        'K_HS_upper': K_upper, 'K_HS_lower': K_lower,
        'G_HS_upper': G_upper, 'G_HS_lower': G_lower,
    }


def compute_interlaminar_shear(abd_results: dict, pattern: List[int],
                                layer_height: float, total_height: float,
                                material_map: Dict[int, str],
                                V_applied: float = 1.0) -> List[dict]:
    """Compute interlaminar shear stress distribution through the thickness.

    Under transverse loading (beam bending), the interlaminar shear stress
    varies through the thickness according to:

        tau(z) = V * Q(z) / (b * EI)

    where V is the applied shear force, Q(z) is the first moment of the
    transformed area above z, b is the width, and EI is the flexural
    rigidity. For a multi-material laminate, the "transformed area" uses
    E_k / E_ref to weight each layer's contribution.

    The shear stress is zero at the free surfaces (top and bottom) and
    maximum at or near the neutral axis. At material interfaces, the shear
    stress can cause delamination if it exceeds the inter-layer bond strength.

    Parameters
    ----------
    abd_results : dict
        Results from compute_abd_matrix.
    pattern : list of int
        Repeating filament pattern.
    layer_height : float
        Layer height in mm.
    total_height : float
        Total part height in mm.
    material_map : dict
        Filament number to material key.
    V_applied : float
        Applied transverse shear force per unit width (N/mm). Default 1.0
        gives stress per unit shear force.

    Returns
    -------
    list of dict
        For each interface between layers:
        'z': interface z position (mm from bottom)
        'tau': interlaminar shear stress (MPa)
        'mat_above': material name above interface
        'mat_below': material name below interface
        'is_material_change': whether this is a material boundary
    """
    num_layers = int(total_height / layer_height)
    pattern_len = len(pattern)
    z_neutral = abd_results['z_neutral']
    EI = abd_results['EI_eff']

    if EI <= 0:
        return []

    interfaces = []

    for i in range(1, num_layers):
        z_interface = i * layer_height

        # First moment of area above the cut plane, weighted by modulus
        # Q(z) = Sum_{k above z} E_k * A_k * (z_k_centroid - z_neutral)
        # where A_k = layer_height * b (per unit width, b=1)
        Q_above = 0.0
        for k in range(i, num_layers):
            filament = pattern[k % pattern_len]
            mat = get_material(material_map.get(filament, 'PLA'))
            z_centroid = (k + 0.5) * layer_height
            Q_above += mat.E * layer_height * (z_centroid - z_neutral)

        # tau = V * Q / (b * EI), with b = 1 (per unit width)
        tau = V_applied * abs(Q_above) / EI

        mat_below_key = material_map.get(pattern[(i - 1) % pattern_len], 'PLA')
        mat_above_key = material_map.get(pattern[i % pattern_len], 'PLA')

        interfaces.append({
            'z': z_interface,
            'tau': tau,
            'mat_below': get_material(mat_below_key).name,
            'mat_above': get_material(mat_above_key).name,
            'is_material_change': mat_below_key != mat_above_key,
        })

    return interfaces


def compute_tsai_wu_failure(pattern: List[int], layer_height: float,
                             total_height: float, material_map: Dict[int, str],
                             Nx: float = 0.0, Ny: float = 0.0, Nxy: float = 0.0,
                             Mx: float = 0.0, My: float = 0.0, Mxy: float = 0.0
                             ) -> List[dict]:
    """First-ply failure analysis using the Tsai-Wu criterion.

    Computes the stress state in each layer under the given loading and
    evaluates the Tsai-Wu failure index. The first layer to reach FI >= 1
    determines the first-ply failure load.

    The Tsai-Wu criterion for plane stress:
        F1*s1 + F2*s2 + F11*s1^2 + F22*s2^2 + F66*t12^2 + 2*F12*s1*s2 = 1

    where the strength coefficients are:
        F1 = 1/Xt - 1/Xc     (different tension/compression strengths)
        F2 = 1/Yt - 1/Yc
        F11 = 1/(Xt*Xc)
        F22 = 1/(Yt*Yc)
        F66 = 1/S^2
        F12 = -0.5 * sqrt(F11 * F22)  (conservative interaction term)

    For FDM materials:
        Xt = sigma_t (tensile strength along rasters)
        Xc = sigma_c (compressive strength along rasters)
        Yt = sigma_t * aniso_ratio (transverse tensile, weaker)
        Yc = sigma_c * aniso_ratio (transverse compressive)
        S = sigma_t / 2 (approximate shear strength)

    Parameters
    ----------
    pattern : list of int
        Repeating filament pattern.
    layer_height, total_height : float
        Layer and total height in mm.
    material_map : dict
        Filament number to material key.
    Nx, Ny, Nxy : float
        Applied in-plane force resultants (N/mm).
    Mx, My, Mxy : float
        Applied moment resultants (N*mm/mm).

    Returns
    -------
    list of dict
        Per-layer failure analysis: 'layer', 'material', 'z_mid',
        'stress' (s1, s2, t12), 'failure_index', 'mode'.
    """
    abd = compute_abd_matrix(pattern, layer_height, total_height, material_map)
    num_layers = int(total_height / layer_height)
    pattern_len = len(pattern)
    h_total = num_layers * layer_height
    z_ref = h_total / 2.0

    # Solve for midplane strains and curvatures
    load = np.array([Nx, Ny, Nxy, Mx, My, Mxy])
    try:
        response = abd['abd_inv'] @ load
    except Exception:
        return []

    eps0 = response[:3]
    kappa = response[3:]

    results = []
    for k in range(num_layers):
        filament = pattern[k % pattern_len]
        mat_key = material_map.get(filament, 'PLA')
        mat = get_material(mat_key)
        Q = compute_layer_stiffness_matrix(mat)

        # Strain at layer midpoint
        z_mid = (k + 0.5) * layer_height - z_ref
        strain = eps0 + kappa * z_mid

        # Stress in laminate coordinates
        stress = Q @ strain
        s1, s2, t12 = stress[0], stress[1], stress[2]

        # Tsai-Wu strength parameters
        aniso = FDM_ANISOTROPY_RATIO.get(mat.category, 0.85)
        Xt = mat.sigma_t
        Xc = mat.sigma_c
        Yt = mat.sigma_t * aniso  # Weaker transverse to rasters
        Yc = mat.sigma_c * aniso
        S = mat.sigma_t * 0.5  # Approximate shear strength

        F1 = 1.0 / Xt - 1.0 / Xc
        F2 = 1.0 / Yt - 1.0 / Yc
        F11 = 1.0 / (Xt * Xc)
        F22 = 1.0 / (Yt * Yc)
        F66 = 1.0 / S**2
        # Conservative interaction term (Tsai-Wu recommendation)
        F12 = -0.5 * np.sqrt(F11 * F22)

        # Failure index
        FI = (F1 * s1 + F2 * s2 +
              F11 * s1**2 + F22 * s2**2 + F66 * t12**2 +
              2 * F12 * s1 * s2)

        # Determine dominant failure mode
        if FI > 0:
            contributions = {
                'fiber_tension':  F11 * s1**2 if s1 > 0 else 0,
                'fiber_compression': F11 * s1**2 if s1 < 0 else 0,
                'matrix_tension': F22 * s2**2 if s2 > 0 else 0,
                'matrix_compression': F22 * s2**2 if s2 < 0 else 0,
                'shear': F66 * t12**2,
            }
            mode = max(contributions, key=contributions.get)
        else:
            mode = 'none'

        results.append({
            'layer': k,
            'material': mat.name,
            'z_mid': (k + 0.5) * layer_height,
            'stress': (s1, s2, t12),
            'failure_index': FI,
            'mode': mode,
        })

    return results


def analyze_layup(pattern: List[int], total_height: float,
                  layer_height: float, material_map: Dict[int, str],
                  verbose: bool = True) -> dict:
    """Comprehensive mechanical analysis of a multi-material layup.

    Performs:
    - Voigt/Reuss bounds (Rule of Mixtures)
    - Hashin-Shtrikman bounds (tighter than Voigt/Reuss)
    - Full CLT analysis (ABD matrix, effective properties)
    - Through-thickness properties using E_z
    - Interface stress concentration analysis
    - Adhesion compatibility assessment

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
        Comprehensive analysis results.
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

    materials = {}
    for filament in layer_counts:
        mat_key = material_map.get(filament, 'PLA')
        materials[filament] = get_material(mat_key)

    # ── Rule of Mixtures (Voigt/Reuss) ───────────────────────────────────
    E_voigt = sum(volume_fractions[f] * materials[f].E for f in layer_counts)
    sigma_t_voigt = sum(volume_fractions[f] * materials[f].sigma_t for f in layer_counts)

    E_reuss = 1.0 / sum(volume_fractions[f] / materials[f].E for f in layer_counts)

    # Through-thickness Reuss using E_z (physically correct for Z-loading)
    E_z_reuss = 1.0 / sum(volume_fractions[f] / materials[f].E_z for f in layer_counts)

    density_avg = sum(volume_fractions[f] * materials[f].density for f in layer_counts)

    # ── Hashin-Shtrikman bounds (for 2-material systems) ─────────────────
    hs_bounds = None
    filament_list = sorted(layer_counts.keys())
    if len(filament_list) == 2:
        f1, f2 = filament_list
        hs_bounds = compute_hashin_shtrikman_bounds(
            materials[f1], materials[f2], volume_fractions[f1])

    # ── CLT analysis ─────────────────────────────────────────────────────
    abd_results = compute_abd_matrix(pattern, layer_height, total_height, material_map)

    # ── Interface analysis ───────────────────────────────────────────────
    transitions_per_pattern = 0
    for i in range(pattern_len):
        if pattern[i] != pattern[(i + 1) % pattern_len]:
            transitions_per_pattern += 1

    complete_cycles = num_layers // pattern_len
    remaining = num_layers % pattern_len
    total_interfaces = transitions_per_pattern * complete_cycles
    for i in range(max(remaining - 1, 0)):
        if pattern[i] != pattern[i + 1]:
            total_interfaces += 1

    # ── Adhesion compatibility ───────────────────────────────────────────
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

    # ── Modulus mismatch ─────────────────────────────────────────────────
    E_values = [materials[f].E for f in layer_counts]
    E_ratio = max(E_values) / min(E_values) if min(E_values) > 0 else float('inf')

    results = {
        'E_voigt': E_voigt,
        'E_reuss': E_reuss,
        'E_z_reuss': E_z_reuss,
        'sigma_t': sigma_t_voigt,
        'density': density_avg,
        'specific_stiffness': E_voigt / density_avg,
        'hs_bounds': hs_bounds,
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
        _print_analysis_report(results, pattern, materials, material_map,
                               layer_height, total_height)

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
        aniso = FDM_ANISOTROPY_RATIO.get(mat.category, 0.85)
        print(f"    F{f} ({mat.name}): E1={mat.E} MPa, E2={mat.E*aniso:.0f} MPa, "
              f"E_z={mat.E_z} MPa, CTE={mat.CTE:.1e} 1/K")

    # Bounds
    print(f"\n  Effective Modulus Bounds:")
    print(f"    Voigt (iso-strain, upper):  {results['E_voigt']:.0f} MPa")
    if results['hs_bounds']:
        hs = results['hs_bounds']
        print(f"    Hashin-Shtrikman upper:     {hs['E_HS_upper']:.0f} MPa")
        print(f"    Hashin-Shtrikman lower:     {hs['E_HS_lower']:.0f} MPa")
    print(f"    Reuss (iso-stress, lower):  {results['E_reuss']:.0f} MPa")
    print(f"    Through-thickness (E_z):    {results['E_z_reuss']:.0f} MPa")
    print(f"    Tensile strength:           {results['sigma_t']:.1f} MPa")
    print(f"    Average density:            {results['density']:.2f} g/cm3")
    print(f"    Specific stiffness (E/rho): {results['specific_stiffness']:.0f} MPa*cm3/g")

    # CLT results
    abd = results['abd']
    print(f"\n  Classical Laminate Theory (CLT):")
    print(f"    In-plane Ex:    {abd['Ex_eff']:.0f} MPa")
    print(f"    In-plane Ey:    {abd['Ey_eff']:.0f} MPa")
    print(f"    In-plane Gxy:   {abd['Gxy_eff']:.0f} MPa")
    print(f"    In-plane nuxy:  {abd['nuxy_eff']:.3f}")
    print(f"    Flexural Ex:    {abd['Ef_x']:.0f} MPa")
    print(f"    Flexural D11:   {abd['EI_eff']:.1f} N*mm")
    print(f"    Neutral axis:   {abd['z_neutral']:.3f} mm from bottom")
    sym_str = 'Yes' if abd['symmetric'] else 'No (extension-bending coupling -> warping risk)'
    print(f"    Symmetric:      {sym_str}")

    # Interface analysis
    print(f"\n  Interface Analysis:")
    print(f"    Transitions per pattern: {results['transitions_per_pattern']}")
    print(f"    Total material interfaces: {results['interfaces']}")
    print(f"    Interface density: {results['interfaces'] / total_height:.1f} /mm")
    print(f"    Modulus mismatch ratio: {results['E_ratio']:.1f}x")

    if results['E_ratio'] > 10:
        print(f"    -> HIGH stress concentration risk at interfaces")
    elif results['E_ratio'] > 3:
        print(f"    -> MODERATE stress concentration at interfaces")
    else:
        print(f"    -> LOW stress concentration - smooth load transfer")

    for warn in results['adhesion_warnings']:
        score_str = f"{warn['score']}/5" if warn['score'] is not None else "unknown"
        print(f"    {warn['mat1']} + {warn['mat2']}: adhesion {score_str} - {warn['note']}")

    # Comparison to homogeneous
    print(f"\n  Comparison to Homogeneous Materials:")
    h = total_height
    for f, mat in materials.items():
        EI_homo = mat.E * h**3 / 12.0
        ratio = abd['EI_eff'] / EI_homo if EI_homo > 0 else 0
        print(f"    vs pure {mat.name}: {ratio:.2f}x flexural rigidity")

    print("\n" + "=" * 70)
