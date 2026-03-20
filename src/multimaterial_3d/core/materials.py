"""
Comprehensive material property database for FDM/FFF 3D printing filaments.

Each material entry contains mechanical, thermal, and economic properties needed
for structural analysis, thermal stress prediction, and cost optimization.

Property Definitions
--------------------
E : float
    Young's modulus (tensile) in MPa. Measured in the XY (in-plane) direction
    for FDM parts, which is typically the stiffest printing direction.
E_z : float
    Young's modulus in the Z (build/through-thickness) direction in MPa.
    FDM parts are significantly weaker in Z due to inter-layer bonding.
    Typically 50-80% of the in-plane modulus for well-bonded layers.
sigma_t : float
    Ultimate tensile strength in MPa (XY direction).
sigma_c : float
    Compressive strength in MPa. Printed parts are generally stronger in
    compression than tension because layer interfaces are less critical.
sigma_t_z : float
    Tensile strength in Z direction in MPa. This is the inter-layer bond
    strength, typically the weakest property of FDM parts.
G_xy : float
    In-plane shear modulus in MPa. For isotropic materials, G = E / (2*(1+nu)).
    FDM parts may deviate due to raster orientation effects.
nu : float
    Poisson's ratio (dimensionless). Assumed quasi-isotropic in-plane.
density : float
    Material density in g/cm^3.
CTE : float
    Coefficient of thermal expansion in 1/K (or 1/degC). This is critical
    for predicting thermal stresses and warping at multi-material interfaces.
T_print : float
    Typical nozzle/print temperature in degC.
T_bed : float
    Typical bed temperature in degC.
T_glass : float
    Glass transition temperature in degC. Below this, the material is rigid;
    above it, the polymer chains gain mobility and the material softens.
    Thermal stresses lock in as the part cools through Tg.
thermal_conductivity : float
    Thermal conductivity in W/(m*K). Affects cooling rate and thermal gradients.
cost_per_kg : float
    Approximate material cost in USD/kg for cost optimization.
name : str
    Human-readable material name.
category : str
    Material family classification (e.g., 'rigid', 'flexible', 'engineering',
    'composite').

Sources
-------
- Manufacturer datasheets (Bambu Lab, Polymaker, eSUN, Prusament)
- CES EduPack polymer database
- Published research on FDM mechanical properties
- ASTM D638 (tensile), ASTM D695 (compression) test data from literature

Notes
-----
All mechanical properties assume 100% infill, 0.2mm layer height, and
typical print settings. Actual values depend heavily on:
- Print orientation and raster angle
- Layer height and extrusion width
- Print speed and temperature
- Infill density and pattern
- Part geometry and cooling conditions
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class MaterialProperties:
    """Complete material property set for a 3D printing filament.

    Contains mechanical, thermal, physical, and economic properties needed
    for multi-material analysis including CLT, thermal stress prediction,
    warping estimation, and cost optimization.
    """
    # Identification
    name: str
    category: str  # 'rigid', 'flexible', 'engineering', 'composite'

    # Mechanical - in-plane (XY)
    E: float           # Young's modulus, MPa
    sigma_t: float     # Tensile strength, MPa
    sigma_c: float     # Compressive strength, MPa
    G_xy: float        # Shear modulus, MPa
    nu: float          # Poisson's ratio

    # Mechanical - through-thickness (Z)
    E_z: float         # Z-direction modulus, MPa
    sigma_t_z: float   # Inter-layer bond strength, MPa

    # Physical
    density: float     # g/cm^3

    # Thermal
    CTE: float             # Coefficient of thermal expansion, 1/K
    T_print: float         # Print temperature, degC
    T_bed: float           # Bed temperature, degC
    T_glass: float         # Glass transition temperature, degC
    thermal_conductivity: float  # W/(m*K)

    # Economic
    cost_per_kg: float  # USD/kg


# ─────────────────────────────────────────────────────────────────────────────
# Material Database
#
# Values are representative of well-tuned FDM prints at 0.2mm layer height.
# Inter-layer (Z) properties assume good layer adhesion; poor adhesion can
# reduce sigma_t_z by 50% or more.
# ─────────────────────────────────────────────────────────────────────────────
MATERIAL_DB: Dict[str, MaterialProperties] = {

    'PLA': MaterialProperties(
        name='PLA (Polylactic Acid)',
        category='rigid',
        E=3500, sigma_t=50, sigma_c=70, G_xy=1300, nu=0.36,
        E_z=2450, sigma_t_z=25,
        density=1.24,
        CTE=68e-6, T_print=215, T_bed=60, T_glass=60,
        thermal_conductivity=0.13,
        cost_per_kg=20,
    ),

    'PETG': MaterialProperties(
        name='PETG (Polyethylene Terephthalate Glycol)',
        category='rigid',
        E=2100, sigma_t=45, sigma_c=55, G_xy=780, nu=0.38,
        E_z=1470, sigma_t_z=22,
        density=1.27,
        CTE=60e-6, T_print=240, T_bed=80, T_glass=80,
        thermal_conductivity=0.15,
        cost_per_kg=22,
    ),

    'ABS': MaterialProperties(
        name='ABS (Acrylonitrile Butadiene Styrene)',
        category='rigid',
        E=2300, sigma_t=40, sigma_c=65, G_xy=850, nu=0.35,
        E_z=1610, sigma_t_z=18,
        density=1.04,
        CTE=90e-6, T_print=250, T_bed=100, T_glass=105,
        thermal_conductivity=0.17,
        cost_per_kg=18,
    ),

    'TPU': MaterialProperties(
        name='TPU 95A (Thermoplastic Polyurethane)',
        category='flexible',
        E=26, sigma_t=30, sigma_c=20, G_xy=9, nu=0.48,
        E_z=18, sigma_t_z=15,
        density=1.21,
        CTE=150e-6, T_print=225, T_bed=60, T_glass=-40,
        thermal_conductivity=0.19,
        cost_per_kg=35,
    ),

    'ASA': MaterialProperties(
        name='ASA (Acrylonitrile Styrene Acrylate)',
        category='rigid',
        E=2200, sigma_t=42, sigma_c=60, G_xy=815, nu=0.35,
        E_z=1540, sigma_t_z=20,
        density=1.07,
        CTE=95e-6, T_print=255, T_bed=100, T_glass=100,
        thermal_conductivity=0.17,
        cost_per_kg=25,
    ),

    'PA': MaterialProperties(
        name='Nylon/PA (Polyamide 6)',
        category='engineering',
        E=1800, sigma_t=70, sigma_c=80, G_xy=650, nu=0.39,
        E_z=1260, sigma_t_z=30,
        density=1.14,
        CTE=80e-6, T_print=270, T_bed=80, T_glass=50,
        thermal_conductivity=0.25,
        cost_per_kg=40,
    ),

    'PC': MaterialProperties(
        name='PC (Polycarbonate)',
        category='engineering',
        E=2400, sigma_t=60, sigma_c=75, G_xy=880, nu=0.37,
        E_z=1680, sigma_t_z=28,
        density=1.20,
        CTE=65e-6, T_print=280, T_bed=110, T_glass=147,
        thermal_conductivity=0.20,
        cost_per_kg=35,
    ),

    'PLA-CF': MaterialProperties(
        name='PLA-CF (Carbon Fiber Reinforced PLA)',
        category='composite',
        E=8000, sigma_t=65, sigma_c=90, G_xy=3000, nu=0.33,
        E_z=4000, sigma_t_z=20,
        density=1.30,
        CTE=30e-6, T_print=230, T_bed=60, T_glass=60,
        thermal_conductivity=0.50,
        cost_per_kg=45,
    ),

    'PETG-CF': MaterialProperties(
        name='PETG-CF (Carbon Fiber Reinforced PETG)',
        category='composite',
        E=6000, sigma_t=55, sigma_c=75, G_xy=2200, nu=0.34,
        E_z=3000, sigma_t_z=18,
        density=1.35,
        CTE=35e-6, T_print=260, T_bed=80, T_glass=80,
        thermal_conductivity=0.45,
        cost_per_kg=50,
    ),

    'PA-CF': MaterialProperties(
        name='PA-CF (Carbon Fiber Reinforced Nylon)',
        category='composite',
        E=9500, sigma_t=80, sigma_c=100, G_xy=3500, nu=0.32,
        E_z=4750, sigma_t_z=25,
        density=1.25,
        CTE=25e-6, T_print=280, T_bed=80, T_glass=50,
        thermal_conductivity=0.55,
        cost_per_kg=60,
    ),

    'PVA': MaterialProperties(
        name='PVA (Polyvinyl Alcohol, water-soluble support)',
        category='support',
        E=1500, sigma_t=30, sigma_c=35, G_xy=550, nu=0.38,
        E_z=1050, sigma_t_z=10,
        density=1.23,
        CTE=85e-6, T_print=200, T_bed=50, T_glass=75,
        thermal_conductivity=0.20,
        cost_per_kg=50,
    ),

    'HIPS': MaterialProperties(
        name='HIPS (High Impact Polystyrene, dissolvable support)',
        category='support',
        E=2000, sigma_t=25, sigma_c=40, G_xy=740, nu=0.35,
        E_z=1400, sigma_t_z=12,
        density=1.04,
        CTE=80e-6, T_print=240, T_bed=100, T_glass=100,
        thermal_conductivity=0.16,
        cost_per_kg=20,
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Inter-material adhesion compatibility matrix
#
# Rated on a 0-5 scale:
#   5 = Excellent (same polymer family, chemical bond)
#   4 = Good (compatible polymers, reliable adhesion)
#   3 = Moderate (usable with tuned settings)
#   2 = Poor (weak bond, may delaminate under load)
#   1 = Very poor (will likely separate)
#   0 = Incompatible (do not combine)
#
# Matrix is symmetric. Only upper triangle is stored; lookup function
# handles both orderings.
# ─────────────────────────────────────────────────────────────────────────────
ADHESION_MATRIX: Dict[tuple, dict] = {
    ('ABS', 'ASA'):     {'score': 5, 'note': 'Same polymer family, excellent adhesion'},
    ('ABS', 'PC'):      {'score': 4, 'note': 'Often blended commercially, good adhesion'},
    ('ABS', 'PETG'):    {'score': 3, 'note': 'Moderate adhesion, similar print temps help'},
    ('ABS', 'PLA'):     {'score': 1, 'note': 'Poor adhesion, different shrinkage rates cause warping'},
    ('ABS', 'TPU'):     {'score': 4, 'note': 'Good adhesion, common rigid/flex combo'},
    ('ABS', 'HIPS'):    {'score': 5, 'note': 'Excellent, HIPS is standard ABS support material'},
    ('ASA', 'PLA'):     {'score': 1, 'note': 'Incompatible materials, poor adhesion'},
    ('PA', 'PLA'):      {'score': 1, 'note': 'Very different polymers, poor adhesion'},
    ('PA', 'TPU'):      {'score': 4, 'note': 'Good adhesion, both somewhat flexible'},
    ('PETG', 'PLA'):    {'score': 1, 'note': 'Poor adhesion, different surface energies'},
    ('PETG', 'TPU'):    {'score': 4, 'note': 'Good adhesion, both flexible-friendly'},
    ('PETG', 'PETG-CF'):{'score': 5, 'note': 'Same base material, excellent adhesion'},
    ('PLA', 'PLA-CF'):  {'score': 5, 'note': 'Same base material, excellent adhesion'},
    ('PLA', 'TPU'):     {'score': 5, 'note': 'Excellent adhesion, most popular rigid/flex combo'},
    ('PLA', 'PVA'):     {'score': 5, 'note': 'Excellent, PVA is standard PLA support'},
    ('PA', 'PA-CF'):    {'score': 5, 'note': 'Same base material, excellent adhesion'},
    ('PC', 'ABS'):      {'score': 4, 'note': 'Good adhesion, PC/ABS is a common alloy'},
}


def get_material(key: str) -> MaterialProperties:
    """Look up a material by its short key (e.g., 'PLA', 'TPU').

    Parameters
    ----------
    key : str
        Case-insensitive material identifier.

    Returns
    -------
    MaterialProperties
        The full property set for the requested material.

    Raises
    ------
    KeyError
        If the material is not found in the database. The error message
        lists all available materials to help the user correct typos.
    """
    key_upper = key.upper().strip()
    if key_upper in MATERIAL_DB:
        return MATERIAL_DB[key_upper]
    available = ', '.join(sorted(MATERIAL_DB.keys()))
    raise KeyError(f"Unknown material '{key}'. Available: {available}")


def list_materials() -> None:
    """Print a formatted table of all available materials and key properties."""
    print(f"{'Key':<10} {'Name':<40} {'E(MPa)':<8} {'σt(MPa)':<8} {'ρ(g/cm³)':<8} {'CTE(1/K)':<12} {'$/kg':<6}")
    print("-" * 92)
    for key, mat in sorted(MATERIAL_DB.items()):
        print(f"{key:<10} {mat.name:<40} {mat.E:<8.0f} {mat.sigma_t:<8.0f} {mat.density:<8.2f} {mat.CTE:<12.1e} {mat.cost_per_kg:<6.0f}")


def get_adhesion(mat1: str, mat2: str) -> dict:
    """Look up inter-material adhesion compatibility.

    Parameters
    ----------
    mat1, mat2 : str
        Material keys (order does not matter).

    Returns
    -------
    dict
        Dictionary with 'score' (0-5) and 'note' (str).
        If the pair is not in the database, returns score=None with a note
        to test adhesion experimentally.
    """
    pair = tuple(sorted([mat1.upper(), mat2.upper()]))
    if pair[0] == pair[1]:
        return {'score': 5, 'note': 'Same material, perfect adhesion'}
    if pair in ADHESION_MATRIX:
        return ADHESION_MATRIX[pair]
    return {'score': None, 'note': f'No data for {pair[0]}+{pair[1]} — test adhesion before production use'}
