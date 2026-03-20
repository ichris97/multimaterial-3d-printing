"""
3D visualizations for mechanical and thermal analysis results.

Creates PyVista meshes that represent:
- Layup cross-section (colored slabs per material)
- Through-thickness stress distribution
- Thermal interface stresses
- Warping deformation
"""

import numpy as np
import pyvista as pv
from typing import Dict, List, Tuple


def create_layup_viz(pattern: List[int], layer_height: float,
                     total_height: float, material_map: Dict[int, str],
                     width: float = 40.0, depth: float = 20.0
                     ) -> Tuple[pv.PolyData, dict]:
    """Create a 3D layup cross-section visualization.

    Builds a stack of colored rectangular slabs, one per layer, with each
    slab colored by its material. This shows the physical layup structure.

    Returns (mesh, scalar_info) where scalar_info maps material IDs to names.
    """
    num_layers = round(total_height / layer_height)
    pattern_len = len(pattern)

    blocks = pv.MultiBlock()
    material_ids = []

    for k in range(num_layers):
        z_bot = k * layer_height
        z_top = (k + 1) * layer_height
        filament = pattern[k % pattern_len]
        material_ids.append(filament)

        # Create a thin box for this layer
        slab = pv.Box(bounds=[
            -width / 2, width / 2,
            -depth / 2, depth / 2,
            z_bot, z_top
        ])
        slab.cell_data['material'] = np.full(slab.n_cells, filament)
        slab.cell_data['layer'] = np.full(slab.n_cells, k)
        blocks.append(slab)

    # Merge into single mesh
    combined = blocks.combine()

    # Build material name map
    unique_mats = sorted(set(pattern))
    mat_names = {}
    for f in unique_mats:
        mat_names[f] = material_map.get(f, f'Filament {f}')

    return combined, mat_names


def create_stress_through_thickness_viz(
    pattern: List[int], layer_height: float, total_height: float,
    material_map: Dict[int, str], results: dict,
    width: float = 40.0, depth: float = 20.0
) -> pv.PolyData:
    """Create a layup visualization colored by through-thickness stress.

    Uses the ABD matrix results to compute the stress distribution under
    a reference bending load (unit moment Mx=1 N*mm/mm).

    Each layer slab is colored by its bending stress level:
    - Red = tension (positive)
    - Blue = compression (negative)
    - White = neutral axis (zero stress)
    """
    num_layers = round(total_height / layer_height)
    pattern_len = len(pattern)
    z_neutral = results['abd']['z_neutral']

    blocks = pv.MultiBlock()

    # Compute bending stress: sigma = E * z_from_neutral * kappa
    # For unit curvature kappa=1, sigma = E * (z - z_neutral)
    # Normalize by max for coloring
    from ..core.materials import get_material

    stresses = []
    for k in range(num_layers):
        filament = pattern[k % pattern_len]
        mat = get_material(material_map.get(filament, 'PLA'))
        z_mid = (k + 0.5) * layer_height
        # Bending stress under unit curvature
        sigma = mat.E * (z_mid - z_neutral)
        stresses.append(sigma)

    max_abs = max(abs(s) for s in stresses) if stresses else 1.0

    for k in range(num_layers):
        z_bot = k * layer_height
        z_top = (k + 1) * layer_height

        slab = pv.Box(bounds=[
            -width / 2, width / 2,
            -depth / 2, depth / 2,
            z_bot, z_top
        ])
        # Normalized stress: -1 (compression) to +1 (tension)
        norm_stress = stresses[k] / max_abs if max_abs > 0 else 0
        slab.cell_data['bending_stress'] = np.full(slab.n_cells, norm_stress)
        slab.cell_data['layer'] = np.full(slab.n_cells, k)
        blocks.append(slab)

    combined = blocks.combine()
    return combined


def create_thermal_stress_viz(
    pattern: List[int], layer_height: float, total_height: float,
    material_map: Dict[int, str], thermal_results: dict,
    width: float = 40.0, depth: float = 20.0
) -> pv.PolyData:
    """Create a layup visualization colored by thermal interface stress.

    Each layer is colored by the maximum thermal stress at its boundaries.
    High-stress interfaces (delamination risk) appear bright red.
    """
    num_layers = round(total_height / layer_height)
    pattern_len = len(pattern)

    # Build per-layer stress from interface data
    layer_thermal = np.zeros(num_layers)

    for iface in thermal_results.get('interface_stresses', []):
        idx = iface['layer_index']
        stress = iface['sigma_normal']
        # Assign stress to both layers adjacent to the interface
        if 0 <= idx < num_layers:
            layer_thermal[idx] = max(layer_thermal[idx], stress)
        if 0 <= idx + 1 < num_layers:
            layer_thermal[idx + 1] = max(layer_thermal[idx + 1], stress)

    blocks = pv.MultiBlock()
    for k in range(num_layers):
        z_bot = k * layer_height
        z_top = (k + 1) * layer_height

        slab = pv.Box(bounds=[
            -width / 2, width / 2,
            -depth / 2, depth / 2,
            z_bot, z_top
        ])
        slab.cell_data['thermal_stress'] = np.full(slab.n_cells, layer_thermal[k])
        slab.cell_data['layer'] = np.full(slab.n_cells, k)
        blocks.append(slab)

    return blocks.combine()


def create_warping_viz(
    pattern: List[int], layer_height: float, total_height: float,
    material_map: Dict[int, str], warp_result: dict,
    part_length: float = 100.0, width: float = 40.0
) -> pv.PolyData:
    """Create a visualization of the warped part shape.

    Shows a flat plate deformed by the predicted thermal curvature.
    Left viewport shows flat (original), right shows warped (predicted).

    The warping is exaggerated by a scale factor for visibility.
    """
    curvature = warp_result.get('curvature', 0.0)

    # Create a flat plate mesh
    nx, ny = 60, 20
    x = np.linspace(-part_length / 2, part_length / 2, nx)
    y = np.linspace(-width / 2, width / 2, ny)
    xx, yy = np.meshgrid(x, y)

    # Warped Z: z = kappa * x^2 / 2 (parabolic approximation of arc)
    # Scale factor for visibility (make warping visible even if small)
    deflection = abs(curvature) * part_length**2 / 8.0
    if deflection > 0.001:
        scale = max(1.0, total_height * 2.0 / deflection)
    else:
        scale = 1.0

    zz_warped = curvature * xx**2 / 2.0 * scale

    # Place the plate at the mid-height of the part
    zz_warped += total_height / 2.0

    points = np.column_stack([xx.ravel(), yy.ravel(), zz_warped.ravel()])
    grid = pv.StructuredGrid()
    grid.dimensions = [nx, ny, 1]
    grid.points = points

    # Color by Z displacement from flat
    z_flat = total_height / 2.0
    displacement = zz_warped.ravel() - z_flat
    grid.point_data['warping'] = displacement

    return grid, scale


def create_flat_plate_viz(
    total_height: float, part_length: float = 100.0, width: float = 40.0
) -> pv.PolyData:
    """Create a flat (unwarped) plate for comparison."""
    nx, ny = 60, 20
    x = np.linspace(-part_length / 2, part_length / 2, nx)
    y = np.linspace(-width / 2, width / 2, ny)
    xx, yy = np.meshgrid(x, y)
    zz = np.full_like(xx, total_height / 2.0)

    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    grid = pv.StructuredGrid()
    grid.dimensions = [nx, ny, 1]
    grid.points = points
    grid.point_data['warping'] = np.zeros(len(points))

    return grid
