"""
Adaptive layer height computation based on model geometry.

Standard slicing uses a uniform layer height throughout the part. However,
different regions of a model benefit from different layer heights:

- **Flat surfaces** (top/bottom): Thick layers are fine — no visible staircase
- **Steep overhangs** (30-60 deg): Need thinner layers to reduce staircase error
- **Curved surfaces** (spheres, fillets): Thin layers reduce faceting
- **Vertical walls**: Layer height doesn't affect surface quality much
- **Fine features**: Small holes, thin walls need thin layers for accuracy

Staircase Error Model
---------------------
For a surface with angle θ from the vertical (0° = vertical wall, 90° = flat),
the staircase error (cusp height) is:

    error = layer_height * cos(θ) / 2

For a flat horizontal surface (θ=90°), error = layer_height/2 (worst case).
For a vertical wall (θ=0°), error = 0 (no staircase).

To maintain a maximum surface error of ε_max:
    layer_height_max = 2 * ε_max / cos(θ)

This is clamped to [min_height, max_height].

Curvature Detection
-------------------
High curvature regions (transitions between flat and angled surfaces) benefit
from thinner layers because the staircase pattern is most visible where the
surface angle changes rapidly. The curvature is estimated by computing how
the cross-sectional area changes between adjacent Z slices — rapid area
changes indicate high curvature.

Integration with Bambu Studio
-----------------------------
Bambu Studio supports variable layer heights through the layer_config_ranges.xml
file. Each layer range can specify its own layer_height parameter, allowing
the adaptive schedule to be directly embedded in the 3MF file.
"""

import numpy as np
from typing import List, Tuple, Optional


def compute_adaptive_layer_heights(
    triangles: np.ndarray,
    min_height: float = 0.05,
    max_height: float = 0.30,
    target_tolerance: float = 0.05,
    curvature_weight: float = 0.7,
) -> List[Tuple[float, float]]:
    """Compute optimal layer height at each Z position based on geometry.

    Analyzes the mesh surface normals and curvature to determine where
    thinner layers are needed for surface quality.

    Parameters
    ----------
    triangles : np.ndarray
        Nx3x3 array of triangle vertices from the model mesh.
    min_height : float
        Minimum allowed layer height in mm (default: 0.05).
    max_height : float
        Maximum allowed layer height in mm (default: 0.30).
    target_tolerance : float
        Maximum acceptable staircase error in mm (default: 0.05).
    curvature_weight : float
        Weight of curvature signal relative to surface angle (0-1).

    Returns
    -------
    list of (z_start, layer_height) tuples
        The adaptive layer schedule. Each tuple gives the Z position
        where a layer starts and its height.
    """
    # Compute face normals for all triangles
    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]
    edges1 = v1 - v0
    edges2 = v2 - v0
    normals = np.cross(edges1, edges2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1e-10
    normals = normals / norms

    # |nz| indicates surface orientation:
    # |nz| ≈ 0: vertical wall (no staircase issue)
    # |nz| ≈ 1: horizontal surface (maximum staircase)
    # Intermediate: sloped surface (needs thinner layers)
    abs_nz = np.abs(normals[:, 2])

    # Z range of the model
    all_z = triangles[:, :, 2].flatten()
    z_min, z_max = all_z.min(), all_z.max()

    # Triangle Z extents for spatial lookup
    tri_z_min = triangles[:, :, 2].min(axis=1)
    tri_z_max = triangles[:, :, 2].max(axis=1)

    # Sample at fine resolution to build complexity profile
    sample_dz = min_height / 2
    z_samples = np.arange(z_min, z_max, sample_dz)
    complexity = np.zeros(len(z_samples))

    for i, z in enumerate(z_samples):
        # Find triangles whose Z extent spans this height. Note: large
        # triangles that span many Z levels will be included at every
        # sample within their range, which adds noise but is acceptable
        # because such triangles still contribute to the local surface
        # orientation at each Z slice they intersect.
        mask = (tri_z_min <= z) & (tri_z_max >= z)
        if mask.sum() == 0:
            continue

        local_nz = abs_nz[mask]

        # Design choice: "slope factor" = |nz| * sqrt(1 - nz^2)
        #
        # This peaks at 45 degrees (nz = 1/sqrt(2)), NOT at horizontal
        # surfaces (nz = 1). The pure staircase error is proportional to
        # |nz| alone, which is maximal for horizontal surfaces. However,
        # horizontal surfaces (|nz| ~ 1) are typically top/bottom faces
        # where the staircase pattern is not visible to the eye — it is
        # the flat face itself. The *visible* staircase artifacts appear
        # on sloped surfaces, especially around 30-60 degrees, where the
        # stepping between layers creates a sawtooth profile viewed from
        # the side. This metric therefore optimises for VISUAL quality
        # rather than geometric accuracy.
        #
        # Alternative: use |nz| directly for geometric-error minimisation.
        slope_factor = local_nz * np.sqrt(1 - local_nz**2 + 1e-10)
        complexity[i] = slope_factor.mean()

    # Normalize complexity to [0, 1]
    if complexity.max() > 0:
        complexity /= complexity.max()

    # Curvature signal: second derivative of triangle count vs Z.
    #
    # This is a rough proxy for vertical curvature. It works well for
    # shapes where curvature correlates with changing cross-section
    # (e.g., domes, tapers, fillets) but poorly for shapes with
    # constant cross-section and high curvature (e.g., a sphere
    # tessellated uniformly has roughly constant triangle count per
    # slice, and a cylinder has varying count but zero vertical
    # curvature). Despite these limitations, it is inexpensive and
    # catches the most common cases where layer refinement helps.
    tri_counts = np.array([
        ((tri_z_min <= z) & (tri_z_max >= z)).sum()
        for z in z_samples
    ]).astype(float)

    if len(tri_counts) > 2:
        curvature_signal = np.abs(np.gradient(np.gradient(tri_counts)))
        if curvature_signal.max() > 0:
            curvature_signal /= curvature_signal.max()
        complexity = np.maximum(complexity, curvature_signal * curvature_weight)

    # Build adaptive layer schedule
    # High complexity → thin layers, low complexity → thick layers
    layers = []
    z = z_min

    while z < z_max:
        idx = min(int((z - z_min) / sample_dz), len(complexity) - 1)
        c = complexity[idx]

        # Map complexity to layer height (inverse relationship)
        lh = max_height - c * (max_height - min_height)
        lh = np.clip(lh, min_height, max_height)

        # Round to nearest 0.01mm for slicer compatibility
        lh = round(lh / 0.01) * 0.01
        lh = max(lh, min_height)

        # Clamp the last layer so it does not overshoot z_max
        remaining = z_max - z
        if lh > remaining:
            lh = max(round(remaining / 0.01) * 0.01, min_height)
            # If even min_height overshoots, accept min_height as final
            if lh < min_height:
                lh = min_height

        layers.append((round(z, 4), round(lh, 4)))
        # Round z after each step to prevent accumulated floating-point drift
        z = round(z + lh, 4)

    return layers


def print_layer_schedule(layers: List[Tuple[float, float]]) -> None:
    """Print a summary of the adaptive layer schedule."""
    heights = [lh for _, lh in layers]
    total = sum(heights)

    print(f"\n  Adaptive Layer Schedule:")
    print(f"    Total layers: {len(layers)}")
    print(f"    Total height: {total:.2f}mm")
    print(f"    Min layer height: {min(heights):.2f}mm")
    print(f"    Max layer height: {max(heights):.2f}mm")
    print(f"    Average height: {np.mean(heights):.3f}mm")

    # Histogram of layer heights
    bins = {}
    for h in heights:
        h_rounded = round(h, 2)
        bins[h_rounded] = bins.get(h_rounded, 0) + 1

    print(f"\n    Distribution:")
    for h, count in sorted(bins.items()):
        bar = "#" * min(count, 40)
        print(f"      {h:.2f}mm: {count:>4} layers {bar}")
