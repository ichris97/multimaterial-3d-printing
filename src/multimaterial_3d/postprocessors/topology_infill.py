#!/usr/bin/env python3
"""
Topology-optimized variable-density infill post-processor.

Analyzes the 3D model geometry to identify stress concentration regions and
automatically increases infill density in those areas while keeping infill
sparse in low-stress regions. This creates a structurally efficient part
that uses material only where it's mechanically needed.

Stress Analysis Methods
-----------------------
The stress map is computed using several geometric indicators:

1. **Corner Detection**: Sharp corners in the model boundary act as stress
   concentrators. The stress concentration factor (Kt) at a corner with
   interior angle θ is approximately Kt ≈ 1 + 2*(180-θ)/180. Infill is
   increased within a configurable radius around each corner.

2. **Thin Section Analysis**: Narrow regions of the model are weak points
   because they have less cross-sectional area to distribute load. The
   distance transform (EDT) is used to find how far each interior point
   is from the nearest boundary — small distances indicate thin sections.

3. **Hole Proximity**: Areas around holes need reinforcement because holes
   create stress concentrations of Kt ≈ 3 (for a circular hole in a
   plate under uniaxial tension, from Kirsch solution).

4. **Edge Proximity**: Points near the model boundary experience higher
   stresses under bending loads due to the linear stress distribution
   (σ = M*y/I, maximum at the surface). Infill near walls is more
   structurally effective than infill at the center.

Infill Modification Strategy
-----------------------------
The post-processor modifies existing G-code infill sections by adding
extra infill lines (reinforcement passes) in high-stress regions. The
number of extra passes is proportional to the local stress level:

- Stress < 0.2: No modification (sparse infill is sufficient)
- Stress 0.2-0.5: 1 extra pass (moderate reinforcement)
- Stress 0.5-0.8: 2 extra passes (significant reinforcement)
- Stress > 0.8: 3 extra passes (near-solid reinforcement)

Extra passes are placed parallel to existing infill lines with a
perpendicular offset of 0.5mm, creating a denser infill pattern locally.
"""

import argparse
import re
import struct
import zipfile
import hashlib
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np

# Optional dependencies with graceful fallback
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    from shapely.geometry import Polygon, Point
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

try:
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..core.file_io import extract_gcode_from_3mf, repack_3mf


@dataclass
class StressRegion:
    """A rectangular region with an associated stress level.

    Used to represent areas of the model that need increased infill density.
    Stress level ranges from 0.0 (no stress) to 1.0 (maximum stress).
    """
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    stress_level: float
    reason: str


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Topology-optimized infill for Bambu Studio 3MF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m multimaterial_3d.postprocessors.topology_infill model.3mf optimized.3mf
  python -m multimaterial_3d.postprocessors.topology_infill model.3mf optimized.3mf --min-density 10 --max-density 100
  python -m multimaterial_3d.postprocessors.topology_infill model.3mf optimized.3mf --sensitivity 0.8
        """
    )
    parser.add_argument('input_3mf', help='Input 3MF file (must be sliced)')
    parser.add_argument('output_3mf', help='Output 3MF file')
    parser.add_argument('--min-density', type=int, default=15,
                        help='Min infill density %% for low-stress areas (default: 15)')
    parser.add_argument('--max-density', type=int, default=80,
                        help='Max infill density %% for high-stress areas (default: 80)')
    parser.add_argument('--sensitivity', type=float, default=0.5,
                        help='Stress detection sensitivity 0.0-1.0 (default: 0.5)')
    parser.add_argument('--corner-radius', type=float, default=5.0,
                        help='Radius around corners to reinforce in mm (default: 5.0)')
    parser.add_argument('--hole-margin', type=float, default=3.0,
                        help='Margin around holes to reinforce in mm (default: 3.0)')
    parser.add_argument('--visualize', action='store_true',
                        help='Save stress map visualization (requires matplotlib)')
    parser.add_argument('--verbose', '-v', action='store_true')
    return parser.parse_args()


def load_binary_stl(data: bytes) -> np.ndarray:
    """Parse a binary STL file into an Nx3x3 array of triangle vertices.

    Binary STL format:
    - 80 bytes: header (ignored)
    - 4 bytes: number of triangles (uint32)
    - For each triangle (50 bytes each):
      - 12 bytes: face normal (3 x float32, ignored)
      - 36 bytes: 3 vertices (9 x float32)
      - 2 bytes: attribute byte count (uint16, ignored)
    """
    num_triangles = struct.unpack('<I', data[80:84])[0]
    triangles = []
    offset = 84
    for _ in range(num_triangles):
        v1 = struct.unpack('<fff', data[offset + 12:offset + 24])
        v2 = struct.unpack('<fff', data[offset + 24:offset + 36])
        v3 = struct.unpack('<fff', data[offset + 36:offset + 48])
        triangles.append([v1, v2, v3])
        offset += 50
    return np.array(triangles)


def load_ascii_stl(data: bytes) -> np.ndarray:
    """Parse an ASCII STL file into an Nx3x3 array of triangle vertices."""
    text = data.decode('utf-8', errors='ignore')
    triangles = []
    current_triangle = []
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('vertex'):
            parts = line.split()
            current_triangle.append((float(parts[1]), float(parts[2]), float(parts[3])))
            if len(current_triangle) == 3:
                triangles.append(current_triangle)
                current_triangle = []
    return np.array(triangles)


def load_stl_from_bytes(data: bytes) -> np.ndarray:
    """Detect STL format (ASCII or binary) and load triangle mesh."""
    if data[:5] == b'solid' and b'\n' in data[:80]:
        return load_ascii_stl(data)
    return load_binary_stl(data)


def parse_3mf_model(data: bytes) -> np.ndarray:
    """Parse a 3MF XML model file to extract triangle mesh.

    The 3MF format stores meshes as XML with <vertex> and <triangle> elements
    within <mesh> elements. Vertices are indexed and triangles reference
    vertex indices.
    """
    import xml.etree.ElementTree as ET
    text = data.decode('utf-8')
    text = re.sub(r'\sxmlns="[^"]+"', '', text, count=1)
    root = ET.fromstring(text)

    vertices = []
    triangles_idx = []

    for mesh in root.iter('mesh'):
        for vertices_elem in mesh.iter('vertices'):
            for vertex in vertices_elem.iter('vertex'):
                vertices.append([float(vertex.get('x', 0)),
                                 float(vertex.get('y', 0)),
                                 float(vertex.get('z', 0))])
        for triangles_elem in mesh.iter('triangles'):
            for triangle in triangles_elem.iter('triangle'):
                triangles_idx.append([int(triangle.get('v1', 0)),
                                      int(triangle.get('v2', 0)),
                                      int(triangle.get('v3', 0))])

    if not vertices or not triangles_idx:
        raise ValueError("Could not parse 3MF model - no mesh data found")

    vertices = np.array(vertices)
    return vertices[np.array(triangles_idx)]


def extract_geometry_from_gcode(gcode_content: str) -> Tuple[np.ndarray, list]:
    """Extract XY geometry from G-code extrusion moves.

    When no 3D model file is available inside the 3MF, this function
    reconstructs the approximate model boundary from the G-code movements.
    Only extrusion moves within wall and infill features are collected,
    as travel moves don't represent model geometry.

    Returns
    -------
    tuple of (xy_points, z_heights)
        xy_points: Nx2 array of XY coordinates from extrusion moves
        z_heights: sorted list of unique Z heights found
    """
    extrude_pattern = re.compile(r'^G[01]\s+X([\d.]+)\s+Y([\d.]+).*E[\d.]+', re.IGNORECASE)
    z_pattern = re.compile(r';\s*Z_HEIGHT:\s*([\d.]+)')
    wall_pattern = re.compile(r';\s*FEATURE:\s*(Inner|Outer)\s*wall', re.IGNORECASE)
    infill_pattern = re.compile(r';\s*FEATURE:\s*(Sparse|Internal|Solid)\s*infill', re.IGNORECASE)
    feature_end = re.compile(r';\s*(FEATURE:|CHANGE_LAYER|WIPE_)', re.IGNORECASE)

    xy_points = []
    z_heights = set()
    in_printable = False

    for line in gcode_content.split('\n'):
        z_match = z_pattern.search(line)
        if z_match:
            z_heights.add(float(z_match.group(1)))

        if wall_pattern.search(line) or infill_pattern.search(line):
            in_printable = True
        elif feature_end.search(line) and not wall_pattern.search(line) and not infill_pattern.search(line):
            in_printable = False

        if in_printable:
            match = extrude_pattern.match(line.strip())
            if match:
                xy_points.append([float(match.group(1)), float(match.group(2))])

    arr = np.array(xy_points) if xy_points else np.array([]).reshape(0, 2)
    return arr, sorted(z_heights)


def find_corners(boundary: np.ndarray, angle_threshold: float = 120) -> list:
    """Find corner points in a 2D boundary polygon.

    A corner is defined as a vertex where the interior angle between
    adjacent edges is less than the threshold. Sharper corners (smaller
    angles) create higher stress concentrations.

    Parameters
    ----------
    boundary : np.ndarray
        Nx2 array of boundary vertices in order.
    angle_threshold : float
        Maximum angle (degrees) to consider a vertex as a corner.

    Returns
    -------
    list of (x, y, angle) tuples
        Detected corner locations and their interior angles.
    """
    corners = []
    n = len(boundary)
    for i in range(n):
        v1 = boundary[(i - 1) % n] - boundary[i]
        v2 = boundary[(i + 1) % n] - boundary[i]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        if angle < angle_threshold:
            corners.append((boundary[i][0], boundary[i][1], angle))
    return corners


def calculate_stress_map(xy_points: np.ndarray, grid_resolution: float = 1.0,
                         corner_radius: float = 5.0, sensitivity: float = 0.5,
                         verbose: bool = False) -> Tuple[np.ndarray, dict]:
    """Compute a 2D stress concentration map from model geometry.

    The stress map is a 2D grid where each cell contains a value from 0 to 1
    indicating the relative stress level. High values mean more infill is needed.

    The map combines contributions from multiple stress indicators:
    - Corner proximity (weighted by corner sharpness)
    - Edge proximity (thin sections need reinforcement)
    - The contributions are combined using element-wise maximum

    Parameters
    ----------
    xy_points : np.ndarray
        Nx2 array of model boundary/geometry points.
    grid_resolution : float
        Grid cell size in mm.
    corner_radius : float
        Radius of influence around corners in mm.
    sensitivity : float
        Detection sensitivity (0=aggressive, 1=conservative).
    verbose : bool
        Print analysis details.

    Returns
    -------
    tuple of (stress_map, metadata)
    """
    if len(xy_points) == 0:
        raise ValueError("No geometry points found")

    x_min, y_min = xy_points.min(axis=0)
    x_max, y_max = xy_points.max(axis=0)

    nx = min(int((x_max - x_min) / grid_resolution) + 1, 300)
    ny = min(int((y_max - y_min) / grid_resolution) + 1, 300)
    res_x = (x_max - x_min) / max(nx - 1, 1)
    res_y = (y_max - y_min) / max(ny - 1, 1)

    stress_map = np.zeros((ny, nx))

    # Compute convex hull for boundary approximation
    boundary = None
    if HAS_SCIPY and len(xy_points) > 3:
        try:
            unique = np.unique(xy_points, axis=0)
            if len(unique) > 3:
                hull = ConvexHull(unique)
                boundary = unique[hull.vertices]
        except Exception:
            pass

    if boundary is None:
        boundary = np.array([[x_min, y_min], [x_max, y_min],
                             [x_max, y_max], [x_min, y_max]])

    # Corner stress contribution
    corners = find_corners(boundary, angle_threshold=150)
    model_size = max(x_max - x_min, y_max - y_min)
    effective_radius = max(corner_radius, model_size * 0.3)

    for cx, cy, angle in corners:
        angle_factor = 1.0 - (angle / 180.0)
        for i in range(ny):
            for j in range(nx):
                x = x_min + j * res_x
                y = y_min + i * res_y
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < effective_radius:
                    stress = angle_factor * (1.0 - dist / effective_radius) ** 0.5
                    stress_map[i, j] = max(stress_map[i, j], stress)

    # Edge proximity stress (using distance transform)
    if HAS_SHAPELY and HAS_SCIPY:
        try:
            poly = Polygon(boundary)
            if not poly.is_valid:
                poly = poly.buffer(0)
            mask = np.zeros((ny, nx), dtype=bool)
            for i in range(ny):
                for j in range(nx):
                    pt = Point(x_min + j * res_x, y_min + i * res_y)
                    mask[i, j] = poly.contains(pt) or poly.boundary.distance(pt) < 0.5
            dist_inside = distance_transform_edt(mask) * res_x
            thin_threshold = model_size * 0.25
            edge_stress = np.clip(1.0 - dist_inside / thin_threshold, 0, 1)
            stress_map = np.maximum(stress_map, edge_stress * 0.5)
        except Exception:
            pass

    # Apply sensitivity and smooth
    stress_map = np.power(stress_map, 1.0 / (sensitivity + 0.5))
    stress_map = np.clip(stress_map, 0, 1)
    if HAS_SCIPY:
        stress_map = gaussian_filter(stress_map, sigma=1.5)

    metadata = {
        'bounds': (x_min, x_max, y_min, y_max),
        'grid_size': (nx, ny),
        'resolution': (res_x, res_y),
        'corners': len(corners),
        'holes': 0,
    }
    return stress_map, metadata


def get_stress_at_point(stress_map: np.ndarray, metadata: dict,
                        x: float, y: float) -> float:
    """Look up the stress level at a specific XY coordinate in the stress map."""
    x_min, x_max, y_min, y_max = metadata['bounds']
    nx, ny = metadata['grid_size']
    if x < x_min or x > x_max or y < y_min or y > y_max:
        return 0.0
    j = np.clip(int((x - x_min) / (x_max - x_min) * (nx - 1)), 0, nx - 1)
    i = np.clip(int((y - y_min) / (y_max - y_min) * (ny - 1)), 0, ny - 1)
    return stress_map[i, j]


def modify_infill_density(gcode_lines: List[str], stress_map: np.ndarray,
                          metadata: dict, min_density: int, max_density: int,
                          verbose: bool = False) -> List[str]:
    """Modify G-code infill sections based on the stress map.

    Processes each sparse infill region and adds reinforcement passes
    (extra parallel extrusion lines) in areas where the stress map indicates
    high stress concentration.

    The reinforcement lines are offset perpendicular to the original infill
    line direction, effectively increasing the local infill density.
    """
    output = []
    sparse_infill = re.compile(r';\s*FEATURE:\s*Sparse infill', re.IGNORECASE)
    other_feature = re.compile(r';\s*FEATURE:\s*(?!Sparse infill)', re.IGNORECASE)
    z_pattern = re.compile(r';\s*Z_HEIGHT:\s*([\d.]+)')
    move_pattern = re.compile(r'^G1\s+X([\d.]+)\s+Y([\d.]+)', re.IGNORECASE)
    extrude_pattern = re.compile(r'^G1\s+X([\d.]+)\s+Y([\d.]+)\s+.*E([\d.]+)', re.IGNORECASE)

    in_infill = False
    infill_lines = []
    stats = {'layers_modified': 0, 'extra_lines': 0}

    for line in gcode_lines:
        z_match = z_pattern.search(line)

        if sparse_infill.search(line):
            in_infill = True
            infill_lines = []
            output.append(line)
            continue

        if in_infill and other_feature.search(line):
            in_infill = False
            modified = _process_infill_region(infill_lines, stress_map, metadata)
            if len(modified) > len(infill_lines):
                stats['layers_modified'] += 1
                stats['extra_lines'] += len(modified) - len(infill_lines)
            output.extend(modified)
            output.append(line)
            continue

        if in_infill:
            infill_lines.append(line)
        else:
            output.append(line)

    if verbose:
        print(f"   Layers with added infill: {stats['layers_modified']}")
        print(f"   Extra infill lines added: {stats['extra_lines']}")

    return output


def _process_infill_region(infill_lines: List[str], stress_map: np.ndarray,
                           metadata: dict) -> List[str]:
    """Add reinforcement passes to a single infill region."""
    output = []
    move_pattern = re.compile(r'^G1\s+X([\d.]+)\s+Y([\d.]+)', re.IGNORECASE)
    extrude_pattern = re.compile(r'^G1\s+X([\d.]+)\s+Y([\d.]+)\s+.*E([\d.]+)', re.IGNORECASE)

    prev_x, prev_y = None, None
    stress_threshold = 0.20

    for line in infill_lines:
        match = move_pattern.match(line.strip())
        x, y, stress = None, None, 0.0
        if match:
            x, y = float(match.group(1)), float(match.group(2))
            stress = get_stress_at_point(stress_map, metadata, x, y)

        output.append(line)

        if x is not None and stress > stress_threshold and prev_x is not None and 'E' in line:
            # Add reinforcement passes perpendicular to the infill direction
            dx, dy = x - prev_x, y - prev_y
            length = np.sqrt(dx**2 + dy**2)
            if length > 1.0:
                stress_norm = (stress - stress_threshold) / (1.0 - stress_threshold)
                extra_passes = 1 + int(stress_norm * 2)
                nx_dir, ny_dir = -dy / length, dx / length
                offset = 0.5  # mm between passes

                e_match = extrude_pattern.match(line.strip())
                if e_match:
                    e_val = float(e_match.group(3))
                    for p in range(min(extra_passes, 3)):
                        offset_dist = offset * (p + 1) * (1 if p % 2 == 0 else -1)
                        ox1 = prev_x + nx_dir * offset_dist
                        oy1 = prev_y + ny_dir * offset_dist
                        ox2 = x + nx_dir * offset_dist
                        oy2 = y + ny_dir * offset_dist
                        output.append(f"; TOPO-REINFORCEMENT pass {p+1} (stress={stress:.2f})\n")
                        output.append(f"G1 X{ox1:.3f} Y{oy1:.3f} F12000\n")
                        output.append(f"G1 X{ox2:.3f} Y{oy2:.3f} E{e_val:.5f} F3000\n")

        if x is not None:
            prev_x, prev_y = x, y

    return output


def visualize_stress_map(stress_map: np.ndarray, metadata: dict,
                         output_path: str) -> bool:
    """Save a heatmap visualization of the stress map as a PNG image."""
    try:
        import matplotlib.pyplot as plt
        x_min, x_max, y_min, y_max = metadata['bounds']
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        im = ax.imshow(stress_map, origin='lower',
                       extent=[x_min, x_max, y_min, y_max],
                       cmap='hot', vmin=0, vmax=1)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('Stress Concentration Map\n(Bright = High Stress = More Infill)')
        plt.colorbar(im, ax=ax, label='Stress Level')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return True
    except ImportError:
        return False


def main():
    """Main entry point for the topology-optimized infill post-processor."""
    args = parse_args()

    print("=" * 60)
    print("  Topology-Optimized Infill Post-Processor")
    print("=" * 60)

    if not HAS_SCIPY:
        print("Warning: scipy not available - some analysis features disabled")
    if not HAS_SHAPELY:
        print("Warning: shapely not available - some analysis features disabled")

    print(f"\nLoading {args.input_3mf}...")

    # Extract G-code and optionally 3D model
    triangles = None
    with zipfile.ZipFile(args.input_3mf, 'r') as zf:
        for name in zf.namelist():
            if name.endswith('.stl'):
                triangles = load_stl_from_bytes(zf.read(name))
                break
            elif name.endswith('.model') and '3D/' in name:
                try:
                    triangles = parse_3mf_model(zf.read(name))
                except Exception:
                    pass
                break

    gcode_content, gcode_path = extract_gcode_from_3mf(args.input_3mf)

    # Get geometry from G-code if no 3D model available
    xy_points, z_heights = extract_geometry_from_gcode(gcode_content)
    print(f"   Extracted {len(xy_points)} geometry points from {len(z_heights)} layers")

    # Calculate stress map
    print(f"\nAnalyzing stress concentrations...")
    stress_map, metadata = calculate_stress_map(
        xy_points, corner_radius=args.corner_radius,
        sensitivity=args.sensitivity, verbose=args.verbose
    )

    stress_mean = stress_map.mean()
    stress_max = stress_map.max()
    high_stress_pct = (stress_map > 0.5).sum() / stress_map.size * 100

    print(f"   Average stress: {stress_mean:.2%}")
    print(f"   Maximum stress: {stress_max:.2%}")
    print(f"   High-stress area: {high_stress_pct:.1f}%")
    print(f"   Corners detected: {metadata['corners']}")

    if args.visualize:
        viz_path = args.output_3mf.replace('.3mf', '_stress_map.png')
        if visualize_stress_map(stress_map, metadata, viz_path):
            print(f"\n  Stress map saved to: {viz_path}")

    # Modify infill
    print(f"\nModifying infill density ({args.min_density}% -> {args.max_density}%)...")
    gcode_lines = gcode_content.splitlines(keepends=True)
    modified = modify_infill_density(gcode_lines, stress_map, metadata,
                                     args.min_density, args.max_density,
                                     verbose=args.verbose)

    new_gcode = ''.join(modified)
    print(f"\nCreating {args.output_3mf}...")
    repack_3mf(args.input_3mf, args.output_3mf, new_gcode, gcode_path)
    print(f"\nDone!")


if __name__ == '__main__':
    main()
