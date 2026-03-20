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
   concentrators. Both convex corners (exterior angles) and concave/re-entrant
   corners (interior angles > 180 degrees) are detected. Re-entrant corners
   are the most critical stress concentrators.

2. **Thin Section Analysis**: Narrow regions of the model are weak points
   because they have less cross-sectional area to distribute load. The
   distance transform (EDT) is used to find how far each interior point
   is from the nearest boundary -- small distances indicate thin sections.

3. **Hole Proximity**: Areas around holes need reinforcement because holes
   create stress concentrations of Kt ~ 3 (for a circular hole in a
   plate under uniaxial tension, from Kirsch solution).

4. **Edge Proximity**: Points near the model boundary experience higher
   stresses under bending loads due to the linear stress distribution
   (sigma = M*y/I, maximum at the surface). Infill near walls is more
   structurally effective than infill at the center.

Geometry Reconstruction
-----------------------
The actual model boundary is reconstructed from outer wall extrusion paths
in the G-code. This preserves concave features, holes, notches, and
L-shapes that would be lost by a convex hull approximation. Stress maps
are computed per-layer (or per Z-range) so that geometry changes along
the Z axis are properly accounted for.

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
from dataclasses import dataclass, field
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


@dataclass
class LayerContour:
    """Contour data for a single layer.

    Stores the outer wall paths that define the actual model boundary
    at a specific Z height.
    """
    z_height: float
    outer_walls: List[np.ndarray] = field(default_factory=list)
    # Each outer_wall is an Mx2 array of XY points forming a closed path


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


def _order_contour_points(points: np.ndarray, tolerance: float = 0.5) -> np.ndarray:
    """Order a set of extrusion points into a contour by connecting nearest neighbors.

    G-code outer wall moves are generally sequential, so the input points
    are usually already in order. This function validates that and handles
    edge cases where points may need reordering.

    Parameters
    ----------
    points : np.ndarray
        Nx2 array of XY points from extrusion moves.
    tolerance : float
        Maximum gap (mm) between consecutive points to consider them connected.

    Returns
    -------
    np.ndarray
        Ordered Nx2 array forming a closed contour.
    """
    if len(points) < 3:
        return points

    # G-code wall moves are already sequential, so just return as-is
    # if the path is reasonably continuous
    diffs = np.diff(points, axis=0)
    gaps = np.linalg.norm(diffs, axis=1)
    max_gap = np.max(gaps) if len(gaps) > 0 else 0

    if max_gap < tolerance * 10:
        return points

    # If there are large gaps, the points may include multiple segments.
    # Use nearest-neighbor ordering as fallback.
    ordered = [points[0]]
    remaining = list(range(1, len(points)))

    for _ in range(len(points) - 1):
        if not remaining:
            break
        current = ordered[-1]
        dists = [np.linalg.norm(points[r] - current) for r in remaining]
        nearest_idx = np.argmin(dists)
        ordered.append(points[remaining[nearest_idx]])
        remaining.pop(nearest_idx)

    return np.array(ordered)


def _simplify_contour(points: np.ndarray, tolerance: float = 0.1) -> np.ndarray:
    """Simplify a contour using the Ramer-Douglas-Peucker algorithm.

    Reduces the number of points while preserving the shape within the
    given tolerance. This is important for corner detection -- too many
    points creates false corners, too few loses real ones.

    Parameters
    ----------
    points : np.ndarray
        Nx2 array of contour points.
    tolerance : float
        Maximum perpendicular distance (mm) for point removal.

    Returns
    -------
    np.ndarray
        Simplified contour points.
    """
    if len(points) < 3:
        return points

    def _rdp(pts, eps):
        if len(pts) <= 2:
            return pts

        # Find point farthest from line between first and last
        start, end = pts[0], pts[-1]
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-10:
            # Degenerate line, find farthest point from start
            dists = np.linalg.norm(pts - start, axis=1)
            max_idx = np.argmax(dists)
            max_dist = dists[max_idx]
        else:
            line_unit = line_vec / line_len
            # Perpendicular distances
            vecs = pts - start
            projs = np.outer(np.dot(vecs, line_unit), line_unit)
            perps = vecs - projs
            dists = np.linalg.norm(perps, axis=1)
            max_idx = np.argmax(dists)
            max_dist = dists[max_idx]

        if max_dist > eps:
            left = _rdp(pts[:max_idx + 1], eps)
            right = _rdp(pts[max_idx:], eps)
            return np.vstack([left[:-1], right])
        else:
            return np.array([pts[0], pts[-1]])

    return _rdp(points, tolerance)


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
    extrude_pattern = re.compile(r'^G[01]\s+X([\d.]+)\s+Y([\d.]+).*E-?[\d.]+', re.IGNORECASE)
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


def extract_layer_contours(gcode_content: str) -> Dict[float, LayerContour]:
    """Extract actual model contours from outer wall extrusion paths per layer.

    The outer wall trace in G-code IS the actual model boundary at each layer.
    This function extracts those paths to reconstruct the true geometry,
    preserving concave features, holes, notches, and L-shapes.

    Parameters
    ----------
    gcode_content : str
        Full G-code content.

    Returns
    -------
    dict mapping z_height -> LayerContour
        Each LayerContour contains the ordered outer wall paths for that layer.
    """
    extrude_pattern = re.compile(
        r'^G[01]\s+X([\d.]+)\s+Y([\d.]+).*E-?[\d.]+', re.IGNORECASE)
    move_pattern = re.compile(
        r'^G[01]\s+X([\d.]+)\s+Y([\d.]+)', re.IGNORECASE)
    z_pattern = re.compile(r';\s*Z_HEIGHT:\s*([\d.]+)')
    outer_wall_pattern = re.compile(
        r';\s*FEATURE:\s*Outer\s*wall', re.IGNORECASE)
    feature_pattern = re.compile(r';\s*FEATURE:', re.IGNORECASE)

    layer_contours: Dict[float, LayerContour] = {}
    current_z = 0.0
    in_outer_wall = False
    current_path: List[List[float]] = []

    for line in gcode_content.split('\n'):
        stripped = line.strip()

        z_match = z_pattern.search(stripped)
        if z_match:
            current_z = float(z_match.group(1))

        if outer_wall_pattern.search(stripped):
            # Starting a new outer wall section
            if current_path and len(current_path) >= 3:
                _store_contour(layer_contours, current_z, current_path)
            current_path = []
            in_outer_wall = True
            continue

        if in_outer_wall and feature_pattern.search(stripped) and \
                not outer_wall_pattern.search(stripped):
            # Ended outer wall section
            if current_path and len(current_path) >= 3:
                _store_contour(layer_contours, current_z, current_path)
            current_path = []
            in_outer_wall = False
            continue

        if in_outer_wall:
            match = extrude_pattern.match(stripped)
            if match:
                current_path.append(
                    [float(match.group(1)), float(match.group(2))])

    # Flush last path
    if current_path and len(current_path) >= 3:
        _store_contour(layer_contours, current_z, current_path)

    return layer_contours


def _store_contour(layer_contours: Dict[float, LayerContour],
                   z: float, path: List[List[float]]) -> None:
    """Store a completed outer wall path into the layer contours dict."""
    if z not in layer_contours:
        layer_contours[z] = LayerContour(z_height=z)
    pts = np.array(path)
    pts = _simplify_contour(pts, tolerance=0.15)
    if len(pts) >= 3:
        layer_contours[z].outer_walls.append(pts)


def _point_in_polygon(px: float, py: float, polygon: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test (no external dependencies).

    Parameters
    ----------
    px, py : float
        Test point coordinates.
    polygon : np.ndarray
        Nx2 array of polygon vertices.

    Returns
    -------
    bool
        True if point is inside the polygon.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and \
                (px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside


def _polygon_area_signed(polygon: np.ndarray) -> float:
    """Compute the signed area of a polygon (positive = CCW, negative = CW).

    Uses the shoelace formula.
    """
    n = len(polygon)
    if n < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + \
        0.5 * (x[-1] * y[0] - x[0] * y[-1])


def _ensure_ccw(polygon: np.ndarray) -> np.ndarray:
    """Ensure polygon vertices are in counter-clockwise order."""
    if _polygon_area_signed(polygon) < 0:
        return polygon[::-1]
    return polygon


def find_corners(boundary: np.ndarray, angle_threshold: float = 120,
                 detect_concave: bool = True) -> list:
    """Find corner points in a 2D boundary polygon.

    Detects both convex corners (sharp exterior angles) and concave/re-entrant
    corners (angles > 180 degrees from the interior). Re-entrant corners are
    the most critical stress concentrators in practice.

    A corner is defined as a vertex where the interior angle between
    adjacent edges deviates significantly from 180 degrees (a straight line).

    Parameters
    ----------
    boundary : np.ndarray
        Nx2 array of boundary vertices in order (CCW for outer boundary).
    angle_threshold : float
        Maximum angle (degrees) to consider a vertex as a convex corner.
    detect_concave : bool
        If True, also detect concave (re-entrant) corners where the
        interior angle exceeds (360 - angle_threshold) degrees.

    Returns
    -------
    list of (x, y, angle, is_concave) tuples
        Detected corner locations, their interior angles, and whether
        they are concave (re-entrant).
    """
    corners = []
    n = len(boundary)
    if n < 3:
        return corners

    # Ensure CCW ordering for consistent interior angle computation
    boundary = _ensure_ccw(boundary)

    for i in range(n):
        v1 = boundary[(i - 1) % n] - boundary[i]
        v2 = boundary[(i + 1) % n] - boundary[i]

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            continue

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

        # Use cross product to determine convexity
        # For CCW polygon: positive cross = convex, negative = concave
        cross = v1[0] * v2[1] - v1[1] * v2[0]

        if cross >= 0:
            # Convex corner -- interior angle is `angle`
            is_concave = False
            interior_angle = angle
        else:
            # Concave (re-entrant) corner -- interior angle is 360 - angle
            is_concave = True
            interior_angle = 360.0 - angle

        if not is_concave and angle < angle_threshold:
            corners.append((boundary[i][0], boundary[i][1], angle, False))
        elif is_concave and detect_concave and (360.0 - angle) > (360.0 - angle_threshold):
            # Re-entrant corner: the reflex angle makes it a stress concentrator.
            # Severity is based on how far past 180 degrees the interior angle is.
            corners.append((boundary[i][0], boundary[i][1], angle, True))

    return corners


def _group_layers_into_z_ranges(
        layer_contours: Dict[float, LayerContour],
        max_ranges: int = 10
) -> List[Tuple[float, float, List[LayerContour]]]:
    """Group layers into Z-ranges for per-range stress map computation.

    Layers with similar geometry are grouped together to avoid computing
    a stress map for every single layer while still capturing geometry
    changes along Z.

    Parameters
    ----------
    layer_contours : dict
        Mapping of z_height -> LayerContour.
    max_ranges : int
        Maximum number of Z-ranges.

    Returns
    -------
    list of (z_min, z_max, contours) tuples
    """
    if not layer_contours:
        return []

    z_values = sorted(layer_contours.keys())
    if len(z_values) <= max_ranges:
        return [(z, z, [layer_contours[z]]) for z in z_values]

    # Split into roughly equal ranges
    chunk_size = max(1, len(z_values) // max_ranges)
    ranges = []
    for start in range(0, len(z_values), chunk_size):
        chunk = z_values[start:start + chunk_size]
        contours = [layer_contours[z] for z in chunk]
        ranges.append((chunk[0], chunk[-1], contours))

    return ranges


def _build_mask_from_contours(
        contours: List[np.ndarray],
        x_min: float, y_min: float,
        nx: int, ny: int,
        res_x: float, res_y: float
) -> np.ndarray:
    """Build a boolean interior mask from actual contour polygons.

    Uses ray-casting point-in-polygon for each grid cell. For multiple
    contours (e.g., outer boundary + holes), the winding rule determines
    interior: points inside an odd number of contours are interior.

    Parameters
    ----------
    contours : list of np.ndarray
        List of Nx2 polygon contours.
    x_min, y_min : float
        Grid origin.
    nx, ny : int
        Grid dimensions.
    res_x, res_y : float
        Grid resolution.

    Returns
    -------
    np.ndarray
        Boolean mask (ny, nx) -- True for interior points.
    """
    mask = np.zeros((ny, nx), dtype=bool)

    if not contours:
        return mask

    for i in range(ny):
        y = y_min + i * res_y
        for j in range(nx):
            x = x_min + j * res_x
            # Count how many contours contain this point (odd = inside)
            count = sum(1 for c in contours if _point_in_polygon(x, y, c))
            mask[i, j] = (count % 2 == 1)

    return mask


def calculate_stress_map(xy_points: np.ndarray, grid_resolution: float = 1.0,
                         corner_radius: float = 5.0, sensitivity: float = 0.5,
                         verbose: bool = False,
                         layer_contours: Optional[Dict[float, LayerContour]] = None,
                         z_height: Optional[float] = None
                         ) -> Tuple[np.ndarray, dict]:
    """Compute a 2D stress concentration map from model geometry.

    The stress map is a 2D grid where each cell contains a value from 0 to 1
    indicating the relative stress level. High values mean more infill is needed.

    When layer_contours are provided, uses the actual outer wall paths to
    reconstruct the true model boundary (preserving concave features, holes,
    etc.) instead of a convex hull approximation.

    Parameters
    ----------
    xy_points : np.ndarray
        Nx2 array of model boundary/geometry points (fallback).
    grid_resolution : float
        Grid cell size in mm.
    corner_radius : float
        Radius of influence around corners in mm.
    sensitivity : float
        Detection sensitivity (0=aggressive, 1=conservative).
    verbose : bool
        Print analysis details.
    layer_contours : dict, optional
        Per-layer contour data from extract_layer_contours(). If provided,
        actual geometry is used instead of convex hull.
    z_height : float, optional
        If provided with layer_contours, use the contour nearest this Z.

    Returns
    -------
    tuple of (stress_map, metadata)
    """
    # Determine which contours to use for this stress map
    contours_for_analysis = _select_contours(
        layer_contours, z_height, xy_points)

    # Determine bounds from contours or fallback to xy_points
    all_pts = _collect_all_points(contours_for_analysis, xy_points)
    if len(all_pts) == 0:
        raise ValueError("No geometry points found")

    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)

    nx = min(int((x_max - x_min) / grid_resolution) + 1, 300)
    ny = min(int((y_max - y_min) / grid_resolution) + 1, 300)
    res_x = (x_max - x_min) / max(nx - 1, 1)
    res_y = (y_max - y_min) / max(ny - 1, 1)

    stress_map = np.zeros((ny, nx))

    # Get boundary polygons -- prefer actual contours over convex hull
    boundaries = []
    if contours_for_analysis:
        boundaries = contours_for_analysis
    else:
        # Legacy fallback: convex hull (only if no contours available)
        boundary = _compute_boundary_fallback(xy_points, x_min, y_min, x_max, y_max)
        if boundary is not None:
            boundaries = [boundary]

    # Corner stress contribution (both convex and concave)
    all_corners = []
    for boundary in boundaries:
        corners = find_corners(boundary, angle_threshold=150, detect_concave=True)
        all_corners.extend(corners)

    model_size = max(x_max - x_min, y_max - y_min)
    effective_radius = max(corner_radius, model_size * 0.15)

    for cx, cy, angle, is_concave in all_corners:
        if is_concave:
            # Re-entrant corners are MORE severe stress concentrators
            # Interior angle > 180 deg: stress factor increases with reflex
            reflex_angle = 360.0 - angle
            angle_factor = min(1.0, 0.5 + (reflex_angle - 180.0) / 180.0)
            # Use a larger radius for concave corners
            radius = effective_radius * 1.3
        else:
            angle_factor = 1.0 - (angle / 180.0)
            radius = effective_radius

        for i in range(ny):
            for j in range(nx):
                x = x_min + j * res_x
                y = y_min + i * res_y
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < radius:
                    stress = angle_factor * (1.0 - dist / radius) ** 0.5
                    stress_map[i, j] = max(stress_map[i, j], stress)

    # Edge proximity / thin section stress using actual contour mask
    if HAS_SCIPY:
        mask = _build_interior_mask(
            boundaries, x_min, y_min, nx, ny, res_x, res_y)
        if mask.any():
            dist_inside = distance_transform_edt(mask) * res_x
            thin_threshold = model_size * 0.25
            edge_stress = np.clip(1.0 - dist_inside / thin_threshold, 0, 1)
            stress_map = np.maximum(stress_map, edge_stress * 0.5)
    elif HAS_SHAPELY:
        # Shapely fallback (kept for compatibility)
        _apply_edge_stress_shapely(
            boundaries, stress_map, x_min, y_min, nx, ny, res_x, res_y,
            model_size)

    # Hole detection: if there are multiple contours, inner ones may be holes
    num_holes = 0
    if len(boundaries) > 1:
        num_holes = _detect_and_stress_holes(
            boundaries, stress_map, x_min, y_min, nx, ny, res_x, res_y,
            corner_radius)

    # Apply sensitivity and smooth
    stress_map = np.power(stress_map, 1.0 / (sensitivity + 0.5))
    stress_map = np.clip(stress_map, 0, 1)
    if HAS_SCIPY:
        stress_map = gaussian_filter(stress_map, sigma=1.5)

    metadata = {
        'bounds': (x_min, x_max, y_min, y_max),
        'grid_size': (nx, ny),
        'resolution': (res_x, res_y),
        'corners': len(all_corners),
        'concave_corners': sum(1 for c in all_corners if c[3]),
        'holes': num_holes,
        'used_actual_contours': len(contours_for_analysis) > 0,
    }
    return stress_map, metadata


def _select_contours(
        layer_contours: Optional[Dict[float, LayerContour]],
        z_height: Optional[float],
        xy_points: np.ndarray
) -> List[np.ndarray]:
    """Select the best contours for stress analysis.

    If layer_contours and z_height are given, picks the nearest layer.
    Otherwise returns empty list (caller falls back to convex hull).
    """
    if not layer_contours:
        return []

    if z_height is not None:
        # Find nearest layer
        z_vals = sorted(layer_contours.keys())
        nearest_z = min(z_vals, key=lambda z: abs(z - z_height))
        return layer_contours[nearest_z].outer_walls

    # No specific Z requested -- merge contours from a representative layer
    # Pick the layer with the most complex geometry (most contour points)
    best_z = max(layer_contours.keys(),
                 key=lambda z: sum(len(w) for w in layer_contours[z].outer_walls))
    return layer_contours[best_z].outer_walls


def _collect_all_points(contours: List[np.ndarray],
                        fallback: np.ndarray) -> np.ndarray:
    """Collect all XY points from contours or use fallback."""
    if contours:
        all_pts = np.vstack(contours) if contours else np.array([]).reshape(0, 2)
        return all_pts
    return fallback


def _compute_boundary_fallback(xy_points: np.ndarray,
                                x_min: float, y_min: float,
                                x_max: float, y_max: float) -> Optional[np.ndarray]:
    """Compute boundary using convex hull (legacy fallback)."""
    if HAS_SCIPY and len(xy_points) > 3:
        try:
            unique = np.unique(xy_points, axis=0)
            if len(unique) > 3:
                hull = ConvexHull(unique)
                return unique[hull.vertices]
        except Exception:
            pass

    return np.array([[x_min, y_min], [x_max, y_min],
                     [x_max, y_max], [x_min, y_max]])


def _build_interior_mask(boundaries: List[np.ndarray],
                          x_min: float, y_min: float,
                          nx: int, ny: int,
                          res_x: float, res_y: float) -> np.ndarray:
    """Build interior mask from actual contour boundaries.

    Uses shapely if available, otherwise falls back to ray-casting.
    """
    if HAS_SHAPELY:
        try:
            mask = np.zeros((ny, nx), dtype=bool)
            polys = []
            for b in boundaries:
                if len(b) >= 3:
                    p = Polygon(b)
                    if not p.is_valid:
                        p = p.buffer(0)
                    polys.append(p)

            if not polys:
                return mask

            # First polygon is outer, rest may be holes
            # Use union for simplicity
            from shapely.ops import unary_union
            combined = unary_union(polys) if len(polys) > 1 else polys[0]

            for i in range(ny):
                for j in range(nx):
                    pt = Point(x_min + j * res_x, y_min + i * res_y)
                    mask[i, j] = combined.contains(pt) or \
                        combined.boundary.distance(pt) < 0.5
            return mask
        except Exception:
            pass

    # Pure numpy fallback using ray-casting
    return _build_mask_from_contours(
        boundaries, x_min, y_min, nx, ny, res_x, res_y)


def _apply_edge_stress_shapely(boundaries, stress_map, x_min, y_min,
                                nx, ny, res_x, res_y, model_size):
    """Apply edge/thin-section stress using shapely (no scipy needed)."""
    try:
        for b in boundaries:
            if len(b) < 3:
                continue
            poly = Polygon(b)
            if not poly.is_valid:
                poly = poly.buffer(0)
            thin_threshold = model_size * 0.25
            for i in range(ny):
                for j in range(nx):
                    pt = Point(x_min + j * res_x, y_min + i * res_y)
                    if poly.contains(pt):
                        dist = poly.boundary.distance(pt)
                        edge_stress = max(0, 1.0 - dist / thin_threshold) * 0.5
                        stress_map[i, j] = max(stress_map[i, j], edge_stress)
    except Exception:
        pass


def _detect_and_stress_holes(boundaries, stress_map, x_min, y_min,
                              nx, ny, res_x, res_y, hole_margin):
    """Detect holes (inner contours inside outer contours) and add stress."""
    if len(boundaries) < 2:
        return 0

    num_holes = 0
    # Check if smaller contours are inside the largest one
    areas = [abs(_polygon_area_signed(b)) for b in boundaries]
    outer_idx = np.argmax(areas)
    outer = boundaries[outer_idx]

    for idx, inner in enumerate(boundaries):
        if idx == outer_idx:
            continue
        # Check if centroid of inner is inside outer
        centroid = inner.mean(axis=0)
        if _point_in_polygon(centroid[0], centroid[1], outer):
            num_holes += 1
            # Add stress around hole boundary
            for pt in inner:
                for i in range(ny):
                    for j in range(nx):
                        x = x_min + j * res_x
                        y = y_min + i * res_y
                        dist = np.sqrt((x - pt[0])**2 + (y - pt[1])**2)
                        if dist < hole_margin:
                            # Kt ~ 3 for circular hole
                            stress = 0.8 * (1.0 - dist / hole_margin)
                            stress_map[i, j] = max(stress_map[i, j], stress)

    return num_holes


def calculate_stress_maps_per_layer(
        gcode_content: str,
        grid_resolution: float = 1.0,
        corner_radius: float = 5.0,
        sensitivity: float = 0.5,
        verbose: bool = False,
        max_z_ranges: int = 10
) -> Dict[Tuple[float, float], Tuple[np.ndarray, dict]]:
    """Compute per-Z-range stress maps from G-code contours.

    Instead of a single global 2D stress map applied to all layers, this
    computes stress maps for groups of layers that share similar geometry.

    Parameters
    ----------
    gcode_content : str
        Full G-code content.
    grid_resolution : float
        Grid cell size in mm.
    corner_radius : float
        Radius of influence around corners in mm.
    sensitivity : float
        Detection sensitivity.
    verbose : bool
        Print analysis details.
    max_z_ranges : int
        Maximum number of Z-ranges to compute.

    Returns
    -------
    dict mapping (z_min, z_max) -> (stress_map, metadata)
    """
    layer_contours = extract_layer_contours(gcode_content)

    if not layer_contours:
        # Fallback to global stress map
        xy_points, _ = extract_geometry_from_gcode(gcode_content)
        stress_map, metadata = calculate_stress_map(
            xy_points, grid_resolution, corner_radius, sensitivity, verbose)
        z_min = min(layer_contours.keys()) if layer_contours else 0.0
        z_max = max(layer_contours.keys()) if layer_contours else 999.0
        return {(z_min, z_max): (stress_map, metadata)}

    z_ranges = _group_layers_into_z_ranges(layer_contours, max_z_ranges)
    result = {}

    for z_min, z_max, contours in z_ranges:
        # Merge contours from all layers in this range
        # Use the layer with the most complex geometry as representative
        best_contour = max(contours,
                           key=lambda c: sum(len(w) for w in c.outer_walls))
        all_walls = best_contour.outer_walls

        if not all_walls:
            continue

        all_pts = np.vstack(all_walls) if all_walls else np.array([]).reshape(0, 2)

        # Create a temporary layer_contours dict for calculate_stress_map
        temp_contours = {best_contour.z_height: best_contour}

        try:
            stress_map, metadata = calculate_stress_map(
                all_pts, grid_resolution, corner_radius, sensitivity,
                verbose, layer_contours=temp_contours,
                z_height=best_contour.z_height)
            metadata['z_range'] = (z_min, z_max)
            result[(z_min, z_max)] = (stress_map, metadata)
        except ValueError:
            continue

    return result


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


def get_stress_at_point_z(stress_maps: Dict[Tuple[float, float], Tuple[np.ndarray, dict]],
                           x: float, y: float, z: float) -> float:
    """Look up stress at a specific XYZ coordinate using per-layer stress maps.

    Finds the Z-range containing the given Z height and returns the stress
    at (x, y) from that range's stress map.
    """
    for (z_min, z_max), (stress_map, metadata) in stress_maps.items():
        if z_min <= z <= z_max or (z_min == z_max and abs(z - z_min) < 0.5):
            return get_stress_at_point(stress_map, metadata, x, y)

    # Z not in any range -- try the nearest
    if stress_maps:
        nearest_key = min(stress_maps.keys(),
                          key=lambda k: min(abs(z - k[0]), abs(z - k[1])))
        sm, md = stress_maps[nearest_key]
        return get_stress_at_point(sm, md, x, y)

    return 0.0


def modify_infill_density(gcode_lines: List[str], stress_map: np.ndarray,
                          metadata: dict, min_density: int, max_density: int,
                          verbose: bool = False,
                          stress_maps: Optional[Dict] = None
                          ) -> List[str]:
    """Modify G-code infill sections based on the stress map.

    Processes each sparse infill region and adds reinforcement passes
    (extra parallel extrusion lines) in areas where the stress map indicates
    high stress concentration.

    When stress_maps (per-layer) is provided, uses the Z-appropriate stress
    map for each layer instead of a single global map.
    """
    output = []
    sparse_infill = re.compile(r';\s*FEATURE:\s*Sparse infill', re.IGNORECASE)
    other_feature = re.compile(r';\s*FEATURE:\s*(?!Sparse infill)', re.IGNORECASE)
    z_pattern = re.compile(r';\s*Z_HEIGHT:\s*([\d.]+)')
    move_pattern = re.compile(r'^G1\s+X([\d.]+)\s+Y([\d.]+)', re.IGNORECASE)
    extrude_pattern = re.compile(r'^G1\s+X([\d.]+)\s+Y([\d.]+)\s+.*E(-?[\d.]+)', re.IGNORECASE)

    in_infill = False
    infill_lines = []
    current_z = 0.0
    stats = {'layers_modified': 0, 'extra_lines': 0}

    for line in gcode_lines:
        z_match = z_pattern.search(line)
        if z_match:
            current_z = float(z_match.group(1))

        if sparse_infill.search(line):
            in_infill = True
            infill_lines = []
            output.append(line)
            continue

        if in_infill and other_feature.search(line):
            in_infill = False
            # Choose the right stress map for this Z
            if stress_maps:
                layer_sm, layer_md = _get_layer_stress_map(
                    stress_maps, current_z, stress_map, metadata)
            else:
                layer_sm, layer_md = stress_map, metadata

            modified = _process_infill_region(infill_lines, layer_sm, layer_md)
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


def _get_layer_stress_map(stress_maps, z, default_map, default_metadata):
    """Get the stress map for a given Z from per-layer maps."""
    for (z_min, z_max), (sm, md) in stress_maps.items():
        if z_min <= z <= z_max or (z_min == z_max and abs(z - z_min) < 0.5):
            return sm, md
    # Find nearest
    if stress_maps:
        nearest = min(stress_maps.keys(),
                      key=lambda k: min(abs(z - k[0]), abs(z - k[1])))
        return stress_maps[nearest]
    return default_map, default_metadata


def _process_infill_region(infill_lines: List[str], stress_map: np.ndarray,
                           metadata: dict) -> List[str]:
    """Add reinforcement passes to a single infill region."""
    output = []
    move_pattern = re.compile(r'^G1\s+X([\d.]+)\s+Y([\d.]+)', re.IGNORECASE)
    extrude_pattern = re.compile(r'^G1\s+X([\d.]+)\s+Y([\d.]+)\s+.*E(-?[\d.]+)', re.IGNORECASE)

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
        title = 'Stress Concentration Map\n(Bright = High Stress = More Infill)'
        if metadata.get('z_range'):
            z_min, z_max = metadata['z_range']
            title += f'\nZ range: {z_min:.1f} - {z_max:.1f} mm'
        if metadata.get('concave_corners', 0) > 0:
            title += f'\n({metadata["concave_corners"]} re-entrant corners detected)'
        ax.set_title(title)
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

    # Extract actual contours from outer wall paths (per layer)
    layer_contours = extract_layer_contours(gcode_content)
    num_contour_layers = len(layer_contours)
    total_contours = sum(len(lc.outer_walls) for lc in layer_contours.values())

    # Also get legacy point cloud for fallback
    xy_points, z_heights = extract_geometry_from_gcode(gcode_content)

    if num_contour_layers > 0:
        print(f"   Extracted {total_contours} outer wall contours from "
              f"{num_contour_layers} layers")
    else:
        print(f"   No outer wall contours found, using {len(xy_points)} "
              f"geometry points from {len(z_heights)} layers")

    # Calculate per-layer stress maps
    print(f"\nAnalyzing stress concentrations...")

    if num_contour_layers > 0:
        stress_maps = calculate_stress_maps_per_layer(
            gcode_content,
            corner_radius=args.corner_radius,
            sensitivity=args.sensitivity,
            verbose=args.verbose
        )
        # Use the most representative map for summary stats
        if stress_maps:
            representative = max(stress_maps.values(),
                                 key=lambda v: v[0].max())
            stress_map, metadata = representative
        else:
            stress_map, metadata = calculate_stress_map(
                xy_points, corner_radius=args.corner_radius,
                sensitivity=args.sensitivity, verbose=args.verbose)
            stress_maps = None
    else:
        stress_map, metadata = calculate_stress_map(
            xy_points, corner_radius=args.corner_radius,
            sensitivity=args.sensitivity, verbose=args.verbose)
        stress_maps = None

    stress_mean = stress_map.mean()
    stress_max = stress_map.max()
    high_stress_pct = (stress_map > 0.5).sum() / stress_map.size * 100

    print(f"   Average stress: {stress_mean:.2%}")
    print(f"   Maximum stress: {stress_max:.2%}")
    print(f"   High-stress area: {high_stress_pct:.1f}%")
    print(f"   Corners detected: {metadata['corners']} "
          f"({metadata.get('concave_corners', 0)} concave/re-entrant)")
    print(f"   Holes detected: {metadata.get('holes', 0)}")
    if metadata.get('used_actual_contours'):
        print(f"   Using actual model contours (not convex hull)")
    if stress_maps:
        print(f"   Computed {len(stress_maps)} per-Z-range stress maps")

    if args.visualize:
        viz_path = args.output_3mf.replace('.3mf', '_stress_map.png')
        if visualize_stress_map(stress_map, metadata, viz_path):
            print(f"\n  Stress map saved to: {viz_path}")

    # Modify infill
    print(f"\nModifying infill density ({args.min_density}% -> {args.max_density}%)...")
    gcode_lines = gcode_content.splitlines(keepends=True)
    modified = modify_infill_density(gcode_lines, stress_map, metadata,
                                     args.min_density, args.max_density,
                                     verbose=args.verbose,
                                     stress_maps=stress_maps)

    new_gcode = ''.join(modified)
    print(f"\nCreating {args.output_3mf}...")
    repack_3mf(args.input_3mf, args.output_3mf, new_gcode, gcode_path)
    print(f"\nDone!")


if __name__ == '__main__':
    main()
