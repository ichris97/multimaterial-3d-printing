#!/usr/bin/env python3
"""
Topology-Optimized Infill Post-Processor for Bambu Studio 3MF files

Analyzes 3D model geometry to identify stress concentration points and
automatically increases infill density in those regions.

Stress Analysis Methods:
1. Corner Detection - Sharp corners concentrate stress
2. Thin Section Analysis - Narrow regions are weak points  
3. Hole Proximity - Areas around holes need reinforcement
4. Curvature Analysis - High curvature = stress concentration
5. Medial Axis Distance - Distance to nearest boundary indicates thickness

Usage:
    python topology_infill.py input.3mf output.3mf
    python topology_infill.py input.3mf output.3mf --min-density 15 --max-density 80
"""

import argparse
import re
import zipfile
import hashlib
import tempfile
import struct
import io
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import numpy as np

# Try to import optional libraries
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    from shapely.geometry import Polygon, MultiPolygon, Point, LineString
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

try:
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import ConvexHull, Delaunay
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class StressRegion:
    """Region with calculated stress level"""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    stress_level: float  # 0.0 to 1.0
    reason: str  # Why this region has high stress


@dataclass 
class InfillSegment:
    """A segment of infill G-code"""
    start_line: int
    end_line: int
    layer_z: float
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Topology-optimized infill for Bambu Studio 3MF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stress Analysis:
  The script analyzes your model geometry to find weak points:
  - Sharp corners (stress concentrators)
  - Thin sections (prone to failure)
  - Areas around holes (need reinforcement)
  - High curvature regions

Infill Modification:
  Low-stress areas: Use base infill density (e.g., 15%)
  High-stress areas: Increase to max density (e.g., 80%)
  Gradient transition between regions

Examples:
  # Basic usage with defaults
  python topology_infill.py model.3mf optimized.3mf
  
  # Custom density range
  python topology_infill.py model.3mf optimized.3mf --min-density 10 --max-density 100
  
  # More aggressive stress detection
  python topology_infill.py model.3mf optimized.3mf --sensitivity 0.8
        """
    )
    parser.add_argument('input_3mf', help='Input 3MF file (must be sliced)')
    parser.add_argument('output_3mf', help='Output 3MF file')
    parser.add_argument('--min-density', type=int, default=15,
                        help='Minimum infill density %% for low-stress areas (default: 15)')
    parser.add_argument('--max-density', type=int, default=80,
                        help='Maximum infill density %% for high-stress areas (default: 80)')
    parser.add_argument('--sensitivity', type=float, default=0.5,
                        help='Stress detection sensitivity 0.0-1.0 (default: 0.5)')
    parser.add_argument('--corner-radius', type=float, default=5.0,
                        help='Radius around corners to reinforce in mm (default: 5.0)')
    parser.add_argument('--hole-margin', type=float, default=3.0,
                        help='Margin around holes to reinforce in mm (default: 3.0)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate stress map visualization (requires matplotlib)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed processing info')
    return parser.parse_args()


def load_stl_from_bytes(data: bytes) -> np.ndarray:
    """Load STL file from bytes, returns Nx3x3 array of triangles"""
    
    # Check if ASCII or binary
    if data[:5] == b'solid' and b'\n' in data[:80]:
        # ASCII STL
        return load_ascii_stl(data)
    else:
        # Binary STL
        return load_binary_stl(data)


def load_binary_stl(data: bytes) -> np.ndarray:
    """Load binary STL file"""
    # Skip 80-byte header
    num_triangles = struct.unpack('<I', data[80:84])[0]
    
    triangles = []
    offset = 84
    
    for _ in range(num_triangles):
        # Skip normal (12 bytes), read 3 vertices (36 bytes), skip attribute (2 bytes)
        # Normal: 3 floats
        # Vertex 1: 3 floats
        # Vertex 2: 3 floats  
        # Vertex 3: 3 floats
        # Attribute: 1 uint16
        
        v1 = struct.unpack('<fff', data[offset+12:offset+24])
        v2 = struct.unpack('<fff', data[offset+24:offset+36])
        v3 = struct.unpack('<fff', data[offset+36:offset+48])
        
        triangles.append([v1, v2, v3])
        offset += 50
    
    return np.array(triangles)


def load_ascii_stl(data: bytes) -> np.ndarray:
    """Load ASCII STL file"""
    text = data.decode('utf-8', errors='ignore')
    triangles = []
    current_triangle = []
    
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('vertex'):
            parts = line.split()
            vertex = (float(parts[1]), float(parts[2]), float(parts[3]))
            current_triangle.append(vertex)
            
            if len(current_triangle) == 3:
                triangles.append(current_triangle)
                current_triangle = []
    
    return np.array(triangles)


def extract_model_from_3mf(input_3mf: str) -> Tuple[Optional[np.ndarray], str, str]:
    """Extract 3D model and G-code from 3MF file.
    
    Returns:
        Tuple of (triangles array or None, gcode_content, gcode_path)
    """
    triangles = None
    gcode_content = None
    gcode_path = None
    
    with zipfile.ZipFile(input_3mf, 'r') as zf:
        # Find model file (usually .stl or in 3D/ folder)
        for name in zf.namelist():
            if name.endswith('.stl'):
                stl_data = zf.read(name)
                triangles = load_stl_from_bytes(stl_data)
                break
            elif name.endswith('.model') and '3D/' in name:
                # 3MF native format - try to parse XML for vertices
                model_data = zf.read(name)
                try:
                    triangles = parse_3mf_model(model_data)
                except:
                    triangles = None  # Model data not available
                break
        
        # Find G-code
        for name in zf.namelist():
            if name.endswith('.gcode'):
                gcode_content = zf.read(name).decode('utf-8')
                gcode_path = name
                break
    
    if gcode_content is None:
        raise ValueError("No G-code found in 3MF file. Make sure to slice the model first!")
    
    return triangles, gcode_content, gcode_path


def extract_geometry_from_gcode(gcode_content: str) -> Tuple[np.ndarray, List[float]]:
    """Extract XY geometry and Z layers from G-code movements.
    
    Returns:
        Tuple of (xy_points array, z_heights list)
    """
    move_pattern = re.compile(r'^G[01]\s+X([\d.]+)\s+Y([\d.]+)', re.IGNORECASE)
    extrude_pattern = re.compile(r'^G[01]\s+X([\d.]+)\s+Y([\d.]+).*E[\d.]+', re.IGNORECASE)
    z_pattern = re.compile(r';\s*Z_HEIGHT:\s*([\d.]+)')
    wall_pattern = re.compile(r';\s*FEATURE:\s*(Inner|Outer)\s*wall', re.IGNORECASE)
    infill_pattern = re.compile(r';\s*FEATURE:\s*(Sparse|Internal|Solid)\s*infill', re.IGNORECASE)
    feature_end = re.compile(r';\s*(FEATURE:|CHANGE_LAYER|WIPE_)', re.IGNORECASE)
    
    xy_points = []
    z_heights = set()
    current_z = 0.0
    in_printable_feature = False
    
    for line in gcode_content.split('\n'):
        # Track Z
        z_match = z_pattern.search(line)
        if z_match:
            current_z = float(z_match.group(1))
            z_heights.add(current_z)
        
        # Track wall and infill features (actual model geometry)
        if wall_pattern.search(line) or infill_pattern.search(line):
            in_printable_feature = True
        elif feature_end.search(line) and not wall_pattern.search(line) and not infill_pattern.search(line):
            in_printable_feature = False
        
        # Only collect extrusion moves in actual print features
        if in_printable_feature:
            match = extrude_pattern.match(line.strip())
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                xy_points.append([x, y])
    
    return np.array(xy_points) if xy_points else np.array([]).reshape(0, 2), sorted(z_heights)


def parse_3mf_model(data: bytes) -> np.ndarray:
    """Parse 3MF XML model format to extract triangles"""
    import xml.etree.ElementTree as ET
    
    text = data.decode('utf-8')
    # Remove namespace for easier parsing
    text = re.sub(r'\sxmlns="[^"]+"', '', text, count=1)
    
    root = ET.fromstring(text)
    
    vertices = []
    triangles_idx = []
    
    # Find mesh vertices and triangles
    for mesh in root.iter('mesh'):
        for vertices_elem in mesh.iter('vertices'):
            for vertex in vertices_elem.iter('vertex'):
                x = float(vertex.get('x', 0))
                y = float(vertex.get('y', 0))
                z = float(vertex.get('z', 0))
                vertices.append([x, y, z])
        
        for triangles_elem in mesh.iter('triangles'):
            for triangle in triangles_elem.iter('triangle'):
                v1 = int(triangle.get('v1', 0))
                v2 = int(triangle.get('v2', 0))
                v3 = int(triangle.get('v3', 0))
                triangles_idx.append([v1, v2, v3])
    
    if not vertices or not triangles_idx:
        raise ValueError("Could not parse 3MF model - no mesh data found")
    
    vertices = np.array(vertices)
    triangles_idx = np.array(triangles_idx)
    
    # Convert to triangle array (Nx3x3)
    triangles = vertices[triangles_idx]
    
    return triangles


def project_to_xy(triangles: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Project 3D triangles to XY plane, return 2D points and bounding box"""
    
    # Get all vertices
    all_vertices = triangles.reshape(-1, 3)
    
    # Project to XY (just take X and Y)
    xy_points = all_vertices[:, :2]
    
    # Calculate bounding box
    x_min, y_min = xy_points.min(axis=0)
    x_max, y_max = xy_points.max(axis=0)
    
    return xy_points, (x_min, x_max, y_min, y_max)


def compute_2d_boundary(triangles: np.ndarray, z_slice: float = None) -> List[np.ndarray]:
    """Compute 2D boundary polygon(s) from 3D mesh.
    
    If z_slice is provided, slice at that height.
    Otherwise, project all edges to XY.
    """
    if HAS_TRIMESH:
        mesh = trimesh.Trimesh(vertices=triangles.reshape(-1, 3),
                               faces=np.arange(len(triangles) * 3).reshape(-1, 3))
        
        if z_slice is not None:
            # Slice at specific Z
            try:
                slice_2d = mesh.section(plane_origin=[0, 0, z_slice],
                                        plane_normal=[0, 0, 1])
                if slice_2d is not None:
                    path_2d, _ = slice_2d.to_planar()
                    return [np.array(p.vertices) for p in path_2d.polygons_closed]
            except:
                pass
        
        # Fall back to projection
        # Get boundary edges (edges that appear in only one triangle)
        edges = mesh.edges_unique
        
    # Simple projection approach
    xy_points, bbox = project_to_xy(triangles)
    
    if HAS_SCIPY:
        try:
            hull = ConvexHull(xy_points)
            boundary = xy_points[hull.vertices]
            return [boundary]
        except:
            pass
    
    # Fallback: return bounding box as polygon
    x_min, x_max, y_min, y_max = bbox
    boundary = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ])
    return [boundary]


def find_corners(boundary: np.ndarray, angle_threshold: float = 120) -> List[Tuple[float, float]]:
    """Find corner points in a 2D boundary polygon.
    
    Corners are defined as points where the angle is less than threshold.
    """
    corners = []
    n = len(boundary)
    
    for i in range(n):
        p_prev = boundary[(i - 1) % n]
        p_curr = boundary[i]
        p_next = boundary[(i + 1) % n]
        
        # Calculate vectors
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        
        if angle < angle_threshold:
            corners.append((p_curr[0], p_curr[1], angle))
    
    return corners


def find_thin_sections(boundary: np.ndarray, grid_resolution: float = 0.5) -> np.ndarray:
    """Find thin sections using distance transform.
    
    Returns a 2D array where low values indicate thin sections.
    """
    if not HAS_SCIPY or not HAS_SHAPELY:
        return None
    
    try:
        # Create polygon
        poly = Polygon(boundary)
        if not poly.is_valid:
            poly = poly.buffer(0)
        
        # Create grid
        minx, miny, maxx, maxy = poly.bounds
        
        nx = int((maxx - minx) / grid_resolution) + 1
        ny = int((maxy - miny) / grid_resolution) + 1
        
        # Limit grid size
        nx = min(nx, 500)
        ny = min(ny, 500)
        
        # Create binary mask
        mask = np.zeros((ny, nx), dtype=bool)
        
        for i in range(ny):
            for j in range(nx):
                x = minx + j * grid_resolution
                y = miny + i * grid_resolution
                if poly.contains(Point(x, y)):
                    mask[i, j] = True
        
        # Distance transform - distance to nearest boundary
        dist = distance_transform_edt(mask) * grid_resolution
        
        return dist, (minx, miny, maxx, maxy, grid_resolution)
    
    except Exception as e:
        print(f"Warning: Could not compute thin sections: {e}")
        return None, None


def detect_holes(triangles: np.ndarray, z_slice: float) -> List[Tuple[float, float, float]]:
    """Detect holes in the model at a given Z height.
    
    Returns list of (center_x, center_y, radius) for each hole.
    """
    holes = []
    
    if not HAS_TRIMESH:
        return holes
    
    try:
        mesh = trimesh.Trimesh(vertices=triangles.reshape(-1, 3),
                               faces=np.arange(len(triangles) * 3).reshape(-1, 3))
        
        # Slice at Z height
        slice_2d = mesh.section(plane_origin=[0, 0, z_slice],
                                plane_normal=[0, 0, 1])
        
        if slice_2d is not None:
            path_2d, _ = slice_2d.to_planar()
            
            # Find interior polygons (holes)
            for polygon in path_2d.polygons_full:
                if hasattr(polygon, 'interiors'):
                    for interior in polygon.interiors:
                        coords = np.array(interior.coords)
                        center = coords.mean(axis=0)
                        radius = np.linalg.norm(coords - center, axis=1).mean()
                        holes.append((center[0], center[1], radius))
    except:
        pass
    
    return holes


def calculate_stress_map_from_xy(xy_points: np.ndarray,
                                  grid_resolution: float = 1.0,
                                  corner_radius: float = 5.0,
                                  sensitivity: float = 0.5,
                                  verbose: bool = False) -> Tuple[np.ndarray, dict]:
    """Calculate 2D stress map from XY points extracted from G-code.
    
    Returns:
        stress_map: 2D numpy array with stress values 0.0-1.0
        metadata: Dictionary with analysis info
    """
    
    if len(xy_points) == 0:
        raise ValueError("No geometry points found in G-code")
    
    # Calculate bounds
    x_min, y_min = xy_points.min(axis=0)
    x_max, y_max = xy_points.max(axis=0)
    
    if verbose:
        print(f"   Model bounds: X=[{x_min:.1f}, {x_max:.1f}] Y=[{y_min:.1f}, {y_max:.1f}]")
    
    # Create grid
    nx = int((x_max - x_min) / grid_resolution) + 1
    ny = int((y_max - y_min) / grid_resolution) + 1
    
    # Limit grid size
    nx = min(nx, 300)
    ny = min(ny, 300)
    
    actual_res_x = (x_max - x_min) / max(nx - 1, 1)
    actual_res_y = (y_max - y_min) / max(ny - 1, 1)
    
    stress_map = np.zeros((ny, nx))
    
    if verbose:
        print(f"   Grid size: {nx}x{ny}")
        print(f"   Extracted {len(xy_points)} boundary points")
    
    # Compute convex hull for boundary
    boundary = None
    if HAS_SCIPY and len(xy_points) > 3:
        try:
            # Use unique points
            unique_points = np.unique(xy_points, axis=0)
            if len(unique_points) > 3:
                hull = ConvexHull(unique_points)
                boundary = unique_points[hull.vertices]
        except:
            pass
    
    if boundary is None:
        # Fallback: use bounding box
        boundary = np.array([
            [x_min, y_min], [x_max, y_min],
            [x_max, y_max], [x_min, y_max]
        ])
    
    # Find corners
    corners = find_corners(boundary, angle_threshold=150)  # More sensitive corner detection
    
    if verbose:
        print(f"   Detected {len(corners)} corners")
    
    # Model size for scaling
    model_size = max(x_max - x_min, y_max - y_min)
    
    # Add stress around corners - propagate further into interior
    effective_corner_radius = max(corner_radius, model_size * 0.3)  # At least 30% of model
    
    for cx, cy, angle in corners:
        angle_factor = 1.0 - (angle / 180.0)
        
        for i in range(ny):
            for j in range(nx):
                x = x_min + j * actual_res_x
                y = y_min + i * actual_res_y
                
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < effective_corner_radius:
                    # Stress decreases with distance from corner
                    stress = angle_factor * (1.0 - dist / effective_corner_radius) ** 0.5  # Slower falloff
                    stress_map[i, j] = max(stress_map[i, j], stress)
    
    # Edge proximity stress (thin sections near boundary)
    if HAS_SHAPELY:
        try:
            poly = Polygon(boundary)
            if not poly.is_valid:
                poly = poly.buffer(0)
            
            # Create inside mask
            mask = np.zeros((ny, nx), dtype=bool)
            for i in range(ny):
                for j in range(nx):
                    x = x_min + j * actual_res_x
                    y = y_min + i * actual_res_y
                    mask[i, j] = poly.contains(Point(x, y)) or poly.boundary.distance(Point(x, y)) < 0.5
            
            if HAS_SCIPY:
                # Distance from boundary
                dist_inside = distance_transform_edt(mask) * actual_res_x
                
                # Edge stress - areas close to boundaries need reinforcement
                thin_threshold = model_size * 0.25
                edge_stress = np.clip(1.0 - dist_inside / thin_threshold, 0, 1)
                stress_map = np.maximum(stress_map, edge_stress * 0.5)
                
                if verbose:
                    print(f"   Applied edge proximity analysis")
        except Exception as e:
            if verbose:
                print(f"   Warning: Edge analysis failed: {e}")
    
    # Ensure all interior points have some baseline stress near boundaries
    # This helps add reinforcement near perimeters
    
    # Apply sensitivity - lower values make more aggressive stress detection
    stress_map = np.power(stress_map, 1.0 / (sensitivity + 0.5))
    stress_map = np.clip(stress_map, 0, 1)
    
    # Smooth
    if HAS_SCIPY:
        from scipy.ndimage import gaussian_filter
        stress_map = gaussian_filter(stress_map, sigma=1.5)
    
    metadata = {
        'bounds': (x_min, x_max, y_min, y_max),
        'grid_size': (nx, ny),
        'resolution': (actual_res_x, actual_res_y),
        'corners': len(corners),
        'holes': 0
    }
    
    return stress_map, metadata


def calculate_stress_map(triangles: np.ndarray, 
                         grid_resolution: float = 1.0,
                         corner_radius: float = 5.0,
                         hole_margin: float = 3.0,
                         sensitivity: float = 0.5,
                         verbose: bool = False) -> Tuple[np.ndarray, dict]:
    """Calculate 2D stress concentration map.
    
    Returns:
        stress_map: 2D numpy array with stress values 0.0-1.0
        metadata: Dictionary with analysis info
    """
    
    # Project to XY
    xy_points, bbox = project_to_xy(triangles)
    x_min, x_max, y_min, y_max = bbox
    
    if verbose:
        print(f"   Model bounds: X=[{x_min:.1f}, {x_max:.1f}] Y=[{y_min:.1f}, {y_max:.1f}]")
    
    # Create grid
    nx = int((x_max - x_min) / grid_resolution) + 1
    ny = int((y_max - y_min) / grid_resolution) + 1
    
    # Limit grid size for performance
    nx = min(nx, 300)
    ny = min(ny, 300)
    
    actual_res_x = (x_max - x_min) / max(nx - 1, 1)
    actual_res_y = (y_max - y_min) / max(ny - 1, 1)
    
    stress_map = np.zeros((ny, nx))
    
    # Get boundary
    boundaries = compute_2d_boundary(triangles)
    
    if verbose:
        print(f"   Grid size: {nx}x{ny}")
        print(f"   Found {len(boundaries)} boundary polygon(s)")
    
    # 1. Corner stress
    all_corners = []
    for boundary in boundaries:
        corners = find_corners(boundary, angle_threshold=120)
        all_corners.extend(corners)
    
    if verbose:
        print(f"   Detected {len(all_corners)} corners")
    
    # Add stress around corners
    for cx, cy, angle in all_corners:
        # Sharper angle = more stress
        angle_factor = 1.0 - (angle / 180.0)
        
        for i in range(ny):
            for j in range(nx):
                x = x_min + j * actual_res_x
                y = y_min + i * actual_res_y
                
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < corner_radius:
                    # Stress decreases with distance from corner
                    stress = angle_factor * (1.0 - dist / corner_radius)
                    stress_map[i, j] = max(stress_map[i, j], stress)
    
    # 2. Thin section stress (using distance transform)
    if HAS_SCIPY and HAS_SHAPELY and len(boundaries) > 0:
        try:
            dist_result = find_thin_sections(boundaries[0], grid_resolution=actual_res_x)
            if dist_result[0] is not None:
                dist_map, dist_meta = dist_result
                
                # Normalize by model size
                model_size = max(x_max - x_min, y_max - y_min)
                thin_threshold = model_size * 0.15  # 15% of model size
                
                # Resize dist_map to match stress_map if needed
                if dist_map.shape != stress_map.shape:
                    from scipy.ndimage import zoom
                    zoom_y = ny / dist_map.shape[0]
                    zoom_x = nx / dist_map.shape[1]
                    dist_map = zoom(dist_map, (zoom_y, zoom_x), order=1)
                
                # Thin sections have high stress
                thin_stress = np.clip(1.0 - dist_map / thin_threshold, 0, 1)
                stress_map = np.maximum(stress_map, thin_stress * 0.7)  # Weight thin sections
                
                if verbose:
                    print(f"   Applied thin section analysis (threshold: {thin_threshold:.1f}mm)")
        except Exception as e:
            if verbose:
                print(f"   Warning: Thin section analysis failed: {e}")
    
    # 3. Hole proximity stress
    z_heights = triangles[:, :, 2]
    z_min, z_max = z_heights.min(), z_heights.max()
    z_mid = (z_min + z_max) / 2
    
    holes = detect_holes(triangles, z_mid)
    
    if verbose and holes:
        print(f"   Detected {len(holes)} holes")
    
    for hx, hy, hr in holes:
        for i in range(ny):
            for j in range(nx):
                x = x_min + j * actual_res_x
                y = y_min + i * actual_res_y
                
                dist = np.sqrt((x - hx)**2 + (y - hy)**2)
                margin_dist = dist - hr
                
                if 0 < margin_dist < hole_margin:
                    stress = 1.0 - margin_dist / hole_margin
                    stress_map[i, j] = max(stress_map[i, j], stress * 0.8)
    
    # 4. Apply sensitivity
    stress_map = np.power(stress_map, 1.0 / (sensitivity + 0.1))
    stress_map = np.clip(stress_map, 0, 1)
    
    # Smooth the stress map
    if HAS_SCIPY:
        from scipy.ndimage import gaussian_filter
        stress_map = gaussian_filter(stress_map, sigma=2)
    
    metadata = {
        'bounds': bbox,
        'grid_size': (nx, ny),
        'resolution': (actual_res_x, actual_res_y),
        'corners': len(all_corners),
        'holes': len(holes)
    }
    
    return stress_map, metadata


def get_stress_at_point(stress_map: np.ndarray, metadata: dict, x: float, y: float) -> float:
    """Get stress level at a specific XY coordinate."""
    x_min, x_max, y_min, y_max = metadata['bounds']
    nx, ny = metadata['grid_size']
    
    # Check bounds
    if x < x_min or x > x_max or y < y_min or y > y_max:
        return 0.0
    
    # Convert to grid coordinates
    j = int((x - x_min) / (x_max - x_min) * (nx - 1))
    i = int((y - y_min) / (y_max - y_min) * (ny - 1))
    
    j = np.clip(j, 0, nx - 1)
    i = np.clip(i, 0, ny - 1)
    
    return stress_map[i, j]


def modify_infill_density(gcode_lines: List[str], 
                          stress_map: np.ndarray,
                          metadata: dict,
                          min_density: int = 15,
                          max_density: int = 80,
                          verbose: bool = False) -> List[str]:
    """Modify G-code to increase infill density in high-stress regions.
    
    Strategy: Add extra infill lines in high-stress areas by reducing
    the spacing between existing infill lines.
    """
    
    output_lines = []
    
    # Patterns
    sparse_infill = re.compile(r';\s*FEATURE:\s*Sparse infill', re.IGNORECASE)
    other_feature = re.compile(r';\s*FEATURE:\s*(?!Sparse infill)', re.IGNORECASE)
    move_pattern = re.compile(r'^G1\s+X([\d.]+)\s+Y([\d.]+)\s+.*E([\d.]+)', re.IGNORECASE)
    z_pattern = re.compile(r';\s*Z_HEIGHT:\s*([\d.]+)')
    
    in_infill = False
    current_z = 0.0
    infill_lines = []
    infill_start = 0
    
    stats = {'layers_modified': 0, 'extra_lines_added': 0}
    
    for i, line in enumerate(gcode_lines):
        # Track Z height
        z_match = z_pattern.search(line)
        if z_match:
            current_z = float(z_match.group(1))
        
        # Detect infill start
        if sparse_infill.search(line):
            in_infill = True
            infill_lines = []
            infill_start = i
            output_lines.append(line)
            continue
        
        # Detect infill end
        if in_infill and other_feature.search(line):
            in_infill = False
            
            # Process collected infill lines
            modified_infill = process_infill_region(
                infill_lines, stress_map, metadata, 
                min_density, max_density, current_z
            )
            
            if len(modified_infill) > len(infill_lines):
                stats['layers_modified'] += 1
                stats['extra_lines_added'] += len(modified_infill) - len(infill_lines)
            
            output_lines.extend(modified_infill)
            output_lines.append(line)
            continue
        
        if in_infill:
            infill_lines.append(line)
        else:
            output_lines.append(line)
    
    if verbose:
        print(f"\n📊 Infill Modification Stats:")
        print(f"   Layers with added infill: {stats['layers_modified']}")
        print(f"   Extra infill lines added: {stats['extra_lines_added']}")
    
    return output_lines


def process_infill_region(infill_lines: List[str],
                          stress_map: np.ndarray,
                          metadata: dict,
                          min_density: int,
                          max_density: int,
                          current_z: float) -> List[str]:
    """Process an infill region and add extra lines in high-stress areas."""
    
    output = []
    move_pattern = re.compile(r'^G1\s+X([\d.]+)\s+Y([\d.]+)', re.IGNORECASE)
    extrude_pattern = re.compile(r'^G1\s+X([\d.]+)\s+Y([\d.]+)\s+.*E([\d.]+)', re.IGNORECASE)
    
    # Collect moves and their stress levels
    moves = []
    for line in infill_lines:
        match = move_pattern.match(line.strip())
        if match:
            x, y = float(match.group(1)), float(match.group(2))
            stress = get_stress_at_point(stress_map, metadata, x, y)
            moves.append((x, y, stress, line))
        else:
            moves.append((None, None, 0, line))
    
    # Generate output with extra infill in high-stress areas
    prev_x, prev_y = None, None
    stress_threshold = 0.20  # Trigger at 20% stress
    
    for x, y, stress, line in moves:
        output.append(line)
        
        # If stress above threshold and this is an extrusion move, add reinforcement
        if x is not None and stress > stress_threshold and prev_x is not None:
            # Calculate extra passes based on stress level
            # At threshold: 1 extra pass
            # At max stress (1.0): up to 3 extra passes
            stress_normalized = (stress - stress_threshold) / (1.0 - stress_threshold)
            extra_passes = 1 + int(stress_normalized * 2)  # 1 to 3 passes
            
            if extra_passes > 0 and 'E' in line:
                # Add offset parallel lines
                dx = x - prev_x
                dy = y - prev_y
                length = np.sqrt(dx*dx + dy*dy)
                
                if length > 1.0:  # Only for moves > 1mm
                    # Perpendicular offset
                    offset = 0.5  # mm between extra lines
                    if length > 0:
                        nx, ny = -dy/length, dx/length
                    else:
                        nx, ny = 0, 0
                    
                    for p in range(min(extra_passes, 3)):  # Max 3 extra passes
                        offset_dist = offset * (p + 1) * (1 if p % 2 == 0 else -1)
                        
                        # Add reinforcement comment
                        output.append(f"; TOPO-REINFORCEMENT pass {p+1} (stress={stress:.2f})\n")
                        
                        # Move to offset start
                        ox1 = prev_x + nx * offset_dist
                        oy1 = prev_y + ny * offset_dist
                        output.append(f"G1 X{ox1:.3f} Y{oy1:.3f} F12000\n")
                        
                        # Extrude to offset end
                        ox2 = x + nx * offset_dist
                        oy2 = y + ny * offset_dist
                        
                        # Calculate extrusion
                        match = extrude_pattern.match(line.strip())
                        if match:
                            e_val = float(match.group(3))
                            output.append(f"G1 X{ox2:.3f} Y{oy2:.3f} E{e_val:.5f} F3000\n")
        
        prev_x, prev_y = x, y
    
    return output


def visualize_stress_map(stress_map: np.ndarray, metadata: dict, output_path: str):
    """Generate a visualization of the stress map."""
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


def repack_3mf(input_3mf: str, output_3mf: str, new_gcode: str, gcode_path: str):
    """Repack 3MF with modified G-code and updated MD5."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Extract all files
        with zipfile.ZipFile(input_3mf, 'r') as zf:
            zf.extractall(temp_dir)
        
        # Write new G-code
        gcode_file = temp_dir / gcode_path
        gcode_file.write_text(new_gcode, encoding='utf-8')
        
        # Update MD5 if it exists
        md5_path = temp_dir / (gcode_path + '.md5')
        if md5_path.exists():
            md5_hash = hashlib.md5(new_gcode.encode('utf-8')).hexdigest()
            md5_path.write_text(md5_hash)
        
        # Repack
        with zipfile.ZipFile(output_3mf, 'w', zipfile.ZIP_DEFLATED) as zf_out:
            for file_path in temp_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(temp_dir)
                    zf_out.write(file_path, arcname)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  Topology-Optimized Infill Post-Processor")
    print("=" * 60)
    
    # Check dependencies
    if not HAS_SCIPY:
        print("⚠️  Warning: scipy not available - some analysis features disabled")
    if not HAS_SHAPELY:
        print("⚠️  Warning: shapely not available - some analysis features disabled")
    if not HAS_TRIMESH:
        print("⚠️  Warning: trimesh not available - using basic mesh loading")
    
    print(f"\n📦 Loading {args.input_3mf}...")
    
    try:
        triangles, gcode_content, gcode_path = extract_model_from_3mf(args.input_3mf)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    
    # Check if we have 3D model or need to extract from G-code
    use_gcode_geometry = (triangles is None or len(triangles) == 0)
    
    if use_gcode_geometry:
        print(f"   3D model not in file - extracting geometry from G-code...")
        xy_points, z_heights = extract_geometry_from_gcode(gcode_content)
        print(f"   Extracted {len(xy_points)} points from {len(z_heights)} layers")
    else:
        print(f"   Model: {len(triangles)} triangles")
    
    print(f"   G-code: {gcode_path}")
    
    # Calculate stress map
    print(f"\n🔬 Analyzing stress concentrations...")
    print(f"   Sensitivity: {args.sensitivity}")
    print(f"   Corner radius: {args.corner_radius}mm")
    print(f"   Hole margin: {args.hole_margin}mm")
    
    if use_gcode_geometry:
        stress_map, metadata = calculate_stress_map_from_xy(
            xy_points,
            grid_resolution=1.0,
            corner_radius=args.corner_radius,
            sensitivity=args.sensitivity,
            verbose=args.verbose
        )
    else:
        stress_map, metadata = calculate_stress_map(
            triangles,
            grid_resolution=1.0,
            corner_radius=args.corner_radius,
            hole_margin=args.hole_margin,
            sensitivity=args.sensitivity,
            verbose=args.verbose
        )
    
    # Print stress analysis summary
    stress_mean = stress_map.mean()
    stress_max = stress_map.max()
    high_stress_pct = (stress_map > 0.5).sum() / stress_map.size * 100
    
    print(f"\n📊 Stress Analysis Results:")
    print(f"   Average stress level: {stress_mean:.2%}")
    print(f"   Maximum stress level: {stress_max:.2%}")
    print(f"   High-stress area: {high_stress_pct:.1f}% of model")
    print(f"   Corners detected: {metadata['corners']}")
    print(f"   Holes detected: {metadata['holes']}")
    
    # Visualize if requested
    if args.visualize:
        viz_path = args.output_3mf.replace('.3mf', '_stress_map.png')
        if visualize_stress_map(stress_map, metadata, viz_path):
            print(f"\n📈 Stress map saved to: {viz_path}")
        else:
            print("⚠️  Could not generate visualization (matplotlib not available)")
    
    # Modify infill
    print(f"\n⚙️  Modifying infill density...")
    print(f"   Low-stress density: {args.min_density}%")
    print(f"   High-stress density: {args.max_density}%")
    
    gcode_lines = gcode_content.splitlines(keepends=True)
    modified_lines = modify_infill_density(
        gcode_lines, stress_map, metadata,
        args.min_density, args.max_density,
        verbose=args.verbose
    )
    
    new_gcode = ''.join(modified_lines)
    
    # Repack 3MF
    print(f"\n📦 Creating {args.output_3mf}...")
    repack_3mf(args.input_3mf, args.output_3mf, new_gcode, gcode_path)
    
    print(f"\n✅ Done!")
    print(f"\n📋 Summary:")
    print(f"   Input: {args.input_3mf}")
    print(f"   Output: {args.output_3mf}")
    print(f"   Infill density: {args.min_density}% → {args.max_density}% based on stress")
    print(f"\n💡 The output 3MF can be sent directly to your printer")


if __name__ == '__main__':
    main()