"""
3D mesh viewer widget using PyVista embedded in Qt.

Handles loading 3MF files, extracting meshes, and rendering them with
per-layer or per-material coloring. Supports interactive rotation, zoom,
and pan via VTK's built-in interactor.
"""

import re
import zipfile
import tempfile
import numpy as np
from pathlib import Path

import pyvista as pv

# Suppress VTK warnings in the GUI
pv.global_theme.allow_empty_mesh = True

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def load_mesh_from_3mf(filepath: str) -> pv.PolyData:
    """Extract and load the 3D mesh from a 3MF file.

    Tries trimesh first (best 3MF support), then falls back to manual
    XML parsing of the 3MF model data.

    Returns a PyVista PolyData mesh.
    """
    if HAS_TRIMESH:
        try:
            scene = trimesh.load(filepath, force='scene')
            if isinstance(scene, trimesh.Scene):
                mesh_tm = trimesh.util.concatenate(scene.dump())
            else:
                mesh_tm = scene
            return pv.wrap(mesh_tm)
        except Exception:
            pass

    # Fallback: manual 3MF XML parsing
    return _parse_3mf_xml(filepath)


def _parse_3mf_xml(filepath: str) -> pv.PolyData:
    """Parse 3MF XML model file to extract mesh as PyVista PolyData."""
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(filepath, 'r') as zf:
        model_file = None
        stl_file = None
        for name in zf.namelist():
            if name.endswith('.model') and '3D/' in name:
                model_file = name
            elif name.endswith('.stl'):
                stl_file = name

        if model_file:
            data = zf.read(model_file).decode('utf-8')
            data = re.sub(r'\sxmlns="[^"]+"', '', data, count=1)
            root = ET.fromstring(data)

            vertices = []
            faces_idx = []

            for mesh in root.iter('mesh'):
                for v_elem in mesh.iter('vertices'):
                    for v in v_elem.iter('vertex'):
                        vertices.append([
                            float(v.get('x', 0)),
                            float(v.get('y', 0)),
                            float(v.get('z', 0))
                        ])
                for t_elem in mesh.iter('triangles'):
                    for t in t_elem.iter('triangle'):
                        faces_idx.append([
                            int(t.get('v1', 0)),
                            int(t.get('v2', 0)),
                            int(t.get('v3', 0))
                        ])

            if vertices and faces_idx:
                verts = np.array(vertices, dtype=np.float32)
                faces = np.array(faces_idx, dtype=np.int64)
                # PyVista face format: [n_pts, p1, p2, p3, ...]
                pv_faces = np.column_stack([
                    np.full(len(faces), 3, dtype=np.int64),
                    faces
                ]).ravel()
                return pv.PolyData(verts, pv_faces)

        if stl_file:
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
                tmp.write(zf.read(stl_file))
                tmp_path = tmp.name
            mesh = pv.read(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)
            return mesh

    raise ValueError(f"No mesh found in {filepath}")


def color_mesh_by_layers(mesh: pv.PolyData, layer_height: float,
                          pattern: list = None,
                          material_map: dict = None) -> pv.PolyData:
    """Assign layer and material scalars to mesh faces for visualization.

    Colors each face based on which layer its centroid falls in, and
    optionally maps layers to material IDs from a pattern.
    """
    centers = mesh.cell_centers().points
    z_min = centers[:, 2].min()
    layer_ids = ((centers[:, 2] - z_min) / layer_height).astype(int)
    mesh.cell_data["layer"] = layer_ids

    if pattern:
        pattern_len = len(pattern)
        material_ids = np.array([pattern[lid % pattern_len] for lid in layer_ids])
        mesh.cell_data["material"] = material_ids

    return mesh


def parse_gcode_paths(gcode_content: str, max_layers: int = None):
    """Parse G-code into renderable line segments with metadata.

    Returns points, line connectivity, and per-segment scalars for
    coloring by feature type or material.
    """
    move_re = re.compile(r'^G[01]\s', re.IGNORECASE)
    feature_re = re.compile(r';\s*FEATURE:\s*(.+)', re.IGNORECASE)
    layer_re = re.compile(r';\s*CHANGE_LAYER', re.IGNORECASE)
    tool_re = re.compile(r'^T(\d+)', re.IGNORECASE)

    FEATURE_COLORS = {
        'outer wall': 0, 'inner wall': 1, 'sparse infill': 2,
        'solid infill': 3, 'bridge': 4, 'support': 5,
        'overhang wall': 6, 'top surface': 7, 'bottom surface': 8,
    }

    x, y, z = 0.0, 0.0, 0.0
    current_feature = 2  # default: infill
    current_layer = 0
    current_tool = 0

    points = []
    lines = []
    feature_scalars = []
    layer_scalars = []
    tool_scalars = []
    idx = 0

    for line in gcode_content.split('\n'):
        stripped = line.strip()

        if layer_re.search(stripped):
            current_layer += 1
            if max_layers and current_layer > max_layers:
                break
            continue

        fm = feature_re.search(stripped)
        if fm:
            feat_name = fm.group(1).strip().lower()
            current_feature = FEATURE_COLORS.get(feat_name, 2)
            continue

        tm = tool_re.match(stripped)
        if tm:
            current_tool = int(tm.group(1))
            continue

        if not move_re.match(stripped):
            continue

        # Parse coordinates
        params = {}
        for m in re.finditer(r'([XYZEF])(-?[\d.]+)', stripped, re.IGNORECASE):
            params[m.group(1).upper()] = float(m.group(2))

        nx = params.get('X', x)
        ny = params.get('Y', y)
        nz = params.get('Z', z)
        has_e = 'E' in params and params['E'] > 0

        if has_e and (abs(nx - x) > 0.01 or abs(ny - y) > 0.01):
            points.append([x, y, z])
            points.append([nx, ny, nz])
            lines.append([2, idx, idx + 1])
            feature_scalars.append(current_feature)
            layer_scalars.append(current_layer)
            tool_scalars.append(current_tool)
            idx += 2

        x, y, z = nx, ny, nz

    if not points:
        return None

    pts = np.array(points, dtype=np.float32)
    mesh = pv.PolyData(pts)
    mesh.lines = np.array([item for sublist in lines for item in sublist], dtype=np.int64)
    mesh.cell_data["feature"] = np.array(feature_scalars)
    mesh.cell_data["layer"] = np.array(layer_scalars)
    mesh.cell_data["tool"] = np.array(tool_scalars)

    return mesh
