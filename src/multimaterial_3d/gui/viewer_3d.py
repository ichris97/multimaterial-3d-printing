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
    """Parse 3MF XML model file to extract mesh as PyVista PolyData.

    Bambu Lab 3MF files store the actual mesh in 3D/Objects/*.model,
    while 3D/3dmodel.model is an assembly file with no geometry.
    We try all .model files and use the first one containing mesh data.
    """
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(filepath, 'r') as zf:
        # Collect all candidate model files, preferring Objects/ subfolder
        model_files = []
        stl_file = None
        for name in zf.namelist():
            if name.endswith('.model') and '3D/' in name:
                model_files.append(name)
            elif name.endswith('.stl'):
                stl_file = name

        # Sort so Objects/*.model files come first (they have actual geometry)
        model_files.sort(key=lambda n: (0 if 'Objects/' in n else 1, n))

        for model_file in model_files:
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


def _subdivide_mesh_at_layers(mesh: pv.PolyData, layer_height: float) -> pv.PolyData:
    """Subdivide a solid mesh at layer-height intervals along Z.

    Clips the mesh at each layer boundary using Z-normal clip planes
    to create new triangles at every layer height. This converts a
    coarse solid mesh (e.g. a cube with 12 faces) into one where
    every cell lies within exactly one layer and can be colored
    independently.

    For meshes spanning many layers, limits to at most 500 clip
    operations to keep performance reasonable.
    """
    z_min = mesh.bounds[4]
    z_max = mesh.bounds[5]
    z_range = z_max - z_min

    if z_range < layer_height:
        return mesh

    n_layers = int(np.ceil(z_range / layer_height))

    # Cap the number of clip operations for performance
    max_clips = 500
    if n_layers > max_clips:
        # Use a coarser effective layer height for clipping, but still
        # assign proper layer IDs by centroid afterward
        effective_lh = z_range / max_clips
        n_layers = max_clips
    else:
        effective_lh = layer_height

    # Ensure mesh is triangulated
    tri_mesh = mesh.triangulate()

    # Iteratively clip the mesh at each layer boundary from bottom up
    slabs = []
    remaining = tri_mesh.copy()

    for i in range(n_layers - 1):
        z_cut = z_min + (i + 1) * effective_lh

        if remaining is None or remaining.n_cells == 0:
            break

        try:
            # Below the cut = this layer's slab (invert=True keeps below)
            below = remaining.clip(
                normal='z', origin=(0, 0, z_cut), invert=True
            )
            above = remaining.clip(
                normal='z', origin=(0, 0, z_cut), invert=False
            )
        except Exception:
            break

        if below is not None and below.n_cells > 0:
            below.cell_data['layer'] = np.full(
                below.n_cells, i, dtype=np.int32
            )
            slabs.append(below)

        remaining = above

    # Add the topmost remaining slab
    if remaining is not None and remaining.n_cells > 0:
        remaining.cell_data['layer'] = np.full(
            remaining.n_cells, n_layers - 1, dtype=np.int32
        )
        slabs.append(remaining)

    if not slabs:
        return mesh

    # Combine all slabs into one mesh
    combined = slabs[0]
    for slab in slabs[1:]:
        combined = combined.merge(slab)

    # If we used coarser clipping, refine layer IDs by centroid
    if effective_lh != layer_height:
        centers = combined.cell_centers().points
        combined.cell_data['layer'] = (
            (centers[:, 2] - z_min) / layer_height
        ).astype(np.int32)

    return combined


def color_mesh_by_layers(mesh: pv.PolyData, layer_height: float,
                          pattern: list = None,
                          material_map: dict = None) -> pv.PolyData:
    """Assign layer and material scalars to mesh faces for visualization.

    For solid meshes (few faces spanning many layers), the mesh is first
    subdivided at layer-height intervals so each resulting cell belongs
    to exactly one layer. This produces proper horizontal color bands.

    Optionally maps layers to material IDs from a pattern.
    """
    z_min = mesh.bounds[4]
    z_max = mesh.bounds[5]
    z_range = z_max - z_min
    expected_layers = max(1, int(np.ceil(z_range / layer_height)))

    # Check if mesh needs subdivision: if the number of unique centroid
    # layers is much less than expected, the mesh is too coarse.
    centers = mesh.cell_centers().points
    coarse_layers = len(np.unique(
        ((centers[:, 2] - z_min) / layer_height).astype(int)
    ))

    if coarse_layers < expected_layers * 0.5 and expected_layers > 3:
        # Mesh is too coarse -- subdivide at layer boundaries
        mesh = _subdivide_mesh_at_layers(mesh, layer_height)
    else:
        # Mesh is already fine enough, assign layer by centroid
        centers = mesh.cell_centers().points
        layer_ids = ((centers[:, 2] - z_min) / layer_height).astype(int)
        mesh.cell_data["layer"] = layer_ids

    if pattern:
        layers = mesh.cell_data.get("layer")
        if layers is not None:
            pattern_len = len(pattern)
            material_ids = np.array(
                [pattern[lid % pattern_len] for lid in layers]
            )
            mesh.cell_data["material"] = material_ids

    return mesh


def parse_gcode_paths(gcode_content: str, max_layers: int = None):
    """Parse G-code into renderable line segments with metadata.

    Returns a PyVista PolyData with line cells and per-segment scalars
    for coloring by feature type, layer, or tool (material).

    Handles Bambu Studio G-code format with:
    - ; CHANGE_LAYER markers
    - ; FEATURE: <type> markers
    - T<n> tool change commands
    - Relative E values (Bambu default)
    """
    move_re = re.compile(r'^G[01]\s', re.IGNORECASE)
    feature_re = re.compile(r';\s*FEATURE:\s*(.+)', re.IGNORECASE)
    layer_re = re.compile(r';\s*CHANGE_LAYER', re.IGNORECASE)
    tool_re = re.compile(r'^T(\d+)', re.IGNORECASE)
    # Also detect Z changes from G1 Z<value> lines (layer height updates)
    z_change_re = re.compile(r'^G1\s+Z([\d.]+)\s+F', re.IGNORECASE)

    FEATURE_MAP = {
        'outer wall': 0, 'inner wall': 1, 'sparse infill': 2,
        'solid infill': 3, 'internal solid infill': 3,
        'bridge': 4, 'support': 5, 'support interface': 5,
        'overhang wall': 6, 'top surface': 7, 'bottom surface': 8,
        'prime tower': 9, 'custom': 10, 'skirt': 11,
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
            current_feature = FEATURE_MAP.get(feat_name, 2)
            continue

        tm = tool_re.match(stripped)
        if tm:
            tool_num = int(tm.group(1))
            # Bambu uses T255, T1000 etc. as special commands, not real
            # tool changes. Only accept tool numbers 0-15.
            if tool_num <= 15:
                current_tool = tool_num
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
    lines_arr = np.array(
        [item for sublist in lines for item in sublist], dtype=np.int64
    )
    # Build PolyData with lines specified upfront so cell count is correct
    mesh = pv.PolyData(pts, lines=lines_arr)
    mesh.cell_data["feature"] = np.array(feature_scalars)
    mesh.cell_data["layer"] = np.array(layer_scalars)
    mesh.cell_data["tool"] = np.array(tool_scalars)

    return mesh
