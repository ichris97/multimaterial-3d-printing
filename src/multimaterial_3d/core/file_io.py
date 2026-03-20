"""
File I/O utilities for 3MF and G-code file handling.

The 3MF format is a ZIP archive containing:
- 3D model data (STL or XML mesh in 3D/ folder)
- Metadata (layer configs, thumbnails, print settings)
- Sliced G-code (in Metadata/ folder for Bambu Studio)
- MD5 checksums for G-code integrity verification

This module provides functions to extract, modify, and repack 3MF archives
while preserving all metadata and updating checksums correctly.
"""

import hashlib
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, Optional


def extract_gcode_from_3mf(input_3mf: str) -> Tuple[str, str]:
    """Extract G-code content from a sliced 3MF file.

    Bambu Studio stores sliced G-code inside the 3MF archive, typically at
    Metadata/plate_1.gcode. This function finds and extracts it.

    Parameters
    ----------
    input_3mf : str
        Path to the input 3MF file. Must be a sliced file (i.e., it must
        contain at least one .gcode file inside the archive).

    Returns
    -------
    tuple of (str, str)
        (gcode_content, gcode_archive_path) where gcode_content is the full
        G-code text and gcode_archive_path is the path within the ZIP archive
        (e.g., 'Metadata/plate_1.gcode').

    Raises
    ------
    ValueError
        If no .gcode file is found inside the 3MF archive. This means the
        model was not sliced before being saved.
    """
    with zipfile.ZipFile(input_3mf, 'r') as zf:
        gcode_files = [f for f in zf.namelist() if f.endswith('.gcode')]

        if not gcode_files:
            raise ValueError(
                f"No G-code found in {input_3mf}. "
                "Make sure to slice the model in Bambu Studio before saving."
            )

        # Bambu Studio convention: Metadata/plate_N.gcode
        gcode_path = gcode_files[0]
        gcode_content = zf.read(gcode_path).decode('utf-8')

        return gcode_content, gcode_path


def repack_3mf(input_3mf: str, output_3mf: str,
               new_gcode: str, gcode_path: str) -> None:
    """Repack a 3MF archive with modified G-code.

    Extracts the original 3MF to a temporary directory, replaces the G-code
    file, updates the MD5 checksum (if present), and creates a new 3MF.

    The MD5 checksum update is important: Bambu Studio and Bambu Lab printers
    verify G-code integrity using an adjacent .md5 file. If the checksum
    doesn't match, the printer may refuse to print.

    Parameters
    ----------
    input_3mf : str
        Path to the original 3MF file.
    output_3mf : str
        Path for the new 3MF file with modified G-code.
    new_gcode : str
        The modified G-code content.
    gcode_path : str
        The archive-internal path of the G-code file (as returned by
        extract_gcode_from_3mf).
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        with zipfile.ZipFile(input_3mf, 'r') as zf:
            zf.extractall(temp_dir)

        # Write the modified G-code
        gcode_file = temp_dir / gcode_path
        gcode_file.write_text(new_gcode, encoding='utf-8')

        # Update MD5 checksum if the slicer generated one
        md5_path = temp_dir / (gcode_path + '.md5')
        if md5_path.exists():
            md5_hash = hashlib.md5(new_gcode.encode('utf-8')).hexdigest()
            md5_path.write_text(md5_hash)

        # Repack into a new ZIP/3MF archive
        with zipfile.ZipFile(output_3mf, 'w', zipfile.ZIP_DEFLATED) as zf_out:
            for file_path in temp_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(temp_dir)
                    zf_out.write(file_path, arcname)


def get_model_height(zip_ref: zipfile.ZipFile) -> Optional[float]:
    """Auto-detect the total model height from 3MF geometry data.

    Parses the 3MF XML model files to find the Z extent of the mesh.
    This is used when the user doesn't specify --total-height, allowing
    automatic layer count calculation.

    Parameters
    ----------
    zip_ref : zipfile.ZipFile
        An open ZipFile handle to the 3MF archive.

    Returns
    -------
    float or None
        The model height in mm (max_z - min_z), or None if the height
        could not be determined from the model data.
    """
    try:
        for name in zip_ref.namelist():
            if 'Objects' in name and name.endswith('.model'):
                content = zip_ref.read(name).decode('utf-8')
                root = ET.fromstring(content)
                z_values = []
                for vertex in root.iter():
                    if 'vertex' in vertex.tag.lower():
                        z = vertex.get('z')
                        if z:
                            z_values.append(float(z))
                if z_values:
                    return max(z_values) - min(z_values)
    except Exception as e:
        print(f"Warning: Could not auto-detect height: {e}")
    return None


def inject_layer_config(input_3mf: str, output_3mf: str,
                        xml_content: str) -> None:
    """Inject a layer_config_ranges.xml into a 3MF file.

    This is used by the layer pattern tool to assign different filaments
    to specific layer height ranges. Bambu Studio reads this XML to
    determine per-layer extruder overrides.

    Parameters
    ----------
    input_3mf : str
        Path to the original 3MF file.
    output_3mf : str
        Path for the output 3MF file.
    xml_content : str
        The XML content for Metadata/layer_config_ranges.xml.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        with zipfile.ZipFile(input_3mf, 'r') as zf:
            zf.extractall(temp_dir)

        # Write the layer config XML
        config_path = temp_dir / 'Metadata' / 'layer_config_ranges.xml'
        config_path.parent.mkdir(parents=True, exist_ok=True)

        full_xml = '<?xml version="1.0" encoding="utf-8"?>\n' + xml_content
        config_path.write_text(full_xml, encoding='utf-8')

        # Repack
        with zipfile.ZipFile(output_3mf, 'w', zipfile.ZIP_DEFLATED) as zf_out:
            for file_path in temp_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(temp_dir)
                    zf_out.write(file_path, arcname)
