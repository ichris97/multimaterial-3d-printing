"""
Core module: material database, data structures, and file I/O utilities.
"""
from .materials import MATERIAL_DB, get_material, list_materials
from .file_io import extract_gcode_from_3mf, repack_3mf, get_model_height
