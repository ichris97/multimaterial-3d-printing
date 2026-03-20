#!/usr/bin/env python3
"""
Layer-by-layer filament pattern assignment for Bambu Studio 3MF files.

This post-processor modifies the layer_config_ranges.xml inside a 3MF file
to assign different extruders (filaments) to each layer according to a
user-defined repeating pattern. This enables multi-material printing where
different materials are used on different layers.

How It Works
------------
Bambu Studio supports per-layer extruder overrides via an XML file stored at
Metadata/layer_config_ranges.xml inside the 3MF archive. Each <range> element
specifies a Z height range and the extruder to use for that range. This script
generates this XML for every layer according to the desired pattern.

Pattern Specification
---------------------
Patterns can be specified in several formats:

1. Simple list: "1,2,2,1" — each number is a filament, one per layer
2. Repeat notation: "1x3,2x2" → [1,1,1,2,2] (filament × count)
3. Clear notation: "F1:3,F2:2" → [1,1,1,2,2] (Filament N for M layers)

The pattern repeats cyclically until all layers are assigned.

Examples
--------
    # Alternating PLA/TPU layers (sandwich composite)
    python -m multimaterial_3d.postprocessors.layer_pattern input.3mf output.3mf \\
        --pattern 1,2 --layer-height 0.2 --total-height 10

    # 3 layers PLA, 2 layers TPU, repeating (laminate)
    python -m multimaterial_3d.postprocessors.layer_pattern input.3mf output.3mf \\
        --pattern F1:3,F2:2 --analyze --materials "1:PLA,2:TPU"

    # Sandwich structure: stiff skin, soft core
    python -m multimaterial_3d.postprocessors.layer_pattern input.3mf output.3mf \\
        --pattern F1:5,F2:10,F1:5 --analyze --materials "1:PLA-CF,2:TPU"

Use Cases
---------
- **Impact absorption**: Alternate stiff and flexible layers to create a
  composite that absorbs energy through controlled deformation
- **Thermal management**: Use materials with different thermal properties
  on different layers
- **Cost optimization**: Use expensive high-performance material only where
  needed (e.g., outer layers for stiffness, cheap material for core)
- **Aesthetic effects**: Layer colors for visual patterns visible from the side
"""

import argparse
import re
import xml.etree.ElementTree as ET

from ..core.file_io import get_model_height, inject_layer_config
from ..analysis.mechanical import analyze_layup
from ..analysis.thermal import thermal_stress_analysis


def parse_args():
    """Parse command-line arguments for the layer pattern tool."""
    parser = argparse.ArgumentParser(
        description='Assign custom layer filament patterns to 3MF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pattern formats:
  Simple list:     --pattern 1,2,2,1,1,1
  Clear notation:  --pattern F1:1,F2:2,F1:3
                   (Filament 1 for 1 layer, Filament 2 for 2 layers, Filament 1 for 3 layers)
  Legacy notation: --pattern 1x1,2x2,1x3

The pattern repeats until total height is reached.

Examples:
  Alternating:     --pattern 1,2              -> [1, 2, 1, 2, ...]
  Sandwich core:   --pattern F1:5,F2:10,F1:5  -> stiff skin, soft core
  Gradient:        --pattern F1:3,F2:3         -> gradual transition
        """
    )
    parser.add_argument('input_3mf', help='Input 3MF file path')
    parser.add_argument('output_3mf', help='Output 3MF file path')
    parser.add_argument('--layer-height', type=float, default=0.1,
                        help='Layer height in mm (default: 0.1)')
    parser.add_argument('--total-height', type=float, default=None,
                        help='Total print height in mm (auto-detected if not specified)')
    parser.add_argument('--pattern', type=str, default='1,2',
                        help='Layer pattern (default: 1,2)')
    parser.add_argument('--object-id', type=int, default=1,
                        help='Object ID in the 3MF (default: 1)')
    parser.add_argument('--analyze', action='store_true',
                        help='Run mechanical and thermal analysis on the layup')
    parser.add_argument('--materials', type=str, default=None,
                        help='Material types for analysis, e.g., "1:PLA,2:TPU"')
    parser.add_argument('--gradient', type=int, default=0,
                        help='Add N gradient transition layers between materials (default: 0)')
    return parser.parse_args()


def parse_pattern(pattern_str: str) -> list:
    """Parse a pattern string into a list of filament numbers.

    Supports three notations:
    - Simple: "1,2,2,1" → [1, 2, 2, 1]
    - Repeat: "1x3,2x2" → [1, 1, 1, 2, 2]
    - Clear:  "F1:3,F2:2" → [1, 1, 1, 2, 2]

    Parameters
    ----------
    pattern_str : str
        Pattern specification string.

    Returns
    -------
    list of int
        Expanded list of filament numbers.

    Raises
    ------
    ValueError
        If the pattern string cannot be parsed.
    """
    pattern = []
    parts = pattern_str.replace(' ', '').split(',')

    for part in parts:
        # Format: F1:3 means "Filament 1 for 3 layers"
        if part.upper().startswith('F') and ':' in part:
            match = re.match(r'[Ff](\d+):(\d+)', part)
            if match:
                filament = int(match.group(1))
                count = int(match.group(2))
                pattern.extend([filament] * count)
            else:
                raise ValueError(f"Invalid F notation: {part}")
        # Format: 1x3 means "filament 1 repeated 3 times"
        elif 'x' in part.lower():
            match = re.match(r'(\d+)x(\d+)', part.lower())
            if match:
                filament = int(match.group(1))
                count = int(match.group(2))
                pattern.extend([filament] * count)
            else:
                raise ValueError(f"Invalid repeat notation: {part}")
        else:
            pattern.append(int(part))

    return pattern


def generate_layer_ranges_xml(object_id: int, layer_height: float,
                               total_height: float, pattern: list) -> str:
    """Generate Bambu Studio layer_config_ranges.xml content.

    Creates an XML document that assigns extruders to layer height ranges.
    Bambu Studio reads this to determine which filament to use per layer.

    Parameters
    ----------
    object_id : int
        The object ID within the 3MF model.
    layer_height : float
        Layer height in mm.
    total_height : float
        Total model height in mm.
    pattern : list of int
        Repeating filament pattern.

    Returns
    -------
    str
        XML content as string.
    """
    root = ET.Element('objects')
    obj = ET.SubElement(root, 'object', id=str(object_id))

    num_layers = int(total_height / layer_height)
    pattern_len = len(pattern)

    for i in range(num_layers):
        min_z = round(i * layer_height, 4)
        max_z = round((i + 1) * layer_height, 4)
        filament = pattern[i % pattern_len]

        range_elem = ET.SubElement(obj, 'range', min_z=str(min_z), max_z=str(max_z))
        ET.SubElement(range_elem, 'option', opt_key='extruder').text = str(filament)
        ET.SubElement(range_elem, 'option', opt_key='layer_height').text = str(layer_height)

    return ET.tostring(root, encoding='unicode')


def parse_materials(materials_str: str) -> dict:
    """Parse materials string like '1:PLA,2:TPU' into a dict."""
    material_map = {}
    if materials_str:
        for pair in materials_str.split(','):
            parts = pair.strip().split(':')
            if len(parts) == 2:
                filament = int(parts[0])
                material = parts[1].upper()
                material_map[filament] = material
    return material_map


def main():
    """Main entry point for the layer pattern post-processor."""
    args = parse_args()

    # Parse pattern
    try:
        pattern = parse_pattern(args.pattern)
    except ValueError as e:
        print(f"Error parsing pattern: {e}")
        return

    print(f"Pattern: {pattern} (repeating)")
    print(f"Pattern length: {len(pattern)} layers")

    # Auto-detect height if needed
    import zipfile
    total_height = args.total_height
    if total_height is None:
        with zipfile.ZipFile(args.input_3mf, 'r') as zf:
            total_height = get_model_height(zf)
        if total_height is None:
            raise ValueError("Could not auto-detect model height. Please specify --total-height")
        print(f"Auto-detected model height: {total_height}mm")

    # Generate and inject layer config
    xml_content = generate_layer_ranges_xml(args.object_id, args.layer_height,
                                            total_height, pattern)
    inject_layer_config(args.input_3mf, args.output_3mf, xml_content)

    num_layers = int(total_height / args.layer_height)
    print(f"Generated {num_layers} layer ranges")
    print(f"Created: {args.output_3mf}")

    # Run analysis if requested
    if args.analyze:
        material_map = parse_materials(args.materials)
        unique_filaments = set(pattern)
        for f in unique_filaments:
            if f not in material_map:
                material_map[f] = 'PLA'
                print(f"Note: Using PLA as default for filament {f}. "
                      f"Specify with --materials '{f}:MATERIAL'")

        # Mechanical analysis
        analyze_layup(pattern, total_height, args.layer_height, material_map)

        # Thermal stress analysis
        thermal_stress_analysis(pattern, args.layer_height, total_height, material_map)

    print("\nDone! Open the output file in Bambu Studio to see the pattern.")


if __name__ == '__main__':
    main()
