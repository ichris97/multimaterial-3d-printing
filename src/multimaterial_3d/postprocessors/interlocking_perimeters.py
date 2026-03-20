#!/usr/bin/env python3
"""
Interlocking perimeter post-processor for Bambu Studio 3MF files.

Creates mechanical interlocking between layers by offsetting alternating
wall (perimeter) loops to different Z heights. This dramatically improves
the Z-direction (inter-layer) strength of FDM prints.

Concept
-------
In a standard FDM print with 3 wall loops, all walls are printed at the
same Z height within each layer:

    Standard:       [Wall1][Wall2][Wall3]  ← all at Z = 0.20mm

With interlocking, the middle wall is printed at a different Z height,
creating a physical overlap between adjacent layers:

    Layer N:        [Wall1]       [Wall3]  at Z = 0.20mm
                           [Wall2]         at Z = 0.30mm  (offset by +0.10mm)

    Layer N+1:      [Wall1]       [Wall3]  at Z = 0.40mm
                           [Wall2]         at Z = 0.50mm

This means Wall2 of layer N physically overlaps with the Wall1/Wall3 zone
of layer N+1, creating a mechanical interlock that resists delamination.

The Z offset is specified as a fraction of the layer height:
- 0.5 (default): Half-layer offset, maximum interlock
- 0.3: Gentler offset, may work better for some materials
- 0.0: No offset (standard behavior)

Implementation
--------------
The post-processor works on sliced G-code inside the 3MF:

1. Parse G-code to identify layer boundaries and wall loops
2. Classify walls by type (inner/outer) and loop index
3. For designated walls, add a Z-move command before printing
4. After the offset wall is printed, return to the normal Z height
5. Repack the modified G-code into the 3MF

The parser handles Bambu Studio's G-code comment format:
- "; FEATURE: Inner wall" marks the start of inner perimeters
- "; FEATURE: Outer wall" marks the start of the outer perimeter
- "; CHANGE_LAYER" marks layer transitions
- "; Z_HEIGHT: X.XX" indicates the current Z position

Wall Loop Detection
-------------------
Within an inner wall section, individual loops are detected by travel moves
(G1 moves with high feedrate and no extrusion). When the printhead makes a
rapid non-extruding move, it's transitioning between separate wall loops.
"""

import argparse
import re
from dataclasses import dataclass, field
from typing import List, Tuple

from ..core.file_io import extract_gcode_from_3mf, repack_3mf


@dataclass
class WallLoop:
    """Represents a single wall perimeter loop with its G-code commands.

    Attributes
    ----------
    lines : list of str
        The G-code lines that make up this wall loop.
    wall_type : str
        Either 'inner' or 'outer'. Bambu Studio prints inner walls first,
        then the outer wall for better surface quality.
    loop_index : int
        Sequential wall number within the layer (1, 2, 3, ...).
        Wall 1 is typically the innermost perimeter.
    z_height : float
        The Z height at which this wall is printed (before offset).
    """
    lines: List[str] = field(default_factory=list)
    wall_type: str = ""
    loop_index: int = 0
    z_height: float = 0.0


@dataclass
class Layer:
    """Represents a complete print layer with its wall loops and other content.

    A layer is divided into three sections:
    1. pre_walls: G-code before any wall commands (Z moves, temp changes, etc.)
    2. wall_loops: The perimeter wall loops (inner + outer)
    3. post_walls: G-code after walls (infill, top/bottom surfaces, etc.)

    Attributes
    ----------
    layer_num : int
        Zero-indexed layer number.
    z_height : float
        The nominal Z height of this layer in mm.
    """
    layer_num: int = 0
    z_height: float = 0.0
    pre_walls: List[str] = field(default_factory=list)
    wall_loops: List[WallLoop] = field(default_factory=list)
    post_walls: List[str] = field(default_factory=list)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Create interlocking perimeters in Bambu Studio 3MF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
How it works:
  With 3 walls, walls are numbered 1 (innermost), 2 (middle), 3 (outermost).
  By default, wall 2 is offset by half a layer height, creating interlock.

Examples:
  python -m multimaterial_3d.postprocessors.interlocking_perimeters input.3mf output.3mf
  python -m multimaterial_3d.postprocessors.interlocking_perimeters input.3mf output.3mf --offset 0.3
  python -m multimaterial_3d.postprocessors.interlocking_perimeters input.3mf output.3mf --walls 1,3
        """
    )
    parser.add_argument('input_3mf', help='Input 3MF file (must be sliced)')
    parser.add_argument('output_3mf', help='Output 3MF file')
    parser.add_argument('--offset', type=float, default=0.5,
                        help='Z offset as fraction of layer height (default: 0.5)')
    parser.add_argument('--walls', type=str, default='2',
                        help='Wall numbers to offset, comma-separated (default: 2)')
    parser.add_argument('--layer-height', type=float, default=None,
                        help='Layer height in mm (auto-detected if not specified)')
    parser.add_argument('--min-layer', type=int, default=3,
                        help='Start interlocking from this layer (default: 3)')
    parser.add_argument('--verbose', '-v', action='store_true')
    return parser.parse_args()


def detect_layer_height(content: str) -> float:
    """Auto-detect layer height from G-code comments.

    Looks for Bambu Studio's layer_height parameter or the LAYER_HEIGHT
    comment. Falls back to 0.2mm if neither is found.

    Parameters
    ----------
    content : str
        G-code content to search.

    Returns
    -------
    float
        Detected layer height in mm.
    """
    match = re.search(r';\s*layer_height\s*=\s*([\d.]+)', content)
    if match:
        return float(match.group(1))
    match = re.search(r';\s*LAYER_HEIGHT:\s*([\d.]+)', content)
    if match:
        return float(match.group(1))
    return 0.2


def parse_gcode_content(lines: List[str], verbose: bool = False
                        ) -> Tuple[List[str], List[Layer], List[str]]:
    """Parse G-code lines into structured layers with separated wall loops.

    This parser identifies layer boundaries, wall features, and individual
    wall loops within each layer. It classifies each line of G-code as
    belonging to the header, a specific layer section, or the footer.

    The wall loop detection within inner wall sections works by identifying
    high-speed travel moves (feedrate > 10000 mm/min) that indicate the
    printhead is moving to the start of a new loop.

    Parameters
    ----------
    lines : list of str
        Raw G-code lines (with newlines preserved).
    verbose : bool
        Print debug info about parsed structure.

    Returns
    -------
    tuple of (header, layers, footer)
        header: list of str — G-code lines before the first layer
        layers: list of Layer — parsed layer objects
        footer: list of str — G-code lines after the last layer
    """
    # Compiled regex patterns for performance
    layer_change = re.compile(r';\s*CHANGE_LAYER|;\s*layer\s*#\d+', re.IGNORECASE)
    inner_wall = re.compile(r';\s*FEATURE:\s*Inner wall', re.IGNORECASE)
    outer_wall = re.compile(r';\s*FEATURE:\s*Outer wall', re.IGNORECASE)
    other_feature = re.compile(r';\s*FEATURE:\s*(?!Inner wall|Outer wall)', re.IGNORECASE)
    z_height_comment = re.compile(r';\s*Z_HEIGHT:\s*([\d.]+)')
    z_move = re.compile(r'G1\s+.*Z([\d.]+)')
    travel_move = re.compile(r'^G1\s+X[\d.]+\s+Y[\d.]+\s+F(\d+)\s*$', re.IGNORECASE)

    header = []
    layers = []
    footer = []

    current_layer = None
    current_wall = None
    wall_count = 0
    in_header = True
    in_walls = False
    in_inner_walls = False
    last_z = 0.0

    for line in lines:
        # Detect layer change
        if layer_change.search(line):
            in_header = False
            if current_layer is not None:
                if current_wall and current_wall.lines:
                    current_layer.wall_loops.append(current_wall)
                layers.append(current_layer)

            current_layer = Layer()
            current_layer.layer_num = len(layers)
            current_wall = None
            wall_count = 0
            in_walls = False
            in_inner_walls = False
            current_layer.pre_walls.append(line)
            continue

        if in_header:
            header.append(line)
            continue

        if current_layer is None:
            header.append(line)
            continue

        # Track Z height from comments and moves
        z_match = z_height_comment.search(line)
        if z_match:
            current_layer.z_height = float(z_match.group(1))
            last_z = current_layer.z_height

        z_move_match = z_move.search(line)
        if z_move_match and not in_walls:
            last_z = float(z_move_match.group(1))
            if current_layer.z_height == 0:
                current_layer.z_height = last_z

        # Inner wall start — begin a new wall loop
        if inner_wall.search(line):
            if current_wall and current_wall.lines:
                current_layer.wall_loops.append(current_wall)
            wall_count += 1
            current_wall = WallLoop(wall_type="inner", loop_index=wall_count,
                                    z_height=last_z)
            current_wall.lines.append(line)
            in_walls = True
            in_inner_walls = True
            continue

        # Outer wall start
        if outer_wall.search(line):
            if current_wall and current_wall.lines:
                current_layer.wall_loops.append(current_wall)
            wall_count += 1
            current_wall = WallLoop(wall_type="outer", loop_index=wall_count,
                                    z_height=last_z)
            current_wall.lines.append(line)
            in_walls = True
            in_inner_walls = False
            continue

        # End of walls (other feature starts)
        if other_feature.search(line):
            if current_wall and current_wall.lines:
                current_layer.wall_loops.append(current_wall)
                current_wall = None
            in_walls = False
            in_inner_walls = False
            current_layer.post_walls.append(line)
            continue

        # Detect new inner wall loop via high-speed travel move
        if in_inner_walls and current_wall is not None:
            travel_match = travel_move.match(line.strip())
            if travel_match and int(travel_match.group(1)) > 10000:
                if current_wall.lines:
                    current_layer.wall_loops.append(current_wall)
                wall_count += 1
                current_wall = WallLoop(wall_type="inner", loop_index=wall_count,
                                        z_height=last_z)
                current_wall.lines.append(line)
                continue

        # Accumulate lines into the appropriate section
        if in_walls and current_wall is not None:
            current_wall.lines.append(line)
        elif current_layer.wall_loops or in_walls:
            if current_wall is not None:
                current_wall.lines.append(line)
            else:
                current_layer.post_walls.append(line)
        else:
            current_layer.pre_walls.append(line)

    # Save the final layer
    if current_layer is not None:
        if current_wall and current_wall.lines:
            current_layer.wall_loops.append(current_wall)
        layers.append(current_layer)

    if verbose:
        print(f"   Parsed {len(layers)} layers")
        for i, layer in enumerate(layers[:5]):
            print(f"     Layer {i}: Z={layer.z_height:.2f}, {len(layer.wall_loops)} wall loops")
            for w in layer.wall_loops:
                print(f"       Wall {w.loop_index} ({w.wall_type}): {len(w.lines)} lines")

    return header, layers, footer


def generate_output(header: List[str], layers: List[Layer], footer: List[str],
                    walls_to_offset: List[int], z_offset_fraction: float,
                    layer_height: float, min_layer: int,
                    verbose: bool) -> List[str]:
    """Generate modified G-code with interlocking wall Z offsets.

    For each layer (above min_layer), walls designated for interlocking are
    printed at Z + offset instead of the nominal Z. The offset creates a
    physical overlap between adjacent layers.

    Parameters
    ----------
    header : list of str
        G-code header lines.
    layers : list of Layer
        Parsed layer objects.
    footer : list of str
        G-code footer lines.
    walls_to_offset : list of int
        Which wall loop indices to offset.
    z_offset_fraction : float
        Offset as fraction of layer height (e.g., 0.5 = half layer).
    layer_height : float
        Layer height in mm.
    min_layer : int
        First layer to apply interlocking (skip initial layers for adhesion).
    verbose : bool
        Print processing statistics.

    Returns
    -------
    list of str
        Modified G-code lines.
    """
    output = header.copy()
    stats = {'layers': 0, 'walls_offset': 0}

    for layer in layers:
        output.extend(layer.pre_walls)

        if layer.layer_num < min_layer or not layer.wall_loops:
            for wall in layer.wall_loops:
                output.extend(wall.lines)
            output.extend(layer.post_walls)
            continue

        stats['layers'] += 1
        z_offset = layer_height * z_offset_fraction
        offset_z = layer.z_height + z_offset

        normal_walls = []
        offset_walls = []

        for wall in layer.wall_loops:
            if wall.loop_index in walls_to_offset:
                offset_walls.append(wall)
                stats['walls_offset'] += 1
            else:
                normal_walls.append(wall)

        # Print normal walls at standard Z
        for wall in normal_walls:
            output.append(f"; Wall {wall.loop_index} ({wall.wall_type}) at Z={layer.z_height:.3f}\n")
            output.extend(wall.lines)

        # Print offset walls at Z + offset
        if offset_walls:
            output.append(f"; === INTERLOCKING WALLS at Z={offset_z:.3f} ===\n")
            output.append(f"G1 Z{offset_z:.3f} F600 ; Interlocking Z\n")

            for wall in offset_walls:
                output.append(f"; Wall {wall.loop_index} ({wall.wall_type}) OFFSET +{z_offset:.3f}mm\n")
                for line in wall.lines:
                    # Skip standalone Z moves within the wall section
                    # (they would conflict with our offset Z)
                    if re.match(r'G1\s+Z[\d.]+\s+F\d+\s*$', line.strip()):
                        continue
                    output.append(line)

            # Return to nominal layer Z
            output.append(f"G1 Z{layer.z_height:.3f} F600 ; Interlocking Z\n")
            output.append("; === END INTERLOCKING ===\n")

        output.extend(layer.post_walls)

    output.extend(footer)

    if verbose:
        print(f"\n  Statistics:")
        print(f"   Layers with interlocking: {stats['layers']}")
        print(f"   Walls offset: {stats['walls_offset']}")

    return output


def main():
    """Main entry point for the interlocking perimeters post-processor."""
    args = parse_args()

    walls_to_offset = [int(w.strip()) for w in args.walls.split(',')]
    print(f"Walls to offset: {walls_to_offset}")
    print(f"Z offset fraction: {args.offset}")

    print(f"\nReading {args.input_3mf}...")
    try:
        gcode_content, gcode_path = extract_gcode_from_3mf(args.input_3mf)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"   Found G-code: {gcode_path}")
    print(f"\nParsing G-code...")
    lines = gcode_content.splitlines(keepends=True)
    header, layers, footer = parse_gcode_content(lines, args.verbose)
    print(f"   Found {len(layers)} layers")

    layer_height = args.layer_height or detect_layer_height(gcode_content)
    print(f"   Layer height: {layer_height}mm")
    print(f"   Z offset: {layer_height * args.offset:.3f}mm")

    if args.verbose:
        print("\n  Wall structure (layers 3-5):")
        for layer in layers[3:6]:
            if layer.wall_loops:
                print(f"  Layer {layer.layer_num} (Z={layer.z_height:.2f}):")
                for w in layer.wall_loops:
                    mark = " [OFFSET]" if w.loop_index in walls_to_offset else ""
                    print(f"    Wall {w.loop_index} ({w.wall_type}){mark}")

    print(f"\nGenerating interlocking G-code...")
    output_lines = generate_output(header, layers, footer, walls_to_offset,
                                   args.offset, layer_height, args.min_layer,
                                   args.verbose)
    new_gcode = ''.join(output_lines)

    print(f"\nCreating {args.output_3mf}...")
    repack_3mf(args.input_3mf, args.output_3mf, new_gcode, gcode_path)

    print(f"\nDone! Created: {args.output_3mf}")
    print(f"   Wall {walls_to_offset} offset by +{layer_height * args.offset:.3f}mm")


if __name__ == '__main__':
    main()
