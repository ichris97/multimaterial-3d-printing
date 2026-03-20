#!/usr/bin/env python3
"""
Interlocking Perimeters Post-Processor for Bambu Studio 3MF files

Creates interlocking wall structures by offsetting alternating wall loops
to different Z heights, creating mechanical interlocking between layers.

Concept (3 walls):
  Normal:       [Wall1][Wall2][Wall3] all at Z=0.2
  Interlocking: [Wall1][Wall3] at Z=0.2, [Wall2] at Z=0.25

Usage:
    python interlocking_perimeters.py input.3mf output.3mf
    python interlocking_perimeters.py input.3mf output.3mf --offset 0.5 --walls 2
"""

import argparse
import re
import zipfile
import hashlib
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class WallLoop:
    """Single wall loop with its G-code lines"""
    lines: List[str] = field(default_factory=list)
    wall_type: str = ""  # "inner" or "outer"
    loop_index: int = 0  # 1, 2, 3... within the layer
    z_height: float = 0.0


@dataclass
class Layer:
    """Complete layer with wall loops and other content"""
    layer_num: int = 0
    z_height: float = 0.0
    pre_walls: List[str] = field(default_factory=list)
    wall_loops: List[WallLoop] = field(default_factory=list)
    post_walls: List[str] = field(default_factory=list)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create interlocking perimeters in Bambu Studio 3MF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
How it works:
  With 3 walls, walls are numbered 1 (innermost), 2 (middle), 3 (outermost).
  By default, wall 2 is offset by half a layer height, creating interlock.

Examples:
  # Basic usage - offset middle wall by half layer height
  python interlocking_perimeters.py input.3mf output.3mf
  
  # Offset wall 2 by 0.3 of layer height  
  python interlocking_perimeters.py input.3mf output.3mf --offset 0.3
  
  # Offset walls 1 and 3 instead of wall 2
  python interlocking_perimeters.py input.3mf output.3mf --walls 1,3

Notes:
  - Input 3MF must be sliced (contain G-code)
  - Output 3MF can be sent directly to printer or opened in Bambu Studio
        """
    )
    parser.add_argument('input_3mf', help='Input 3MF file (must be sliced)')
    parser.add_argument('output_3mf', help='Output 3MF file')
    parser.add_argument('--offset', type=float, default=0.5,
                        help='Z offset as fraction of layer height (default: 0.5)')
    parser.add_argument('--walls', type=str, default='2',
                        help='Which wall numbers to offset, comma-separated (default: 2)')
    parser.add_argument('--layer-height', type=float, default=None,
                        help='Layer height in mm (auto-detected if not specified)')
    parser.add_argument('--min-layer', type=int, default=3,
                        help='Start interlocking from this layer (default: 3)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed processing info')
    return parser.parse_args()


def detect_layer_height(content: str) -> float:
    """Auto-detect layer height from G-code comments"""
    # Try Bambu Studio format first
    match = re.search(r';\s*layer_height\s*=\s*([\d.]+)', content)
    if match:
        return float(match.group(1))
    
    # Try LAYER_HEIGHT comment
    match = re.search(r';\s*LAYER_HEIGHT:\s*([\d.]+)', content)
    if match:
        return float(match.group(1))
    
    return 0.2  # Default



def add_z_move(z: float, feedrate: int = 600) -> str:
    """Generate G-code for Z move"""
    return f"G1 Z{z:.3f} F{feedrate} ; Interlocking Z\n"


def generate_output(header: List[str], layers: List[Layer], footer: List[str],
                    walls_to_offset: List[int], z_offset_fraction: float,
                    layer_height: float, min_layer: int, verbose: bool) -> List[str]:
    """Generate modified G-code with interlocking walls"""
    
    output = header.copy()
    
    stats = {'layers': 0, 'walls_offset': 0}
    
    for layer in layers:
        # Add pre-wall content
        output.extend(layer.pre_walls)
        
        if layer.layer_num < min_layer or not layer.wall_loops:
            # No interlocking for early layers or layers without walls
            for wall in layer.wall_loops:
                output.extend(wall.lines)
            output.extend(layer.post_walls)
            continue
        
        stats['layers'] += 1
        z_offset = layer_height * z_offset_fraction
        offset_z = layer.z_height + z_offset
        
        # Separate walls into normal and offset groups
        normal_walls = []
        offset_walls = []
        
        for wall in layer.wall_loops:
            if wall.loop_index in walls_to_offset:
                offset_walls.append(wall)
                stats['walls_offset'] += 1
            else:
                normal_walls.append(wall)
        
        # Print normal walls first at layer Z
        for wall in normal_walls:
            output.append(f"; Wall {wall.loop_index} ({wall.wall_type}) at Z={layer.z_height:.3f}\n")
            output.extend(wall.lines)
        
        # Print offset walls at Z + offset
        if offset_walls:
            output.append(f"; === INTERLOCKING WALLS at Z={offset_z:.3f} ===\n")
            output.append(add_z_move(offset_z))
            
            for wall in offset_walls:
                output.append(f"; Wall {wall.loop_index} ({wall.wall_type}) OFFSET +{z_offset:.3f}mm\n")
                for line in wall.lines:
                    # Skip pure Z moves within wall section
                    if re.match(r'G1\s+Z[\d.]+\s+F\d+\s*$', line.strip()):
                        continue
                    output.append(line)
            
            # Return to layer Z
            output.append(add_z_move(layer.z_height))
            output.append("; === END INTERLOCKING ===\n")
        
        output.extend(layer.post_walls)
    
    output.extend(footer)
    
    if verbose:
        print(f"\n📊 Statistics:")
        print(f"   Layers with interlocking: {stats['layers']}")
        print(f"   Walls offset: {stats['walls_offset']}")
    
    return output


def extract_gcode_from_3mf(input_3mf: str) -> Tuple[str, str]:
    """Extract G-code content from a 3MF file.
    
    Returns:
        Tuple of (gcode_content, gcode_filename)
    """
    with zipfile.ZipFile(input_3mf, 'r') as zf:
        # Find the G-code file
        gcode_files = [f for f in zf.namelist() if f.endswith('.gcode')]
        
        if not gcode_files:
            raise ValueError(f"No G-code found in {input_3mf}. Make sure to slice the model first!")
        
        # Usually it's Metadata/plate_1.gcode
        gcode_path = gcode_files[0]
        gcode_content = zf.read(gcode_path).decode('utf-8')
        
        return gcode_content, gcode_path


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
    
    # Parse walls to offset
    walls_to_offset = [int(w.strip()) for w in args.walls.split(',')]
    print(f"Walls to offset: {walls_to_offset}")
    print(f"Z offset fraction: {args.offset}")
    
    # Extract G-code from 3MF
    print(f"\n📦 Reading {args.input_3mf}...")
    try:
        gcode_content, gcode_path = extract_gcode_from_3mf(args.input_3mf)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    
    print(f"   Found G-code: {gcode_path}")
    
    # Parse G-code
    print(f"\n🔍 Parsing G-code...")
    lines = gcode_content.splitlines(keepends=True)
    
    # Create a temporary structure for parsing
    header, layers, footer = parse_gcode_content(lines, args.verbose)
    print(f"   Found {len(layers)} layers")
    
    # Detect layer height
    if args.layer_height:
        layer_height = args.layer_height
    else:
        layer_height = detect_layer_height(gcode_content)
    
    print(f"   Layer height: {layer_height}mm")
    print(f"   Z offset: {layer_height * args.offset:.3f}mm")
    
    # Show wall structure for verification
    if args.verbose:
        print("\n📋 Wall structure (layers 3-5):")
        for layer in layers[3:6]:
            if layer.wall_loops:
                print(f"  Layer {layer.layer_num} (Z={layer.z_height:.2f}):")
                for w in layer.wall_loops:
                    mark = " [OFFSET]" if w.loop_index in walls_to_offset else ""
                    print(f"    Wall {w.loop_index} ({w.wall_type}){mark}")
    
    # Generate modified G-code
    print(f"\n⚙️  Generating interlocking G-code...")
    output_lines = generate_output(header, layers, footer, walls_to_offset,
                                   args.offset, layer_height, args.min_layer, args.verbose)
    
    new_gcode = ''.join(output_lines)
    
    # Repack 3MF
    print(f"\n📦 Creating {args.output_3mf}...")
    repack_3mf(args.input_3mf, args.output_3mf, new_gcode, gcode_path)
    
    print(f"\n✅ Done! Created: {args.output_3mf}")
    print(f"\n📋 Summary:")
    print(f"   - Wall {walls_to_offset} offset by +{layer_height * args.offset:.3f}mm")
    print(f"   - Creates mechanical interlocking between layers")
    print(f"\n💡 You can send this 3MF directly to your printer or open in Bambu Studio")


def parse_gcode_content(lines: List[str], verbose: bool = False) -> Tuple[List[str], List[Layer], List[str]]:
    """Parse G-code lines into layers with separated wall loops"""
    
    # Patterns
    layer_change = re.compile(r';\s*CHANGE_LAYER|;\s*layer\s*#\d+', re.IGNORECASE)
    inner_wall = re.compile(r';\s*FEATURE:\s*Inner wall', re.IGNORECASE)
    outer_wall = re.compile(r';\s*FEATURE:\s*Outer wall', re.IGNORECASE)
    other_feature = re.compile(r';\s*FEATURE:\s*(?!Inner wall|Outer wall)', re.IGNORECASE)
    z_height_comment = re.compile(r';\s*Z_HEIGHT:\s*([\d.]+)')
    z_move = re.compile(r'G1\s+.*Z([\d.]+)')
    # Travel move pattern: G1 with X/Y but no E, high feedrate
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
    
    for i, line in enumerate(lines):
        # Detect layer change
        if layer_change.search(line):
            in_header = False
            
            # Save previous layer
            if current_layer is not None:
                if current_wall and current_wall.lines:
                    current_layer.wall_loops.append(current_wall)
                layers.append(current_layer)
            
            # Start new layer
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
        
        # Track Z height
        z_match = z_height_comment.search(line)
        if z_match:
            current_layer.z_height = float(z_match.group(1))
            last_z = current_layer.z_height
        
        z_move_match = z_move.search(line)
        if z_move_match and not in_walls:
            last_z = float(z_move_match.group(1))
            if current_layer.z_height == 0:
                current_layer.z_height = last_z
        
        # Detect inner wall start
        if inner_wall.search(line):
            # Save previous wall if exists
            if current_wall and current_wall.lines:
                current_layer.wall_loops.append(current_wall)
            
            wall_count += 1
            current_wall = WallLoop()
            current_wall.wall_type = "inner"
            current_wall.loop_index = wall_count
            current_wall.z_height = last_z
            current_wall.lines.append(line)
            in_walls = True
            in_inner_walls = True
            continue
        
        # Detect outer wall start
        if outer_wall.search(line):
            # Save previous wall if exists
            if current_wall and current_wall.lines:
                current_layer.wall_loops.append(current_wall)
            
            wall_count += 1
            current_wall = WallLoop()
            current_wall.wall_type = "outer"
            current_wall.loop_index = wall_count
            current_wall.z_height = last_z
            current_wall.lines.append(line)
            in_walls = True
            in_inner_walls = False
            continue
        
        # Detect end of walls (other feature starts)
        if other_feature.search(line):
            if current_wall and current_wall.lines:
                current_layer.wall_loops.append(current_wall)
                current_wall = None
            in_walls = False
            in_inner_walls = False
            current_layer.post_walls.append(line)
            continue
        
        # Within inner walls, detect loop transitions via travel moves
        if in_inner_walls and current_wall is not None:
            travel_match = travel_move.match(line.strip())
            if travel_match:
                feedrate = int(travel_match.group(1))
                # High feedrate travel = new loop starting
                if feedrate > 10000:
                    # Save current loop and start new one
                    if current_wall.lines:
                        current_layer.wall_loops.append(current_wall)
                    wall_count += 1
                    current_wall = WallLoop()
                    current_wall.wall_type = "inner"
                    current_wall.loop_index = wall_count
                    current_wall.z_height = last_z
                    current_wall.lines.append(line)
                    continue
        
        # Accumulate lines
        if in_walls and current_wall is not None:
            current_wall.lines.append(line)
        elif current_layer.wall_loops or in_walls:
            if current_wall is not None:
                current_wall.lines.append(line)
            else:
                current_layer.post_walls.append(line)
        else:
            current_layer.pre_walls.append(line)
    
    # Save final layer
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


if __name__ == '__main__':
    main()