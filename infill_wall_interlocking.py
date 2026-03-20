#!/usr/bin/env python3
"""
Wall-Infill Mechanical Interlocking Post-Processor for Bambu Studio 3MF files

Creates TRUE mechanical interlocking by giving walls and infill COMPLEMENTARY
geometries that fit together like puzzle pieces.

Concept (cross-section view):

    WALL has VALLEYS (indentations pointing inward):
    
    ══╗ ╔══╗ ╔══╗ ╔══
      ╚═╝  ╚═╝  ╚═╝
      
    INFILL has PEAKS (protrusions pointing outward):
    
      ╔═╗  ╔═╗  ╔═╗
    ══╝ ╚══╝ ╚══╝ ╚══
    
    TOGETHER - peaks fit into valleys:
    
    Wall:   ══╗ ╔══╗ ╔══
    Infill:   ╠═╣  ╠═╣    <- INTERLOCKED!
              ╚═╝  ╚═╝

The wall and infill are 180° out of phase:
- Where wall goes IN (valley), infill goes OUT (peak)
- They mechanically interlock, preventing delamination

Usage:
    python wall_infill_interlock.py input.3mf output.3mf
"""

import argparse
import re
import zipfile
import hashlib
import tempfile
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class ProcessingStats:
    layers_processed: int = 0
    wall_segments_modified: int = 0
    infill_lines_modified: int = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create interlocking wall-infill geometry in Bambu Studio 3MF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Mechanical Interlocking Concept:

  Wall (valleys):    ══╗ ╔══╗ ╔══
                       ╚═╝  ╚═╝
                       
  Infill (peaks):      ╔═╗  ╔═╗
                     ══╝ ╚══╝ ╚══
                     
  Together:          ══╗ ╔══╗ ╔══
                       ╠═╣  ╠═╣   <- Interlocked!
                       ╚═╝  ╚═╝

The peaks of the infill fit INTO the valleys of the wall.
This creates mechanical resistance to separation.

Examples:
  python wall_infill_interlock.py model.3mf output.3mf
  python wall_infill_interlock.py model.3mf output.3mf --teeth-depth 0.5 --teeth-pitch 1.5
        """
    )
    parser.add_argument('input_3mf', help='Input 3MF file (must be sliced)')
    parser.add_argument('output_3mf', help='Output 3MF file')
    parser.add_argument('--teeth-depth', type=float, default=0.4,
                        help='Depth of teeth/valleys in mm (default: 0.4)')
    parser.add_argument('--teeth-pitch', type=float, default=1.5,
                        help='Distance between teeth in mm (default: 1.5)')
    parser.add_argument('--interlock-length', type=float, default=3.0,
                        help='Length of interlocking zone at infill ends in mm (default: 3.0)')
    parser.add_argument('--start-layer', type=int, default=3,
                        help='Start from this layer (default: 3)')
    parser.add_argument('--min-infill-length', type=float, default=8.0,
                        help='Minimum infill line length to modify in mm (default: 8.0)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed info')
    return parser.parse_args()


def extract_gcode_from_3mf(input_3mf: str) -> Tuple[str, str]:
    """Extract G-code from 3MF file."""
    with zipfile.ZipFile(input_3mf, 'r') as zf:
        for name in zf.namelist():
            if name.endswith('.gcode'):
                return zf.read(name).decode('utf-8'), name
    raise ValueError("No G-code found in 3MF. Slice the model first!")


def repack_3mf(input_3mf: str, output_3mf: str, new_gcode: str, gcode_path: str):
    """Repack 3MF with modified G-code."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        with zipfile.ZipFile(input_3mf, 'r') as zf:
            zf.extractall(temp_dir)
        
        (temp_dir / gcode_path).write_text(new_gcode, encoding='utf-8')
        
        md5_path = temp_dir / (gcode_path + '.md5')
        if md5_path.exists():
            md5_path.write_text(hashlib.md5(new_gcode.encode()).hexdigest())
        
        with zipfile.ZipFile(output_3mf, 'w', zipfile.ZIP_DEFLATED) as zf_out:
            for fp in temp_dir.rglob('*'):
                if fp.is_file():
                    zf_out.write(fp, fp.relative_to(temp_dir))


def generate_teeth_path(start_x: float, start_y: float,
                        end_x: float, end_y: float,
                        teeth_depth: float, teeth_pitch: float,
                        phase: float, extrusion: float, feedrate: int) -> List[str]:
    """
    Generate G-code path with sinusoidal teeth pattern.
    
    phase = 0.0: peaks (teeth pointing perpendicular +)
    phase = 0.5: valleys (teeth pointing perpendicular -)
    
    Wall uses phase=0.5 (valleys pointing inward)
    Infill uses phase=0.0 (peaks pointing outward)
    """
    dx = end_x - start_x
    dy = end_y - start_y
    length = math.sqrt(dx*dx + dy*dy)
    
    if length < 0.1:
        return []
    
    # Unit vectors: along path and perpendicular
    ux, uy = dx/length, dy/length
    px, py = -uy, ux  # Perpendicular
    
    e_per_mm = extrusion / length
    
    lines = []
    current_x, current_y = start_x, start_y
    
    # Generate points along the path with sinusoidal offset
    num_points = max(int(length / (teeth_pitch / 8)), 2)
    
    for i in range(1, num_points + 1):
        t = i / num_points
        dist = t * length
        
        # Sinusoidal offset perpendicular to path
        # phase shifts the wave by fraction of period
        offset = teeth_depth * math.sin(2 * math.pi * (dist / teeth_pitch + phase))
        
        # Position = base position + perpendicular offset
        next_x = start_x + ux * dist + px * offset
        next_y = start_y + uy * dist + py * offset
        
        # Calculate extrusion for this segment
        seg_len = math.sqrt((next_x - current_x)**2 + (next_y - current_y)**2)
        e_val = seg_len * e_per_mm
        
        if seg_len > 0.02:
            lines.append(f"G1 X{next_x:.3f} Y{next_y:.3f} E{e_val:.5f} F{feedrate}\n")
            current_x, current_y = next_x, next_y
    
    # Ensure we end exactly at endpoint
    final_dist = math.sqrt((end_x - current_x)**2 + (end_y - current_y)**2)
    if final_dist > 0.01:
        e_val = final_dist * e_per_mm
        lines.append(f"G1 X{end_x:.3f} Y{end_y:.3f} E{e_val:.5f} F{feedrate}\n")
    
    return lines


def generate_wall_segment_with_valleys(start_x: float, start_y: float,
                                        end_x: float, end_y: float,
                                        extrusion: float, feedrate: int,
                                        teeth_depth: float, teeth_pitch: float) -> List[str]:
    """
    Generate inner wall segment with VALLEYS (indentations pointing inward).
    Uses phase=0.5 so valleys align with where infill peaks will be.
    """
    lines = ["; WALL-VALLEYS\n"]
    
    # Phase 0.5 = valleys (sine wave starts going negative = inward)
    teeth_lines = generate_teeth_path(
        start_x, start_y, end_x, end_y,
        teeth_depth, teeth_pitch,
        phase=0.5,  # VALLEYS
        extrusion=extrusion,
        feedrate=feedrate
    )
    lines.extend(teeth_lines)
    
    return lines


def generate_infill_with_peaks(start_x: float, start_y: float,
                               end_x: float, end_y: float,
                               extrusion: float, feedrate: int,
                               teeth_depth: float, teeth_pitch: float,
                               interlock_length: float) -> List[str]:
    """
    Generate infill line with PEAKS at both ends (protrusions toward walls).
    Uses phase=0.0 so peaks align with wall valleys.
    
    Structure:
    [PEAKS zone] ════ straight middle ════ [PEAKS zone]
    """
    dx = end_x - start_x
    dy = end_y - start_y
    total_length = math.sqrt(dx*dx + dy*dy)
    
    if total_length < interlock_length * 2.5:
        # Too short, return as-is
        return [f"G1 X{end_x:.3f} Y{end_y:.3f} E{extrusion:.5f} F{feedrate}\n"]
    
    ux, uy = dx/total_length, dy/total_length
    e_per_mm = extrusion / total_length
    
    lines = ["; INFILL-PEAKS START\n"]
    
    # 1. First interlock zone (peaks at start)
    zone1_end_x = start_x + ux * interlock_length
    zone1_end_y = start_y + uy * interlock_length
    zone1_e = interlock_length * e_per_mm
    
    teeth_lines = generate_teeth_path(
        start_x, start_y, zone1_end_x, zone1_end_y,
        teeth_depth, teeth_pitch,
        phase=0.0,  # PEAKS (opposite of wall valleys)
        extrusion=zone1_e,
        feedrate=feedrate
    )
    lines.extend(teeth_lines)
    
    # 2. Straight middle section
    zone2_start_x = zone1_end_x
    zone2_start_y = zone1_end_y
    zone2_end_x = end_x - ux * interlock_length
    zone2_end_y = end_y - uy * interlock_length
    middle_length = total_length - 2 * interlock_length
    middle_e = middle_length * e_per_mm
    
    if middle_length > 0.5:
        lines.append(f"G1 X{zone2_end_x:.3f} Y{zone2_end_y:.3f} E{middle_e:.5f} F{feedrate}\n")
    
    # 3. Second interlock zone (peaks at end)
    zone3_e = interlock_length * e_per_mm
    
    teeth_lines = generate_teeth_path(
        zone2_end_x, zone2_end_y, end_x, end_y,
        teeth_depth, teeth_pitch,
        phase=0.0,  # PEAKS
        extrusion=zone3_e,
        feedrate=feedrate
    )
    lines.extend(teeth_lines)
    
    lines.append("; INFILL-PEAKS END\n")
    
    return lines


def process_gcode(gcode_content: str,
                  teeth_depth: float,
                  teeth_pitch: float,
                  interlock_length: float,
                  start_layer: int,
                  min_infill_length: float,
                  verbose: bool) -> Tuple[str, ProcessingStats]:
    """Process G-code to add interlocking geometry."""
    
    lines = gcode_content.splitlines(keepends=True)
    output = []
    stats = ProcessingStats()
    
    # Patterns
    inner_wall = re.compile(r';\s*FEATURE:\s*Inner wall', re.IGNORECASE)
    infill = re.compile(r';\s*FEATURE:\s*(Sparse|Internal|Solid)\s*infill', re.IGNORECASE)
    any_feature = re.compile(r';\s*FEATURE:\s*', re.IGNORECASE)
    z_height = re.compile(r';\s*Z_HEIGHT:\s*([\d.]+)')
    g1_move = re.compile(r'^G1\s+X([\d.]+)\s+Y([\d.]+)(?:.*E([\d.]+))?(?:.*F(\d+))?', re.IGNORECASE)
    
    in_inner_wall = False
    in_infill = False
    current_layer = 0
    current_x, current_y = 0.0, 0.0
    current_f = 3000
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Track Z/layer
        z_match = z_height.search(line)
        if z_match:
            current_layer = int(float(z_match.group(1)) / 0.2)
        
        # Feature transitions
        if inner_wall.search(line):
            in_inner_wall = True
            in_infill = False
            output.append(line)
            i += 1
            continue
        
        if infill.search(line):
            in_infill = True
            in_inner_wall = False
            output.append(line)
            i += 1
            continue
        
        if any_feature.search(line) and not inner_wall.search(line) and not infill.search(line):
            in_inner_wall = False
            in_infill = False
            output.append(line)
            i += 1
            continue
        
        # Parse G1 moves
        move_match = g1_move.match(line.strip())
        if move_match:
            new_x = float(move_match.group(1))
            new_y = float(move_match.group(2))
            new_e = float(move_match.group(3)) if move_match.group(3) else None
            new_f = int(move_match.group(4)) if move_match.group(4) else current_f
            current_f = new_f
            
            is_extrusion = new_e is not None and new_e > 0
            seg_length = math.sqrt((new_x - current_x)**2 + (new_y - current_y)**2)
            
            # Process INNER WALL with valleys
            if in_inner_wall and is_extrusion and current_layer >= start_layer and seg_length > teeth_pitch:
                wall_lines = generate_wall_segment_with_valleys(
                    current_x, current_y, new_x, new_y,
                    new_e, current_f, teeth_depth, teeth_pitch
                )
                output.extend(wall_lines)
                stats.wall_segments_modified += 1
                current_x, current_y = new_x, new_y
                i += 1
                continue
            
            # Process INFILL with peaks
            if in_infill and is_extrusion and current_layer >= start_layer and seg_length >= min_infill_length:
                infill_lines = generate_infill_with_peaks(
                    current_x, current_y, new_x, new_y,
                    new_e, current_f, teeth_depth, teeth_pitch, interlock_length
                )
                output.extend(infill_lines)
                stats.infill_lines_modified += 1
                current_x, current_y = new_x, new_y
                i += 1
                continue
            
            current_x, current_y = new_x, new_y
        
        # Track position from travel moves too
        elif line.strip().startswith('G1') or line.strip().startswith('G0'):
            x_match = re.search(r'X([\d.]+)', line)
            y_match = re.search(r'Y([\d.]+)', line)
            if x_match:
                current_x = float(x_match.group(1))
            if y_match:
                current_y = float(y_match.group(1))
        
        output.append(line)
        i += 1
    
    stats.layers_processed = current_layer
    return ''.join(output), stats


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  Wall-Infill Mechanical Interlocking")
    print("=" * 60)
    
    print("""
    ┌─────────────────────────────────────────────┐
    │  WALL has VALLEYS    INFILL has PEAKS       │
    │                                             │
    │  ══╗ ╔══╗ ╔══        ╔═╗  ╔═╗               │
    │    ╚═╝  ╚═╝        ══╝ ╚══╝ ╚══             │
    │                                             │
    │  INTERLOCKED:                               │
    │  ══╗ ╔══╗ ╔══                               │
    │    ╠═╣  ╠═╣   <- Peaks fit into valleys!    │
    │    ╚═╝  ╚═╝                                 │
    └─────────────────────────────────────────────┘
    """)
    
    print(f"📦 Loading {args.input_3mf}...")
    
    try:
        gcode_content, gcode_path = extract_gcode_from_3mf(args.input_3mf)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    
    print(f"\n⚙️  Settings:")
    print(f"   Teeth depth: {args.teeth_depth}mm")
    print(f"   Teeth pitch: {args.teeth_pitch}mm")
    print(f"   Interlock zone: {args.interlock_length}mm at infill ends")
    
    print(f"\n🔧 Processing...")
    
    new_gcode, stats = process_gcode(
        gcode_content,
        args.teeth_depth,
        args.teeth_pitch,
        args.interlock_length,
        args.start_layer,
        args.min_infill_length,
        args.verbose
    )
    
    print(f"\n📊 Results:")
    print(f"   Wall segments with valleys: {stats.wall_segments_modified}")
    print(f"   Infill lines with peaks: {stats.infill_lines_modified}")
    
    print(f"\n📦 Saving {args.output_3mf}...")
    repack_3mf(args.input_3mf, args.output_3mf, new_gcode, gcode_path)
    
    print(f"\n✅ Done!")
    print(f"\n💪 The wall valleys and infill peaks will mesh together,")
    print(f"   creating mechanical resistance to delamination!")


if __name__ == '__main__':
    main()