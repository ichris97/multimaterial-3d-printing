#!/usr/bin/env python3
"""
Wall-infill mechanical interlocking post-processor.

Creates TRUE mechanical interlocking between wall perimeters and internal
infill by giving them complementary sinusoidal geometries that mesh together
like gear teeth. This dramatically improves the bond between walls and infill,
which is one of the weakest interfaces in FDM prints.

The Problem
-----------
In standard FDM printing, walls (perimeters) and infill meet at a flat
interface. The bond between them relies entirely on thermal adhesion — the
molten infill line bonds to the already-cooled wall. Under load, this flat
interface is prone to delamination because there's no mechanical resistance
to separation, only the material bond strength.

The Solution: Complementary Sinusoidal Interlocking
---------------------------------------------------
This post-processor modifies the G-code so that:

1. **Inner walls** are printed with a sinusoidal wave pattern (VALLEYS)
   along their length. The wave goes inward (toward the infill side),
   creating indentations in the wall.

2. **Infill lines** are printed with the opposite sinusoidal pattern (PEAKS)
   at their endpoints where they meet the walls. The peaks protrude outward
   toward the wall.

3. The wall valleys and infill peaks are 180° out of phase, so the peaks
   fit INTO the valleys, creating a mechanical interlock.

Cross-section view:

    Wall (valleys):    ══╗ ╔══╗ ╔══
                         ╚═╝  ╚═╝

    Infill (peaks):      ╔═╗  ╔═╗
                       ══╝ ╚══╝ ╚══

    Together:          ══╗ ╔══╗ ╔══
                         ╠═╣  ╠═╣     ← Mechanically interlocked!
                         ╚═╝  ╚═╝

Sinusoidal Wave Parameters
--------------------------
- **teeth_depth** (default: 0.4mm): Amplitude of the sine wave. Larger values
  create stronger interlocking but may affect surface quality. Should be
  comparable to the nozzle diameter (typically 0.4mm).

- **teeth_pitch** (default: 1.5mm): Wavelength of the sine wave. Shorter
  pitch = more teeth per mm = stronger interlock but more complex toolpath.
  Should be at least 3x the nozzle diameter for reliable printing.

- **interlock_length** (default: 3.0mm): Length of the interlocking zone at
  each end of infill lines. Only the ends of infill lines (where they meet
  walls) get the peak pattern; the middle section remains straight for speed.

- **phase**: Walls use phase=0.5 (sine starts going negative = inward),
  infill uses phase=0.0 (sine starts going positive = outward). This
  ensures the two patterns are complementary.
"""

import argparse
import math
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..core.file_io import extract_gcode_from_3mf, repack_3mf


@dataclass
class ProcessingStats:
    """Tracks how many G-code elements were modified during processing."""
    layers_processed: int = 0
    wall_segments_modified: int = 0
    infill_lines_modified: int = 0


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Create interlocking wall-infill geometry in 3MF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Wall-Infill Interlocking Concept:

  Wall (valleys):    ══╗ ╔══╗ ╔══
                       ╚═╝  ╚═╝

  Infill (peaks):      ╔═╗  ╔═╗
                     ══╝ ╚══╝ ╚══

  Interlocked:       ══╗ ╔══╗ ╔══
                       ╠═╣  ╠═╣   <- Peaks fit into valleys!
                       ╚═╝  ╚═╝

Examples:
  python -m multimaterial_3d.postprocessors.wall_infill_interlock model.3mf output.3mf
  python -m multimaterial_3d.postprocessors.wall_infill_interlock model.3mf output.3mf --teeth-depth 0.5 --teeth-pitch 1.5
        """
    )
    parser.add_argument('input_3mf', help='Input 3MF file (must be sliced)')
    parser.add_argument('output_3mf', help='Output 3MF file')
    parser.add_argument('--teeth-depth', type=float, default=0.4,
                        help='Depth of teeth/valleys in mm (default: 0.4)')
    parser.add_argument('--teeth-pitch', type=float, default=1.5,
                        help='Distance between teeth in mm (default: 1.5)')
    parser.add_argument('--interlock-length', type=float, default=3.0,
                        help='Interlocking zone at infill ends in mm (default: 3.0)')
    parser.add_argument('--start-layer', type=int, default=3,
                        help='Start from this layer (default: 3)')
    parser.add_argument('--min-infill-length', type=float, default=8.0,
                        help='Min infill line length to modify in mm (default: 8.0)')
    parser.add_argument('--verbose', '-v', action='store_true')
    return parser.parse_args()


def generate_teeth_path(start_x: float, start_y: float,
                        end_x: float, end_y: float,
                        teeth_depth: float, teeth_pitch: float,
                        phase: float, extrusion: float,
                        feedrate: int) -> List[str]:
    """Generate G-code for a sinusoidal teeth path between two points.

    The path follows a sine wave perpendicular to the line connecting
    start and end points. The phase parameter controls whether the
    sine wave starts going up (peaks, phase=0) or down (valleys, phase=0.5).

    The sine wave equation for displacement perpendicular to the path:
        d(s) = A * sin(2π * (s/λ + phase))

    where:
        s = distance along the path
        A = teeth_depth (amplitude)
        λ = teeth_pitch (wavelength)
        phase = 0.0 for peaks, 0.5 for valleys

    Parameters
    ----------
    start_x, start_y : float
        Starting point coordinates.
    end_x, end_y : float
        Ending point coordinates.
    teeth_depth : float
        Sine wave amplitude (mm).
    teeth_pitch : float
        Sine wave wavelength (mm).
    phase : float
        Phase shift (0.0 = peaks toward +perpendicular, 0.5 = valleys).
    extrusion : float
        Total extrusion amount (E value) for the entire segment.
    feedrate : int
        Print speed in mm/min.

    Returns
    -------
    list of str
        G-code lines for the sinusoidal path.
    """
    dx = end_x - start_x
    dy = end_y - start_y
    length = math.sqrt(dx**2 + dy**2)

    if length < 0.1:
        return []

    # Unit vectors: along path and perpendicular
    ux, uy = dx / length, dy / length
    px, py = -uy, ux  # 90° rotation for perpendicular direction

    e_per_mm = extrusion / length
    lines = []
    current_x, current_y = start_x, start_y

    # Sample the sine wave with enough points for smooth curves
    # At least 8 points per wavelength for visual smoothness
    num_points = max(int(length / (teeth_pitch / 8)), 2)

    for i in range(1, num_points + 1):
        t = i / num_points
        dist = t * length

        # Sinusoidal perpendicular offset
        offset = teeth_depth * math.sin(2 * math.pi * (dist / teeth_pitch + phase))

        # New position = base position + perpendicular offset
        next_x = start_x + ux * dist + px * offset
        next_y = start_y + uy * dist + py * offset

        # Extrusion proportional to actual path length (not straight-line)
        seg_len = math.sqrt((next_x - current_x)**2 + (next_y - current_y)**2)
        e_val = seg_len * e_per_mm

        if seg_len > 0.02:
            lines.append(f"G1 X{next_x:.3f} Y{next_y:.3f} E{e_val:.5f} F{feedrate}\n")
            current_x, current_y = next_x, next_y

    # Ensure exact endpoint
    final_dist = math.sqrt((end_x - current_x)**2 + (end_y - current_y)**2)
    if final_dist > 0.01:
        e_val = final_dist * e_per_mm
        lines.append(f"G1 X{end_x:.3f} Y{end_y:.3f} E{e_val:.5f} F{feedrate}\n")

    return lines


def generate_wall_valleys(start_x: float, start_y: float,
                          end_x: float, end_y: float,
                          extrusion: float, feedrate: int,
                          teeth_depth: float, teeth_pitch: float) -> List[str]:
    """Generate an inner wall segment with VALLEY pattern.

    Phase = 0.5 means the sine wave starts going negative (inward),
    creating valleys that will receive infill peaks.
    """
    lines = ["; WALL-VALLEYS\n"]
    lines.extend(generate_teeth_path(
        start_x, start_y, end_x, end_y,
        teeth_depth, teeth_pitch,
        phase=0.5, extrusion=extrusion, feedrate=feedrate
    ))
    return lines


def generate_infill_peaks(start_x: float, start_y: float,
                          end_x: float, end_y: float,
                          extrusion: float, feedrate: int,
                          teeth_depth: float, teeth_pitch: float,
                          interlock_length: float) -> List[str]:
    """Generate an infill line with PEAK pattern at both ends.

    The infill line structure is:
        [PEAKS zone] ═══ straight middle ═══ [PEAKS zone]

    Only the ends (where infill meets walls) get the sinusoidal pattern.
    The middle section remains straight for faster printing.

    Phase = 0.0 means peaks point outward (toward walls), complementing
    the wall valleys at phase = 0.5.
    """
    dx = end_x - start_x
    dy = end_y - start_y
    total_length = math.sqrt(dx**2 + dy**2)

    if total_length < interlock_length * 2.5:
        return [f"G1 X{end_x:.3f} Y{end_y:.3f} E{extrusion:.5f} F{feedrate}\n"]

    ux, uy = dx / total_length, dy / total_length
    e_per_mm = extrusion / total_length
    lines = ["; INFILL-PEAKS START\n"]

    # Zone 1: peaks at start
    z1_end_x = start_x + ux * interlock_length
    z1_end_y = start_y + uy * interlock_length
    lines.extend(generate_teeth_path(
        start_x, start_y, z1_end_x, z1_end_y,
        teeth_depth, teeth_pitch,
        phase=0.0, extrusion=interlock_length * e_per_mm, feedrate=feedrate
    ))

    # Zone 2: straight middle
    z2_end_x = end_x - ux * interlock_length
    z2_end_y = end_y - uy * interlock_length
    middle_length = total_length - 2 * interlock_length
    if middle_length > 0.5:
        lines.append(f"G1 X{z2_end_x:.3f} Y{z2_end_y:.3f} E{middle_length * e_per_mm:.5f} F{feedrate}\n")

    # Zone 3: peaks at end
    lines.extend(generate_teeth_path(
        z2_end_x, z2_end_y, end_x, end_y,
        teeth_depth, teeth_pitch,
        phase=0.0, extrusion=interlock_length * e_per_mm, feedrate=feedrate
    ))

    lines.append("; INFILL-PEAKS END\n")
    return lines


def process_gcode(gcode_content: str, teeth_depth: float, teeth_pitch: float,
                  interlock_length: float, start_layer: int,
                  min_infill_length: float,
                  verbose: bool) -> Tuple[str, ProcessingStats]:
    """Process entire G-code to add interlocking wall-infill geometry.

    Scans through the G-code, identifies inner wall and infill sections,
    and replaces qualifying extrusion moves with sinusoidal teeth patterns.

    Wall segments get valleys (phase=0.5), infill segments get peaks
    (phase=0.0) at their endpoints.
    """
    lines = gcode_content.splitlines(keepends=True)
    output = []
    stats = ProcessingStats()

    inner_wall = re.compile(r';\s*FEATURE:\s*Inner wall', re.IGNORECASE)
    infill = re.compile(r';\s*FEATURE:\s*(Sparse|Internal|Solid)\s*infill', re.IGNORECASE)
    any_feature = re.compile(r';\s*FEATURE:\s*', re.IGNORECASE)
    z_height = re.compile(r';\s*Z_HEIGHT:\s*([\d.]+)')
    g1_move = re.compile(r'^G1\s+X([\d.]+)\s+Y([\d.]+)', re.IGNORECASE)
    e_pattern = re.compile(r'E(-?[\d.]+)', re.IGNORECASE)
    f_pattern = re.compile(r'F(\d+)', re.IGNORECASE)

    layer_change = re.compile(r';\s*CHANGE_LAYER', re.IGNORECASE)

    in_inner_wall = False
    in_infill = False
    current_layer = 0
    current_x, current_y = 0.0, 0.0
    current_f = 3000

    for line in lines:
        if layer_change.search(line):
            current_layer += 1

        # Feature transitions
        if inner_wall.search(line):
            in_inner_wall, in_infill = True, False
            output.append(line)
            continue
        if infill.search(line):
            in_infill, in_inner_wall = True, False
            output.append(line)
            continue
        if any_feature.search(line) and not inner_wall.search(line) and not infill.search(line):
            in_inner_wall, in_infill = False, False
            output.append(line)
            continue

        # Parse G1 extrusion moves
        move_match = g1_move.match(line.strip())
        if move_match:
            new_x = float(move_match.group(1))
            new_y = float(move_match.group(2))
            e_m = e_pattern.search(line)
            f_m = f_pattern.search(line)
            new_e = float(e_m.group(1)) if e_m else None
            new_f = int(f_m.group(1)) if f_m else current_f
            current_f = new_f

            is_extrusion = new_e is not None and new_e > 0
            seg_length = math.sqrt((new_x - current_x)**2 + (new_y - current_y)**2)

            # Apply wall valleys
            if (in_inner_wall and is_extrusion and
                    current_layer >= start_layer and seg_length > teeth_pitch):
                output.extend(generate_wall_valleys(
                    current_x, current_y, new_x, new_y,
                    new_e, current_f, teeth_depth, teeth_pitch
                ))
                stats.wall_segments_modified += 1
                current_x, current_y = new_x, new_y
                continue

            # Apply infill peaks
            if (in_infill and is_extrusion and
                    current_layer >= start_layer and seg_length >= min_infill_length):
                output.extend(generate_infill_peaks(
                    current_x, current_y, new_x, new_y,
                    new_e, current_f, teeth_depth, teeth_pitch, interlock_length
                ))
                stats.infill_lines_modified += 1
                current_x, current_y = new_x, new_y
                continue

            current_x, current_y = new_x, new_y

        elif line.strip().startswith(('G1', 'G0')):
            x_match = re.search(r'X([\d.]+)', line)
            y_match = re.search(r'Y([\d.]+)', line)
            if x_match:
                current_x = float(x_match.group(1))
            if y_match:
                current_y = float(y_match.group(1))

        output.append(line)

    stats.layers_processed = current_layer
    return ''.join(output), stats


def main():
    """Main entry point for the wall-infill interlocking post-processor."""
    args = parse_args()

    print("=" * 60)
    print("  Wall-Infill Mechanical Interlocking")
    print("=" * 60)

    print(f"\nLoading {args.input_3mf}...")
    try:
        gcode_content, gcode_path = extract_gcode_from_3mf(args.input_3mf)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"\nSettings:")
    print(f"   Teeth depth: {args.teeth_depth}mm")
    print(f"   Teeth pitch: {args.teeth_pitch}mm")
    print(f"   Interlock zone: {args.interlock_length}mm")

    print(f"\nProcessing...")
    new_gcode, stats = process_gcode(
        gcode_content, args.teeth_depth, args.teeth_pitch,
        args.interlock_length, args.start_layer,
        args.min_infill_length, args.verbose
    )

    print(f"\nResults:")
    print(f"   Wall segments with valleys: {stats.wall_segments_modified}")
    print(f"   Infill lines with peaks: {stats.infill_lines_modified}")

    print(f"\nSaving {args.output_3mf}...")
    repack_3mf(args.input_3mf, args.output_3mf, new_gcode, gcode_path)
    print(f"\nDone!")


if __name__ == '__main__':
    main()
