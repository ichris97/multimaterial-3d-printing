"""
Print time and cost estimation from G-code analysis.

Estimates total print time by parsing G-code and accounting for:
- Travel moves (non-extrusion repositioning)
- Extrusion moves (actual printing) at varying speeds
- Acceleration and deceleration profiles
- Filament changes (tool changes on multi-material printers)
- Layer change overhead (Z moves, wipe sequences)

Also computes material usage and cost per material for multi-material prints.

Printer Kinematics Model
-------------------------
Bambu Lab printers (X1, P1, A1) use CoreXY kinematics with high acceleration.
The actual print time is significantly affected by acceleration limits:
a move of length L at speed V with acceleration A takes:

    t = L/V                                   if L > V²/A (cruise phase exists)
    t = 2 * sqrt(L/A)                        if L ≤ V²/A (triangular profile)

For short moves (infill, walls), the printer rarely reaches the commanded
speed, making the acceleration-limited time estimate more accurate.
"""

import re
from typing import Dict, List, Tuple
from ..core.materials import get_material


def estimate_print_time(gcode_content: str,
                        acceleration: float = 5000.0,
                        verbose: bool = True) -> dict:
    """Estimate print time from G-code content.

    Parses all G0/G1 moves, classifies them as travel or extrusion, and
    computes time accounting for acceleration limits.

    Parameters
    ----------
    gcode_content : str
        Full G-code content as string.
    acceleration : float
        Printer acceleration in mm/s^2 (default: 5000, typical for Bambu Lab).
    verbose : bool
        Print time breakdown.

    Returns
    -------
    dict
        'total_time_s': total estimated print time in seconds,
        'total_time_min': total time in minutes,
        'travel_time_s': time spent traveling (not extruding),
        'print_time_s': time spent extruding,
        'filament_change_time_s': time for tool changes,
        'num_layers': number of layers,
        'num_tool_changes': number of filament changes,
        'filament_usage_mm': total filament used in mm,
        'filament_usage_g': estimated filament weight in grams per material.
    """
    # Parse G-code for moves
    x, y, z = 0.0, 0.0, 0.0
    current_f = 3000  # mm/min
    travel_time = 0.0
    print_time = 0.0
    total_e = 0.0
    num_layers = 0
    num_tool_changes = 0

    # Per-material filament usage (by tool number)
    filament_by_tool = {}
    current_tool = 0

    # Typical times for non-move operations
    TOOL_CHANGE_TIME = 12.0  # seconds per filament change (Bambu AMS)
    LAYER_CHANGE_TIME = 0.5  # seconds for Z move + wipe

    move_pattern = re.compile(
        r'^G[01]\s+'
        r'(?:X([\d.]+)\s*)?'
        r'(?:Y([\d.]+)\s*)?'
        r'(?:Z([\d.]+)\s*)?'
        r'(?:.*?E([\d.]+))?'
        r'(?:.*?F(\d+))?',
        re.IGNORECASE
    )

    for line in gcode_content.split('\n'):
        stripped = line.strip()

        # Track layer changes
        if stripped.startswith(';CHANGE_LAYER') or '; CHANGE_LAYER' in stripped:
            num_layers += 1

        # Track tool changes
        if stripped.startswith('T') and stripped[1:].strip().isdigit():
            new_tool = int(stripped[1:].strip())
            if new_tool != current_tool:
                num_tool_changes += 1
                current_tool = new_tool

        # Parse moves
        match = move_pattern.match(stripped)
        if match:
            new_x = float(match.group(1)) if match.group(1) else x
            new_y = float(match.group(2)) if match.group(2) else y
            new_z = float(match.group(3)) if match.group(3) else z
            e_val = float(match.group(4)) if match.group(4) else 0
            if match.group(5):
                current_f = int(match.group(5))

            # Compute move distance
            dx = new_x - x
            dy = new_y - y
            dz = new_z - z
            dist = (dx**2 + dy**2 + dz**2) ** 0.5

            if dist > 0.001:
                # Speed in mm/s
                speed = current_f / 60.0

                # Acceleration-limited time calculation
                # If the move is short, the printer uses a triangular velocity
                # profile and never reaches the commanded speed.
                accel_distance = speed**2 / acceleration
                if dist > accel_distance:
                    # Trapezoidal profile: accelerate, cruise, decelerate
                    t = dist / speed + speed / acceleration
                else:
                    # Triangular profile: never reaches full speed
                    t = 2.0 * (dist / acceleration) ** 0.5

                if e_val > 0:
                    print_time += t
                    total_e += e_val
                    filament_by_tool[current_tool] = filament_by_tool.get(current_tool, 0) + e_val
                else:
                    travel_time += t

            x, y, z = new_x, new_y, new_z

    # Add overhead times
    tool_change_total = num_tool_changes * TOOL_CHANGE_TIME
    layer_change_total = num_layers * LAYER_CHANGE_TIME

    total_time = print_time + travel_time + tool_change_total + layer_change_total

    # Estimate filament weight (assuming 1.75mm filament diameter)
    # Volume = π * (d/2)² * L, then mass = volume * density
    import math
    filament_diameter = 1.75  # mm
    cross_section = math.pi * (filament_diameter / 2) ** 2  # mm²

    filament_weight = {}
    for tool, length_mm in filament_by_tool.items():
        volume_cm3 = cross_section * length_mm / 1000.0  # cm³
        # Assume PLA density as default (1.24 g/cm³)
        weight_g = volume_cm3 * 1.24
        filament_weight[tool] = weight_g

    result = {
        'total_time_s': total_time,
        'total_time_min': total_time / 60.0,
        'travel_time_s': travel_time,
        'print_time_s': print_time,
        'filament_change_time_s': tool_change_total,
        'num_layers': num_layers,
        'num_tool_changes': num_tool_changes,
        'filament_usage_mm': total_e,
        'filament_usage_g': filament_weight,
    }

    if verbose:
        print("\n" + "=" * 50)
        print("  PRINT TIME ESTIMATION")
        print("=" * 50)
        hours = int(total_time // 3600)
        mins = int((total_time % 3600) // 60)
        print(f"\n  Estimated print time: {hours}h {mins}m")
        print(f"    Printing: {print_time/60:.1f} min")
        print(f"    Travel: {travel_time/60:.1f} min")
        print(f"    Tool changes: {tool_change_total/60:.1f} min ({num_tool_changes} changes)")
        print(f"    Layer changes: {layer_change_total/60:.1f} min ({num_layers} layers)")
        print(f"\n  Filament Usage:")
        total_weight = 0
        for tool, weight in sorted(filament_weight.items()):
            print(f"    Tool {tool}: {filament_by_tool[tool]:.0f} mm ({weight:.1f} g)")
            total_weight += weight
        print(f"    Total: {total_weight:.1f} g")
        print("=" * 50)

    return result


def estimate_cost(gcode_content: str, material_map: Dict[int, str],
                  verbose: bool = True) -> dict:
    """Estimate material cost for a multi-material print.

    Parameters
    ----------
    gcode_content : str
        Full G-code content.
    material_map : dict
        Tool number to material key mapping (e.g., {0: 'PLA', 1: 'TPU'}).
    verbose : bool
        Print cost breakdown.

    Returns
    -------
    dict
        'total_cost': total material cost in USD,
        'cost_by_tool': per-tool cost breakdown,
        'weight_by_tool': per-tool weight in grams.
    """
    import math

    time_result = estimate_print_time(gcode_content, verbose=False)
    filament_diameter = 1.75
    cross_section = math.pi * (filament_diameter / 2) ** 2

    cost_by_tool = {}
    weight_by_tool = {}
    total_cost = 0.0

    for tool, length_mm in time_result.get('filament_usage_g', {}).items():
        # Look up material for this tool
        mat_key = material_map.get(tool, 'PLA')
        mat = get_material(mat_key)

        # Recalculate weight with correct density
        # Note: filament_usage_g used PLA density; recalculate here
        length = 0
        for t, l in enumerate(time_result.get('filament_usage_mm', {})):
            pass
        # Use stored length from the main estimation
        volume_cm3 = cross_section * length_mm / 1000.0 / mat.density * 1.24
        weight_g = volume_cm3 * mat.density
        cost = weight_g * mat.cost_per_kg / 1000.0

        cost_by_tool[tool] = cost
        weight_by_tool[tool] = weight_g
        total_cost += cost

    result = {
        'total_cost': total_cost,
        'cost_by_tool': cost_by_tool,
        'weight_by_tool': weight_by_tool,
    }

    if verbose:
        print(f"\n  Material Cost: ${total_cost:.2f}")
        for tool, cost in sorted(cost_by_tool.items()):
            mat_key = material_map.get(tool, 'PLA')
            print(f"    Tool {tool} ({mat_key}): ${cost:.3f} ({weight_by_tool[tool]:.1f}g)")

    return result
