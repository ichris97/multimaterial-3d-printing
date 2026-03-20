#!/usr/bin/env python3
"""
Advanced Layer Filament Pattern Script for Bambu Studio 3MF files

Supports custom repeating patterns for multi-material layer printing.

Usage examples:
    # Simple alternating (1,2,1,2,...)
    python alternating_layers_v2.py input.3mf output.3mf --layer-height 0.1 --total-height 18 --pattern 1,2
    
    # Custom pattern: 1 layer of F1, 2 layers of F2, 3 layers of F1 (repeating)
    python alternating_layers_v2.py input.3mf output.3mf --layer-height 0.1 --total-height 18 --pattern 1,2,2,1,1,1
    
    # Using repeat notation: 1x3,2x2 = 1,1,1,2,2
    python alternating_layers_v2.py input.3mf output.3mf --layer-height 0.1 --total-height 18 --pattern 1x3,2x2
    
    # Analyze mechanical properties
    python alternating_layers_v2.py input.3mf output.3mf --layer-height 0.1 --total-height 18 --pattern 1,2,2,1,1,1 --analyze
"""

import argparse
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import tempfile
import re


# Material properties database (typical values in MPa for tensile modulus, tensile strength)
MATERIAL_PROPERTIES = {
    'PLA': {'E': 3500, 'sigma_t': 50, 'sigma_c': 70, 'density': 1.24, 'name': 'PLA'},
    'PETG': {'E': 2100, 'sigma_t': 45, 'sigma_c': 55, 'density': 1.27, 'name': 'PETG'},
    'ABS': {'E': 2300, 'sigma_t': 40, 'sigma_c': 65, 'density': 1.04, 'name': 'ABS'},
    'TPU': {'E': 26, 'sigma_t': 30, 'sigma_c': 20, 'density': 1.21, 'name': 'TPU (95A)'},
    'ASA': {'E': 2200, 'sigma_t': 42, 'sigma_c': 60, 'density': 1.07, 'name': 'ASA'},
    'PA': {'E': 1800, 'sigma_t': 70, 'sigma_c': 80, 'density': 1.14, 'name': 'Nylon/PA'},
    'PC': {'E': 2400, 'sigma_t': 60, 'sigma_c': 75, 'density': 1.20, 'name': 'Polycarbonate'},
    'PLA-CF': {'E': 8000, 'sigma_t': 65, 'sigma_c': 90, 'density': 1.30, 'name': 'PLA-CF'},
    'PETG-CF': {'E': 6000, 'sigma_t': 55, 'sigma_c': 75, 'density': 1.35, 'name': 'PETG-CF'},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Add custom layer filament patterns to 3MF file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pattern formats:
  Simple list:     --pattern 1,2,2,1,1,1
  
  Clear notation:  --pattern F1:1,F2:2,F1:3
                   (Filament 1 for 1 layer, Filament 2 for 2 layers, Filament 1 for 3 layers)
  
  Legacy notation: --pattern 1x1,2x2,1x3  (same as above)
  
The pattern repeats until total height is reached.

Examples:
  Your example:    --pattern F1:1,F2:2,F1:3   → [1, 2, 2, 1, 1, 1]
  Alternating:     --pattern 1,2              → [1, 2, 1, 2, ...]
  Sandwich core:   --pattern F1:5,F2:10,F1:5  → stiff skin, soft core
  Gradient:        --pattern F1:3,F2:3        → gradual transition
        """
    )
    parser.add_argument('input_3mf', help='Input 3MF file path')
    parser.add_argument('output_3mf', help='Output 3MF file path')
    parser.add_argument('--layer-height', type=float, default=0.1,
                        help='Layer height in mm (default: 0.1)')
    parser.add_argument('--total-height', type=float, default=None,
                        help='Total print height in mm (auto-detected if not specified)')
    parser.add_argument('--pattern', type=str, default='1,2',
                        help='Layer pattern - see examples above (default: 1,2)')
    parser.add_argument('--object-id', type=int, default=1,
                        help='Object ID in the 3MF (default: 1)')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze mechanical properties of the layup')
    parser.add_argument('--materials', type=str, default=None,
                        help='Material types for analysis, e.g., "1:PLA,2:TPU"')
    return parser.parse_args()


def parse_pattern(pattern_str):
    """
    Parse pattern string into list of filament numbers.
    
    Supports:
    - Simple: "1,2,2,1,1,1"
    - Repeat notation: "1x3,2x2" -> [1,1,1,2,2] (filament X repeated N times)
    - Clearer notation: "F1:3,F2:2" -> [1,1,1,2,2] (Filament N for M layers)
    """
    pattern = []
    parts = pattern_str.replace(' ', '').split(',')
    
    for part in parts:
        # New clearer format: F1:3 means "Filament 1 for 3 layers"
        if part.upper().startswith('F') and ':' in part:
            match = re.match(r'[Ff](\d+):(\d+)', part)
            if match:
                filament = int(match.group(1))
                count = int(match.group(2))
                pattern.extend([filament] * count)
            else:
                raise ValueError(f"Invalid F notation: {part}")
        elif 'x' in part.lower():
            # Legacy format: "1x3" means filament 1 repeated 3 times
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


def get_model_height(zip_ref):
    """Try to auto-detect model height from the 3MF geometry"""
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


def generate_layer_ranges_xml(object_id, layer_height, total_height, pattern):
    """Generate the layer_config_ranges.xml content with custom pattern"""
    
    root = ET.Element('objects')
    obj = ET.SubElement(root, 'object', id=str(object_id))
    
    num_layers = int(total_height / layer_height)
    pattern_length = len(pattern)
    
    for i in range(num_layers):
        min_z = round(i * layer_height, 4)
        max_z = round((i + 1) * layer_height, 4)
        
        # Get filament from repeating pattern
        filament = pattern[i % pattern_length]
        
        range_elem = ET.SubElement(obj, 'range', min_z=str(min_z), max_z=str(max_z))
        ET.SubElement(range_elem, 'option', opt_key='extruder').text = str(filament)
        ET.SubElement(range_elem, 'option', opt_key='layer_height').text = str(layer_height)
    
    return ET.tostring(root, encoding='unicode')


def analyze_layup(pattern, total_height, layer_height, material_map):
    """
    Analyze mechanical properties of the multi-material layup.
    
    Uses classical laminate theory simplified for 3D printed parts.
    """
    num_layers = int(total_height / layer_height)
    pattern_length = len(pattern)
    
    # Count layers per material
    layer_counts = {}
    for i in range(num_layers):
        filament = pattern[i % pattern_length]
        layer_counts[filament] = layer_counts.get(filament, 0) + 1
    
    # Calculate volume fractions
    total_layers = sum(layer_counts.values())
    volume_fractions = {f: count / total_layers for f, count in layer_counts.items()}
    
    print("\n" + "="*70)
    print("MECHANICAL PROPERTY ANALYSIS")
    print("="*70)
    
    print(f"\n📊 Layer Distribution:")
    print(f"   Total layers: {total_layers}")
    print(f"   Pattern length: {pattern_length} layers")
    print(f"   Pattern repeats: {num_layers / pattern_length:.1f}x")
    
    for filament, count in sorted(layer_counts.items()):
        mat_name = material_map.get(filament, f"Filament {filament}")
        print(f"   Filament {filament} ({mat_name}): {count} layers ({volume_fractions[filament]*100:.1f}%)")
    
    # Get material properties
    materials = {}
    for filament in layer_counts.keys():
        mat_key = material_map.get(filament, 'PLA')
        if mat_key in MATERIAL_PROPERTIES:
            materials[filament] = MATERIAL_PROPERTIES[mat_key]
        else:
            print(f"   ⚠️  Unknown material '{mat_key}' for filament {filament}, using PLA defaults")
            materials[filament] = MATERIAL_PROPERTIES['PLA']
    
    print(f"\n📐 Material Properties Used:")
    for filament, props in materials.items():
        print(f"   Filament {filament} ({props['name']}): E={props['E']} MPa, σt={props['sigma_t']} MPa")
    
    # Rule of Mixtures calculations
    print(f"\n🔬 Composite Properties (Rule of Mixtures):")
    
    # In-plane (Voigt) - upper bound, loading parallel to layers
    E_voigt = sum(volume_fractions[f] * materials[f]['E'] for f in layer_counts.keys())
    sigma_t_voigt = sum(volume_fractions[f] * materials[f]['sigma_t'] for f in layer_counts.keys())
    
    # Through-thickness (Reuss) - lower bound, loading perpendicular to layers
    E_reuss = 1 / sum(volume_fractions[f] / materials[f]['E'] for f in layer_counts.keys())
    
    # Average density
    density_avg = sum(volume_fractions[f] * materials[f]['density'] for f in layer_counts.keys())
    
    print(f"\n   🔹 In-plane loading (parallel to layers):")
    print(f"      Effective modulus (Voigt): {E_voigt:.0f} MPa")
    print(f"      Effective tensile strength: {sigma_t_voigt:.1f} MPa")
    
    print(f"\n   🔹 Through-thickness loading (perpendicular to layers):")
    print(f"      Effective modulus (Reuss): {E_reuss:.0f} MPa")
    
    print(f"\n   🔹 General properties:")
    print(f"      Average density: {density_avg:.2f} g/cm³")
    print(f"      Specific stiffness (E/ρ): {E_voigt/density_avg:.0f} MPa·cm³/g")
    
    # Stress analysis for bending
    print(f"\n📏 Bending Analysis (Flexural Loading):")
    
    # Calculate neutral axis and flexural rigidity
    # For a beam with layers, we need to consider layer positions
    h = total_height  # total thickness
    z_positions = []
    
    # Build layer stack with positions (z from bottom)
    for i in range(num_layers):
        z_bottom = i * layer_height
        z_top = (i + 1) * layer_height
        z_mid = (z_bottom + z_top) / 2
        filament = pattern[i % pattern_length]
        E = materials[filament]['E']
        z_positions.append((z_bottom, z_top, z_mid, E, filament))
    
    # Find neutral axis (weighted by E*A)
    sum_EA_z = sum(E * layer_height * z_mid for _, _, z_mid, E, _ in z_positions)
    sum_EA = sum(E * layer_height for _, _, _, E, _ in z_positions)
    z_neutral = sum_EA_z / sum_EA
    
    # Calculate equivalent flexural rigidity (EI per unit width)
    EI = 0
    for z_bottom, z_top, z_mid, E, _ in z_positions:
        # Parallel axis theorem: I = I_centroid + A*d²
        I_layer = layer_height**3 / 12  # per unit width
        d = z_mid - z_neutral
        EI += E * (I_layer + layer_height * d**2)
    
    print(f"   Neutral axis position: {z_neutral:.3f} mm from bottom")
    print(f"   Equivalent flexural rigidity (EI): {EI:.1f} MPa·mm⁴/mm")
    
    # Compare to homogeneous materials
    print(f"\n📊 Comparison to Homogeneous Materials:")
    for mat_key, props in MATERIAL_PROPERTIES.items():
        if mat_key in [material_map.get(f) for f in layer_counts.keys()]:
            I_homo = h**3 / 12
            EI_homo = props['E'] * I_homo
            ratio = EI / EI_homo
            print(f"   vs pure {props['name']}: {ratio:.2f}x flexural rigidity")
    
    # Interface analysis
    print(f"\n⚠️  Interface Considerations:")
    
    # Count material transitions per pattern cycle
    transitions_per_pattern = 0
    for i in range(pattern_length):
        f1 = pattern[i]
        f2 = pattern[(i + 1) % pattern_length]
        if f1 != f2:
            transitions_per_pattern += 1
    
    # Total interfaces in the part
    complete_cycles = num_layers // pattern_length
    remaining_layers = num_layers % pattern_length
    
    total_interfaces = transitions_per_pattern * complete_cycles
    # Add interfaces from partial cycle
    for i in range(remaining_layers - 1):
        if pattern[i] != pattern[i + 1]:
            total_interfaces += 1
    
    print(f"   Material transitions per pattern: {transitions_per_pattern}")
    print(f"   Total interfaces in part: ~{total_interfaces}")
    print(f"   Interface density: {total_interfaces / total_height:.1f} interfaces/mm")
    
    # Check material compatibility
    filament_list = list(layer_counts.keys())
    if len(filament_list) > 1:
        mat1 = material_map.get(filament_list[0], 'PLA')
        mat2 = material_map.get(filament_list[1], 'PLA')
        
        print(f"\n   Material compatibility notes:")
        
        # Known compatibility issues (based on real-world printing experience)
        # Keys are sorted alphabetically for consistent lookup
        compatibility_warnings = {
            ('ABS', 'ASA'): "✅ Excellent adhesion - same polymer family",
            ('ABS', 'PC'): "✅ Good adhesion - often blended together",
            ('ABS', 'PETG'): "⚠️ Moderate adhesion - similar temps help but not ideal",
            ('ABS', 'PLA'): "❌ Poor adhesion - different shrinkage rates, warping risk",
            ('ABS', 'TPU'): "✅ Good adhesion",
            ('ASA', 'PLA'): "❌ Poor adhesion - incompatible materials",
            ('PA', 'PLA'): "❌ Poor adhesion - very different materials",
            ('PA', 'TPU'): "✅ Good adhesion - both flexible materials",
            ('PETG', 'PLA'): "❌ Poor adhesion - different surface energies, layers may separate",
            ('PETG', 'TPU'): "✅ Good adhesion - both flexible-friendly",
            ('PETG-CF', 'PETG'): "✅ Excellent adhesion - same base material",
            ('PLA', 'PLA-CF'): "✅ Excellent adhesion - same base material",
            ('PLA', 'TPU'): "✅ Excellent adhesion - commonly used combo for flexible/rigid hybrids",
        }
        
        pair = tuple(sorted([mat1, mat2]))
        if pair in compatibility_warnings:
            print(f"   {compatibility_warnings[pair]}")
        else:
            print(f"   ℹ️  {mat1} + {mat2}: No data - test adhesion before production use")
    
    # Delamination risk assessment - this is about MECHANICAL stress, not adhesion
    E_values = [materials[f]['E'] for f in layer_counts.keys()]
    E_ratio = max(E_values) / min(E_values) if min(E_values) > 0 else float('inf')
    
    print(f"\n   Modulus mismatch ratio: {E_ratio:.1f}x")
    print(f"   (Note: This affects stress concentration under load, separate from adhesion)")
    if E_ratio > 10:
        print(f"   🔴 HIGH stress concentration risk - stiff/flexible interface will see high shear stress under load")
        print(f"       → Good for impact absorption, but weak point under sustained bending/tension")
    elif E_ratio > 3:
        print(f"   🟡 MODERATE stress concentration - some load transfer issues at interfaces")
    else:
        print(f"   🟢 LOW stress concentration - similar stiffness, smooth load transfer")
    
    print("\n" + "="*70)
    
    return {
        'E_voigt': E_voigt,
        'E_reuss': E_reuss,
        'sigma_t': sigma_t_voigt,
        'density': density_avg,
        'EI': EI,
        'z_neutral': z_neutral,
        'volume_fractions': volume_fractions,
        'interfaces': total_interfaces
    }


def modify_3mf(input_path, output_path, layer_height, total_height, pattern, object_id):
    """Modify the 3MF file to add custom layer pattern"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
            if total_height is None:
                total_height = get_model_height(zip_ref)
                if total_height is None:
                    raise ValueError("Could not auto-detect model height. Please specify --total-height")
                print(f"Auto-detected model height: {total_height}mm")
        
        # Generate layer config
        layer_config_path = temp_dir / 'Metadata' / 'layer_config_ranges.xml'
        layer_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        xml_content = generate_layer_ranges_xml(object_id, layer_height, total_height, pattern)
        xml_content = '<?xml version="1.0" encoding="utf-8"?>\n' + xml_content
        
        with open(layer_config_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        num_layers = int(total_height / layer_height)
        print(f"Generated {num_layers} layer ranges with pattern {pattern[:10]}{'...' if len(pattern) > 10 else ''}")
        
        # Repack
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_out:
            for file_path in temp_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(temp_dir)
                    zip_out.write(file_path, arcname)
        
        print(f"Created: {output_path}")
        
        return total_height


def parse_materials(materials_str):
    """Parse materials string like '1:PLA,2:TPU' into dict"""
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
    args = parse_args()
    
    # Parse pattern
    try:
        pattern = parse_pattern(args.pattern)
    except ValueError as e:
        print(f"Error parsing pattern: {e}")
        return
    
    print(f"Pattern: {pattern} (repeating)")
    print(f"Pattern length: {len(pattern)} layers")
    
    # Modify 3MF
    total_height = modify_3mf(
        args.input_3mf,
        args.output_3mf,
        args.layer_height,
        args.total_height,
        pattern,
        args.object_id
    )
    
    # Analyze if requested
    if args.analyze:
        material_map = parse_materials(args.materials)
        
        # Default materials if not specified
        unique_filaments = set(pattern)
        for f in unique_filaments:
            if f not in material_map:
                material_map[f] = 'PLA'
                print(f"Note: Using PLA as default for filament {f}. Specify with --materials '{f}:MATERIAL'")
        
        analyze_layup(pattern, total_height, args.layer_height, material_map)
    
    print("\n✅ Done! Open the output file in Bambu Studio to see the pattern.")


if __name__ == '__main__':
    main()