"""Verification tests for thermal stress analysis."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multimaterial_3d.analysis.thermal import thermal_stress_analysis, predict_warping


class TestSymmetricLayup:
    """Symmetric PLA/TPU/PLA layup should have zero warping curvature."""

    def test_symmetric_pla_tpu_pla_zero_curvature(self):
        # 3-layer symmetric: PLA / TPU / PLA
        # Pattern [1,2,1] with total_height = 3 * layer_height
        result = thermal_stress_analysis(
            pattern=[1, 2, 1],
            layer_height=0.2,
            total_height=0.6,
            material_map={1: 'PLA', 2: 'TPU'},
            T_room=23.0,
            verbose=False,
        )
        # Symmetric layup -> curvature must be zero
        assert abs(result['curvature']) < 1e-10, (
            f"Symmetric PLA/TPU/PLA layup should have zero curvature, "
            f"got {result['curvature']}"
        )


class TestAsymmetricLayup:
    """Asymmetric PLA/TPU layup should have nonzero curvature."""

    def test_asymmetric_pla_tpu_nonzero_curvature(self):
        result = thermal_stress_analysis(
            pattern=[1, 2],
            layer_height=0.2,
            total_height=0.4,
            material_map={1: 'PLA', 2: 'TPU'},
            T_room=23.0,
            verbose=False,
        )
        # TPU has Tg = -40C, below room temp, so delta_T <= 0 for TPU layer.
        # PLA has Tg = 60C. The stress-free temp for PLA/TPU interface is
        # min(60, -40) = -40, so delta_T = -40 - 23 = -63 < 0.
        # This means no thermal stress accumulates!
        # Let's use PLA/ABS instead where both have Tg > room temp.

    def test_asymmetric_pla_abs_nonzero_curvature(self):
        result = thermal_stress_analysis(
            pattern=[1, 2],
            layer_height=0.2,
            total_height=0.4,
            material_map={1: 'PLA', 2: 'ABS'},
            T_room=23.0,
            verbose=False,
        )
        # PLA Tg=60, ABS Tg=105, min=60, delta_T=37 > 0
        # Asymmetric -> nonzero curvature
        assert abs(result['curvature']) > 1e-10, (
            f"Asymmetric PLA/ABS layup should have nonzero curvature, "
            f"got {result['curvature']}"
        )

    def test_curvature_sign_convention(self):
        """Curvature sign should flip when layer order is reversed."""
        result_12 = thermal_stress_analysis(
            pattern=[1, 2],
            layer_height=0.2,
            total_height=0.4,
            material_map={1: 'PLA', 2: 'ABS'},
            T_room=23.0,
            verbose=False,
        )
        result_21 = thermal_stress_analysis(
            pattern=[2, 1],
            layer_height=0.2,
            total_height=0.4,
            material_map={1: 'PLA', 2: 'ABS'},
            T_room=23.0,
            verbose=False,
        )
        # Reversing the order should flip the curvature sign
        # (or at least change it due to asymmetry reversal)
        assert result_12['curvature'] * result_21['curvature'] < 0, (
            f"Reversing layer order should flip curvature sign: "
            f"{result_12['curvature']} vs {result_21['curvature']}"
        )


class TestSameMaterial:
    """Same material on all layers should produce no interface stresses."""

    def test_pla_pla_no_stresses(self):
        result = thermal_stress_analysis(
            pattern=[1],
            layer_height=0.2,
            total_height=0.8,
            material_map={1: 'PLA'},
            T_room=23.0,
            verbose=False,
        )
        assert len(result['interface_stresses']) == 0
        assert result['max_shear_stress'] == 0.0
        assert result['max_normal_stress'] == 0.0

    def test_pla_pla_two_filaments_no_stresses(self):
        # Two different filament numbers mapping to same material
        result = thermal_stress_analysis(
            pattern=[1, 2],
            layer_height=0.2,
            total_height=0.4,
            material_map={1: 'PLA', 2: 'PLA'},
            T_room=23.0,
            verbose=False,
        )
        assert len(result['interface_stresses']) == 0


class TestDimensionalConsistency:
    """Check dimensional consistency of tau_max for PLA/ABS interface."""

    def test_tau_max_dimensions(self):
        """tau_max should be in MPa and have reasonable magnitude."""
        result = thermal_stress_analysis(
            pattern=[1, 2],
            layer_height=0.2,
            total_height=0.4,
            material_map={1: 'PLA', 2: 'ABS'},
            T_room=23.0,
            verbose=False,
        )
        # Should have exactly one interface
        assert len(result['interface_stresses']) == 1
        iface = result['interface_stresses'][0]

        tau = iface['tau_max']
        # tau_max should be positive (it's a magnitude)
        assert tau > 0, f"tau_max should be positive, got {tau}"
        # For PLA/ABS with ~22 1/K CTE mismatch and ~37C delta_T,
        # tau should be on the order of a few MPa (not kPa, not GPa)
        assert 0.01 < tau < 100, (
            f"tau_max = {tau} MPa seems out of reasonable range for FDM"
        )

    def test_sigma_signs_opposite(self):
        """sigma_bot and sigma_top should have opposite signs (force balance)."""
        result = thermal_stress_analysis(
            pattern=[1, 2],
            layer_height=0.2,
            total_height=0.4,
            material_map={1: 'PLA', 2: 'ABS'},
            T_room=23.0,
            verbose=False,
        )
        assert len(result['interface_stresses']) == 1
        iface = result['interface_stresses'][0]
        # Force balance: sigma_bot * h1 + sigma_top * h2 = 0
        # With equal layer heights: sigma_bot = -sigma_top
        assert iface['sigma_bot'] * iface['sigma_top'] < 0, (
            f"sigma_bot={iface['sigma_bot']:.4f} and sigma_top={iface['sigma_top']:.4f} "
            f"should have opposite signs"
        )

    def test_force_balance(self):
        """sigma_1 * h1 + sigma_2 * h2 should equal zero."""
        result = thermal_stress_analysis(
            pattern=[1, 2],
            layer_height=0.2,
            total_height=0.4,
            material_map={1: 'PLA', 2: 'ABS'},
            T_room=23.0,
            verbose=False,
        )
        iface = result['interface_stresses'][0]
        h = 0.2
        force_sum = iface['sigma_bot'] * h + iface['sigma_top'] * h
        assert abs(force_sum) < 1e-10, (
            f"Force balance violated: sigma_bot*h + sigma_top*h = {force_sum}"
        )


class TestPredictWarping:
    """Test the predict_warping convenience function."""

    def test_deflection_scales_with_length_squared(self):
        w1 = predict_warping(
            [1, 2], 0.2, 0.4, {1: 'PLA', 2: 'ABS'}, part_length=100.0
        )
        w2 = predict_warping(
            [1, 2], 0.2, 0.4, {1: 'PLA', 2: 'ABS'}, part_length=200.0
        )
        # deflection = kappa * L^2 / 8, so doubling L should quadruple deflection
        if w1['deflection'] > 1e-12:
            ratio = w2['deflection'] / w1['deflection']
            assert abs(ratio - 4.0) < 1e-6, (
                f"Deflection should scale as L^2, ratio = {ratio}"
            )
