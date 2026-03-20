"""Tests for the mechanical analysis module."""

import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multimaterial_3d.analysis.mechanical import (
    compute_layer_stiffness_matrix,
    compute_abd_matrix,
    analyze_layup,
)
from multimaterial_3d.core.materials import get_material


class TestStiffnessMatrix:
    """Test the reduced stiffness matrix computation."""

    def test_pla_stiffness_symmetric(self):
        """Stiffness matrix must be symmetric."""
        mat = get_material('PLA')
        Q = compute_layer_stiffness_matrix(mat)
        np.testing.assert_array_almost_equal(Q, Q.T)

    def test_pla_stiffness_positive_definite(self):
        """Stiffness matrix must be positive definite."""
        mat = get_material('PLA')
        Q = compute_layer_stiffness_matrix(mat)
        eigenvalues = np.linalg.eigvals(Q)
        assert all(ev > 0 for ev in eigenvalues)

    def test_stiffer_material_higher_q11(self):
        """PLA-CF should have higher Q11 than PLA."""
        Q_pla = compute_layer_stiffness_matrix(get_material('PLA'))
        Q_cf = compute_layer_stiffness_matrix(get_material('PLA-CF'))
        assert Q_cf[0, 0] > Q_pla[0, 0]


class TestABDMatrix:
    """Test the ABD laminate stiffness matrix computation."""

    def test_single_material_symmetric(self):
        """A single-material layup should be symmetric (B = 0)."""
        result = compute_abd_matrix(
            pattern=[1], layer_height=0.2, total_height=2.0,
            material_map={1: 'PLA'}
        )
        assert result['symmetric'] is True
        np.testing.assert_array_almost_equal(result['B'], np.zeros((3, 3)), decimal=6)

    def test_symmetric_layup(self):
        """A symmetric pattern like [1,2,2,1] should have B ≈ 0."""
        result = compute_abd_matrix(
            pattern=[1, 2, 2, 1], layer_height=0.2, total_height=0.8,
            material_map={1: 'PLA', 2: 'TPU'}
        )
        # Symmetric layup should have near-zero coupling
        assert result['symmetric'] is True

    def test_asymmetric_layup_has_coupling(self):
        """An asymmetric pattern like [1,2] should have B ≠ 0."""
        result = compute_abd_matrix(
            pattern=[1, 2], layer_height=0.2, total_height=2.0,
            material_map={1: 'PLA', 2: 'TPU'}
        )
        # With very different materials, asymmetric layup has coupling
        B_norm = np.linalg.norm(result['B'])
        # B should be nonzero for asymmetric layup with different materials
        assert B_norm > 0

    def test_effective_modulus_between_constituents(self):
        """Effective modulus should be between the two material moduli."""
        result = compute_abd_matrix(
            pattern=[1, 2], layer_height=0.2, total_height=2.0,
            material_map={1: 'PLA', 2: 'TPU'}
        )
        E_pla = get_material('PLA').E
        E_tpu = get_material('TPU').E
        assert E_tpu < result['Ex_eff'] < E_pla

    def test_neutral_axis_at_midplane_for_single_material(self):
        """Single material should have neutral axis at geometric center."""
        result = compute_abd_matrix(
            pattern=[1], layer_height=0.2, total_height=2.0,
            material_map={1: 'PLA'}
        )
        assert abs(result['z_neutral'] - 1.0) < 0.01


class TestAnalyzeLayup:
    """Test the full layup analysis function."""

    def test_voigt_bound_higher_than_reuss(self):
        """Voigt (iso-strain) bound should always exceed Reuss (iso-stress)."""
        result = analyze_layup(
            pattern=[1, 2], total_height=2.0, layer_height=0.2,
            material_map={1: 'PLA', 2: 'TPU'}, verbose=False
        )
        assert result['E_voigt'] > result['E_reuss']

    def test_volume_fractions_sum_to_one(self):
        result = analyze_layup(
            pattern=[1, 2, 2], total_height=3.0, layer_height=0.2,
            material_map={1: 'PLA', 2: 'TPU'}, verbose=False
        )
        total = sum(result['volume_fractions'].values())
        assert abs(total - 1.0) < 1e-10

    def test_interface_count_correct(self):
        """Pattern [1,2] has 2 transitions per cycle (1->2 and 2->1)."""
        result = analyze_layup(
            pattern=[1, 2], total_height=2.0, layer_height=0.2,
            material_map={1: 'PLA', 2: 'TPU'}, verbose=False
        )
        assert result['transitions_per_pattern'] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
