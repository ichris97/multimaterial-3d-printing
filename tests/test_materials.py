"""Tests for the material database and lookup functions."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multimaterial_3d.core.materials import (
    MATERIAL_DB, get_material, get_adhesion, list_materials, MaterialProperties
)


class TestMaterialDatabase:
    """Verify the material database contains valid, consistent data."""

    def test_all_materials_have_positive_modulus(self):
        """Young's modulus must be positive for all materials."""
        for key, mat in MATERIAL_DB.items():
            assert mat.E > 0, f"{key}: E must be positive, got {mat.E}"

    def test_z_modulus_less_than_xy(self):
        """Z-direction modulus should be less than in-plane modulus for FDM."""
        for key, mat in MATERIAL_DB.items():
            assert mat.E_z <= mat.E, (
                f"{key}: E_z ({mat.E_z}) should be <= E ({mat.E}) for FDM parts"
            )

    def test_all_materials_have_positive_density(self):
        for key, mat in MATERIAL_DB.items():
            assert 0.5 < mat.density < 3.0, f"{key}: unrealistic density {mat.density}"

    def test_cte_is_positive(self):
        """CTE should be positive for all polymer filaments."""
        for key, mat in MATERIAL_DB.items():
            assert mat.CTE > 0, f"{key}: CTE must be positive"

    def test_print_temp_above_bed_temp(self):
        """Nozzle temperature should always exceed bed temperature."""
        for key, mat in MATERIAL_DB.items():
            assert mat.T_print > mat.T_bed, (
                f"{key}: T_print ({mat.T_print}) must exceed T_bed ({mat.T_bed})"
            )

    def test_carbon_fiber_composites_stiffer_than_base(self):
        """CF composites must have higher modulus than their base material."""
        assert MATERIAL_DB['PLA-CF'].E > MATERIAL_DB['PLA'].E
        assert MATERIAL_DB['PETG-CF'].E > MATERIAL_DB['PETG'].E

    def test_tpu_is_flexible(self):
        """TPU should be significantly less stiff than rigid materials."""
        assert MATERIAL_DB['TPU'].E < 100  # Much less than PLA's ~3500


class TestMaterialLookup:
    """Test the get_material() lookup function."""

    def test_lookup_existing_material(self):
        mat = get_material('PLA')
        assert mat.name == 'PLA (Polylactic Acid)'
        assert mat.E == 3500

    def test_lookup_case_insensitive(self):
        mat = get_material('pla')
        assert mat.E == 3500

    def test_lookup_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown material"):
            get_material('UNOBTAINIUM')


class TestAdhesionMatrix:
    """Test inter-material adhesion compatibility lookups."""

    def test_same_material_perfect(self):
        result = get_adhesion('PLA', 'PLA')
        assert result['score'] == 5

    def test_symmetric_lookup(self):
        """Adhesion should be the same regardless of order."""
        r1 = get_adhesion('PLA', 'TPU')
        r2 = get_adhesion('TPU', 'PLA')
        assert r1['score'] == r2['score']

    def test_pla_tpu_good_adhesion(self):
        result = get_adhesion('PLA', 'TPU')
        assert result['score'] >= 4

    def test_pla_petg_poor_adhesion(self):
        result = get_adhesion('PLA', 'PETG')
        assert result['score'] <= 2

    def test_unknown_pair_returns_none_score(self):
        result = get_adhesion('PLA', 'PA-CF')
        assert result['score'] is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
