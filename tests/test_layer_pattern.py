"""Tests for the layer pattern post-processor."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multimaterial_3d.postprocessors.layer_pattern import parse_pattern


class TestParsePattern:
    """Test pattern parsing for all supported formats."""

    def test_simple_list(self):
        assert parse_pattern("1,2") == [1, 2]
        assert parse_pattern("1,2,2,1") == [1, 2, 2, 1]

    def test_repeat_notation(self):
        assert parse_pattern("1x3,2x2") == [1, 1, 1, 2, 2]

    def test_f_notation(self):
        assert parse_pattern("F1:3,F2:2") == [1, 1, 1, 2, 2]
        assert parse_pattern("f1:1,f2:2,f1:3") == [1, 2, 2, 1, 1, 1]

    def test_mixed_notation(self):
        # Simple numbers mixed with F notation
        assert parse_pattern("1,F2:3") == [1, 2, 2, 2]

    def test_single_element(self):
        assert parse_pattern("1") == [1]

    def test_spaces_ignored(self):
        assert parse_pattern("1 , 2 , 3") == [1, 2, 3]

    def test_invalid_f_notation_raises(self):
        with pytest.raises(ValueError):
            parse_pattern("FX:3")

    def test_invalid_repeat_raises(self):
        with pytest.raises(ValueError):
            parse_pattern("axb")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
