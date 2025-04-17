"""Test for utilities module."""

import pytest

from aurora_cycler_manager.utils import c_to_float, run_from_sample


class TestRunFromSample:
    """Test the run_from_sample function."""

    def test_valid_sample_id(self) -> None:
        """With valid Sample ID."""
        assert run_from_sample("240620_svfe_gen4_05") == "240620_svfe_gen4"
        assert run_from_sample("date_name_gennum_17") == "date_name_gennum"

    def test_invalid_sample_id(self) -> None:
        """With invalid Sample ID."""
        assert run_from_sample("sample") == "misc"
        assert run_from_sample("sample_123_456_789") == "sample_123_456"
        assert run_from_sample("sample_123_456_789_0") == "sample_123_456_789"

    def test_empty_sample_id(self) -> None:
        """With empty Sample ID."""
        assert run_from_sample("") == "misc"

    def test_non_string_sample_id(self) -> None:
        """With non-string Sample ID."""
        assert run_from_sample(123) == "misc"
        assert run_from_sample(None) == "misc"


class TestCToFloat:
    """Test the c_to_float function."""

    def test_valid_c_rate(self) -> None:
        """Valid C-rate strings."""
        assert c_to_float("C/2") == 0.5
        assert c_to_float("0.5C") == 0.5
        assert c_to_float("C  / 3") == 1 / 3
        assert c_to_float("5C/0.5") == 10
        assert c_to_float("5/10C") == 0.5
        assert c_to_float("3D/5") == -0.6
        assert c_to_float("1/2 D") == -0.5

    def test_invalid_c_rate(self) -> None:
        """Invalid C-rate strings."""
        with pytest.raises(ValueError):
            c_to_float("invalidC")
        with pytest.raises(ValueError):
            c_to_float("invalidD")
        with pytest.raises(ValueError):
            c_to_float("0.5/1C/2")
        with pytest.raises(ValueError):
            c_to_float("3C/4D")

    def test_empty_c_rate(self) -> None:
        """Empty C-rate string."""
        with pytest.raises(ValueError):
            c_to_float("")

    def test_non_string_c_rate(self) -> None:
        """Non-string inputs."""
        with pytest.raises(TypeError):
            c_to_float(123)
        with pytest.raises(TypeError):
            c_to_float(None)
        with pytest.raises(ValueError):
            c_to_float(["in", "valid"])
        with pytest.raises(AttributeError):
            c_to_float(["C", "/2"])
        with pytest.raises(AttributeError):
            c_to_float({"C": 1, "/": 2})
        with pytest.raises(ValueError):
            c_to_float(["C/2"])
