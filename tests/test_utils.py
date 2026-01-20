"""Test for utilities module."""

import pytest

from aurora_cycler_manager.stdlib_utils import c_to_float, check_illegal_text, run_from_sample
from aurora_cycler_manager.utils import weighted_median


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


class TestWeightedMedian:
    """Test the weighted_median function."""

    def test_valid_input(self) -> None:
        """Valid inputs."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        assert weighted_median(values, weights) == 3.0

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        weights = [0.0, 0.0, 1.0, 1.0, 1.0]
        assert weighted_median(values, weights) == 4.0

    def test_empty_input(self) -> None:
        """Empty input."""
        with pytest.raises(ValueError):
            weighted_median([], [])

    def test_different_length(self) -> None:
        """Different length of values and weights."""
        with pytest.raises(ValueError):
            weighted_median([1.0, 2.0], [1.0])


class TestIllegalText:
    """Test check_illegal_text function."""

    def test_legal(self) -> None:
        """These should not raise errors."""
        tests = [
            "hello",
            "this-is-fine",
            "this_is also--fine ",
            "dots. are. fine. if. . spacedwé hävé löts öf ümläüts",
            "(brackets)[are]{okay}",
            "emoji are valid in sample IDs 🤖",
        ]
        for test in tests:
            check_illegal_text(test)

    def test_illegal(self) -> None:
        """These should raise errors."""
        tests = [
            "cant/use",
            r"cant\use",
            "naughty \0",
            "nope: not allowed",
            "can i use this? no",
            "How about >",
            "< or that",
            "oh|no",
            ".. is a big no no",
            ".............. too",
        ]
        for test in tests:
            with pytest.raises(ValueError, match="Illegal character or sequence"):
                check_illegal_text(test)

        # Check the error printing works
        with pytest.raises(ValueError, match=r"Illegal character or sequence in text: '>'"):
            check_illegal_text("aaaaa>aaaa")
        with pytest.raises(ValueError, match=r"Illegal character or sequence in text: '..'"):
            check_illegal_text("aaaaa..aaaa")
