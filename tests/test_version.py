"""Test version module."""

from aurora_cycler_manager.version import (
    __author__,
    __copyright__,
    __description__,
    __license__,
    __title__,
    __url__,
    __version__,
)


class TestVersion:
    """Version test class."""

    def test_metadata(self) -> None:
        """Test the metadata."""
        attrs = [
            __author__,
            __copyright__,
            __description__,
            __license__,
            __title__,
            __url__,
            __version__,
        ]
        assert all(isinstance(attr, str) and len(attr) > 0 for attr in attrs)
        # Should follow semantic versioning e.g "1.0.0" or "1.0.0-dev" or "1.0.0-rc.2"
        digits = __version__.split("-")[0].split(".")
        assert len(digits) == 3
        assert all(d.isdigit() for d in digits)
