"""Test for config module."""

from pathlib import Path

from aurora_cycler_manager.config import get_config


class TestGetConfig:
    """Test the get_config function."""

    def test_get_config(self) -> None:
        """Test the get_config function."""
        config = get_config()
        assert isinstance(config, dict)
        assert "Shared config path" in config
        assert "SSH private key path" in config
        assert "Snapshots folder path" in config

    def test_config_structure(self) -> None:
        """Test the structure of the config dictionary."""
        config = get_config()
        assert isinstance(config, dict)
        assert all(isinstance(k, str) for k in config)
        path_keys = [
            "Shared config path",
            "SSH private key path",
            "Database path",
            "Database backup folder path",
            "Processed snapshots folder path",
        ]
        assert all(key in config for key in path_keys)
        assert all(isinstance(config[key], Path) for key in path_keys)
        assert all(config[key].is_absolute() for key in path_keys)
