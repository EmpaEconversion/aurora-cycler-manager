"""Test for config module."""

from pathlib import Path

from aurora_cycler_manager.config import _read_config_file, get_config


class TestGetConfig:
    """Test the get_config and _read_config_file functions."""

    def test_read_config_file(self) -> None:
        """Test the _read_config_file function."""
        config = _read_config_file()
        assert isinstance(config, dict)
        assert "Shared config path" in config
        assert "SSH private key path" in config
        assert "Snapshots folder path" in config

    def check_file_read_once(self) -> None:
        """Check that all configs are the same object in memory."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
        config3 = get_config(reload=True)
        assert config1 is not config3

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
        path_keys = [k for k in config if "path" in k.lower() and "ssh" not in k.lower()]
        assert all(isinstance(config[key], Path) for key in path_keys)
        assert all(config[key].is_absolute() for key in path_keys)
