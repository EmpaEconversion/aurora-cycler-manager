# Copyright © 2025-2026, Empa.
"""Test for config module."""

import getpass
from pathlib import Path

import pytest

from aurora_cycler_manager.config import (
    _convert_legacy_servers,
    _fixup_config,
    _read_config_file,
    get_config,
    normalise_username,
)


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

    def test_username(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Username comes from the config, otherwise the OS account name."""
        monkeypatch.setattr(getpass, "getuser", lambda: "OSUser")
        assert _fixup_config({"Snapshots folder path": "x", "Username": "AbCd"})["Username"] == "abcd"
        assert _fixup_config({"Snapshots folder path": "x"})["Username"] == "osuser"

        def no_user() -> str:
            raise OSError

        monkeypatch.setattr(getpass, "getuser", no_user)
        assert _fixup_config({"Snapshots folder path": "x"})["Username"] is None

    def test_invalid_username(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid supplied usernames raise, invalid OS account names are dropped."""
        with pytest.raises(ValueError, match="Invalid 'Username' in user config"):
            _fixup_config({"Snapshots folder path": "x", "Username": "-oProxyCommand=hackserver"})
        monkeypatch.setattr(getpass, "getuser", lambda: "Graham Kimbell")
        assert _fixup_config({"Snapshots folder path": "x"})["Username"] is None

    @pytest.mark.parametrize(
        "value",
        ["-oProxyCommand=x", ".hidden", "a b", "a@evil.host", "a;b", "a\nb", "a\\b", "$USERNAME", "é", "x" * 33, 123],
    )
    def test_normalise_username_rejects(self, value: object) -> None:
        """Values outside the allowed character set raise."""
        with pytest.raises(ValueError, match="must be 1-32"):
            normalise_username(value)

    def test_normalise_username(self) -> None:
        """Valid usernames are lowercased."""
        assert normalise_username("KiGr") == "kigr"
        assert normalise_username("a.b-c_1") == "a.b-c_1"
        assert normalise_username("x" * 32) == "x" * 32

    def test_convert_legacy_servers(self) -> None:
        """Test converting old style server configs."""
        config = {
            "Servers": [
                {"label": "a", "server_type": "biologic"},
                {"label": "b", "server_type": "biologic"},
            ]
        }
        res = _convert_legacy_servers(config)
        assert res == {
            "a": {"label": "a", "server_type": "biologic"},
            "b": {"label": "b", "server_type": "biologic"},
        }

        config = {
            "Servers": [
                {"label": "a", "server_type": "biologic"},
                {"label": "b", "server_type": "biologic"},
            ],
            "Neware harvester": {
                "Servers": [{"label": "c"}],
            },
            "EC-lab harvester": {
                "Servers": [{"label": "d"}],
            },
        }
        res = _convert_legacy_servers(config)
        assert res == {
            "a": {"label": "a", "server_type": "biologic"},
            "b": {"label": "b", "server_type": "biologic"},
            "c": {"label": "c", "server_type": "neware_harvester"},
            "d": {"label": "d", "server_type": "biologic_harvester"},
        }

        config = {
            "Servers": {
                "a": {"label": "a", "server_type": "neware"},
                "b": {"label": "b", "server_type": "biologic"},
            },
            "Neware harvester": {
                "Servers": [{"label": "a"}],
            },
            "EC-lab harvester": {
                "Servers": [{"label": "b"}],
            },
        }
        res = _convert_legacy_servers(config)
        assert res == {
            "a": {"label": "a", "server_type": "neware"},
            "b": {"label": "b", "server_type": "biologic"},
        }
