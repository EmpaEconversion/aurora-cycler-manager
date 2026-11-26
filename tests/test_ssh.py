# Copyright © 2026, Empa.
"""Tests for the ssh module."""

import pytest

from aurora_cycler_manager import ssh
from aurora_cycler_manager.ssh import _resolve_username


class TestResolveUsername:
    """Test the $USERNAME sentinel in SSH usernames."""

    @pytest.fixture(autouse=True)
    def _username(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(ssh.CONFIG, "Username", "kigr")

    def test_no_sentinel(self) -> None:
        """Other usernames pass through unchanged."""
        assert _resolve_username("labuser") == "labuser"
        assert _resolve_username("DOMAIN\\labuser") == "DOMAIN\\labuser"
        assert _resolve_username("user@domain.com") == "user@domain.com"

    def test_sentinel(self) -> None:
        """The sentinel is replaced by the configured username."""
        assert _resolve_username("$USERNAME") == "kigr"

    def test_sentinel_lowercased(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The configured username is lowercased."""
        monkeypatch.setitem(ssh.CONFIG, "Username", "KiGr")
        assert _resolve_username("$USERNAME") == "kigr"

    def test_sentinel_must_be_whole_field(self) -> None:
        """The sentinel is not substituted inside longer strings."""
        assert _resolve_username("x-$USERNAME") == "x-$USERNAME"
        assert _resolve_username("$username") == "$username"

    @pytest.mark.parametrize(
        "value",
        ["-oProxyCommand=x", ".hidden", "a b", "a@evil.host", "a;b", "a\nb", "a\\b", "$USERNAME", "é", "x" * 33, 123],
    )
    def test_bad_values(self, monkeypatch: pytest.MonkeyPatch, value: object) -> None:
        """Values outside the allowed character set raise."""
        monkeypatch.setitem(ssh.CONFIG, "Username", value)
        with pytest.raises(ValueError, match="must be 1-32"):
            _resolve_username("$USERNAME")

    @pytest.mark.parametrize("value", [None, ""])
    def test_missing(self, monkeypatch: pytest.MonkeyPatch, value: object) -> None:
        """The sentinel raises if no username is configured."""
        monkeypatch.setitem(ssh.CONFIG, "Username", value)
        assert _resolve_username("labuser") == "labuser"
        with pytest.raises(ValueError, match="not set"):
            _resolve_username("$USERNAME")
