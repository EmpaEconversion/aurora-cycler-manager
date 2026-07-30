# Copyright © 2026, Empa.
"""Functions for connecting to instrument servers with SSH."""

import atexit
import base64
import logging
import posixpath
import threading
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path, PureWindowsPath
from time import monotonic, sleep

import paramiko
from typing_extensions import Self

from aurora_cycler_manager.config import get_config

CONFIG = get_config()

logger = logging.getLogger(__name__)

# Close a connection that is not checked-out and idle for this long
_IDLE_TIMEOUT_S = 15 * 60
# Close a connection and force a refresh after this time, if it is not checked-out
_MAX_LIFETIME_S = 12 * 60 * 60
# Check for any idle/max lifetime connections and close
_SWEEP_INTERVAL_S = 60


class _PoolEntry:
    """One SSH client in the _ClientPool, also handles checkout count and lifecycle clocks."""

    def __init__(
        self,
        key: tuple[str, str],
        client: paramiko.SSHClient,
        on_close: Callable[[], None] | None,
    ) -> None:
        """Create an entry that starts checked out once."""
        self.key = key
        self.client = client
        self.on_close = on_close
        self.created_at = monotonic()
        self.last_used = self.created_at
        self.in_use = 1
        self.closed = False


class _ClientPool:
    """Thread-safe cache of live paramiko SSHClients.

    Each key gets its own lock, so connecting under different keys can run in
    parallel, while concurrent connects to the same key are serialised.
    """

    def __init__(self, *, idle_timeout: float, max_lifetime: float) -> None:
        """Initialise empty pool."""
        self._idle_timeout = idle_timeout
        self._max_lifetime = max_lifetime
        self._entries: dict[tuple[str, str], _PoolEntry] = {}
        self._key_locks: dict[tuple[str, str], threading.Lock] = {}
        self._key_locks_guard = threading.Lock()

    def _lock_for(self, key: tuple[str, str]) -> threading.Lock:
        with self._key_locks_guard:
            return self._key_locks.setdefault(key, threading.Lock())

    def _expired(self, entry: _PoolEntry, now: float) -> bool:
        return now - entry.created_at > self._max_lifetime or now - entry.last_used > self._idle_timeout

    def _close_entry(self, entry: _PoolEntry) -> None:
        """Close an entry's client and run its on_close hook."""
        if entry.closed:
            return
        entry.closed = True
        try:
            entry.client.close()
        except Exception:
            logger.exception("Error closing SSH connection %s", entry.key)
        if entry.on_close is not None:
            try:
                entry.on_close()
            except Exception:
                logger.exception("Error in on_close hook for SSH connection %s", entry.key)

    def get(
        self,
        key: tuple[str, str],
        connect: Callable[[], tuple[paramiko.SSHClient, Callable[[], None] | None]],
        *,
        force_reconnect: bool = False,
    ) -> _PoolEntry:
        """Return a checked-out entry for `key`, calling `connect` to (re)create the client if needed.

        `connect` must return (client, on_close); on_close runs once, when the pool closes that
        client. While checked out an entry is never replaced for being idle, past max lifetime,
        or force-reconnected - only a dead transport is replaced regardless.
        Every `get` must have a `release` of the returned entry.
        """
        with self._lock_for(key):
            now = monotonic()
            entry = self._entries.get(key)
            # If the connection exists and doesnt need reconnecting, return it
            if entry is not None:
                transport = entry.client.get_transport()
                alive = transport is not None and transport.is_active()
                busy = entry.in_use > 0
                if alive and (busy or (not force_reconnect and not self._expired(entry, now))):
                    entry.last_used = now
                    entry.in_use += 1
                    return entry
                # The connection exists but needs reconnecting, close it now it is idle
                # If it is busy, the last release will close it
                del self._entries[key]
                if not busy:
                    self._close_entry(entry)
            # (Re)connect to the client and register as a entry in the pool
            client, on_close = connect()
            new_entry = _PoolEntry(key, client, on_close)
            self._entries[key] = new_entry
            return new_entry

    def release(self, entry: _PoolEntry) -> None:
        """Return one checkout of `entry`, refreshing its idle clock."""
        with self._lock_for(entry.key):
            if entry.in_use > 0:
                entry.in_use -= 1
            entry.last_used = monotonic()
            displaced = self._entries.get(entry.key) is not entry
            if displaced and entry.in_use == 0:
                self._close_entry(entry)

    def sweep_expired(self) -> None:
        """Close any cached client past its idle timeout or max lifetime, skipping checked-out ones."""
        with self._key_locks_guard:
            keys = list(self._key_locks)
        now = monotonic()
        for key in keys:
            with self._lock_for(key):
                entry = self._entries.get(key)
                if entry is not None and entry.in_use == 0 and self._expired(entry, now):
                    del self._entries[key]
                    self._close_entry(entry)

    def close_all(self) -> None:
        """Close and forget every cached client, even ones still checked out."""
        with self._key_locks_guard:
            keys = list(self._key_locks)
        for key in keys:
            with self._lock_for(key):
                entry = self._entries.pop(key, None)
                if entry is not None:
                    self._close_entry(entry)


# Shared, reused jump-host connections, keyed by (proxy_hostname, proxy_username)
_jump_pool = _ClientPool(idle_timeout=_IDLE_TIMEOUT_S, max_lifetime=_MAX_LIFETIME_S)
# Shared, reused connections to target machines, keyed by (hostname, username)
_target_pool = _ClientPool(idle_timeout=_IDLE_TIMEOUT_S, max_lifetime=_MAX_LIFETIME_S)


def _sweep_expired_connections() -> None:
    while True:
        sleep(_SWEEP_INTERVAL_S)
        try:
            # Sweep targets first, as they may release jump checkouts
            _target_pool.sweep_expired()
            _jump_pool.sweep_expired()
        except Exception:
            logger.exception("Error sweeping SSH connection pools")


threading.Thread(target=_sweep_expired_connections, daemon=True, name="aurora-ssh-pool-sweeper").start()


def _get_jump_entry(proxy_hostname: str, proxy_username: str, *, force_reconnect: bool = False) -> _PoolEntry:
    """Return a checked-out pool entry for the shared jump connection to the given proxy."""

    def connect() -> tuple[paramiko.SSHClient, None]:
        jump = paramiko.SSHClient()
        jump.load_host_keys(CONFIG["SSH known hosts path"])
        jump.load_system_host_keys()
        jump.set_missing_host_key_policy(paramiko.RejectPolicy())
        jump.connect(
            hostname=proxy_hostname,
            username=proxy_username,
            key_filename=CONFIG.get("SSH private key path"),
        )
        transport = jump.get_transport()
        if transport is None:
            msg = f"Connected to jump host {proxy_hostname} but no transport was created."
            raise paramiko.SSHException(msg)
        transport.set_keepalive(30)
        return jump, None

    return _jump_pool.get((proxy_hostname, proxy_username), connect, force_reconnect=force_reconnect)


def close_all_connections() -> None:
    """Close every cached target and jump-host connection."""
    _target_pool.close_all()
    _jump_pool.close_all()


# Close connections if Python exits
atexit.register(close_all_connections)


def _ps_to_cmd(ps_command: str) -> str:
    """Convert powershell command to command prompt."""
    encoded_ps_command = base64.b64encode(ps_command.encode("utf-16le")).decode("ascii")
    return f"powershell.exe -EncodedCommand {encoded_ps_command}"


def _sftp_mkdir_p(sftp: paramiko.SFTPClient, remote_directory: str) -> None:
    """SFTP mkdir with parents and exist okay."""
    remote_directory = posixpath.normpath(remote_directory)
    current = ""
    for part in remote_directory.split("/"):
        if not part:
            current = "/"
            continue
        current = posixpath.join(current, part)
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


class SSHConnection:
    """Wrapper around paramiko SSHClient."""

    def __init__(self, server: dict) -> None:
        """Store server info."""
        self.server = server
        self.client: paramiko.SSHClient
        self._entry: _PoolEntry | None = None

    def _open_sock(self) -> tuple[paramiko.Channel | None, _PoolEntry | None]:
        """Open a tunnel channel through the shared jump connection, if a proxy is needed."""
        proxy = self.server.get("proxy_hostname")
        if not proxy:
            return None, None

        proxy = proxy.lower()
        proxy_username = self.server.get("proxy_username", self.server["username"]).lower()
        target = (self.server["hostname"].lower(), 22)

        # Shared jump connection may have died since the last check - retry once with a fresh one.
        for force_reconnect in (False, True):
            jump = _get_jump_entry(proxy, proxy_username, force_reconnect=force_reconnect)
            handed_off = False
            try:
                transport = jump.client.get_transport()
                if transport is None:
                    continue
                channel = transport.open_channel("direct-tcpip", target, ("127.0.0.1", 0))
            except (OSError, paramiko.SSHException):
                if force_reconnect:
                    raise
                continue
            else:
                handed_off = True
                return channel, jump
            finally:
                if not handed_off:
                    _jump_pool.release(jump)
        msg = f"Could not establish a transport to jump host {proxy}."
        raise paramiko.SSHException(msg)

    def connect(self) -> Self:
        """Attach a cached, live SSH connection to the target machine, connecting if needed."""
        if self._entry is not None:  # reconnect on a live instance: return the old checkout first
            self.close()
        key = (self.server["hostname"].lower(), self.server["username"].lower())

        def make_client() -> tuple[paramiko.SSHClient, Callable[[], None] | None]:
            sock, jump = self._open_sock()
            handed_off = False
            try:
                client = paramiko.SSHClient()
                client.load_host_keys(CONFIG["SSH known hosts path"])
                client.load_system_host_keys()
                client.set_missing_host_key_policy(paramiko.RejectPolicy())
                client.connect(
                    hostname=self.server["hostname"].lower(),
                    username=self.server["username"].lower(),
                    key_filename=CONFIG.get("SSH private key path"),
                    sock=sock,
                )
                transport = client.get_transport()
                if transport is not None:
                    transport.set_keepalive(30)
                handed_off = True
            finally:
                if not handed_off and jump is not None:
                    _jump_pool.release(jump)
            # The jump checkout now belongs to the pooled target client: the pool releases it
            # when it closes this client, so the jump can't idle-expire under a live tunnel.
            on_close = partial(_jump_pool.release, jump) if jump is not None else None
            return client, on_close

        self._entry = _target_pool.get(key, make_client)
        self.client = self._entry.client
        return self

    def close(self) -> None:
        """Release this connection's pool checkout; the connection stays cached for reuse."""
        if self._entry is not None:
            _target_pool.release(self._entry)
            self._entry = None

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Context manager exit."""
        self.close()

    def get_files(self, local_files: list[Path], remote_files: list[str], *, missing_ok: bool = False) -> None:
        """Copy the files across with SFTP."""
        with self.client.open_sftp() as sftp:
            for remote_file, local_file in zip(remote_files, local_files, strict=True):
                local_file.parent.mkdir(parents=True, exist_ok=True)
                logger.info("Downloading file %s to %s", remote_file, local_file)
                try:
                    sftp.get(remote_file, str(local_file))
                except FileNotFoundError:
                    if missing_ok:
                        logger.warning("Remote file not found: %s", remote_file)
                    else:
                        raise

    def put_file(self, local_path: str | Path, remote_path: str | Path | PureWindowsPath) -> None:
        """Send file to Windows PC."""
        remote_path = PureWindowsPath(remote_path)
        with self.client.open_sftp() as sftp:
            _sftp_mkdir_p(sftp, remote_path.parent.as_posix())
            sftp.put(str(local_path), str(remote_path.as_posix()))

    def check_new_files(
        self,
        remote_folder: str,
        extensions: list,
        since_uts: float,
    ) -> list[str]:
        """Get list of modified files from Windows PC."""
        # Cannot use timezone or ISO8061 - not supported in PowerShell 5.1
        cutoff_date_str = datetime.fromtimestamp(since_uts).strftime("%Y-%m-%d %H:%M:%S")  # noqa: DTZ006
        extensions_str = ",".join([f"'{e}'" for e in extensions])
        command = (
            f"Get-ChildItem -Path '{remote_folder}' -Recurse "
            f"| Where-Object {{ $_.LastWriteTime -gt '{cutoff_date_str}' -and ($_.Extension -in {extensions_str})}} "
            "| Select-Object -ExpandProperty FullName"
        )
        _stdin, stdout, stderr = self.exec_command(command)
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            msg = f"Command failed with exit status {exit_status}: {stderr.read().decode('utf-8', errors='replace')}"
            raise RuntimeError(msg)
        output = stdout.read().decode("utf-8", errors="replace").strip()
        modified_files = output.splitlines()
        logger.info("Found %d modified files since %s", len(modified_files), cutoff_date_str)
        return modified_files

    def exec_command(
        self,
        ps_command: str,
        **kwargs,  # noqa: ANN003
    ) -> tuple[paramiko.ChannelFile, paramiko.ChannelFile, paramiko.ChannelFile]:
        """Execute Powershell command, convert to cmd automatically if needed."""
        return self.client.exec_command(self._normalise_command(ps_command), **kwargs)

    def _normalise_command(self, ps_command: str) -> str:
        """Normalises command to work on either powershell or cmd."""
        if self.server["shell_type"] == "powershell":
            return ps_command
        if self.server["shell_type"] == "cmd":
            return _ps_to_cmd(ps_command)
        msg = f"Unsupported shell type '{self.server['shell_type']}' for server {self.server['label']}."
        raise ValueError(msg)
