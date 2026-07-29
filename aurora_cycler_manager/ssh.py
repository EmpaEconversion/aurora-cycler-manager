# Copyright © 2026, Empa.
"""Functions for connecting to instrument servers with SSH."""

import atexit
import base64
import logging
import posixpath
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path, PureWindowsPath
from time import monotonic, sleep

import paramiko
from typing_extensions import Self

from aurora_cycler_manager.config import get_config

CONFIG = get_config()

logger = logging.getLogger(__name__)

# If a connection does nothing for this long, close it
_IDLE_TIMEOUT_S = 15 * 60
# Always close a connection and force a refresh after this time
_MAX_LIFETIME_S = 12 * 60 * 60
# Check for any idle/max lifetime connections and close
_SWEEP_INTERVAL_S = 60


class _ClientPool:
    """Thread-safe cache of live paramiko SSHClients.

    Each key gets its own lock, so connecting under different keys can run in
    parallel, while concurrent connects to the same key are serialised.
    """

    def __init__(self, *, idle_timeout: float, max_lifetime: float) -> None:
        """Initialise empty pool."""
        self._idle_timeout = idle_timeout
        self._max_lifetime = max_lifetime
        self._clients: dict[tuple, paramiko.SSHClient] = {}
        self._created_at: dict[tuple, float] = {}
        self._last_used: dict[tuple, float] = {}
        self._key_locks: dict[tuple, threading.Lock] = {}
        self._key_locks_guard = threading.Lock()

    def _lock_for(self, key: tuple) -> threading.Lock:
        with self._key_locks_guard:
            return self._key_locks.setdefault(key, threading.Lock())

    def _is_expired(self, key: tuple, now: float) -> bool:
        return now - self._created_at[key] > self._max_lifetime or now - self._last_used[key] > self._idle_timeout

    def get(
        self,
        key: tuple,
        connect: Callable[[], paramiko.SSHClient],
        *,
        force_reconnect: bool = False,
    ) -> paramiko.SSHClient:
        """Return a cached, live client for `key`, calling `connect` to (re)create it if needed."""
        with self._lock_for(key):
            now = monotonic()
            client = self._clients.get(key)
            transport = client.get_transport() if client else None
            if (
                client is not None
                and not force_reconnect
                and transport is not None
                and transport.is_active()
                and not self._is_expired(key, now)
            ):
                self._last_used[key] = now
                return client
            if client is not None:
                client.close()
            client = connect()
            self._clients[key] = client
            self._created_at[key] = now
            self._last_used[key] = now
            return client

    def sweep_expired(self) -> None:
        """Proactively close any cached client past its idle timeout or max lifetime."""
        with self._key_locks_guard:
            keys = list(self._key_locks)
        for key in keys:
            with self._lock_for(key):
                client = self._clients.get(key)
                if client is not None and self._is_expired(key, monotonic()):
                    del self._clients[key]
                    del self._created_at[key]
                    del self._last_used[key]
                    client.close()

    def close_all(self) -> None:
        """Close and forget every cached client."""
        with self._key_locks_guard:
            for key, lock in self._key_locks.items():
                with lock:
                    client = self._clients.pop(key, None)
                    self._created_at.pop(key, None)
                    self._last_used.pop(key, None)
                    if client is not None:
                        client.close()


# Shared, reused jump-host connections, keyed by (proxy_hostname, proxy_username)
_jump_pool = _ClientPool(idle_timeout=_IDLE_TIMEOUT_S, max_lifetime=_MAX_LIFETIME_S)
# Shared, reused connections to target machines, keyed by (hostname, username)
_target_pool = _ClientPool(idle_timeout=_IDLE_TIMEOUT_S, max_lifetime=_MAX_LIFETIME_S)


def _sweep_expired_connections() -> None:
    while True:
        sleep(_SWEEP_INTERVAL_S)
        _jump_pool.sweep_expired()
        _target_pool.sweep_expired()


threading.Thread(target=_sweep_expired_connections, daemon=True, name="aurora-ssh-pool-sweeper").start()


def _get_jump_client(proxy_hostname: str, proxy_username: str, *, force_reconnect: bool = False) -> paramiko.SSHClient:
    """Return a cached, connected jump SSHClient for the given proxy, reconnecting if needed."""

    def connect() -> paramiko.SSHClient:
        jump = paramiko.SSHClient()
        jump.load_system_host_keys()
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
        return jump

    return _jump_pool.get((proxy_hostname, proxy_username), connect, force_reconnect=force_reconnect)


def close_all_connections() -> None:
    """Close every cached jump-host and target connection."""
    _jump_pool.close_all()
    _target_pool.close_all()


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

    def get_sock(self) -> paramiko.Channel | None:
        """Return a tunnel channel through the shared jump connection, if a proxy is needed."""
        proxy = self.server.get("proxy_hostname")
        if not proxy:
            return None

        proxy = proxy.lower()
        proxy_username = self.server.get("proxy_username", self.server["username"]).lower()
        target = (self.server["hostname"].lower(), 22)

        # Shared jump connection may have died since the last check - retry once with a fresh one.
        for force_reconnect in (False, True):
            jump = _get_jump_client(proxy, proxy_username, force_reconnect=force_reconnect)
            transport = jump.get_transport()
            if transport is None:
                continue
            try:
                return transport.open_channel("direct-tcpip", target, ("127.0.0.1", 0))
            except (OSError, paramiko.SSHException):
                if force_reconnect:
                    raise
        msg = f"Could not establish a transport to jump host {proxy}."
        raise paramiko.SSHException(msg)

    def connect(self) -> Self:
        """Attach a cached, live SSH connection to the target machine, connecting if needed."""
        key = (self.server["hostname"].lower(), self.server["username"].lower())

        def make_client() -> paramiko.SSHClient:
            client = paramiko.SSHClient()
            client.load_host_keys(CONFIG["SSH known hosts path"])
            client.load_system_host_keys()
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
            client.connect(
                hostname=self.server["hostname"].lower(),
                username=self.server["username"].lower(),
                key_filename=CONFIG.get("SSH private key path"),
                sock=self.get_sock(),
            )
            transport = client.get_transport()
            if transport is not None:
                transport.set_keepalive(30)
            return client

        self.client = _target_pool.get(key, make_client)
        return self

    def close(self) -> None:
        """Do nothing, the underlying connection is managed by the _ClientPool.

        Use close_all_connections() to actually tear down cached connections.
        This function is kept for backwards compatibility.
        """

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
