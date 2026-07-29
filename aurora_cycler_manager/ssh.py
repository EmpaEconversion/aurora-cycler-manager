# Copyright © 2026, Empa.
"""Functions for connecting to instrument servers with SSH."""

import atexit
import base64
import logging
import posixpath
import threading
from datetime import datetime
from pathlib import Path, PureWindowsPath

import paramiko
from typing_extensions import Self

from aurora_cycler_manager.config import get_config

CONFIG = get_config()

logger = logging.getLogger(__name__)

# Cache of shared, reused jump-host connections, keyed by (proxy_hostname, proxy_username)
_jump_clients: dict[tuple[str, str], paramiko.SSHClient] = {}
_jump_clients_lock = threading.Lock()


def _get_jump_client(proxy_hostname: str, proxy_username: str, *, force_reconnect: bool = False) -> paramiko.SSHClient:
    """Return a cached, connected jump SSHClient for the given proxy, reconnecting if needed."""
    key = (proxy_hostname, proxy_username)
    with _jump_clients_lock:
        jump = _jump_clients.get(key)
        transport = jump.get_transport() if jump else None
        if jump is not None and not force_reconnect and transport is not None and transport.is_active():
            return jump
        if jump is not None:
            jump.close()
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
        _jump_clients[key] = jump
        return jump


def close_jump_clients() -> None:
    """Close all cached jump-host connections."""
    with _jump_clients_lock:
        for jump in _jump_clients.values():
            jump.close()
        _jump_clients.clear()


# Close connections if Python exits
atexit.register(close_jump_clients)


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
        """Establish SSH connection."""
        self.client = paramiko.SSHClient()
        self.client.load_host_keys(CONFIG["SSH known hosts path"])
        self.client.load_system_host_keys()
        self.client.set_missing_host_key_policy(paramiko.RejectPolicy())
        self.client.connect(
            hostname=self.server["hostname"].lower(),
            username=self.server["username"].lower(),
            key_filename=CONFIG.get("SSH private key path"),
            sock=self.get_sock(),
        )
        return self

    def close(self) -> None:
        """Close SSH connection."""
        if self.client:
            self.client.close()

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
