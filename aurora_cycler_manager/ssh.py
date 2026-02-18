"""Copyright © 2026, Empa.

Functions for connecting to instrument servers with SSH.
"""

import base64
import logging
import posixpath
from datetime import datetime
from pathlib import Path, PureWindowsPath

import paramiko
from typing_extensions import Self

from aurora_cycler_manager.config import get_config

CONFIG = get_config()

logger = logging.getLogger(__name__)


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
        assert isinstance(self.client, paramiko.SSHClient)  # noqa: S101
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
        assert isinstance(self.client, paramiko.SSHClient)  # noqa: S101
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
        assert isinstance(self.client, paramiko.SSHClient)  # noqa: S101
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
