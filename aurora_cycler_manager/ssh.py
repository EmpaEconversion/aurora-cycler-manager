"""Copyright © 2026, Empa.

Functions for connecting to instruments with SSH.
"""

import base64
import logging
from contextlib import suppress
from datetime import datetime
from pathlib import Path, PureWindowsPath

import paramiko

from aurora_cycler_manager.config import get_config

CONFIG = get_config()

logger = logging.getLogger(__name__)


def ssh_connect(ssh: paramiko.SSHClient, server: dict) -> None:
    """Connect via SSH."""
    ssh.load_host_keys(CONFIG["SSH known hosts path"])
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
    # Paramiko is case sensitive
    ssh.connect(
        hostname=server["hostname"].lower(),
        username=server["username"].lower(),
        key_filename=CONFIG.get("SSH private key path"),
    )


def _ps_to_cmd(ps_command: str) -> str:
    """Convert powershell command to command prompt."""
    encoded_ps_command = base64.b64encode(ps_command.encode("utf-16le")).decode("ascii")
    return f"powershell.exe -EncodedCommand {encoded_ps_command}"


def normalise_command(server: dict, ps_command: str) -> str:
    """Normalises command to work on either powershell or cmd."""
    if server["shell_type"] == "powershell":
        return ps_command
    if server["shell_type"] == "cmd":
        return _ps_to_cmd(ps_command)
    msg = f"Unsupported shell type '{server['shell_type']}' for server {server['label']}."
    raise ValueError(msg)


def check_new_files(
    ssh: paramiko.SSHClient,
    server: dict,
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
    _stdin, stdout, stderr = ssh.exec_command(normalise_command(server, command))
    exit_status = stdout.channel.recv_exit_status()
    if exit_status != 0:
        msg = f"Command failed with exit status {exit_status}: {stderr.read().decode('utf-8')}"
        raise RuntimeError(msg)
    output = stdout.read().decode("utf-8").strip()
    modified_files = output.splitlines()
    logger.info("Found %d modified files since %s", len(modified_files), cutoff_date_str)
    return modified_files


def put_file(ssh: paramiko.SSHClient, local_path: str | Path, remote_path: str | Path | PureWindowsPath) -> None:
    """Send file to Windows PC."""
    remote_path = PureWindowsPath(remote_path)
    with ssh.open_sftp() as sftp:
        with suppress(IOError):  # Already exists
            sftp.mkdir(str(remote_path.parent))
        sftp.put(str(local_path), str(remote_path))


def get_files(ssh: paramiko.SSHClient, local_files: list[Path], remote_files: list[str]) -> None:
    """Copy the files across with SFTP."""
    with ssh.open_sftp() as sftp:
        for remote_file, local_file in zip(remote_files, local_files, strict=True):
            local_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading file %s to %s", remote_file, local_file)
            sftp.get(remote_file, str(local_file))
