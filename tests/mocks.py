"""Testing ssh module."""

from pathlib import Path
from unittest.mock import Mock


class MockSSHClient:
    """Mock ssh.SSHClient with configurable command responses."""

    def __init__(self) -> None:
        """Initialize."""
        self.responses: dict[str, dict] = {}
        self.sftp_files: dict[str, bytes] = {}
        self.connected = False

    def add_command_response(self, command: str, stdout: str = "", stderr: str = "", exit_code: int = 0) -> None:
        """Add a command response."""
        self.responses[command] = {"stdout": stdout, "stderr": stderr, "exit_code": exit_code}

    def add_sftp_file(self, remote_path: str, content: bytes) -> None:
        """Add a file that can be downloaded with SFTP."""
        self.sftp_files[remote_path] = content

    def load_host_keys(self, path) -> None:
        """Mock load_host_keys."""

    def load_system_host_keys(self) -> None:
        """Mock load_system_host_keys."""

    def set_missing_host_key_policy(self, policy) -> None:
        """Mock set_missing_host_key_policy."""

    def connect(self, hostname, username, key_filename=None) -> None:
        """Mock connect."""
        self.connected = True

    def close(self) -> None:
        """Mock close."""
        self.connected = False

    def exec_command(self, command: str, **kwargs) -> tuple[Mock, Mock, Mock]:  # noqa: ANN003
        """Mock exec_command with configured responses."""
        # Find matching response (exact match or contains)
        response = None
        for cmd_pattern, resp in self.responses.items():
            if cmd_pattern in command or command == cmd_pattern:
                response = resp
                break

        if response is None:
            # Default response for unmocked commands
            response = {"stdout": "", "stderr": f"Unmocked command: {command}", "exit_code": 1}

        # Create mock stdin, stdout, stderr
        stdin = Mock()

        stdout = Mock()
        stdout.read.return_value = response["stdout"].encode("utf-8")
        stdout.channel.recv_exit_status.return_value = response["exit_code"]

        stderr = Mock()
        stderr.read.return_value = response["stderr"].encode("utf-8")

        return stdin, stdout, stderr

    def open_sftp(self) -> Mock:
        """Mock SFTP context manager."""
        sftp = Mock()

        def mock_get(remote_path, local_path) -> None:
            """Get a file from fake server."""
            if remote_path in self.sftp_files:
                Path(local_path).write_bytes(self.sftp_files[remote_path])
            else:
                msg = f"Mock file not found: {remote_path}"
                raise FileNotFoundError(msg)

        def mock_put(local_path, remote_path) -> None:
            """Put a file on fake server."""
            self.sftp_files[remote_path] = Path(local_path).read_bytes()

        sftp.get = mock_get
        sftp.put = mock_put
        sftp.mkdir = Mock()
        sftp.__enter__ = Mock(return_value=sftp)
        sftp.__exit__ = Mock(return_value=False)

        return sftp
