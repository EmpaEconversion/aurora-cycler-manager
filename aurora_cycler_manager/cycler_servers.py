"""Copyright © 2025, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Server classes used by server_manager, currently only tomato servers are implemented.

Unlike the harvester modules, which can only download the latest data, cycler servers can be used to
interact with the server directly, e.g. to submit a job or get the status of a pipeline.

These classes are used by server_manager.
"""

from __future__ import annotations

import base64
import json
import warnings
from pathlib import Path

import paramiko
from scp import SCPClient


class CyclerServer:
    """Base class for server objects, should not be instantiated directly."""

    def __init__(self, server_config: dict, local_private_key_path: str | Path) -> None:
        """Initialise server object."""
        self.label = server_config["label"]
        self.hostname = server_config["hostname"]
        self.username = server_config["username"]
        self.server_type = server_config["server_type"]
        self.shell_type = server_config.get("shell_type", "")
        self.command_prefix = server_config.get("command_prefix", "")
        self.command_suffix = server_config.get("command_suffix", "")
        self.local_private_key = paramiko.RSAKey.from_private_key_file(local_private_key_path)
        self.last_status = None
        self.last_queue = None
        self.last_queue_all = None
        self.check_connection()

    def command(self, command: str) -> str:
        """Send a command to the server and return the output.

        The command is prefixed with the command_prefix specified in the server_config, is run on
        the server's default shell, the standard output is returned as a string.
        """
        with paramiko.SSHClient() as ssh:
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.hostname, username=self.username, pkey=self.local_private_key)
            stdin, stdout, stderr = ssh.exec_command(self.command_prefix + command + self.command_suffix)
            output = stdout.read().decode("utf-8")
            error = stderr.read().decode("utf-8")
        if error:
            if "Error" in error:
                print(f"Error running '{command}' on {self.label}")
                raise ValueError(error)
            if error.startswith("WARNING"):
                warnings.warn(error, RuntimeWarning, stacklevel=2)
            else:
                print(f"Error running '{command}' on {self.label}")
                raise ValueError(error)
        return output

    def check_connection(self) -> bool:
        """Check if the server is reachable by running a simple command.

        Returns:
            bool: True if the server is reachable

        Raises:
            ValueError: If the server is unreachable

        """
        test_phrase = "hellothere"
        output = self.command(f"echo {test_phrase}").strip()
        if output != test_phrase:
            msg = f"Connection error, expected output '{test_phrase}', got '{output}'"
            raise ValueError(msg)
        print(f"Succesfully connected to {self.label}")
        return True

    def eject(self, pipeline: str) -> str:
        """Remove a sample from a pipeline."""
        raise NotImplementedError

    def load(self, sample: str, pipeline: str) -> str:
        """Load a sample into a pipeline."""
        raise NotImplementedError

    def ready(self, pipeline: str) -> str:
        """Ready a pipeline for use."""
        raise NotImplementedError

    def submit(self, sample: str, capacity_Ah: float, payload: str | dict) -> str:
        """Submit a job to the server."""
        raise NotImplementedError

    def cancel(self, job_id_on_server: str) -> str:
        """Cancel a job on the server."""
        raise NotImplementedError

    def get_pipelines(self) -> dict:
        """Get the status of all pipelines on the server."""
        raise NotImplementedError

    def get_jobs(self) -> dict:
        """Get all jobs from server."""
        raise NotImplementedError

    def snapshot(
            self,
            jobid: str,
            jobid_on_server: str,
            local_save_location: str,
            get_raw: bool,
        ) -> str:
        """Save a snapshot of a job on the server and download it to the local machine."""
        raise NotImplementedError

class TomatoServer(CyclerServer):
    """Server class for Tomato servers, implements all the methods in CyclerServer.

    Used by server_manager to interact with Tomato servers, should not be instantiated directly.

    Attributes:
        save_location (str): The location on the server where snapshots are saved.

    """

    def __init__(self, server_config: dict, local_private_key_path: str | Path) -> None:
        """Initialise server object."""
        super().__init__(server_config, local_private_key_path)
        self.tomato_scripts_path = server_config.get("tomato_scripts_path")
        self.save_location = "C:/tomato/aurora_scratch"
        self.tomato_data_path = server_config.get("tomato_data_path")

    def eject(self, pipeline: str) -> str:
        """Eject any sample from the pipeline."""
        return self.command(f"{self.tomato_scripts_path}ketchup eject {pipeline}")


    def load(self, sample: str, pipeline: str) -> str:
        """Load a sample into a pipeline."""
        return self.command(f"{self.tomato_scripts_path}ketchup load {sample} {pipeline}")


    def ready(self, pipeline: str) -> str:
        """Ready a pipeline for use."""
        return self.command(f"{self.tomato_scripts_path}ketchup ready {pipeline}")

    def unready(self, pipeline: str) -> str:
        """Unready a pipeline - only works if no job submitted yet, otherwise use cancel."""
        return self.command(f"{self.tomato_scripts_path}ketchup unready {pipeline}")

    def submit(
            self,
            sample: str,
            capacity_Ah: float,
            payload: str | dict,
            send_file: bool = False,
        ) -> str:
        """Submit a job to the server.

        Args:
            sample (str): The name of the sample to be tested
            capacity_Ah (float): The capacity of the sample in Ah
            payload (str | dict): The JSON payload to be submitted, can include '$NAME' which is
                replaced with the actual sample ID
            send_file (bool): If True, the payload is written to a file and sent to the server

        Returns:
            str: The jobid of the submitted job with the server prefix
            str: The jobid of the submitted job on the server (without the prefix)
            str: The JSON string of the submitted payload

        """
        # Check if json_file is a string that could be a file path or a JSON string
        if isinstance(payload, str):
            try:
                # Attempt to load json_file as JSON string
                payload = json.loads(payload)
            except json.JSONDecodeError:
                with Path(payload).open(encoding="utf-8") as f:
                    payload = json.load(f)
        # If json_file is already a dictionary, use it directly
        elif not isinstance(payload, dict):
            msg = "json_file must be a file path, a JSON string, or a dictionary"
            raise TypeError(msg)

        # Add the sample name and capacity to the payload
        payload["sample"]["name"] = sample
        payload["sample"]["capacity"] = capacity_Ah
        # Convert the payload to a json string
        json_string = json.dumps(payload)
        # Change all other instances of $NAME to the sample name
        json_string = json_string.replace("$NAME", sample)

        if send_file: # Write the json string to a file, send it, run it on the server
            # Write file locally
            with Path("temp.json").open("w", encoding="utf-8") as f:
                f.write(json_string)

            # Send file to server
            ssh = paramiko.SSHClient()
            ssh.load_system_host_keys()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.hostname, username=self.username, pkey=self.local_private_key)
            with SCPClient(ssh.get_transport(), socket_timeout=120) as scp:
                scp.put("temp.json", f"{self.save_location}/temp.json")
            ssh.close()

            # Submit the file on the server
            output = self.command(f"{self.tomato_scripts_path}ketchup submit {self.save_location}/temp.json")

            # Remove the file locally
            Path("temp.json").unlink()

        else: # Encode the json string to base64 and submit it directly
            encoded_json_string = base64.b64encode(json_string.encode()).decode()
            output = self.command(f"{self.tomato_scripts_path}ketchup submit -J {encoded_json_string}")
        if "jobid: " in output:
            jobid = output.split("jobid: ")[1].splitlines[0]
            print(f"Sample {sample} submitted on server {self.label} with jobid {jobid}")
            full_jobid = f"{self.label}-{jobid}"
            print(f"Full jobid: {full_jobid}")
            return full_jobid, jobid, json_string

        msg = f"Error submitting job: {output}"
        raise ValueError(msg)

    def cancel(self, job_id_on_server: str) -> str:
        """Cancel a job on the server."""
        return self.command(f"{self.tomato_scripts_path}ketchup cancel {job_id_on_server}")

    def get_pipelines(self) -> dict:
        """Get the status of all pipelines on the server."""
        output = self.command(f"{self.tomato_scripts_path}ketchup status -J")
        status_dict = json.loads(output)
        self.last_status = status_dict
        return status_dict

    def get_queue(self) -> dict:
        """Get running and queued jobs from server."""
        output = self.command(f"{self.tomato_scripts_path}ketchup status queue -J")
        queue_dict = json.loads(output)
        self.last_queue = queue_dict
        return queue_dict

    def get_jobs(self) -> dict:
        """Get all jobs from server."""
        output = self.command(f"{self.tomato_scripts_path}ketchup status queue -v -J")
        queue_all_dict = json.loads(output)
        self.last_queue_all = queue_all_dict
        return queue_all_dict

    def snapshot(
            self,
            jobid: str,
            jobid_on_server: str,
            local_save_location: str,
            get_raw: bool = False,
        ) -> str:
        """Save a snapshot of a job on the server and download it to the local machine.

        Args:
            jobid (str): The jobid of the job on the local machine
            jobid_on_server (str): The jobid of the job on the server
            local_save_location (str): The directory to save the snapshot data to
            get_raw (bool): If True, download the raw data as well as the snapshot data

        Returns:
            str: The status of the snapshot (e.g. "c", "r", "ce", "cd")

        """
        # Save a snapshot on the remote machine
        remote_save_location = f"{self.save_location}/{jobid_on_server}"
        if self.shell_type == "powershell":
            self.command(
                f'if (!(Test-Path "{remote_save_location}")) '
                f'{{ New-Item -ItemType Directory -Path "{remote_save_location}" }}',
            )
        elif self.shell_type == "cmd":
            self.command(
                f'if not exist "{remote_save_location}" '
                f'mkdir "{remote_save_location}"',
            )
        else:
            msg = "Shell type not recognised, must be 'powershell' or 'cmd', check config.json"
            raise ValueError(msg)
        output = self.command(f"{self.tomato_scripts_path}ketchup status -J {jobid_on_server}")
        print(f"Got job status on remote server {self.label}")
        json_output = json.loads(output)
        snapshot_status = json_output["status"][0]
        # Catch errors
        try:
            with warnings.catch_warnings(record=True) as w:
                if self.shell_type == "powershell":
                    self.command(
                        f"cd {remote_save_location} ; "
                        f"{self.tomato_scripts_path}ketchup snapshot {jobid_on_server}",
                    )
                elif self.shell_type == "cmd":
                    self.command(
                        f"cd {remote_save_location} && "
                        f"{self.tomato_scripts_path}ketchup snapshot {jobid_on_server}",
                    )
                for warning in w:
                    if "out-of-date version" in str(warning.message) or "has been completed" in str(warning.message):
                        continue  # Ignore these warnings
                    print(f"Warning: {warning.message}")
        except ValueError as e:
            emsg = str(e)
            if "AssertionError" in emsg and "os.path.isdir(jobdir)" in emsg:
                raise FileNotFoundError from e
            raise
        print(f"Snapshotted file on remote server {self.label}")
        # Get local directory to save the snapshot data
        if not Path(local_save_location).exists():
            Path(local_save_location).mkdir(parents=True)

        # Use SCPClient to transfer the file from the remote machine
        ssh = paramiko.SSHClient()
        ssh.load_system_host_keys()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        print(f"Connecting to {self.label}: host {self.hostname} user {self.username}")
        ssh.connect(self.hostname, username=self.username, pkey=self.local_private_key)
        try:
            print(
                f"Downloading file {remote_save_location}/snapshot.{jobid_on_server}.json to "
                f"{local_save_location}/snapshot.{jobid}.json",
            )
            with SCPClient(ssh.get_transport(), socket_timeout=120) as scp:
                scp.get(
                    f"{remote_save_location}/snapshot.{jobid_on_server}.json",
                    f"{local_save_location}/snapshot.{jobid}.json",
                )
                if get_raw:
                    print("Downloading snapshot raw data")
                    scp.get(
                        f"{remote_save_location}/snapshot.{jobid_on_server}.zip",
                        f"{local_save_location}/snapshot.{jobid}.zip",
                    )
        finally:
            ssh.close()

        return snapshot_status

    def get_last_data(self, job_id_on_server: int) -> tuple[str,dict]:
        """Get the last data from a job snapshot.

        Args:
            job_id_on_server : str
                The job ID on the server as an integer

        Returns:
            str: the file name of the latest data json file
            dict: the latest data

        """
        if not self.tomato_data_path:
            msg = "tomato_data_path not set for this server in config file"
            raise ValueError(msg)

        # get the last data file in the job folder and read out the json string
        ps_command = (
            f"$file = Get-ChildItem -Path '{self.tomato_data_path}\\{job_id_on_server}' -Filter 'MPG2*data.json' "
            f"| Sort-Object LastWriteTime -Descending "
            f"| Select-Object -First 1; "
            f"if ($file) {{ Write-Output $file.FullName; Get-Content $file.FullName }}"
        )
        if self.shell_type not in ["powershell", "cmd"]:
            msg = "Shell type not recognised, must be 'powershell' or 'cmd'"
            raise ValueError(msg)
        if self.shell_type == "powershell":
            command = ps_command
        elif self.shell_type == "cmd":
            command = (
                f'powershell.exe -Command "{ps_command}"'
            )

        with paramiko.SSHClient() as ssh:
            ssh.load_system_host_keys()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.hostname, username=self.username, pkey=self.local_private_key)
            stdin, stdout, stderr = ssh.exec_command(command)
            if stderr.read():
                raise ValueError(stderr.read())
        file_name = stdout.readline().strip()
        file_content = stdout.readline().strip()
        file_content_json = json.loads(file_content)
        return file_name, file_content_json

    def get_job_data(self, jobid_on_server: int) -> dict:
        """Get the jobdata dict for a job."""
        if not self.tomato_data_path:
            msg = "tomato_data_path not set for this server in config file"
            raise ValueError(msg)
        ps_command = (
            f"if (Test-Path -Path '{self.tomato_data_path}\\{jobid_on_server}\\jobdata.json') {{ "
            f"Get-Content '{self.tomato_data_path}\\{jobid_on_server}\\jobdata.json' "
            f"}} else {{ "
            f"Write-Output 'File not found.' "
            f"}}"
        )
        if self.shell_type not in ["powershell", "cmd"]:
            msg = "Shell type not recognised, must be 'powershell' or 'cmd'"
            raise ValueError(msg)
        if self.shell_type == "powershell":
            command = ps_command
        elif self.shell_type == "cmd":
            command = (
                f'powershell.exe -Command "{ps_command}"'
            )
        with paramiko.SSHClient() as ssh:
            ssh.load_system_host_keys()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.hostname, username=self.username, pkey=self.local_private_key)
            stdin, stdout, stderr = ssh.exec_command(command)
            stdout = stdout.read().decode("utf-8")
            stderr = stderr.read().decode("utf-8")
        if stderr:
            raise ValueError(stderr)
        if "File not found." in stdout:
            msg = f"jobdata.json not found for job {jobid_on_server}"
            raise FileNotFoundError(msg)
        return json.loads(stdout)
