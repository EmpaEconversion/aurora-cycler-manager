"""Copyright © 2025, Empa.

Server classes used by server_manager, including:
- Neware server, designed for Neware BTS 8.0 with aurora-neware CLI
- Biologic server, designed for Biologic EC-lab with aurora-biologic CLI
- Tomato server, designed for tomato 0.2.3

Unlike the harvester modules, which can only download the latest data, cycler
servers can be used to interact with the server directly, e.g. to submit a job
or get the status of a pipeline.

These classes are used by server_manager.
"""

import base64
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path, PureWindowsPath

import paramiko
from aurora_unicycler import Protocol
from scp import SCPClient

from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.eclab_harvester import convert_mpr, get_eclab_snapshot_folder
from aurora_cycler_manager.neware_harvester import convert_neware_data, snapshot_raw_data
from aurora_cycler_manager.tomato_converter import convert_tomato_json, get_tomato_snapshot_folder, puree_tomato
from aurora_cycler_manager.utils import run_from_sample

logger = logging.getLogger(__name__)
CONFIG = get_config()


class CyclerServer:
    """Base class for server objects, should not be instantiated directly."""

    def __init__(self, server_config: dict) -> None:
        """Initialise server object."""
        self.label = server_config["label"]
        self.hostname = server_config["hostname"]
        self.username = server_config["username"]
        self.server_type = server_config["server_type"]
        self.shell_type = server_config.get("shell_type", "")
        self.command_prefix = server_config.get("command_prefix", "")
        self.command_suffix = server_config.get("command_suffix", "")
        self.last_status = None
        self.last_queue = None
        self.last_queue_all = None
        self.check_connection()

    def command(self, command: str, timeout: float = 300) -> str:
        """Send a command to the server and return the output.

        The command is prefixed with the command_prefix specified in the server_config, is run on
        the server's default shell, the standard output is returned as a string.
        """
        with paramiko.SSHClient() as ssh:
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.hostname, username=self.username, key_filename=CONFIG.get("SSH private key path"))
            stdin, stdout, stderr = ssh.exec_command(
                self.command_prefix + command + self.command_suffix,
                timeout=timeout,
            )
            output = stdout.read().decode("utf-8").strip()
            error = stderr.read().decode("utf-8").strip()
            exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            logger.error("Command '%s' on %s failed with exit status %d", command, self.label, exit_status)
            logger.error("Error: %s", error)
            raise ValueError(f"Command failed with exit status {exit_status}: {error}")
        if error:
            logger.warning("Command completed with warnings running '%s' on %s: %s", command, self.label, error)
        return output

    def check_connection(self) -> bool:
        """Check if the server is reachable by running a simple command.

        Returns:
            bool: True if the server is reachable

        Raises:
            ValueError: If the server is unreachable

        """
        test_phrase = "hellothere"
        output = self.command(f"echo {test_phrase}", timeout=5).strip()
        if output != test_phrase:
            msg = f"Connection error, expected output '{test_phrase}', got '{output}'"
            raise ValueError(msg)
        logger.info("Succesfully connected to %s", self.label)
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

    def unready(self, pipeline: str) -> str:
        """Mark a pipeline not ready for use."""
        raise NotImplementedError

    def submit(self, sample: str, capacity_Ah: float, payload: str | dict, pipeline: str) -> tuple[str, str, str]:
        """Submit a job to the server."""
        raise NotImplementedError

    def cancel(self, job_id_on_server: str, sampleid: str, pipeline: str) -> str:
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
        sample_id: str,
        jobid: str,
        jobid_on_server: str,
        get_raw: bool = False,
    ) -> str | None:
        """Save a snapshot of a job on the server and download it to the local machine."""
        raise NotImplementedError

    def get_last_data(self, job_id_on_server: str) -> dict:
        """Get the last data from a job."""
        raise NotImplementedError

    def get_job_data(self, jobid_on_server: str) -> dict:
        """Get the jobdata dict for a job."""
        raise NotImplementedError


class TomatoServer(CyclerServer):
    """Server class for Tomato servers, implements all the methods in CyclerServer.

    Used by server_manager to interact with Tomato servers, should not be instantiated directly.

    Attributes:
        save_location (str): The location on the server where snapshots are saved.

    """

    def __init__(self, server_config: dict) -> None:
        """Initialise server object."""
        super().__init__(server_config)
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
        payload: str | Path | dict,
        _pipeline: str = "",
        send_file: bool = False,
    ) -> tuple[str, str, str]:
        """Submit a job to the server.

        Args:
            sample (str): The name of the sample to be tested
            capacity_Ah (float): The capacity of the sample in Ah
            payload (str | Path | dict): The JSON protocol to be submitted, either unicycler or tomato
                can be a path to a file or a dictionary
            pipeline (str, optional): The pipeline to submit the job to (not necessary for Tomato servers)
            send_file (bool, default = False): If True, the payload is written to a file and sent to the server

        Returns:
            str: The jobid of the submitted job with the server prefix
            str: The jobid of the submitted job on the server (without the prefix)
            str: The JSON string of the submitted payload

        """
        # Check if json_file is a string that could be a file path or a JSON string
        if isinstance(payload, (str, Path)):
            try:
                # Attempt to load json_file as JSON string
                payload = json.loads(payload)
            except json.JSONDecodeError:
                with Path(payload).open(encoding="utf-8") as f:  # type: ignore[arg-type]
                    payload = json.load(f)

        # If json_file is already a dictionary, use it directly
        elif not isinstance(payload, dict):
            msg = "json_file must be a file path, a JSON string, or a dictionary"
            raise TypeError(msg)

        assert isinstance(payload, dict)  # noqa: S101 for mypy type checking

        # Check if payload is unicycler
        if "tomato" in payload:  # It is already a tomato payload
            # Add the sample name and capacity to the payload
            payload["sample"]["name"] = sample
            payload["sample"]["capacity"] = capacity_Ah
            # Convert the payload to a json string
            json_string = json.dumps(payload)
            # Change all other instances of $NAME to the sample name
            json_string = json_string.replace("$NAME", sample)
        else:
            try:
                json_string = Protocol.from_dict(payload).to_tomato_mpg2(
                    sample_name=sample,
                    capacity_mAh=capacity_Ah * 1000,
                )
            except Exception as e:
                msg = "Payload must be a unicycler protocol or a valid Tomato MPG2 protocol"
                raise ValueError(msg) from e

        if send_file:  # Write the json string to a file, send it, run it on the server
            # Write file locally
            with Path("temp.json").open("w", encoding="utf-8") as f:
                f.write(json_string)

            # Send file to server
            with paramiko.SSHClient() as ssh:
                ssh.load_system_host_keys()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(self.hostname, username=self.username, key_filename=CONFIG.get("SSH private key path"))
                with SCPClient(ssh.get_transport(), socket_timeout=120) as scp:
                    scp.put("temp.json", f"{self.save_location}/temp.json")

            # Submit the file on the server
            output = self.command(f"{self.tomato_scripts_path}ketchup submit {self.save_location}/temp.json")

            # Remove the file locally
            Path("temp.json").unlink()

        else:  # Encode the json string to base64 and submit it directly
            encoded_json_string = base64.b64encode(json_string.encode()).decode()
            output = self.command(f"{self.tomato_scripts_path}ketchup submit -J {encoded_json_string}")
        if "jobid: " in output:
            jobid = output.split("jobid: ")[1].splitlines()[0]
            logger.info("Sample %s submitted on server %s with jobid %s", sample, self.label, jobid)
            full_jobid = f"{self.label}-{jobid}"
            logger.info("Full jobid: %s", full_jobid)
            return full_jobid, jobid, json_string

        msg = f"Error submitting job: {output}"
        raise ValueError(msg)

    def cancel(self, job_id_on_server: str, sampleid: str, pipeline: str) -> str:
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
        sample_id: str,
        jobid: str,
        jobid_on_server: str,
        get_raw: bool = False,
    ) -> str | None:
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
        run_id = run_from_sample(sample_id)
        local_save_location = get_tomato_snapshot_folder() / run_id / sample_id

        if self.shell_type == "powershell":
            self.command(
                f'if (!(Test-Path "{remote_save_location}")) '
                f'{{ New-Item -ItemType Directory -Path "{remote_save_location}" }}',
            )
        elif self.shell_type == "cmd":
            self.command(
                f'if not exist "{remote_save_location}" mkdir "{remote_save_location}"',
            )
        else:
            msg = "Shell type not recognised, must be 'powershell' or 'cmd', check config.json"
            raise ValueError(msg)
        output = self.command(f"{self.tomato_scripts_path}ketchup status -J {jobid_on_server}")
        logger.info("Got job status on remote server %s", self.label)
        json_output = json.loads(output)
        snapshot_status = json_output["status"][0]
        # Catch errors
        try:
            with warnings.catch_warnings(record=True) as w:
                if self.shell_type == "powershell":
                    self.command(
                        f"cd {remote_save_location} ; {self.tomato_scripts_path}ketchup snapshot {jobid_on_server}",
                    )
                elif self.shell_type == "cmd":
                    self.command(
                        f"cd {remote_save_location} && {self.tomato_scripts_path}ketchup snapshot {jobid_on_server}",
                    )
                for warning in w:
                    if "out-of-date version" in str(warning.message) or "has been completed" in str(warning.message):
                        continue  # Ignore these warnings
                    logger.warning("Warning: %s", warning.message)
        except ValueError as e:
            emsg = str(e)
            if "AssertionError" in emsg and "os.path.isdir(jobdir)" in emsg:
                raise FileNotFoundError from e  # TODO make this error more deterministic up the chain
            raise
        logger.info("Snapshotted file on remote server %s", self.label)
        # Get local directory to save the snapshot data
        if not Path(local_save_location).exists():
            Path(local_save_location).mkdir(parents=True)

        # Use SCPClient to transfer the file from the remote machine
        ssh = paramiko.SSHClient()
        ssh.load_system_host_keys()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        logger.info("Connecting to %s: host %s user %s", self.label, self.hostname, self.username)
        ssh.connect(self.hostname, username=self.username, key_filename=CONFIG.get("SSH private key path"))
        try:
            logger.info(
                "Downloading file %s/snapshot.%s.json to %s/snapshot.%s.json",
                remote_save_location,
                jobid_on_server,
                local_save_location,
                jobid,
            )
            with SCPClient(ssh.get_transport(), socket_timeout=120) as scp:
                scp.get(
                    f"{remote_save_location}/snapshot.{jobid_on_server}.json",
                    f"{local_save_location}/snapshot.{jobid}.json",
                )
                if get_raw:
                    logger.info("Downloading shapshot raw data to %s/snapshot.%s.zip", local_save_location, jobid)
                    scp.get(
                        f"{remote_save_location}/snapshot.{jobid_on_server}.zip",
                        f"{local_save_location}/snapshot.{jobid}.zip",
                    )
        finally:
            ssh.close()

        # Compress the local snapshot file
        puree_tomato(f"{local_save_location}/snapshot.{jobid}.json")

        # Convert the snapshot file to hdf5
        convert_tomato_json(
            f"{local_save_location}/snapshot.{jobid}.json",
            output_hdf_file=True,
        )

        return snapshot_status

    def get_last_data(self, job_id_on_server: str) -> dict:
        """Get the last data from a job snapshot.

        Args:
            job_id_on_server : str
                The job ID on the server (an integer for tomato)

        Returns:
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
            command = f'powershell.exe -Command "{ps_command}"'

        with paramiko.SSHClient() as ssh:
            ssh.load_system_host_keys()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.hostname, username=self.username, key_filename=CONFIG.get("SSH private key path"))
            stdin, stdout, stderr = ssh.exec_command(command)
            if stderr.read():
                raise ValueError(stderr.read())
        file_name = stdout.readline().strip()
        file_content = stdout.readline().strip()
        file_content_json = json.loads(file_content)
        file_content_json["file_name"] = file_name
        return file_content_json

    def get_job_data(self, jobid_on_server: str) -> dict:
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
            command = f'powershell.exe -Command "{ps_command}"'
        with paramiko.SSHClient() as ssh:
            ssh.load_system_host_keys()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.hostname, username=self.username, key_filename=CONFIG.get("SSH private key path"))
            stdin, stdout, stderr = ssh.exec_command(command)
            stdout = stdout.read().decode("utf-8")
            stderr = stderr.read().decode("utf-8")
        if stderr:
            raise ValueError(stderr)
        if "File not found." in stdout:
            msg = f"jobdata.json not found for job {jobid_on_server}"
            raise FileNotFoundError(msg)
        return json.loads(stdout)


class NewareServer(CyclerServer):
    """Server class for Neware servers, implements all the methods in CyclerServer.

    Used by server_manager to interact with Neware servers, should not be instantiated directly.

    A Neware server is a PC running Neware BTS 8.0 with the API enabled and aurora-neware CLI
    installed. The 'neware' CLI command should be accessible in the PATH. If it is not by default,
    use the 'command_prefix' in the shared config to add it to the PATH.

    """

    def eject(self, pipeline: str) -> str:
        """Remove a sample from a pipeline.

        Do not need to actually change anything on Neware client, just update the database.
        """
        return f"Ejecting {pipeline}"

    def load(self, sample: str, pipeline: str) -> str:
        """Load a sample onto a pipeline.

        Do not need to actually change anything on Neware client, just update the database.
        """
        return f"Loading {sample} onto {pipeline}"

    def ready(self, pipeline: str) -> str:
        """Readying and unreadying does not exist on Neware."""
        raise NotImplementedError

    def submit(
        self, sample: str, capacity_Ah: float, payload: str | dict | Path, pipeline: str
    ) -> tuple[str, str, str]:
        """Submit a job to the server.

        Use the start command on the aurora-neware CLI installed on Neware machine.
        """
        # Parse the input into an xml string
        if not isinstance(payload, str | Path | dict):
            msg = (
                "For Neware, payload must be a unicycler protocol (dict, or path to JSON file) "
                "or a Neware XML (XML string, or path to XML file)."
            )
            raise TypeError(msg)
        if isinstance(payload, dict):  # assume unicycler dict
            xml_string = Protocol.from_dict(payload, sample, capacity_Ah * 1000).to_neware_xml()
        elif isinstance(payload, str):  # it is a file path
            if payload.startswith("<?xml"):  # it is already an xml string
                xml_string = payload
            else:  # it is probably a file path
                payload = Path(payload)
        elif isinstance(payload, Path):  # it is a file path
            if not payload.exists():
                raise FileNotFoundError
            if payload.suffix == ".xml":
                with payload.open(encoding="utf-8") as f:
                    xml_string = f.read()
            elif payload.suffix == ".json":
                with payload.open(encoding="utf-8") as f:
                    xml_string = Protocol.from_dict(json.load(f), sample, capacity_Ah * 1000).to_neware_xml()
            else:
                msg = "Payload must be a path to an xml or json file or xml string or dict."
                raise TypeError(msg)

        # Check the xml string is valid
        if not xml_string.startswith("<?xml"):
            msg = "Payload does not look like xml, does not start with '<?xml'. "
            raise ValueError(msg)
        if 'config type="Step File"' not in xml_string or 'client_version="BTS Client' not in xml_string:
            msg = "Payload looks like xml, but not a Neware step file."
            raise ValueError(msg)

        # Convert capacity in Ah to capacity in mA s
        capacity_mA_s = round(capacity_Ah * 1000 * 3600)

        # If they still exist, change $NAME and $CAPACITY to appropriate values
        xml_string = xml_string.replace("$NAME", sample)
        xml_string = xml_string.replace("$CAPACITY", str(capacity_mA_s))

        # Write the xml string to a temporary file
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        try:
            with Path("./temp.xml").open("w", encoding="utf-8") as f:
                f.write(xml_string)
            # Transfer the file to the remote PC and start the job
            with paramiko.SSHClient() as ssh:
                ssh.load_system_host_keys()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(self.hostname, username=self.username, key_filename=CONFIG.get("SSH private key path"))
                with SCPClient(ssh.get_transport(), socket_timeout=120) as scp:
                    remote_xml_dir = "C:/submitted_payloads/"
                    remote_xml_path = remote_xml_dir + f"{sample}__{current_datetime}.xml"
                    # Create the directory if it doesn't exist
                    if self.shell_type == "cmd":
                        ssh.exec_command(f'mkdir "{remote_xml_dir!s}"')
                    elif self.shell_type == "powershell":
                        ssh.exec_command(f'New-Item -ItemType Directory -Path "{remote_xml_dir!s}"')
                    scp.put("./temp.xml", remote_xml_path)

            # Submit the file on the remote PC
            output = self.command(f"neware start {pipeline} {sample} {remote_xml_path}")
            # Expect the output to be empty if successful, otherwise raise error
            if output:
                msg = (
                    f"Command 'neware stop {pipeline}' failed with response:\n{output}\n"
                    "Probably an issue with the xml file. "
                    "You must check the Neware client logs for more information."
                )
                raise ValueError(msg)
            logger.info("Submitted job to Neware server %s", self.label)
            # Then ask for the jobid
            jobid_on_server = self._get_job_id(pipeline)
            jobid = f"{self.label}-{jobid_on_server}"
            logger.info("Job started on Neware server with ID %s", jobid)
        finally:
            Path("temp.xml").unlink()  # Remove the file on local machine
        return jobid, jobid_on_server, xml_string

    def cancel(self, job_id_on_server: str, sampleid: str, pipeline: str) -> str:
        """Cancel a job on the server.

        Use the STOP command on the Neware-api.
        """
        # Check that sample ID matches
        output = self.command(f"neware status {pipeline}")
        barcode = json.loads(output).get(pipeline, {}).get("barcode")
        if barcode != sampleid:
            msg = "Barcode on server does not match Sample ID being cancelled"
            raise ValueError(msg)
        # Check that a job is running
        workstatus = json.loads(output).get(pipeline, {}).get("workstatus")
        if workstatus not in ["working", "pause", "protect"]:
            msg = "Pipeline is not running, cannot cancel job"
            raise ValueError(msg)
        # Check that job ID matches
        output = self.command(f"neware testid {pipeline}")
        full_test_id = self._get_job_id(pipeline)
        if full_test_id != job_id_on_server:
            msg = "Job ID on server does not match Job ID being cancelled"
            raise ValueError(msg)
        # Stop the pipeline
        output = self.command(f"neware stop {pipeline}")
        # Expect the output to be empty if successful, otherwise raise error
        if output:
            msg = (
                f"Command 'neware stop {pipeline}' failed with response:\n{output}\n"
                "Check the Neware client logs for more information."
            )
            raise ValueError(output)
        return f"Stopped pipeline {pipeline} on Neware"

    def get_pipelines(self) -> dict:
        """Get the status of all pipelines on the server."""
        result = json.loads(self.command("neware status"))
        # result is a dict with keys=pipeline and value a dict of stuff
        # need to return in list format with keys 'pipeline', 'sampleid', 'ready', 'jobid'
        pipelines, sampleids, readys = [], [], []
        for pip, data in result.items():
            pipelines.append(pip)
            if data["workstatus"] in ["working", "pause", "protect"]:  # working\stop\finish\protect\pause
                sampleids.append(data["barcode"])
                readys.append(False)
            else:
                sampleids.append(None)
                readys.append(True)
        return {"pipeline": pipelines, "sampleid": sampleids, "jobid": [None] * len(pipelines), "ready": readys}

    def get_jobs(self) -> dict:
        """Get all jobs from server.

        Not implemented, could use inquiredf but very slow. Return empty dict for now.
        """
        return {}

    def snapshot(
        self,
        sample_id: str,
        jobid: str,
        jobid_on_server: str,  # noqa: ARG002
        get_raw: bool = False,  # noqa: ARG002
    ) -> str | None:
        """Save a snapshot of a job on the server and download it to the local machine."""
        ndax_path = snapshot_raw_data(jobid)
        if ndax_path:
            convert_neware_data(ndax_path, sample_id, output_hdf5_file=True)

        return None  # Neware does not have a snapshot status like tomato

    def get_job_data(self, jobid_on_server: str) -> dict:
        """Get the jobdata dict for a job."""
        # TODO: This is problematic because Neware XMLs don't easily translate to a dict.
        raise NotImplementedError

    def get_last_data(self, job_id_on_server: str) -> dict:
        """Get the last data from a job snapshot."""
        raise NotImplementedError

    def _get_job_id(self, pipeline: str) -> str:
        """Get the testid for a pipeline."""
        output = self.command(f"neware get-job-id {pipeline} --full-id")
        return json.loads(output).get(pipeline)


class BiologicServer(CyclerServer):
    """Server class for Biologic servers, implements all the methods in CyclerServer.

    Used by server_manager to interact with Biologic servers, should not be instantiated directly.

    A Biologic server is a PC running EC-lab (11.52) with OLE-COM registered and the aurora-biologic
    CLI installed. The 'biologic' CLI command should be accessible in the PATH. If it is not by
    default, use the 'command_prefix' in the shared config to add it to the PATH.
    """

    def __init__(self, server_config: dict) -> None:
        """Initialise server object."""
        super().__init__(server_config)
        # EC-lab can only work on Windows
        self.biologic_data_path = PureWindowsPath(
            server_config.get("biologic_data_path", "C:/aurora/data/"),
        )

    def eject(self, pipeline: str) -> str:
        """Remove a sample from a pipeline.

        Do not need to actually change anything on Biologic client, just update the database.
        """
        return f"Ejecting {pipeline}"

    def load(self, sample: str, pipeline: str) -> str:
        """Load a sample onto a pipeline.

        Do not need to actually change anything on Biologic client, just update the database.
        """
        return f"Loading {sample} onto {pipeline}"

    def ready(self, pipeline: str) -> str:
        """Readying and unreadying does not exist on Biologic."""
        raise NotImplementedError

    def submit(
        self, sample: str, capacity_Ah: float, payload: str | dict | Path, pipeline: str
    ) -> tuple[str, str, str]:
        """Submit a job to the server.

        Uses the start command on the aurora-biologic CLI.
        """
        # Parse the input into an xml string
        if not isinstance(payload, str | Path | dict):
            msg = "For Biologic, payload must be a unicycler protocol, either a dict, or path to a JSON file."
            raise TypeError(msg)
        if isinstance(payload, dict):  # assume unicycler dict
            mps_string = Protocol.from_dict(payload, sample, capacity_Ah * 1000).to_biologic_mps()
        elif isinstance(payload, (Path, str)):  # it is a file path
            payload = Path(payload)
            if not payload.exists():
                raise FileNotFoundError
            if payload.suffix == ".json":
                with payload.open(encoding="utf-8") as f:
                    mps_string = Protocol.from_dict(json.load(f), sample, capacity_Ah * 1000).to_biologic_mps()
            else:
                msg = "Payload must be a path to a unicycler json file or dict."
                raise TypeError(msg)

        # Check the mps string is valid
        if not mps_string.startswith("EC-LAB SETTING FILE"):
            msg = "Payload does not look like EC-lab settings file."
            raise ValueError(msg)

        # If it still exists, change $NAME to appropriate values
        mps_string = mps_string.replace("$NAME", sample)

        # Write the mps string to a temporary file
        # EC-lab has no concept of job IDs - we use the folder as the job ID
        # Job ID is sample ID + unix timestamp in seconds
        run_id = run_from_sample(sample)
        jobid_on_server = f"{sample}__{int(datetime.now().timestamp())}"
        try:
            with Path("./temp.mps").open("w", encoding="utf-8") as f:
                f.write(mps_string)
            # Transfer the file to the remote PC and start the job
            with paramiko.SSHClient() as ssh:
                ssh.load_system_host_keys()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(self.hostname, username=self.username, key_filename=CONFIG.get("SSH private key path"))
                with SCPClient(ssh.get_transport(), socket_timeout=120) as scp:
                    # One folder per job, EC-lab generates multiple files per job
                    # EC-lab will make files with suffix _C01, _C02, etc. and extensions .mpr .mpl etc.
                    remote_output_path = (
                        self.biologic_data_path / run_id / sample / jobid_on_server / f"{jobid_on_server}.mps"
                    )
                    # Create the directory if it doesn't exist - data directory must also exist
                    if self.shell_type == "cmd":
                        ssh.exec_command(f'mkdir "{remote_output_path.parent.as_posix()}"')
                    elif self.shell_type == "powershell":
                        ssh.exec_command(f'New-Item -ItemType Directory -Path "{remote_output_path.parent.as_posix()}"')
                    scp.put("./temp.mps", remote_output_path.as_posix())  # SCP hates Windows \

            # Submit the file on the remote PC
            output = self.command(f"biologic start {pipeline} {remote_output_path!s} {remote_output_path!s} --ssh")
            # Expect the output to be empty if successful, otherwise raise error
            if output:
                msg = (
                    f"Command 'biologic start' failed with response:\n{output}\n"
                    "Probably an issue with the mps input file. "
                    "You must check on the server for more information. "
                    f"Try manually loading the mps file at {remote_output_path}."
                )
                raise ValueError(msg)
            jobid = f"{self.label}-{jobid_on_server}"
            logger.info("Job started on Biologic server with ID %s", jobid)
        finally:
            Path("temp.mps").unlink()  # Remove the file on local machine
        return jobid, jobid_on_server, mps_string

    def cancel(self, job_id_on_server: str, sampleid: str, pipeline: str) -> str:
        """Cancel a job on the server.

        Use the STOP command on the Neware-api.
        """
        # Get job ID on server
        output = self.command(f"biologic get-job-id {pipeline} --ssh")
        job_id_on_biologic = json.loads(output).get(pipeline, {})
        # Check that a job is running
        if not job_id_on_biologic:
            msg = "No job is running on the server, cannot cancel job"
            raise ValueError(msg)
        # Check that a job_id matches
        if job_id_on_server != job_id_on_biologic:
            msg = "Job ID on server does not match job ID being cancelled"
            raise ValueError(msg)
        # Stop the pipeline
        output = self.command(f"biologic stop {pipeline} --ssh")
        # Expect the output to be empty if successful, otherwise raise error
        if output:
            msg = f"Command 'biologic stop {pipeline}' failed with response:\n{output}\n"
            raise ValueError(output)
        return f"Stopped pipeline {pipeline} on Biologic"

    def get_pipelines(self) -> dict:
        """Get the status of all pipelines on the server."""
        result = json.loads(self.command("biologic status --ssh"))
        # Result is a dict with keys=pipeline and value a dict of stuff
        # need to return in list format with keys 'pipeline', 'sampleid', 'ready', 'jobid'
        # Biologic does not give sample ID or job IDs from status
        # The Nones are handled in server_manager.update_pipelines()
        pipelines, readys = [], []
        for pip, data in result.items():
            pipelines.append(pip)
            if data["Status"] in ["Run", "Pause", "Sync"]:  # working\stop\finish\protect\pause
                readys.append(False)  # Job is running - not ready
            else:
                readys.append(True)  # Job is not running - ready
        return {
            "pipeline": pipelines,
            "sampleid": [None] * len(pipelines),
            "jobid": [None] * len(pipelines),
            "ready": readys,
        }

    def get_jobs(self) -> dict:
        """Get all jobs from server.

        Not implemented, could use get-job-id but very slow. Return empty dict for now.
        """
        return {}

    def snapshot(
        self,
        sample_id: str,
        jobid: str,
        jobid_on_server: str,
        get_raw: bool = False,  # noqa: ARG002
    ) -> str | None:
        """Save a snapshot of a job on the server and download it to the local machine."""
        # We know where the job will be on the remote PC
        run_id = run_from_sample(sample_id)
        remote_job_folder = self.biologic_data_path / run_id / sample_id / jobid_on_server

        # Connect to the remote server
        with paramiko.SSHClient() as ssh:
            ssh.load_system_host_keys()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.hostname, username=self.username, key_filename=CONFIG.get("SSH private key path"))

            # Find all the .mpr and .mpl files in the job folder
            ps_command = (
                f"Get-ChildItem -Path '{remote_job_folder}' -Recurse -File "
                f"| Where-Object {{ ($_.Extension -in '.mpl', '.mpr')}} "
                f"| Select-Object -ExpandProperty FullName"
            )
            if self.shell_type == "powershell":
                command = ps_command
            elif self.shell_type == "cmd":
                # Base64 encode the command to avoid quote/semicolon issues
                encoded_ps_command = base64.b64encode(ps_command.encode("utf-16le")).decode("ascii")
                command = f"powershell.exe -EncodedCommand {encoded_ps_command}"
            else:
                msg = f"Unknown shell type {self.shell_type} for server {self.label}"
                raise ValueError(msg)
            stdin, stdout, stderr = ssh.exec_command(command)
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                msg = f"Command failed with exit status {exit_status}: {stderr.read().decode('utf-8')}"
                raise RuntimeError(msg)
            output = stdout.read().decode("utf-8").strip()
            files_to_copy = output.splitlines()
            local_folder = get_eclab_snapshot_folder()

            # Local files will have the same relative path as the remote files
            local_files = [local_folder / run_id / sample_id / jobid / file.split("\\")[-1] for file in files_to_copy]

            # Copy the files across with SFTP
            with ssh.open_sftp() as sftp:
                for remote_file, local_file in zip(files_to_copy, local_files, strict=True):
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    logger.info("Downloading file %s to %s", remote_file, local_file)
                    sftp.get(remote_file, str(local_file))

            # Convert copied files to hdf5
            for local_file in local_files:
                if local_file.suffix == ".mpr":
                    convert_mpr(local_file)

        return None

    def get_job_data(self, jobid_on_server: str) -> dict:
        """Get the jobdata dict for a job."""
        # TODO: Implement getting job data from mps file
        raise NotImplementedError

    def get_last_data(self, job_id_on_server: str) -> dict:
        """Get the last data from a job snapshot."""
        raise NotImplementedError

    def _get_job_id(self, pipeline: str) -> str:
        """Get the testid for a pipeline."""
        output = self.command(f"biologic get-job-id {pipeline} --ssh")
        return json.loads(output).get(pipeline)
