"""Copyright © 2025, Empa.

Server classes used by server_manager, including:
- Neware server, designed for Neware BTS 8.0 with aurora-neware CLI
- Biologic server, designed for Biologic EC-lab with aurora-biologic CLI

Unlike the harvester modules, which can only download the latest data, cycler
servers can be used to interact with the server directly, e.g. to submit a job
or get the status of a pipeline.

These classes are used by server_manager.
"""

import base64
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

import paramiko
from aurora_unicycler import Protocol
from scp import SCPClient
from typing_extensions import override

from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.eclab_harvester import convert_mpr, get_eclab_snapshot_folder
from aurora_cycler_manager.neware_harvester import convert_neware_data, snapshot_raw_data
from aurora_cycler_manager.utils import run_from_sample, ssh_connect

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

    def _command(self, command: str, timeout: float = 300) -> str:
        """Send a command to the server and return the output.

        The command is prefixed with the command_prefix specified in the server_config, is run on
        the server's default shell, the standard output is returned as a string.
        """
        with paramiko.SSHClient() as ssh:
            ssh_connect(ssh, self.username, self.hostname)
            _stdin, stdout, stderr = ssh.exec_command(
                self.command_prefix + command + self.command_suffix,
                timeout=timeout,
            )
            output = stdout.read().decode("utf-8").strip()
            error = stderr.read().decode("utf-8").strip()
            exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            logger.error("Command '%s' on %s failed with exit status %d", command, self.label, exit_status)
            logger.error("Error: %s", error)
            msg = f"Command failed with exit status {exit_status}: {error}"
            raise ValueError(msg)
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
        output = self._command(f"echo {test_phrase}", timeout=5).strip()
        if output != test_phrase:
            msg = f"Connection error, expected output '{test_phrase}', got '{output}'"
            raise ValueError(msg)
        logger.info("Succesfully connected to %s", self.label)
        return True

    def submit(
        self, sample: str, capacity_Ah: float, payload: str | dict | Path, pipeline: str
    ) -> tuple[str, str, str]:
        """Submit a job to the server."""
        raise NotImplementedError

    def cancel(self, jobid: str, job_id_on_server: str, sampleid: str, pipeline: str) -> None:
        """Cancel a job on the server."""
        raise NotImplementedError

    def get_pipelines(self) -> dict:
        """Get the status of all pipelines on the server."""
        raise NotImplementedError

    def snapshot(self, sample_id: str, jobid: str, jobid_on_server: str) -> str | None:
        """Save a snapshot of a job on the server and download it to the local machine."""
        raise NotImplementedError


class NewareServer(CyclerServer):
    """Server class for Neware servers, implements all the methods in CyclerServer.

    Used by server_manager to interact with Neware servers, should not be instantiated directly.

    A Neware server is a PC running Neware BTS 8.0 with the API enabled and aurora-neware CLI
    installed. The 'neware' CLI command should be accessible in the PATH. If it is not by default,
    use the 'command_prefix' in the shared config to add it to the PATH.

    """

    @override
    def submit(
        self,
        sample: str,
        capacity_Ah: float,
        payload: str | dict | Path,
        pipeline: str,
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
        current_datetime = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        try:
            with Path("./temp.xml").open("w", encoding="utf-8") as f:
                f.write(xml_string)
            # Transfer the file to the remote PC and start the job
            with paramiko.SSHClient() as ssh:
                ssh_connect(ssh, self.username, self.hostname)
                with SCPClient(ssh.get_transport(), socket_timeout=120) as scp:
                    remote_xml_dir = PureWindowsPath("C:/submitted_payloads/")
                    remote_xml_path = remote_xml_dir / f"{sample}__{current_datetime}.xml"
                    # Create the directory if it doesn't exist
                    if self.shell_type == "cmd":
                        ssh.exec_command(f'mkdir "{remote_xml_dir!s}"')
                    elif self.shell_type == "powershell":
                        ssh.exec_command(f'New-Item -ItemType Directory -Path "{remote_xml_dir!s}"')
                    scp.put("./temp.xml", remote_xml_path.as_posix())  # SCP hates windows \
            # Submit the file on the remote PC
            output = self._command(f"neware start {pipeline} {sample} {remote_xml_path}")
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

    @override
    def cancel(self, jobid: str, job_id_on_server: str, sampleid: str, pipeline: str) -> None:
        """Cancel a job on the server.

        Use the STOP command on the Neware-api.
        """
        # Check that sample ID matches
        output = self._command(f"neware status {pipeline}")
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
        full_test_id = self._get_job_id(pipeline)
        if full_test_id != job_id_on_server:
            msg = "Job ID on server does not match Job ID being cancelled"
            raise ValueError(msg)
        # Stop the pipeline
        output = self._command(f"neware stop {pipeline}")
        # Expect the output to be empty if successful, otherwise raise error
        if output:
            msg = (
                f"Command 'neware stop {pipeline}' failed with response:\n{output}\n"
                "Check the Neware client logs for more information."
            )
            raise ValueError(output)

    @override
    def get_pipelines(self) -> dict:
        """Get the status of all pipelines on the server."""
        result = json.loads(self._command("neware status"))
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

    @override
    def snapshot(self, sample_id: str, jobid: str, jobid_on_server: str) -> str | None:
        """Save a snapshot of a job on the server and download it to the local machine."""
        ndax_path = snapshot_raw_data(jobid)
        if ndax_path:
            convert_neware_data(ndax_path, sample_id, output_hdf5_file=True)

        return None  # Neware does not have a snapshot status

    def _get_job_id(self, pipeline: str) -> str:
        """Get the testid for a pipeline."""
        output = self._command(f"neware get-job-id {pipeline} --full-id")
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

    @override
    def submit(
        self,
        sample: str,
        capacity_Ah: float,
        payload: str | dict | Path,
        pipeline: str,
    ) -> tuple[str, str, str]:
        """Submit a job to the server.

        Payload can be
        - a unicycler protocol (dict, or path to JSON file)
        - a Biologic mps settings file (string, or path to mps file)

        Uses the start command on the aurora-biologic CLI.
        """
        # Parse the input into an mps string
        if not isinstance(payload, str | Path | dict):
            msg = "For Biologic, payload must be a string, path or dict of a unicycler protocol or mps settings file."
            raise TypeError(msg)
        if isinstance(payload, dict):  # assume unicycler dict
            mps_string = Protocol.from_dict(payload, sample, capacity_Ah * 1000).to_biologic_mps()
        elif isinstance(payload, (Path, str)):  # it is a file path or mps string
            if isinstance(payload, str) and payload.startswith("EC-LAB SETTING FILE"):
                mps_string = payload
            else:
                payload = Path(payload)
                if not payload.exists():
                    raise FileNotFoundError
                if payload.suffix == ".json":
                    with payload.open() as f:
                        mps_string = Protocol.from_dict(json.load(f), sample, capacity_Ah * 1000).to_biologic_mps()
                elif payload.suffix == ".mps":
                    with payload.open(encoding="cp1252") as f:
                        mps_string = f.read()
                else:
                    msg = "Payload path must be a path to a unicycler json file or dict, or path to an mps file."
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
        jobid_on_server = str(uuid.uuid4())
        jobid = jobid_on_server  # Do not need separate IDs
        try:
            with Path("./temp.mps").open("w", encoding="utf-8") as f:
                f.write(mps_string)
            # Transfer the file to the remote PC and start the job
            with paramiko.SSHClient() as ssh:
                ssh_connect(ssh, self.username, self.hostname)
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
            output = self._command(f"biologic start {pipeline} {remote_output_path!s} {remote_output_path!s} --ssh")
            # Expect the output to be empty if successful, otherwise raise error
            if output:
                msg = (
                    f"Command 'biologic start' failed with response:\n{output}\n"
                    "Probably an issue with the mps input file. "
                    "You must check on the server for more information. "
                    f"Try manually loading the mps file at {remote_output_path}."
                )
                raise ValueError(msg)
            logger.info("Job started on Biologic server with ID %s", jobid)
        finally:
            Path("temp.mps").unlink()  # Remove the file on local machine
        return jobid, jobid_on_server, mps_string

    @override
    def cancel(self, jobid: str, job_id_on_server: str, sampleid: str, pipeline: str) -> None:
        """Cancel a job on the server.

        Use the STOP command on the Neware-api.
        """
        # Get job ID on server
        output = self._command(f"biologic get-job-id {pipeline} --ssh")
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
        output = self._command(f"biologic stop {pipeline} --ssh")
        # Expect the output to be empty if successful, otherwise raise error
        if output:
            msg = f"Command 'biologic stop {pipeline}' failed with response:\n{output}\n"
            raise ValueError(output)

    @override
    def get_pipelines(self) -> dict:
        """Get the status of all pipelines on the server."""
        result = json.loads(self._command("biologic status --ssh"))
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

    @override
    def snapshot(self, sample_id: str, jobid: str, jobid_on_server: str) -> str | None:
        """Save a snapshot of a job on the server and download it to the local machine."""
        # We know where the job will be on the remote PC
        run_id = run_from_sample(sample_id)
        remote_job_folder = self.biologic_data_path / run_id / sample_id / jobid_on_server

        # Connect to the remote server
        with paramiko.SSHClient() as ssh:
            ssh_connect(ssh, self.username, self.hostname)

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
            _stdin, stdout, stderr = ssh.exec_command(command)
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
                    try:
                        convert_mpr(local_file, job_id=jobid, update_database=True)
                    except Exception:
                        logger.exception("Error converting %s", local_file.name)

        return None

    def _get_job_id(self, pipeline: str) -> str:
        """Get the testid for a pipeline."""
        output = self._command(f"biologic get-job-id {pipeline} --ssh")
        return json.loads(output).get(pipeline)
