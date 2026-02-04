"""Copyright © 2025-2026, Empa.

server_manager manages a database and communicates with multiple cycler servers.

This module defines a ServerManager class. The ServerManager object communicates
with multiple CyclerServer objects defined in cycler_servers, and manages the
database of samples, pipelines and jobs.

Server manager takes functions like load, submit, snapshot, update etc. sends
commands to the appropriate server, and handles the database updates.
"""

import json
import logging
import sqlite3
import traceback
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from time import sleep, time
from typing import Any, Literal

import paramiko
from aurora_unicycler import Protocol

from aurora_cycler_manager import analysis, config, cycler_servers
from aurora_cycler_manager import database_funcs as dbf
from aurora_cycler_manager.cycler_servers import CyclerServer
from aurora_cycler_manager.stdlib_utils import run_from_sample

SERVER_CORRESPONDENCE = {
    "neware": cycler_servers.NewareServer,
    "biologic": cycler_servers.BiologicServer,
}

SERVER_OBJECTS: dict[str, cycler_servers.CyclerServer] = {}

logger = logging.getLogger(__name__)


def get_servers(*, reload: bool = False) -> dict[str, cycler_servers.CyclerServer]:
    """Create the cycler server objects from the config file."""
    global SERVER_OBJECTS
    if not SERVER_OBJECTS or reload:
        SERVER_OBJECTS = {}
        logger.info("Attempting to connect to to servers")
        if not config.get_config(reload=reload).get("Servers"):
            logger.warning("No servers in the configuration.")
        else:
            for server_label, server_dict in config.get_config()["Servers"].items():
                server_type = server_dict.get("server_type")
                if server_type.endswith("_harvester"):
                    continue  # Skip silently
                if server_type not in SERVER_CORRESPONDENCE:
                    logger.error("Server type %s not recognized, skipping", server_type)
                    continue
                try:
                    server_class = SERVER_CORRESPONDENCE[server_type]
                    SERVER_OBJECTS[server_label] = server_class(server_dict, label=server_label)
                except (OSError, ValueError, TimeoutError, paramiko.SSHException):
                    logger.exception("Server %s could not be created, skipping", server_label)
    return SERVER_OBJECTS


def find_server(label: str) -> cycler_servers.CyclerServer:
    """Get the server object from the label."""
    server = get_servers().get(label, None)
    if not server:
        msg = (
            f"Server with label {label} not found. "
            "Either there is a mistake in the label name or you do not have access to the server."
        )
        raise KeyError(msg)
    return server


class _Pipeline:
    """A class representing a pipeline in the database."""

    def __init__(self, pipeline_name: str, server_label: str) -> None:
        """Initialize the _Pipeline object."""
        self.name = pipeline_name
        self.server_label = server_label

    @cached_property
    def server(self) -> CyclerServer:
        """Lazy-load the server object."""
        return find_server(self.server_label)

    @property
    def sample(self) -> "_Sample | None":
        return _Sample.from_pipeline(self)

    @classmethod
    def from_id(cls, pipeline_name: str) -> "_Pipeline":
        """Create a _Pipeline object from a pipeline ID.

        Args:
            pipeline_name : str
                The pipeline name to create the object for.

        Returns:
            _Pipeline: The Pipeline object.

        """
        result = dbf.execute_sql(
            "SELECT `Pipeline`, `Server label` FROM pipelines WHERE `Pipeline` = ?",
            (pipeline_name,),
        )
        if not result:
            msg = f"Pipeline '{pipeline_name}' not found in the database."
            raise ValueError(msg)
        return cls(pipeline_name, result[0][1])

    @classmethod
    def from_sample(cls, sample: "_Sample") -> "_Pipeline | None":
        """Create a _Pipeline object from a Sample object."""
        result = dbf.execute_sql(
            "SELECT `Pipeline`, `Server label` FROM pipelines WHERE `Sample ID` = ?",
            (sample.id,),
        )
        if not result:
            return None
        return cls(result[0][0], result[0][1])

    def load(self, sample: "_Sample") -> None:
        """Load the sample on a pipeline.

        The appropriate server is found based on the pipeline, and the sample is loaded.

        Args:
            sample (_Sample):
                The sample to load on the pipeline.

        """
        if self.sample is not None:
            msg = f"Pipeline {self.name}, server {self.server.label} already has a sample loaded: {self.sample.id}."
            raise ValueError(msg)

        if sample.pipeline:
            msg = (
                f"Sample {sample.id} is already loaded on pipeline {sample.pipeline.name}, "
                f"server {sample.pipeline.server.label}."
            )
            raise ValueError(msg)

        # Get pipeline and load
        logger.info("Loading sample %s on server %s", sample.id, self.server.label)
        dbf.execute_sql(
            "UPDATE pipelines SET `Sample ID` = ? WHERE `Pipeline` = ?",
            (sample.id, self.name),
        )

    def eject(self, sample_id: str | None = None) -> None:
        """Eject the sample from a pipeline."""
        # Find server associated with pipeline
        logger.info("Ejecting sample from the pipeline %s on server: %s", self.name, self.server.label)
        if self.sample is None:
            msg = f"There is no sample to eject on pipeline {self.name}"
            raise ValueError(msg)
        if sample_id and self.sample and self.sample.id != sample_id:
            msg = (
                f"The pipeline {self.name} on server {self.server.label} has"
                f" sample {self.sample.id} loaded, not {sample_id}."
            )
            raise ValueError(msg)
        result = dbf.execute_sql(
            "SELECT `Job ID` FROM pipelines WHERE `Pipeline` = ?",
            (self.name,),
        )
        if result[0][0]:
            msg = f"Cannot eject sample {self.sample.id} from {self.name}, job is currently running: {result[0][0]}"
            raise ValueError(msg)
        # Eject the sample
        dbf.execute_sql(
            "UPDATE pipelines SET `Sample ID` = NULL, `Flag` = Null, `Ready` = ? WHERE `Pipeline` = ?",
            (True, self.name),
        )

    def set_jobid(self, job_id: str, jobid_on_server: str | None = None) -> None:
        """Set the job ID on the pipeline in the database."""
        dbf.execute_sql(
            "UPDATE pipelines SET `Job ID` = ?, `Job ID on server` = ?, `Ready` = 0 WHERE `Pipeline` = ?",
            (job_id, jobid_on_server, self.name),
        )


class _Sample:
    """A class representing a sample in the database."""

    def __init__(self, sample_id: str) -> None:
        """Initialize the _Sample object."""
        self.id = sample_id
        self._data: dict[str, str] = {}

    def get(self, key: str) -> Any:  # noqa: ANN401
        """Get a property of the sample from the database.

        Args:
            key : str
                The property name to get.

        Returns:
            Any or None: The property value, or None if not found.

        """
        return self._data.get(key)

    def get_sample_capacity(
        self,
        mode: Literal["areal", "mass", "nominal"],
        ignore_anode: bool = True,
    ) -> float:
        """Get the capacity of a sample in Ah based on the mode.

        Args:
            mode : str
                The mode to calculate the capacity. Must be 'areal', 'mass', or 'nominal'
                areal: calculate from anode/cathode C-rate definition areal capacity (mAh/cm2) and
                    anode/cathode Diameter (mm)
                mass: calculate from anode/cathode C-rate definition specific capacity (mAh/g) and
                    anode/cathode active material mass (mg)
                nominal: use C-rate definition capacity (mAh)
            ignore_anode : bool, optional
                If True, only use the cathode capacity. Default is True.

        Returns:
            float: The capacity of the sample in Ah

        """
        if mode == "mass":
            an_cap_mAh_g = self.get("Anode C-rate definition specific capacity (mAh/g)")
            an_mass_mg = self.get("Anode active material mass (mg)")
            an_diam_mm = self.get("Anode diameter (mm)")
            cat_cap_mAh_g = self.get("Cathode C-rate definition specific capacity (mAh/g)")
            cat_mass_mg = self.get("Cathode active material mass (mg)")
            cat_diam_mm = self.get("Cathode diameter (mm)")

            cat_frac_used = min(1, an_diam_mm**2 / cat_diam_mm**2)
            cat_cap_Ah = cat_frac_used * (cat_cap_mAh_g * cat_mass_mg * 1e-6)
            capacity_Ah = cat_cap_Ah
            if not ignore_anode:
                an_frac_used = min(1, cat_diam_mm**2 / an_diam_mm**2)
                an_cap_Ah = an_frac_used * (an_cap_mAh_g * an_mass_mg * 1e-6)
                capacity_Ah = min(an_cap_Ah, cat_cap_Ah)

        elif mode == "areal":
            an_cap_mAh_cm2 = self.get("Anode C-rate definition areal capacity (mAh/cm2)")
            an_diam_mm = self.get("Anode diameter (mm)")
            cat_cap_mAh_cm2 = self.get("Cathode C-rate definition areal capacity (mAh/cm2)")
            cat_diam_mm = self.get("Cathode diameter (mm)")
            cat_frac_used = min(1, an_diam_mm**2 / cat_diam_mm**2)
            cat_cap_Ah = cat_frac_used * cat_cap_mAh_cm2 * (cat_diam_mm / 2) ** 2 * 3.14159 * 1e-5
            capacity_Ah = cat_cap_Ah

            if not ignore_anode:
                an_frac_used = min(1, cat_diam_mm**2 / an_diam_mm**2)
                an_cap_Ah = an_frac_used * an_cap_mAh_cm2 * (an_diam_mm / 2) ** 2 * 3.14159 * 1e-5
                capacity_Ah = min(an_cap_Ah, cat_cap_Ah)

        elif mode == "nominal":
            capacity_Ah = self.get("C-rate definition capacity (mAh)") * 1e-3

        return capacity_Ah

    @classmethod
    def from_id(cls, sample_id: str) -> "_Sample":
        """Create a _Sample object from the database.

        Args:
            sample_id : str
                The sample ID to create the object for.

        Returns:
            _Sample: The _Sample object.

        """
        sample = cls(sample_id)

        # Load sample properties into _data dict
        sample._data = dbf.get_sample_data(sample_id)

        return sample

    @classmethod
    def from_pipeline(cls, pipeline: "_Pipeline") -> "_Sample | None":
        """Create a _Sample object from a _Pipeline object."""
        result = dbf.execute_sql(
            "SELECT `Sample ID` FROM pipelines WHERE `Pipeline` = ?",
            (pipeline.name,),
        )
        if not result[0][0]:
            return None
        return cls(str(result[0][0]))

    @property
    def pipeline(self) -> "_Pipeline | None":
        return _Pipeline.from_sample(self)

    def safe_get_sample_capacity(
        self,
        mode: Literal["areal", "mass", "nominal"],
        ignore_anode: bool = True,
    ) -> float | None:
        """Get the capacity of a sample in Ah based on the mode. Returns None if failed."""
        try:
            return self.get_sample_capacity(mode, ignore_anode)
        except (ValueError, TypeError):
            return None


class _CyclingJob:
    """A class representing a job in the database."""

    def __init__(
        self,
        sample: _Sample,
        job_name: str,
        capacity_Ah: float,
        comment: str,
    ) -> None:
        """Initialize the Job object."""
        self.job_id: str | None = None
        self.jobid_on_server: str | None = None
        self.sample = sample
        self.job_name = job_name
        if not sample.pipeline:
            msg = f"Sample {sample.id} is not loaded on any pipeline."
            raise ValueError(msg)
        self.pipeline: _Pipeline | None = sample.pipeline
        self.capacity_Ah = capacity_Ah
        self.comment = comment
        self.unicycler_protocol: str | None = None
        self.payload: str | Path | dict | None = None

    def submit(self) -> None:
        """Submit the job to the server."""
        # Update the job table in the database

        if self.pipeline and self.pipeline.server:
            self.job_id, self.jobid_on_server, json_string = self.pipeline.server.submit(
                self.sample.id, self.capacity_Ah, self.payload, self.pipeline.name
            )
        else:
            msg = f"Sample {self.sample.id} is not loaded on any pipeline."
            raise ValueError(msg)

        if self.job_id and self.jobid_on_server:
            dbf.execute_sql(
                "INSERT INTO jobs (`Job ID`, `Sample ID`, `Server label`, `Server hostname`, `Job ID on server`, "
                "`Pipeline`, `Submitted`, `Payload`, `Unicycler protocol`, `Capacity (mAh)`, `Comment`) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.job_id,
                    self.sample.id,
                    self.pipeline.server.label,
                    self.pipeline.server.hostname,
                    self.jobid_on_server,
                    self.pipeline.name,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    json_string,
                    self.unicycler_protocol,
                    self.capacity_Ah * 1000,
                    self.comment,
                ),
            )

            self.pipeline.set_jobid(self.job_id, self.jobid_on_server)

    def add_payload(self, payload: str | Path | dict) -> None:
        """Add a payload to the job.

        Args:
            payload : str or Path or dict
                Preferably an aurora-unicycler dictionary - this is auto-converted to the right format for each cycler
                In addition, different cyclers can accept different payload formats
                (Neware) A .xml path or xml string with a Neware protocol
                (Biologic) A .mps path or mps string with a Biologic protocol

        """
        # Try to convert str to Path or dict
        if isinstance(payload, str):
            if payload.endswith((".json", ".mps", ".xml")):
                payload = Path(payload)
            elif payload.startswith("{"):
                payload = json.loads(payload)
        # Convert json path to dict, leave mps and xml as path
        if isinstance(payload, Path):
            if payload.suffix not in {".json", ".mps", ".xml"}:
                msg = "If payload is a path, it must be a json, mps, or xml file."
                raise AssertionError(msg)
            if payload.suffix == ".json":
                with payload.open("r") as f:
                    payload = json.load(f)
        # Convert dict to unicycler
        if isinstance(payload, dict):
            self.unicycler_protocol = Protocol.from_dict(
                payload,
                sample_name=self.sample.id,
                sample_capacity_mAh=self.capacity_Ah * 1000,
            ).model_dump_json(exclude_none=True)
        self.payload = payload

    def cancel(self) -> None:
        """Cancel this job on a server.

        Returns:
            str: The output from the server cancel command

        """
        if not self.pipeline or not self.pipeline.server:
            msg = f"Sample {self.sample.id} is not loaded on any pipeline."
            raise ValueError(msg)

        return self.pipeline.server.cancel(self.job_id, self.jobid_on_server, self.sample.id, self.pipeline.name)

    @classmethod
    def from_id(cls, job_id: str) -> "_CyclingJob":
        """Create a _CyclingJob object from the database.

        Args:
            job_id : str
                The job ID to create the object for.

        Returns:
            _CyclingJob: The _CyclingJob object.

        """
        result = dbf.execute_sql(
            "SELECT `Sample ID`, `Capacity (mAh)`, `Job ID On Server`, `Comment` FROM jobs WHERE `Job ID` = ?",
            (job_id,),
        )
        if not result:
            msg = f"Job '{job_id}' not found in the database."
            raise ValueError(msg)
        sample_id, capacity_mAh, jobid_on_server, comment = result[0]
        sample = _Sample.from_id(sample_id)
        job = cls(
            sample=sample,
            job_name=f"Job for sample {sample.id}",
            capacity_Ah=capacity_mAh * 1e-3,
            comment=comment,
        )
        job.job_id = job_id
        job.jobid_on_server = jobid_on_server

        return job


class ServerManager:
    """The ServerManager class manages the cycling servers."""

    def __init__(self) -> None:
        """Initialize the server manager object."""
        self.config = config.get_config()
        if not self.config.get("Snapshots folder path"):
            msg = "'Snapshots folder path' not found in config file. Cannot save snapshots."
            raise ValueError(msg)
        if not self.config.get("Servers"):
            msg = "No servers in project configuration."
            raise ValueError(msg)
        logger.info("Server manager initialised, consider updating database with update_db()")

    @cached_property
    def servers(self) -> dict[str, CyclerServer]:
        """Get a dictionary of Cycler Servers."""
        return get_servers()

    def update_db(self) -> None:
        """Update all tables in the database."""
        self.update_pipelines()
        self.update_flags()

    def update_pipelines(self) -> None:
        """Update the pipelines table in the database with the current status."""
        for label, server in self.servers.items():
            try:
                status = server.get_pipelines()
            except Exception as e:
                logger.error("Error getting pipeline status from %s: %s", label, e)
                continue
            dt = datetime.now(timezone.utc).isoformat(timespec="seconds")
            if status:
                with sqlite3.connect(self.config["Database path"]) as conn:
                    cursor = conn.cursor()
                    for ready, pipeline, sampleid, jobid_on_server in zip(
                        status["ready"],
                        status["pipeline"],
                        status["sampleid"],
                        status["jobid"],
                        strict=True,
                    ):
                        cursor.execute(
                            "INSERT OR IGNORE INTO pipelines (`Pipeline`) VALUES (?)",
                            (pipeline,),
                        )
                        cursor.execute(
                            "UPDATE pipelines "
                            "SET `Ready` = ?, `Last checked` = ?, `Server label` = ?, "
                            "`Server hostname` = ?, `Server type` = ? "
                            "WHERE `Pipeline` = ?",
                            (ready, dt, label, server.hostname, server.server_type, pipeline),
                        )
                        if sampleid is not None:
                            cursor.execute(
                                "UPDATE pipelines SET `Sample ID` = ? WHERE `Pipeline` = ?",
                                (sampleid, pipeline),
                            )
                        # There is no job running - remove job ids from pipeline
                        if ready == 1:
                            cursor.execute(
                                "UPDATE pipelines SET `Job ID on server` = ?, `Job ID` = ? WHERE `Pipeline` = ?",
                                (None, None, pipeline),
                            )
                        # Update the job id (if it is None, then ignore rather than remove)
                        elif jobid_on_server is not None:
                            cursor.execute(
                                "SELECT `Job ID` FROM jobs "
                                "WHERE `Job ID on server` = ? AND `Sample ID` = ? AND `Pipeline` = ?",
                                (jobid_on_server, sampleid, pipeline),
                            )
                            result = cursor.fetchone()
                            if result:
                                cursor.execute(
                                    "UPDATE pipelines SET `Job ID on server` = ?, `Job ID` = ? WHERE `Pipeline` = ?",
                                    (jobid_on_server, result[0], pipeline),
                                )
                            else:
                                logger.warning(
                                    "No matching Job ID found in database for server '%s' Job ID '%s'.",
                                    label,
                                    jobid_on_server,
                                )
                    conn.commit()

    def update_flags(self) -> None:
        """Update the flags in the pipelines table from the results table."""
        with sqlite3.connect(self.config["Database path"]) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE pipelines SET `Flag` = NULL")
            cursor.execute("SELECT `Pipeline`, `Flag`, `Sample ID` FROM results")
            results = cursor.fetchall()
            for pipeline, flag, sampleid in results:
                cursor.execute(
                    "UPDATE pipelines SET `Flag` = ? WHERE `Pipeline` = ? AND `Sample ID` = ?",
                    (flag, pipeline, sampleid),
                )
            conn.commit()

    def load(self, pipeline_id: str, sample_id: str) -> None:
        """Load a sample on a pipeline.

        The appropriate server is found based on the pipeline, and the sample is loaded.

        Args:
            sample_id (str):
                The sample ID to load. Must exist in samples table of database
            pipeline_id (str):
                The pipeline to load the sample on. Must exist in pipelines table of database

        """
        sample = _Sample.from_id(sample_id)
        pipeline = _Pipeline.from_id(pipeline_id)
        pipeline.load(sample)

    def eject(self, pipeline_id: str, sample_id: str | None = None) -> None:
        """Eject a sample from a pipeline.

        Args:
            pipeline_id (str):
                The pipeline to eject the sample from, must exist in pipelines table of database
            sample_id (str, optional):
                Check that this sample is on the pipeline before ejecting

        """
        _Pipeline.from_id(pipeline_id).eject(sample_id)

    def submit(
        self,
        sample_id: str,
        payload: str | Path | dict,
        capacity_Ah: float | Literal["areal", "mass", "nominal"],
        comment: str = "",
    ) -> None:
        """Submit a job to a server.

        Args:
            sample_id : str
                The sample ID to submit the job for, must exist in samples table of database
            payload : str or Path or dict
                Preferably an aurora-unicycler dictionary - this is auto-converted to the right format for each cycler
                In addition, different cyclers can accept different payload formats
                (Neware) A .xml path or xml string with a Neware protocol
                (Biologic) A .mps path or mps string with a Biologic protocol
            capacity_Ah : float or str
                The capacity of the sample in Ah, if 'areal', 'mass', or 'nominal', the capacity is
                calculated from the sample information
            comment : str, optional
                A comment to add to the job in the database

        """
        sample = _Sample.from_id(sample_id)

        if isinstance(capacity_Ah, str):
            capacity_Ah = sample.get_sample_capacity(capacity_Ah)

        cycling_job = _CyclingJob(
            sample=sample,
            job_name=f"Job for sample {sample.id}",
            capacity_Ah=capacity_Ah,
            comment=comment,
        )
        cycling_job.add_payload(payload)
        cycling_job.submit()

    def cancel(self, jobid: str) -> None:
        """Cancel a job on a server.

        Args:
            jobid : str
                The job ID to cancel, must exist in jobs table of database

        Returns:
            str: The output from the server cancel command

        """
        return _CyclingJob.from_id(jobid).cancel()

    def snapshot(
        self,
        samp_or_jobid: str,
        mode: Literal["always", "new_data", "if_not_exists"] = "new_data",
    ) -> None:
        """Snapshot sample or job, download data, process, and save.

        Args:
            samp_or_jobid : str
                The sample ID or (aurora) job ID to snapshot.
            mode : str, optional
                When to make a new snapshot. Can be one of the following:
                    - 'always': Force a snapshot even if job is already done and data is downloaded.
                    - 'new_data': Snapshot if there is new data.
                    - 'if_not_exists': Snapshot only if the file doesn't exist locally.
                Default is 'new_data'.

        """
        # check if the input is a sample ID
        result = dbf.execute_sql("SELECT `Sample ID` FROM samples WHERE `Sample ID` = ?", (samp_or_jobid,))
        if result:  # it's a sample
            result = dbf.execute_sql(
                "SELECT `Sample ID`, `Status`, `Job ID`, `Job ID on server`, `Server label`, `Snapshot Status` "
                "FROM jobs WHERE `Sample ID` = ? ",
                (samp_or_jobid,),
            )
        else:  # it's a job ID
            result = dbf.execute_sql(
                "SELECT `Sample ID`, `Status`, `Job ID`, `Job ID on server`, `Server label`, `Snapshot Status` "
                "FROM jobs WHERE `Job ID` = ?",
                (samp_or_jobid,),
            )
        if not result:
            msg = f"Sample or job ID '{samp_or_jobid}' not found in the database."
            raise ValueError(msg)

        for sample_id, status, jobid, jobid_on_server, server_label, snapshot_status in result:
            if not sample_id:
                logger.warning("Job %s has no sample, skipping.", jobid)
                continue
            if not jobid_on_server:
                logger.warning("Job %s has no job ID on server, skipping.", jobid)
                continue
            # Check that sample is known
            if sample_id == "Unknown":
                logger.warning("Job %s has no sample name or payload, skipping.", jobid)
                continue
            run_id = run_from_sample(sample_id)

            local_save_location_processed = Path(self.config["Processed snapshots folder path"]) / run_id / sample_id

            files_exist = (local_save_location_processed / f"snapshot.{jobid}.h5").exists()
            if files_exist and mode != "always":
                if mode == "if_not_exists":
                    logger.info("Snapshot for %s already exists, skipping.", jobid)
                    continue
                if mode == "new_data" and snapshot_status is not None and snapshot_status.startswith("c"):
                    logger.info("Snapshot for %s already complete, skipping.", jobid)
                    continue

            # Check that the job has started
            if status in ["q", "qw"]:
                logger.warning("Job %s is still queued, skipping snapshot.", jobid)
                continue

            # Check that the server is accessible
            try:
                server = find_server(server_label)
            except KeyError as e:
                logger.warning("Could not access server %s for job %s: %s", server_label, jobid, e)
                continue

            # Snapshot the job
            try:
                new_snapshot_status = server.snapshot(sample_id, jobid, jobid_on_server)
            except FileNotFoundError as e:
                msg = (
                    f"Error snapshotting {jobid}: {e}\n"
                    "Likely the job was cancelled before starting. "
                    "Setting `Snapshot Status` to 'ce' in the database."
                )
                dbf.execute_sql(
                    "UPDATE jobs SET `Snapshot status` = 'ce' WHERE `Job ID` = ?",
                    (jobid,),
                )
                raise FileNotFoundError(msg) from e

            # Update the snapshot status in the database
            dt = datetime.now(timezone.utc).isoformat(timespec="seconds")
            dbf.execute_sql(
                "INSERT OR IGNORE INTO results (`Sample ID`) VALUES (?)",
                (sample_id,),
            )
            dbf.execute_sql(
                "UPDATE results SET `Last snapshot` = ? WHERE `Sample ID` = ?",
                (dt, sample_id),
            )
            dbf.execute_sql(
                "UPDATE jobs SET `Snapshot status` = ?, `Last snapshot` = ? WHERE `Job ID` = ?",
                (new_snapshot_status, dt, jobid),
            )

        # Analyse the new data (only once per sample)
        for unique_sample_id in {sample_id for sample_id, *_ in result}:
            analysis.analyse_sample(unique_sample_id)

    def snapshot_all(
        self,
        sampleid_contains: str = "",
        mode: Literal["always", "new_data", "if_not_exists"] = "new_data",
    ) -> None:
        """Snapshot all jobs in the database.

        Args:
            sampleid_contains : str, optional
                A string that the sample ID must contain to be snapshot. By default all samples are
                considered for snapshotting.
            mode : str, optional
                When to make a new snapshot. Can be one of the following:
                    - 'always': Force a snapshot even if job is already done and data is downloaded.
                    - 'new_data': Snapshot if there is new data on the server.
                    - 'if_not_exists': Snapshot only if the file doesn't exist locally.
                Default is 'new_data'.

        """
        # Find the jobs that need snapshotting
        if mode not in ["always", "new_data", "if_not_exists"]:
            msg = f"Invalid mode: {mode}. Must be one of 'always', 'new_data', 'if_not_exists'."
            raise ValueError(msg)
        where = "`Status` IN ( 'c', 'r', 'rd', 'cd', 'ce') AND `Sample ID` IS NOT 'Unknown'"
        if mode == "new_data":
            where += " AND (`Snapshot status` NOT LIKE 'c%' OR `Snapshot status` IS NULL)"
        if sampleid_contains:
            result = dbf.execute_sql(
                "SELECT `Job ID` FROM jobs WHERE " + where + " AND `Sample ID` LIKE ?",  # noqa: S608
                (f"%{sampleid_contains}%",),
            )
        else:
            result = dbf.execute_sql("SELECT `Job ID` FROM jobs WHERE " + where)  # noqa: S608
        total_jobs = len(result)
        logger.info("Snapshotting %d jobs with mode '%s'", total_jobs, mode)
        t0 = time()
        # Snapshot each job, ignore errors and continue
        # Sleeps are added after each to not overload the server
        # If snapshotting non-stop for hours, you can stop data transfer in the machine
        # This could result in memory filling up in the cycler and data being lost
        for i, (jobid,) in enumerate(result):
            try:
                self.snapshot(jobid, mode=mode)
            except (KeyError, NotImplementedError, FileNotFoundError) as e:
                logger.exception("Error snapshotting job %s", jobid)
                if isinstance(e, FileNotFoundError):  # Something ran on server, so sleep
                    sleep(10)
                continue
            except Exception as e:
                tb = traceback.format_exc()
                error_message = str(e) or "An error occurred but no message was provided."
                logger.warning("Unexpected error snapshotting %s: %s\n%s", jobid, error_message, tb)
                sleep(10)  # to not overload the server
                continue
            time_elapsed = time() - t0
            time_remaining = time_elapsed / (i + 1) * (total_jobs - i - 1)
            sleep(10)  # to not overload the server
            logger.info("%d/%d jobs done. Approx %d minutes remaining", i + 1, total_jobs, int(time_remaining / 60))

    def _update_neware_jobids(self) -> None:
        """Update all Job IDs on Neware servers.

        Temporary measure until we have a faster way to get Job IDs from Newares
        that can run in update_pipelines. This implementation takes ~1 second
        per channel.

        """
        result = dbf.execute_sql(
            "SELECT `Pipeline`, `Server label` FROM pipelines WHERE "
            "`Sample ID` NOT NULL AND `Ready` = 0 AND `Server type` = 'neware'",
        )
        pipelines = [row[0] for row in result]
        serverids = [row[1] for row in result]
        for serverid, pipeline in zip(serverids, pipelines, strict=True):
            logger.info("Updating job ID for %s on server %s", pipeline, serverid)
            server = find_server(serverid)
            assert isinstance(server, cycler_servers.NewareServer)  # noqa: S101
            jobid_on_server = server._get_job_id(pipeline)  # noqa: SLF001
            full_jobid = f"{server.label}-{jobid_on_server}"
            dbf.execute_sql(
                "UPDATE pipelines SET `Job ID` = ?, `Job ID on server` = ? WHERE `Pipeline` = ?",
                (full_jobid, jobid_on_server, pipeline),
            )
