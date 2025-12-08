"""Copyright © 2025, Empa.

server_manager manages a database and communicates with multiple cycler servers.

This module defines a ServerManager class. The ServerManager object communicates
with multiple CyclerServer objects defined in cycler_servers, and manages the
database of samples, pipelines and jobs.

Server manager takes functions like load, submit, snapshot, update etc. sends
commands to the appropriate server, and handles the database updates.
"""

import contextlib
import logging
import sqlite3
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import sleep, time
from typing import Literal

import pandas as pd
import paramiko
from aurora_unicycler import Protocol

from aurora_cycler_manager import analysis, config, cycler_servers
from aurora_cycler_manager import database_funcs as dbf
from aurora_cycler_manager.utils import run_from_sample

SERVER_CORRESPONDENCE = {
    "neware": cycler_servers.NewareServer,
    "biologic": cycler_servers.BiologicServer,
}

SERVER_OBJECTS: dict[str, cycler_servers.CyclerServer] = {}

logger = logging.getLogger(__name__)


def get_servers() -> dict[str, cycler_servers.CyclerServer]:
    """Create the cycler server objects from the config file."""
    servers: dict[str, cycler_servers.CyclerServer] = {}
    for server_config in config.get_config()["Servers"]:
        if server_config["server_type"] not in SERVER_CORRESPONDENCE:
            logger.error("Server type %s not recognized, skipping", server_config["server_type"])
            continue
        try:
            server_class = SERVER_CORRESPONDENCE[server_config["server_type"]]
            servers[server_config["label"]] = server_class(server_config)
        except (OSError, ValueError, TimeoutError, paramiko.SSHException):
            logger.exception("Server %s could not be created, skipping", server_config["label"])
    return servers


def find_server(label: str) -> cycler_servers.CyclerServer:
    """Get the server object from the label."""
    if not label:
        msg = (
            "No server label found from query, there is probably a mistake in the query. "
            "E.g. if you are searching for a sample ID, the ID might be wrong."
        )
        raise ValueError(msg)
    servers = SERVER_OBJECTS or get_servers()
    server = servers.get(label, None)
    if not server:
        msg = (
            f"Server with label {label} not found. "
            "Either there is a mistake in the label name or you do not have access to the server."
        )
        raise KeyError(msg)
    return server


class Pipeline:
    """A class representing a pipeline in the database."""

    def __init__(self, pipeline_name: str, server_label: str) -> None:
        """Initialize the Pipeline object."""
        self.name = pipeline_name
        self.server = find_server(server_label)
        self.sample: Sample | None = None

    @classmethod
    def from_id(cls, pipeline_name: str) -> "Pipeline":
        """Create a Pipeline object from the database.

        Args:
            pipeline_name : str
                The pipeline name to create the object for.

        Returns:
            Pipeline: The Pipeline object.

        """
        result = dbf.execute_sql(
            "SELECT `Pipeline`, `Server label` FROM pipelines WHERE `Pipeline` = ?",
            (pipeline_name,),
        )
        if not result:
            msg = f"Pipeline '{pipeline_name}' not found in the database."
            raise ValueError(msg)
        return cls(pipeline_name, result[0][1])

    def load(self, sample: "Sample") -> None:
        """Load the sample on a pipeline.

        The appropriate server is found based on the pipeline, and the sample is loaded.

        Args:
            sample (Sample):
                The sample to load on the pipeline.

        """
        if self.sample:
            msg = f"The pipeline {self.name} on server {self.server} already has a sample loaded ({self.sample.id})."
            raise ValueError(msg)

        if sample.pipeline:
            msg = (
                f"Sample {sample.id} is already loaded on pipeline {sample.pipeline.name}, "
                f"server {sample.pipeline.server.label} ."
            )
            raise ValueError(msg)

        self.sample = sample
        sample.pipeline = self

        # Get pipeline and load
        logger.info("Loading sample %s on server %s", self.sample.id, self.server.label)
        dbf.execute_sql(
            "UPDATE pipelines SET `Sample ID` = ? WHERE `Pipeline` = ?",
            (self.sample.id, self.name),
        )

    def eject(self, sample: "Sample | None" = None) -> None:
        """Eject the sample from a pipeline."""
        # Find server associated with pipeline
        logger.info("Ejecting sample from the pipeline %s on server: %s", self.name, self.server.label)
        dbf.execute_sql(
            "UPDATE pipelines SET `Sample ID` = NULL, `Flag` = Null, `Ready` = ? WHERE `Pipeline` = ?",
            (True, self.name),
        )
        if self.sample:
            if sample and self.sample.id != sample.id:
                msg = (
                    f"The pipeline {self.name} on server {self.server.label} has"
                    f" sample {self.sample.id} loaded, not {sample.id}."
                )
                raise ValueError(msg)
            self.sample.pipeline = None
            self.sample = None


class Sample:
    """A class representing a sample in the database."""

    def __init__(self, sample_id: str) -> None:
        """Initialize the Sample object."""
        self.id = sample_id
        self.pipeline = None
        self._properties = {}

    def get_property(self, property_name: str) -> str | None:
        """Get a property of the sample from the database.

        Args:
            property_name : str
                The property name to get.

        Returns:
            str or None: The property value, or None if not found.

        """
        return self._properties.get(property_name)

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
            result = dbf.execute_sql(
                "SELECT "
                "`Anode C-rate definition specific capacity (mAh/g)`, "
                "`Anode active material mass (mg)`, "
                "`Anode diameter (mm)`, "
                "`Cathode C-rate definition specific capacity (mAh/g)`, "
                "`Cathode active material mass (mg)`, "
                "`Cathode diameter (mm)` "
                "FROM samples WHERE `Sample ID` = ?",
                (self.id,),
            )
        elif mode == "areal":
            result = dbf.execute_sql(
                "SELECT "
                "`Anode C-rate definition areal capacity (mAh/cm2)`, "
                "`Anode diameter (mm)`, "
                "`Cathode C-rate definition areal capacity (mAh/cm2)`, "
                "`Cathode diameter (mm)` "
                "FROM samples WHERE `Sample ID` = ?",
                (self.id,),
            )
        elif mode == "nominal":
            result = dbf.execute_sql(
                "SELECT `C-rate definition capacity (mAh)` FROM samples WHERE `Sample ID` = ?",
                (self.id,),
            )
        if not result:
            msg = f"Sample '{self.id}' not found in the database."
            raise ValueError(msg)
        if mode == "mass":
            an_cap_mAh_g, an_mass_mg, an_diam_mm, cat_cap_mAh_g, cat_mass_mg, cat_diam_mm = result[0]
            cat_frac_used = min(1, an_diam_mm**2 / cat_diam_mm**2)
            cat_cap_Ah = cat_frac_used * (cat_cap_mAh_g * cat_mass_mg * 1e-6)
            capacity_Ah = cat_cap_Ah
            if not ignore_anode:
                an_frac_used = min(1, cat_diam_mm**2 / an_diam_mm**2)
                an_cap_Ah = an_frac_used * (an_cap_mAh_g * an_mass_mg * 1e-6)
                capacity_Ah = min(an_cap_Ah, cat_cap_Ah)
        elif mode == "areal":
            an_cap_mAh_cm2, an_diam_mm, cat_cap_mAh_cm2, cat_diam_mm = result[0]
            cat_frac_used = min(1, an_diam_mm**2 / cat_diam_mm**2)
            cat_cap_Ah = cat_frac_used * cat_cap_mAh_cm2 * (cat_diam_mm / 2) ** 2 * 3.14159 * 1e-5
            capacity_Ah = cat_cap_Ah
            if not ignore_anode:
                an_frac_used = min(1, cat_diam_mm**2 / an_diam_mm**2)
                an_cap_Ah = an_frac_used * an_cap_mAh_cm2 * (an_diam_mm / 2) ** 2 * 3.14159 * 1e-5
                capacity_Ah = min(an_cap_Ah, cat_cap_Ah)
        elif mode == "nominal":
            capacity_Ah = result[0][0] * 1e-3
        return capacity_Ah

    @classmethod
    def from_id(cls, sample_id: str) -> "Sample":
        """Create a Sample object from the database.

        Args:
            sample_id : str
                The sample ID to create the object for.

        Returns:
            Sample: The Sample object.

        """
        # Check if sample exists in database
        result = dbf.execute_sql(
            "SELECT `Sample ID` FROM samples WHERE `Sample ID` = ?",
            (sample_id,),
        )
        if not result:
            msg = f"Sample '{sample_id}' not found in the database."
            raise ValueError(msg)

        # Check if the sample is loaded on a pipeline
        result = dbf.execute_sql(
            "SELECT `Server label`, `Pipeline` FROM pipelines WHERE `Sample ID` = ?",
            (sample_id,),
        )

        if result:
            server_label, pipeline = result[0]
            sample = cls(sample_id)
            sample.pipeline = Pipeline.from_id(pipeline)
        else:
            sample = cls(sample_id)

        # TODO: Load sample properties into _properties dict
        sample._properties = {}

        return sample

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


class CyclingJob:
    """A class representing a job in the database."""

    def __init__(
        self,
        sample: Sample,
        job_name: str,
        capacity_Ah: float,
        comment: str,
    ) -> None:
        """Initialize the Job object."""
        self.job_id: str | None = None
        self.jobid_on_server: str | None = None
        self.sample = sample
        self.job_name = job_name
        self.pipeline = sample.pipeline
        self.capacity_Ah = capacity_Ah
        self.comment = comment
        self.unicycler_protocol: str | None = None
        self.payload: str | Path | dict | None = None

    def submit(self) -> None:
        """Submit the job to the server."""
        # Update the job table in the database

        self.job_id, self.jobid_on_server, json_string = self.pipeline.server.submit(
            self.sample.id, self.capacity_Ah, self.payload, self.pipeline.name
        )

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

    def add_payload(self, payload: str | Path | dict) -> None:
        """Add a payload to the job.

        Args:
            payload : str or Path or dict
                Preferably an aurora-unicycler dictionary - this is auto-converted to the right format for each cycler
                In addition, different cyclers can accept different payload formats
                (Neware) A .xml path or xml string with a Neware protocol
                (Biologic) A .mps path or mps string with a Biologic protocol

        """
        # Check if the payload is a unicycler protocol
        if isinstance(payload, dict):
            with contextlib.suppress(ValueError, AttributeError, KeyError):
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
        return self.sample.pipeline.server.cancel(
            self.job_id, self.jobid_on_server, self.sample.id, self.sample.pipeline.name
        )

    @classmethod
    def from_id(cls, job_id: str) -> "CyclingJob":
        """Create a CyclingJob object from the database.

        Args:
            job_id : str
                The job ID to create the object for.

        Returns:
            CyclingJob: The CyclingJob object.

        """
        result = dbf.execute_sql(
            "SELECT `Sample ID`, `Server label`, `Pipeline`, `Capacity (mAh)`, `Job ID On Server`, `Comment` "
            "FROM jobs WHERE `Job ID` = ?",
            (job_id,),
        )
        if not result:
            msg = f"Job '{job_id}' not found in the database."
            raise ValueError(msg)
        sample_id, server_label, pipeline, capacity_mAh, jobid_on_server, comment = result[0]
        sample = Sample.from_id(sample_id)
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
        logger.info("Creating cycler server objects")
        self.config = config.get_config()
        if not self.config.get("Snapshots folder path"):
            msg = "'Snapshots folder path' not found in config file. Cannot save snapshots."
            raise ValueError(msg)
        self.servers = get_servers()
        if not self.servers:
            msg = "No servers found in config file, please check the config file."
            raise ValueError(msg)
        logger.info("Server manager initialised, consider updating database with update_db()")

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

    def load(self, sample_id: str, pipeline: str) -> None:
        """Load a sample on a pipeline.

        The appropriate server is found based on the pipeline, and the sample is loaded.

        Args:
            sample_id (str):
                The sample ID to load. Must exist in samples table of database
            pipeline (str):
                The pipeline to load the sample on. Must exist in pipelines table of database

        """
        sample = Sample.from_id(sample_id)
        pipeline = Pipeline.from_id(pipeline)
        pipeline.load(sample)

    def eject(self, sample_id: str, pipeline_id: str) -> None:
        """Eject a sample from a pipeline.

        Args:
            sample_id (str):
                The sample ID to eject. Must exist in samples table of database
            pipeline_id (str):
                The pipeline to eject the sample from, must exist in pipelines table of database

        """
        sample = Sample.from_id(sample_id)
        pipeline = Pipeline.from_id(pipeline_id)
        pipeline.eject(sample)

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
        sample = Sample.from_id(sample_id)
        cycling_job = CyclingJob(
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
        return CyclingJob.from_id(jobid).cancel()


class ServerManagerX:
    """The ServerManager class manages the database and cycling servers.

    This class is used in the app and the daemon. The class can also be used
    directly in scripts for finer control over the cyclers.

    Typical usage in a script:

        # This will connect to servers and update the database
        sm = ServerManager()

        # Load a sample, submit a job, ready the pipeline
        sm.load("sample_id", "pipeline_name")
        sm.submit("sample_id", payload_dict, sample_capacity_Ah)
        sm.ready("pipeline_name")

        # Update the database to check status of jobs and pipelines
        sm.update_db()

        # Snapshot a job or sample to get the data
        sm.snapshot("sample_id")

        # Use analysis.analyse_sample("sample_id") to analyse the data
        # After this it can be viewed in the app, or read the hdf/json files
        # directly
    """

    def __init__(self) -> None:
        """Initialize the server manager object."""
        logger.info("Creating cycler server objects")
        self.config = config.get_config()
        if not self.config.get("Snapshots folder path"):
            msg = "'Snapshots folder path' not found in config file. Cannot save snapshots."
            raise ValueError(msg)
        self.servers = self.get_servers()
        if not self.servers:
            msg = "No servers found in config file, please check the config file."
            raise ValueError(msg)
        logger.info("Server manager initialised, consider updating database with update_db()")

    def get_servers(self) -> dict[str, cycler_servers.CyclerServer]:
        """Create the cycler server objects from the config file."""
        servers: dict[str, cycler_servers.CyclerServer] = {}
        for server_config in self.config["Servers"]:
            if server_config["server_type"] not in SERVER_CORRESPONDENCE:
                logger.error("Server type %s not recognized, skipping", server_config["server_type"])
                continue
            try:
                server_class = SERVER_CORRESPONDENCE[server_config["server_type"]]
                servers[server_config["label"]] = server_class(server_config)
            except (OSError, ValueError, TimeoutError, paramiko.SSHException):
                logger.exception("Server %s could not be created, skipping", server_config["label"])

        return servers

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

    def update_db(self) -> None:
        """Update all tables in the database."""
        self.update_pipelines()
        self.update_flags()

    def find_server(self, label: str) -> cycler_servers.CyclerServer:
        """Get the server object from the label."""
        if not label:
            msg = (
                "No server label found from query, there is probably a mistake in the query. "
                "E.g. if you are searching for a sample ID, the ID might be wrong."
            )
            raise ValueError(msg)
        server = self.servers.get(label, None)
        if not server:
            msg = (
                f"Server with label {label} not found. "
                "Either there is a mistake in the label name or you do not have access to the server."
            )
            raise KeyError(msg)
        return server

    @staticmethod
    def sort_pipeline(df: pd.DataFrame) -> pd.DataFrame:
        """Sorting pipelines so e.g. MPG2-1-2 comes before MPG2-1-10."""

        def custom_sort(x: str) -> int | str:
            try:
                numbers = x.split("-")[-2:]
                return 1000 * int(numbers[0]) + int(numbers[1])
            except ValueError:
                return x

        df = df.sort_values(by="Pipeline", key=lambda x: x.map(custom_sort))
        return df.reset_index(drop=True)

    def get_pipelines(self) -> pd.DataFrame:
        """Return the status of all pipelines as a DataFrame."""
        columns = ["Pipeline", "Sample ID", "Job ID on server", "Server label"]
        result = dbf.execute_sql(
            "SELECT `Pipeline`, `Sample ID`, `Job ID on server`, `Server label` FROM pipelines",
        )
        return self.sort_pipeline(pd.DataFrame(result, columns=columns))

    def load(self, sample: str, pipeline: str) -> None:
        """Load a sample on a pipeline.

        The appropriate server is found based on the pipeline, and the sample is loaded.

        Args:
            sample (str):
                The sample ID to load. Must exist in samples table of database
            pipeline (str):
                The pipeline to load the sample on. Must exist in pipelines table of database

        """
        # Check if sample exists in database
        result = dbf.execute_sql("SELECT `Sample ID` FROM samples WHERE `Sample ID` = ?", (sample,))
        # Get pipeline and load
        result = dbf.execute_sql("SELECT `Server label` FROM pipelines WHERE `Pipeline` = ?", (pipeline,))
        server = self.find_server(result[0][0])
        logger.info("Loading sample %s on server %s", sample, server.label)
        dbf.execute_sql(
            "UPDATE pipelines SET `Sample ID` = ? WHERE `Pipeline` = ?",
            (sample, pipeline),
        )

    def eject(self, pipeline: str) -> None:
        """Eject a sample from a pipeline.

        Args:
            pipeline (str):
                The pipeline to eject the sample from, must exist in pipelines table of database

        """
        # Find server associated with pipeline
        result = dbf.execute_sql("SELECT `Server label` FROM pipelines WHERE `Pipeline` = ?", (pipeline,))
        server = self.find_server(result[0][0])
        logger.info("Ejecting %s on server: %s", pipeline, server.label)
        dbf.execute_sql(
            "UPDATE pipelines SET `Sample ID` = NULL, `Flag` = Null, `Ready` = ? WHERE `Pipeline` = ?",
            (True, pipeline),
        )

    def submit(
        self,
        sample: str,
        payload: str | Path | dict,
        capacity_Ah: float | Literal["areal", "mass", "nominal"],
        comment: str = "",
    ) -> None:
        """Submit a job to a server.

        Args:
            sample : str
                The sample ID to submit the job for, must exist in samples table of database
            payload : str or dict
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
        # Get the sample capacity
        if isinstance(capacity_Ah, str) and capacity_Ah in ["areal", "mass", "nominal"]:
            capacity_Ah = self.get_sample_capacity(sample, capacity_Ah)
        elif not isinstance(capacity_Ah, float):
            msg = f"Capacity {capacity_Ah} must be 'areal', 'mass', or a float in Ah."
            raise ValueError(msg)
        if capacity_Ah > 0.05:
            msg = f"Capacity {capacity_Ah} too large - value must be in Ah, not mAh"
            raise ValueError(msg)

        # Find the server with the sample loaded, if there is more than one throw an error
        result = dbf.execute_sql("SELECT `Server label`, `Pipeline` FROM pipelines WHERE `Sample ID` = ?", (sample,))
        if len(result) > 1:
            msg = f"Sample {sample} is loaded on more than one server, cannot submit job."
            raise ValueError(msg)
        server = self.find_server(result[0][0])
        pipeline = result[0][1]

        # Check if the payload is a unicycler protocol
        unicycler_protocol: str | None = None
        if isinstance(payload, dict):
            with contextlib.suppress(ValueError, AttributeError, KeyError):
                unicycler_protocol = Protocol.from_dict(
                    payload,
                    sample_name=sample,
                    sample_capacity_mAh=capacity_Ah * 1000,
                ).model_dump_json(exclude_none=True)

        logger.info("Submitting job to %s with capacity %.5f Ah", sample, capacity_Ah)
        full_jobid, jobid_on_server, json_string = server.submit(sample, capacity_Ah, payload, pipeline)
        dt = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # Update the job table in the database
        if full_jobid and jobid_on_server:
            dbf.execute_sql(
                "INSERT INTO jobs (`Job ID`, `Sample ID`, `Server label`, `Server hostname`, `Job ID on server`, "
                "`Pipeline`, `Submitted`, `Payload`, `Unicycler protocol`, `Capacity (mAh)`, `Comment`) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    full_jobid,
                    sample,
                    server.label,
                    server.hostname,
                    jobid_on_server,
                    pipeline,
                    dt,
                    json_string,
                    unicycler_protocol,
                    capacity_Ah * 1000,
                    comment,
                ),
            )
            # Neware and Biologic servers have very expensive job id retrieval
            # It costs around 1 second to get the job id for one channel, so cannot do this in update_pipelines
            # Just do it once on job submission and don't update until job is finished
            if isinstance(server, (cycler_servers.NewareServer, cycler_servers.BiologicServer)):
                dbf.execute_sql(
                    "UPDATE pipelines SET `Job ID` = ?, `Job ID on server` = ?, `Ready` = 0 WHERE `Pipeline` = ?",
                    (full_jobid, jobid_on_server, pipeline),
                )

    def cancel(self, jobid: str) -> None:
        """Cancel a job on a server.

        Args:
            jobid : str
                The job ID to cancel, must exist in jobs table of database

        Returns:
            str: The output from the server cancel command

        """
        result = dbf.execute_sql(
            "SELECT `Server label`, `Job ID on server`, `Sample ID`, `Pipeline` FROM jobs WHERE `Job ID` = ?",
            (jobid,),
        )
        server_label, jobid_on_server, sampleid, pipeline = result[0]
        server = self.find_server(server_label)
        output = server.cancel(jobid, jobid_on_server, sampleid, pipeline)
        # If no error, assume job is cancelled and update the database
        dbf.execute_sql(
            "UPDATE jobs SET `Status` = 'cd' WHERE `Job ID` = ?",
            (jobid,),
        )
        return output

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
                server = self.find_server(server_label)
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
        if mode in ["new_data"]:
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
                error_message = str(e) if str(e) else "An error occurred but no message was provided."
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
            server = self.find_server(serverid)
            assert isinstance(server, cycler_servers.NewareServer)  # noqa: S101
            jobid_on_server = server._get_job_id(pipeline)  # noqa: SLF001
            full_jobid = f"{server.label}-{jobid_on_server}"
            dbf.execute_sql(
                "UPDATE pipelines SET `Job ID` = ?, `Job ID on server` = ? WHERE `Pipeline` = ?",
                (full_jobid, jobid_on_server, pipeline),
            )
