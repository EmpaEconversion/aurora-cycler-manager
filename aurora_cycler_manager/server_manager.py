"""Copyright © 2025, Empa.

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
import warnings
from datetime import datetime
from pathlib import Path
from time import sleep, time
from typing import Literal

import pandas as pd
import paramiko

from aurora_cycler_manager.analysis import analyse_sample
from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.cycler_servers import BiologicServer, CyclerServer, NewareServer, TomatoServer
from aurora_cycler_manager.utils import run_from_sample

CONFIG = get_config()
logger = logging.getLogger(__name__)


class ServerManager:
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
        self.config = CONFIG
        if not self.config.get("Snapshots folder path"):
            msg = "'Snapshots folder path' not found in config file. Cannot save snapshots."
            raise ValueError(msg)
        self.servers = self.get_servers()
        if not self.servers:
            msg = "No servers found in config file, please check the config file."
            raise ValueError(msg)
        logger.info("Server manager initialised, consider updating database with update_db()")

    def get_servers(self) -> dict[str, CyclerServer]:
        """Create the cycler server objects from the config file."""
        servers: dict[str, CyclerServer] = {}
        for server_config in self.config["Servers"]:
            if server_config["server_type"] == "tomato":
                try:
                    servers[server_config["label"]] = TomatoServer(server_config)
                except (OSError, ValueError, TimeoutError, paramiko.SSHException):
                    logger.exception("Server %s could not be created, skipping", server_config["label"])
            elif server_config["server_type"] == "neware":
                try:
                    servers[server_config["label"]] = NewareServer(server_config)
                except (OSError, ValueError, TimeoutError, paramiko.SSHException):
                    logger.exception("Server %s could not be created, skipping", server_config["label"])
            elif server_config["server_type"] == "biologic":
                try:
                    servers[server_config["label"]] = BiologicServer(server_config)
                except (OSError, ValueError, TimeoutError, paramiko.SSHException):
                    logger.exception("Server %s could not be created, skipping", server_config["label"])
            else:
                logger.error("Server type %s not recognized, skipping", server_config["server_type"])
        return servers

    def update_jobs(self) -> None:
        """Update the jobs table in the database with the current job status."""
        for label, server in self.servers.items():
            try:
                jobs = server.get_jobs()
            except Exception as e:
                logger.error("Error getting job status from %s: %s", label, e)
                continue
            dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            hostname = server.hostname
            if jobs:
                with sqlite3.connect(self.config["Database path"]) as conn:
                    cursor = conn.cursor()
                    for jobid_on_server, jobname, status, pipeline in zip(
                        jobs["jobid"],
                        jobs["jobname"],
                        jobs["status"],
                        jobs["pipeline"],
                        strict=True,
                    ):
                        # Insert the job if it does not exist
                        cursor.execute(
                            "INSERT OR IGNORE INTO jobs (`Job ID`,`Job ID on server`) VALUES (?,?)",
                            (f"{label}-{jobid_on_server}", jobid_on_server),
                        )
                        # If pipeline is none, do not update (keep old value)
                        if pipeline is None:
                            cursor.execute(
                                "UPDATE jobs "
                                "SET `Status` = ?, `Jobname` = ?, `Server label` = ?, "
                                "`Server hostname` = ?, `Last checked` = ? "
                                "WHERE `Job ID` = ?",
                                (status, jobname, label, hostname, dt, f"{label}-{jobid_on_server}"),
                            )
                        else:
                            cursor.execute(
                                "UPDATE jobs "
                                "SET `Status` = ?, `Pipeline` = ?, `Jobname` = ?, `Server label` = ?, "
                                "`Server Hostname` = ?, `Job ID on server` = ?, "
                                "`Last Checked` = ? "
                                "WHERE `Job ID` = ?",
                                (
                                    status,
                                    pipeline,
                                    jobname,
                                    label,
                                    hostname,
                                    jobid_on_server,
                                    dt,
                                    f"{label}-{jobid_on_server}",
                                ),
                            )
                    conn.commit()

    def update_pipelines(self) -> None:
        """Update the pipelines table in the database with the current status."""
        for label, server in self.servers.items():
            try:
                status = server.get_pipelines()
            except Exception as e:
                logger.error("Error getting pipeline status from %s: %s", label, e)
                continue
            dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            hostname = server.hostname
            server_type = server.server_type
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
                            (ready, dt, label, hostname, server_type, pipeline),
                        )
                        # Need to treat nulls from tomato and Neware/Biologic differently
                        # Tomato null means sample removed, Neware/Biologic means no update
                        if server_type == "tomato" or sampleid is not None:
                            cursor.execute(
                                "UPDATE pipelines SET `Sample ID` = ? WHERE `Pipeline` = ?",
                                (sampleid, pipeline),
                            )
                        if (
                            server_type == "tomato"
                            or jobid_on_server is not None
                            or (server_type == "neware" and ready == 1)
                            or (server_type == "biologic" and ready == 1)
                        ):
                            jobid = f"{label}-{jobid_on_server}" if jobid_on_server else None
                            cursor.execute(
                                "UPDATE pipelines SET `Job ID on server` = ?, `Job ID` = ? WHERE `Pipeline` = ?",
                                (jobid_on_server, jobid, pipeline),
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
        self.update_jobs()
        self.update_flags()
        self.update_all_payloads()

    def execute_sql(self, query: str, params: tuple | None = None) -> list[tuple]:
        """Execute a query on the database.

        Args:
            query : str
                The query to execute
            params : tuple, optional
                The parameters to pass to the query

        Returns:
            list[tuple] : the result of the query

        """
        commit_keywords = ["UPDATE", "INSERT", "DELETE", "REPLACE", "CREATE", "DROP", "ALTER"]
        commit = any(keyword in query.upper() for keyword in commit_keywords)
        with sqlite3.connect(self.config["Database path"]) as conn:
            cursor = conn.cursor()
            if params is not None:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            if commit:
                conn.commit()
            return cursor.fetchall()

    def find_server(self, label: str) -> CyclerServer:
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

        def custom_sort(x: str):  # noqa: ANN202
            try:
                numbers = x.split("-")[-2:]
                return 1000 * int(numbers[0]) + int(numbers[1])
            except ValueError:
                return x

        df = df.sort_values(by="Pipeline", key=lambda x: x.map(custom_sort))
        return df.reset_index(drop=True)

    @staticmethod
    def sort_job(df: pd.DataFrame) -> pd.DataFrame:
        """Sort jobs so servers are grouped together and jobs are sorted by number."""

        def custom_sort(x: str):  # noqa: ANN202
            try:
                server, number = x.rsplit("-", 1)
                return (server, int(number))
            except ValueError:
                return (x, 0)

        return df.sort_values(by="Job ID", key=lambda x: x.map(custom_sort))

    def get_pipelines(self) -> pd.DataFrame:
        """Return the status of all pipelines as a DataFrame."""
        columns = ["Pipeline", "Sample ID", "Job ID on server", "Server label"]
        result = self.execute_sql("SELECT `Pipeline`, `Sample ID`, `Job ID on server`, `Server label` FROM pipelines")
        return self.sort_pipeline(pd.DataFrame(result, columns=columns))

    def get_queue(self) -> pd.DataFrame:
        """Return all running and queued jobs as a DataFrame."""
        columns = ["Job ID", "Sample ID", "Status", "Server label"]
        result = self.execute_sql(
            "SELECT `Job ID`, `Sample ID`, `Status`, `Server label` FROM jobs WHERE `Status` IN ('q', 'qw', 'r', 'rd')",
        )
        return self.sort_job(pd.DataFrame(result, columns=columns))

    def get_jobs(self) -> pd.DataFrame:
        """Return all jobs as a DataFrame."""
        columns = ["Job ID", "Sample ID", "Status", "Server label"]
        result = self.execute_sql(
            "SELECT `Job ID`, `Sample ID`, `Status`, `Server label` FROM jobs "
            "WHERE `Status` IN ('q', 'qw', 'r', 'rd', 'c', 'cd')",
        )
        return self.sort_job(pd.DataFrame(result, columns=columns))

    def get_sample_capacity(
        self,
        sample: str,
        mode: Literal["areal", "mass", "nominal"],
        ignore_anode: bool = True,
    ) -> float:
        """Get the capacity of a sample in Ah based on the mode.

        Args:
            sample : str
                The sample ID to get the capacity for
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
            result = self.execute_sql(
                "SELECT "
                "`Anode C-rate definition specific capacity (mAh/g)`, "
                "`Anode active material mass (mg)`, "
                "`Anode diameter (mm)`, "
                "`Cathode C-rate definition specific capacity (mAh/g)`, "
                "`Cathode active material mass (mg)`, "
                "`Cathode diameter (mm)` "
                "FROM samples WHERE `Sample ID` = ?",
                (sample,),
            )
        elif mode == "areal":
            result = self.execute_sql(
                "SELECT "
                "`Anode C-rate definition areal capacity (mAh/cm2)`, "
                "`Anode diameter (mm)`, "
                "`Cathode C-rate definition areal capacity (mAh/cm2)`, "
                "`Cathode diameter (mm)` "
                "FROM samples WHERE `Sample ID` = ?",
                (sample,),
            )
        elif mode == "nominal":
            result = self.execute_sql(
                "SELECT `C-rate definition capacity (mAh)` FROM samples WHERE `Sample ID` = ?",
                (sample,),
            )
        if not result:
            msg = f"Sample '{sample}' not found in the database."
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

    def safe_get_sample_capacity(
        self,
        sample: str,
        mode: Literal["areal", "mass", "nominal"],
        ignore_anode: bool = True,
    ) -> float | None:
        """Get the capacity of a sample in Ah based on the mode. Returns None if failed."""
        try:
            return self.get_sample_capacity(sample, mode, ignore_anode)
        except (ValueError, TypeError):
            return None

    def load(self, sample: str, pipeline: str) -> str:
        """Load a sample on a pipeline.

        The appropriate server is found based on the pipeline, and the sample is loaded.

        Args:
            sample (str):
                The sample ID to load. Must exist in samples table of database
            pipeline (str):
                The pipeline to load the sample on. Must exist in pipelines table of database

        Returns:
            The output from the server load command as a string

        """
        # Check if sample exists in database
        result = self.execute_sql("SELECT `Sample ID` FROM samples WHERE `Sample ID` = ?", (sample,))
        # Get pipeline and load
        result = self.execute_sql("SELECT `Server label` FROM pipelines WHERE `Pipeline` = ?", (pipeline,))
        server = self.find_server(result[0][0])
        logger.info("Loading sample %s on server %s", sample, server.label)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = server.load(sample, pipeline)
            if w:
                for warning in w:
                    logger.warning(warning.message)
            else:
                # Update database preemtively if no warnings were caught
                self.execute_sql(
                    "UPDATE pipelines SET `Sample ID` = ? WHERE `Pipeline` = ?",
                    (sample, pipeline),
                )
        return output

    def eject(self, pipeline: str) -> str:
        """Eject a sample from a pipeline.

        Args:
            pipeline (str):
                The pipeline to eject the sample from, must exist in pipelines table of database

        Returns:
            The output from the server eject command as a string

        """
        # Find server associated with pipeline
        result = self.execute_sql("SELECT `Server label` FROM pipelines WHERE `Pipeline` = ?", (pipeline,))
        server = self.find_server(result[0][0])
        logger.info("Ejecting %s on server: %s", pipeline, server.label)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = server.eject(pipeline)
            if w:
                for warning in w:
                    logger.warning(warning.message)
            else:
                # Update database preemtively if no warnings were caught
                ready = server.server_type in ["neware", "biologic"]  # Neware/Biologic -> ready if no sample
                self.execute_sql(
                    "UPDATE pipelines SET `Sample ID` = NULL, `Flag` = Null, `Ready` = ? WHERE `Pipeline` = ?",
                    (ready, pipeline),
                )
        return output

    def ready(self, pipeline: str) -> str:
        """Ready a pipeline for a new job.

        Args:
            pipeline (str):
                The pipeline to ready, must exist in pipelines table of database

        Returns:
            The output from the server ready command as a string

        """
        # find server with pipeline, if there is more than one throw an error
        result = self.execute_sql("SELECT `Server label` FROM pipelines WHERE `Pipeline` = ?", (pipeline,))
        server = self.find_server(result[0][0])
        logger.info("Readying %s on server: %s", pipeline, server.label)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = server.ready(pipeline)
            if w:
                for warning in w:
                    logger.warning(warning.message)
            else:
                # Update database preemtively if no warnings caught
                self.execute_sql(
                    "UPDATE pipelines SET `Ready` = 1 WHERE `Pipeline` = ?",
                    (pipeline,),
                )
        return output

    def unready(self, pipeline: str) -> str:
        """Unready a pipeline, only works if no job running, if job is running user must cancel.

        Args:
            pipeline (str):
                The pipeline to unready, must exist in pipelines table of database

        Returns:
            The output from the server unready command as a string

        """
        # Find server with pipeline, if there is more than one throw an error
        result = self.execute_sql("SELECT `Server label` FROM pipelines WHERE `Pipeline` = ?", (pipeline,))
        server = self.find_server(result[0][0])
        logger.info("Unreadying %s on server: %s", pipeline, server.label)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = server.unready(pipeline)
            if w:
                for warning in w:
                    logger.warning(warning.message)
            else:
                # Update database preemtively if no warnings caught
                self.execute_sql(
                    "UPDATE pipelines SET `Ready` = 0 WHERE `Pipeline` = ?",
                    (pipeline,),
                )
        return output

    def submit(
        self,
        sample: str,
        payload: str | dict,
        capacity_Ah: float | Literal["areal", "mass", "nominal"],
        comment: str = "",
    ) -> None:
        """Submit a job to a server.

        Args:
            sample : str
                The sample ID to submit the job for, must exist in samples table of database
            payload : str or dict
                (tomato) A .json path, json string, or dictionary with payload to submit to the server
                (Neware) A .xml path or xml string with payload to submit to the server
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
        result = self.execute_sql("SELECT `Server label`, `Pipeline` FROM pipelines WHERE `Sample ID` = ?", (sample,))
        if len(result) > 1:
            msg = f"Sample {sample} is loaded on more than one server, cannot submit job."
            raise ValueError(msg)
        server = self.find_server(result[0][0])
        pipeline = result[0][1]

        logger.info("Submitting job to %s with capacity %.5f Ah", sample, capacity_Ah)
        full_jobid, jobid_on_server, json_string = server.submit(sample, capacity_Ah, payload, pipeline)
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Update the job table in the database
        if full_jobid and jobid_on_server:
            self.execute_sql(
                "INSERT INTO jobs (`Job ID`, `Sample ID`, `Server label`, `Server hostname`, `Job ID on server`, "
                "`Pipeline`, `Submitted`, `Payload`, `Comment`) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    full_jobid,
                    sample,
                    server.label,
                    server.hostname,
                    jobid_on_server,
                    pipeline,
                    dt,
                    json_string,
                    comment,
                ),
            )
            # Neware and Biologic servers have very expensive job id retrieval
            # It costs around 1 second to get the job id for one channel, so cannot do this in update_pipelines
            # Just do it once on job submission and don't update until job is finished
            if isinstance(server, (NewareServer, BiologicServer)):
                self.execute_sql(
                    "UPDATE pipelines SET `Job ID` = ?, `Job ID on server` = ?, `Ready` = 0 WHERE `Pipeline` = ?",
                    (full_jobid, jobid_on_server, pipeline),
                )

    def cancel(self, jobid: str) -> str:
        """Cancel a job on a server.

        Args:
            jobid : str
                The job ID to cancel, must exist in jobs table of database

        Returns:
            str: The output from the server cancel command

        """
        result = self.execute_sql(
            "SELECT `Server label`, `Job ID on server`, `Sample ID`, `Pipeline` FROM jobs WHERE `Job ID` = ?",
            (jobid,),
        )
        server_label, jobid_on_server, sampleid, pipeline = result[0]
        server = self.find_server(server_label)
        output = server.cancel(jobid_on_server, sampleid, pipeline)
        # If no error, assume job is cancelled and update the database
        self.execute_sql(
            "UPDATE jobs SET `Status` = 'cd' WHERE `Job ID` = ?",
            (jobid,),
        )
        return output

    def snapshot(
        self,
        samp_or_jobid: str,
        get_raw: bool = False,
        mode: Literal["always", "new_data", "if_not_exists"] = "new_data",
    ) -> None:
        """Snapshot sample or job, download data, process, and save.

        Args:
            samp_or_jobid : str
                The sample ID or (aurora) job ID to snapshot.
            get_raw : bool, optional
                If True, also download the raw data as a zip.
            mode : str, optional
                When to make a new snapshot. Can be one of the following:
                    - 'always': Force a snapshot even if job is already done and data is downloaded.
                    - 'new_data': Snapshot if there is new data.
                    - 'if_not_exists': Snapshot only if the file doesn't exist locally.
                Default is 'new_data'.

        """
        # check if the input is a sample ID
        result = self.execute_sql("SELECT `Sample ID` FROM samples WHERE `Sample ID` = ?", (samp_or_jobid,))
        if result:  # it's a sample
            result = self.execute_sql(
                "SELECT `Sample ID`, `Status`, `Job ID`, `Job ID on server`, `Server label`, `Snapshot Status` "
                "FROM jobs WHERE `Sample ID` = ? ",
                (samp_or_jobid,),
            )
        else:  # it's a job ID
            result = self.execute_sql(
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
                new_snapshot_status = server.snapshot(sample_id, jobid, jobid_on_server, get_raw=get_raw)
            except FileNotFoundError as e:
                msg = (
                    f"Error snapshotting {jobid}: {e}\n"
                    "Likely the job was cancelled before starting. "
                    "Setting `Snapshot Status` to 'ce' in the database."
                )
                self.execute_sql(
                    "UPDATE jobs SET `Snapshot status` = 'ce' WHERE `Job ID` = ?",
                    (jobid,),
                )
                raise FileNotFoundError(msg) from e

            # Update the snapshot status in the database
            dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.execute_sql(
                "INSERT OR IGNORE INTO results (`Sample ID`) VALUES (?)",
                (sample_id,),
            )
            self.execute_sql(
                "UPDATE results SET `Last snapshot` = ? WHERE `Sample ID` = ?",
                (dt, sample_id),
            )
            self.execute_sql(
                "UPDATE jobs SET `Snapshot status` = ?, `Last snapshot` = ? WHERE `Job ID` = ?",
                (new_snapshot_status, dt, jobid),
            )

        # Analyse the new data (only once per sample)
        for unique_sample_id in {sample_id for sample_id, *_ in result}:
            analyse_sample(unique_sample_id)

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
            result = self.execute_sql(
                "SELECT `Job ID` FROM jobs WHERE " + where + " AND `Sample ID` LIKE ?",  # noqa: S608
                (f"%{sampleid_contains}%",),
            )
        else:
            result = self.execute_sql("SELECT `Job ID` FROM jobs WHERE " + where)  # noqa: S608
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
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc()
                error_message = str(e) if str(e) else "An error occurred but no message was provided."
                logger.warning("Unexpected error snapshotting %s: %s\n%s", jobid, error_message, tb)
                sleep(10)  # to not overload the server
                continue
            time_elapsed = time() - t0
            time_remaining = time_elapsed / (i + 1) * (total_jobs - i - 1)
            sleep(10)  # to not overload the server
            logger.info("%d/%d jobs done. Approx %d minutes remaining", i + 1, total_jobs, int(time_remaining / 60))

    def get_last_data(self, samp_or_jobid: str) -> dict:
        """Get the last data from a sample or job.

        Args:
            samp_or_jobid : str
                The sample ID or job ID (with server label) to get the last data for

        Returns:
            str: The filename of the last json
            dict: The last data as a dictionary

        """
        # check if the input is a sample ID
        result = self.execute_sql("SELECT `Sample ID` FROM samples WHERE `Sample ID` = ?", (samp_or_jobid,))
        if result:  # it's a sample
            result = self.execute_sql(
                "SELECT `Job ID on server`, `Server label` FROM jobs WHERE `Sample ID` = ? "
                "ORDER BY `Submitted` DESC LIMIT 1",
                (samp_or_jobid,),
            )
        else:  # it's a job ID
            result = self.execute_sql(
                "SELECT `Job ID on server`, `Server label` FROM jobs WHERE `Job ID` = ?",
                (samp_or_jobid,),
            )
        if not result:
            msg = f"Job {samp_or_jobid} not found in the database"
            raise ValueError(msg)

        jobid_on_server, server_label = result[0]
        server = self.find_server(server_label)
        return server.get_last_data(jobid_on_server)

    def update_payload(self, jobid: str) -> None:
        """Get the payload information from a job ID."""
        result = self.execute_sql("SELECT `Job ID on server`, `Server label` FROM jobs WHERE `Job ID` = ?", (jobid,))
        jobid_on_server, server_label = result[0]
        server = self.find_server(server_label)
        try:
            jobdata = server.get_job_data(jobid_on_server)
        except FileNotFoundError:
            logger.exception("Job data not found on remote PC for %s", jobid)
            self.execute_sql(
                "UPDATE jobs SET `Payload` = ?, `Sample ID` = ? WHERE `Job ID` = ?",
                (json.dumps("Unknown"), "Unknown", jobid),
            )
            raise
        except NotImplementedError as e:
            msg = f"Server type {server.server_type} not supported for getting job data."
            raise NotImplementedError(msg) from e
        payload = jobdata["payload"]
        sampleid = jobdata["payload"]["sample"]["name"]
        self.execute_sql(
            "UPDATE jobs SET `Payload` = ?, `Sample ID` = ? WHERE `Job ID` = ?",
            (json.dumps(payload), sampleid, jobid),
        )

    def update_all_payloads(self, force_retry: bool = False) -> None:
        """Update the payload information for all jobs in the database."""
        if force_retry:
            result = self.execute_sql("SELECT `Job ID` FROM jobs WHERE `Payload` IS NULL OR `Payload` = '\"Unknown\"'")
        else:
            result = self.execute_sql("SELECT `Job ID` FROM jobs WHERE `Payload` IS NULL")
        for (jobid,) in result:
            try:
                self.update_payload(jobid)
            except Exception as e:
                logger.error("Error updating payload for job %s: %s", jobid, e)

    def _update_neware_jobids(self) -> None:
        """Update all Job IDs on Neware servers.

        Temporary measure until we have a faster way to get Job IDs from Newares
        that can run in update_pipelines. This implementation takes ~1 second
        per channel.

        """
        result = self.execute_sql(
            "SELECT `Pipeline`, `Server label` FROM pipelines WHERE "
            "`Sample ID` NOT NULL AND `Ready` = 0 AND `Server type` = 'neware'",
        )
        pipelines = [row[0] for row in result]
        serverids = [row[1] for row in result]
        for serverid, pipeline in zip(serverids, pipelines, strict=True):
            logger.info("Updating job ID for %s on server %s", pipeline, serverid)
            server = self.find_server(serverid)
            assert isinstance(server, NewareServer)  # noqa: S101
            jobid_on_server = server._get_job_id(pipeline)  # noqa: SLF001
            full_jobid = f"{server.label}-{jobid_on_server}"
            self.execute_sql(
                "UPDATE pipelines SET `Job ID` = ?, `Job ID on server` = ? WHERE `Pipeline` = ?",
                (full_jobid, jobid_on_server, pipeline),
            )


if __name__ == "__main__":
    pass
