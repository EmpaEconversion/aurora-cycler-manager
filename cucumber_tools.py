""" cucumber_tools manages a database and tomato servers for battery cycling

This module contains the Cucumber class which communicates with multiple tomato
servers and manages a database of samples, pipelines and jobs from all servers.

Cucumber can do all ketchup functions (load, submit, eject, ready, cancel,
snapshot) without the user having to know which server samples are on.

Jobs can be submitted with C-rates, and cucumber will automatically calculate
the capacity based on the sample information in the database.

Cucumber can also take snapshots of all jobs in the database, and save the data
locally as a json and hdf5 file. The data can then be processed and plotted.
See the cucumber_daemon.py script for how to run this process automatically.
"""

import os
import warnings
import json
import sqlite3
from time import time, sleep
import traceback
from typing import Literal
import pandas as pd
import paramiko
from cycler_servers import TomatoServer
from database_setup import create_config, create_database
from cucumber_analysis import convert_tomato_json


class Cucumber:
    """ The Cucumber class is the only class in the cucumber_tools module.

    Typical usage:

        # This will connect to servers and update the database
        # Add sample files to ./samples folder and they will be added to the database automatically
        cucumber = Cucumber()

        # Load a sample, submit a job, ready the pipeline
        cucumber.load("sample_id", "pipeline_name")
        cucumber.submit("sample_id", payload_dict, sample_capacity_Ah)
        cucumber.ready("pipeline_name")

        # Update the database to check status of jobs and pipelines
        cucumber.update_db()

        # Snapshot a job or sample to get the data
        cucumber.snapshot("sample_id")
    """

    def __init__(self):
        """ Initialize the cucumber object.
        
        Reads configuration from './config.json' to connect to the database and servers.
        """
        if not os.path.exists("./config.json") or not os.path.exists("./database.db"):
            create_config()
            create_database()
            raise ValueError(
                "Config file or database not found. Created new config.json and database.db files."
                "Please check the config file and restart the program, "
                "or change and rerun database_setup.py as needed."
            )
        with open("./config.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.db = self.config["Database Path"]

        # get the private key
        if not self.config["SSH Private Key Path"]:
            warnings.warn(
                "No SSH private key path specified in config.json."
                "Using default path ~/.ssh/id_rsa",
                RuntimeWarning
            )
            private_key_path = os.path.join(os.path.expanduser("~"), ".ssh", "id_rsa")
        else:
            private_key_path = self.config["SSH Private Key Path"]
        self.private_key = paramiko.RSAKey.from_private_key_file(private_key_path)

        self.status = None
        self.queue = None
        self.queue_all = None

        print("Creating cycler server objects")
        self.get_servers()
        print("Updating the database")
        self.update_db()
        print("Cucumber complete")

    def get_servers(self) -> None:
        """ Create the cycler server objects from the config file. """
        with open("./config.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)
        server_list = self.config["Servers"]

        self.servers=[]

        for server_config in server_list:
            if server_config["server_type"] == "tomato":
                self.servers.append(
                    TomatoServer(
                        server_config,
                        self.private_key
                    )
                )
            else:
                print(f"Server type {server_config['server_type']} not recognized, skipping")

    def insert_sample_file(self, csv_file: str) -> None:
        """ Add a sample csv file to the database.

        The csv file must have a header row with the column names. The columns should match the
        columns defined in the config.json file. At least a 'Sample ID' column is required.
        
        Args:
            csv_file : str
                The path to the csv file to insert
        """
        df = pd.read_csv(csv_file,delimiter=';')
        with open("./config.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)
        column_config = self.config["Sample Database"]

        # Create a dictionary for easy lookup of alternative names
        col_names = [col["Name"] for col in column_config]
        alt_name_dict = {
            alt_name: item["Name"] for item in column_config for alt_name in item["Alternative Names"]
        }

        # Check each column in the DataFrame
        for column in df.columns:
            if column in alt_name_dict:
                # The column is an alternative name, change to the corresponding main name
                df.rename(columns={column: alt_name_dict[column]}, inplace=True)

        # Skip columns that do not exist in col_names
        df = df[[col for col in df.columns if col in col_names]]
        # Warn if there are columns that are not in the database
        for col in df.columns:
            if col not in col_names:
                warnings.warn(
                    f"Column '{col}' in the sample file {csv_file} is not in the database. "
                    "Skipping this column.",
                    RuntimeWarning
                )

        # Check that all essential columns exist
        essential_keys = ["Sample ID"]
        for key in essential_keys:
            if key not in df.columns:
                raise ValueError(
                    f"Essential column '{key}' was not found in the sample file {csv_file}. "
                    "Please double check the file."
                )

        # Insert the new data into the database
        with sqlite3.connect(self.db) as conn:
            cursor = conn.cursor()
            for _, row in df.iterrows():
                # Remove empty columns from the row
                row = row.dropna()
                if row.empty:
                    continue
                # Check if the row has sample ID and cathode capacity
                if "Sample ID" not in row:
                    continue
                placeholders = ', '.join('?' * len(row))
                columns = ', '.join(f"`{key}`" for key in row.keys())
                # Insert or ignore the row
                sql = f"INSERT OR IGNORE INTO samples ({columns}) VALUES ({placeholders})"
                cursor.execute(sql, tuple(row))
                # Update the row
                updates = ", ".join(f"`{column}` = ?" for column in row.keys())
                sql = f"UPDATE samples SET {updates} WHERE `Sample ID` = ?"
                cursor.execute(sql, (*tuple(row), row['Sample ID']))
            conn.commit()

    def delete_sample(self, samples: str | list) -> None:
        """ Remove a sample from the database.
        
        Args:
            samples : str or list
                The sample ID or list of sample IDs to remove from the database
        """
        if not isinstance(samples, list):
            samples = [samples]
        with sqlite3.connect(self.db) as conn:
            cursor = conn.cursor()
            for sample in samples:
                cursor.execute("DELETE FROM samples WHERE `Sample ID` = ?", (sample,))
            conn.commit()

    def update_samples(self) -> None:
        """ Add all csv files in samples folder to the db. """
        samples_folder = self.config["Samples Folder Path"]
        if not os.path.exists(samples_folder):
            os.makedirs(samples_folder)
        for file in os.listdir(samples_folder):
            if file.endswith(".csv"):
                self.insert_sample_file(os.path.join(samples_folder,file))
            else:
                warnings.warn(
                    f"File {file} in samples folder is not a csv file, skipping",
                    RuntimeWarning
                )

    def update_jobs(self) -> None:
        """ Update the jobs table in the database with the current job status. """
        for server in self.servers:
            jobs = server.get_jobs()
            label = server.label
            hostname = server.hostname
            if jobs:
                with sqlite3.connect(self.db) as conn:
                    cursor = conn.cursor()
                    for jobid, jobname, status, pipeline in zip(
                        jobs['jobid'], jobs['jobname'], jobs['status'], jobs['pipeline']
                        ):
                        cursor.execute(
                            "UPDATE jobs "
                            "SET `Status` = ?, `Pipeline` = ?, `Jobname` = ?, `Server Label` = ?, "
                            "`Server Hostname` = ?, `Job ID on Server` = ?, "
                            "`Last Checked` = datetime('now') "
                            "WHERE `Job ID` = ?",
                            (status, pipeline, jobname, label, hostname, jobid, f"{label}-{jobid}")
                        )
                    conn.commit()

    def update_status(self) -> None:
        """ Update the pipelines table in the database with the current status """
        for server in self.servers:
            status = server.get_pipelines()
            label = server.label
            hostname = server.hostname
            if status:
                with sqlite3.connect(self.db) as conn:
                    cursor = conn.cursor()
                    for pipeline, sampleid, jobid in zip(status['pipeline'], status['sampleid'], status['jobid']):
                        cursor.execute(
                            "INSERT OR REPLACE INTO pipelines "
                            "(`Pipeline`, `Sample ID`, `Job ID`, `Server Label`, "
                            "`Server Hostname`, `Last Checked`) "
                            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
                            (pipeline, sampleid, jobid, label, hostname)
                        )
                    conn.commit()

    def update_db(self) -> None:
        """ Update all tables in the database """
        self.update_samples()
        self.update_status()
        self.update_jobs()

    def get_from_db(self, table: str, columns: str = "*", where: str = "") -> list:
        """ Get data from the database.
        
        Args:
            table : str
                The table to get data from
            columns : str, optional
                The columns to get from the table, by default all columns
            where : str, optional
                The where clause to filter the data
        
        Returns:
            The result of the query as a list of tuples
        """
        with sqlite3.connect(self.db) as conn:
            cursor = conn.cursor()
            if not where:
                cursor.execute(f"SELECT {columns} FROM {table}")
            else:
                cursor.execute(f"SELECT {columns} FROM {table} WHERE {where}")
            result = cursor.fetchall()
        if result is None:
            raise ValueError(
                f"No results found in table '{table}' with columns '{columns}' and where '{where}'"
            )
        return result

    @staticmethod
    def sort_pipeline(df: pd.DataFrame) -> pd.DataFrame:
        """ For sorting pipelines so e.g. MPG2-1-2 comes before MPG2-1-10."""
        def custom_sort(x):
            try:
                numbers = x.split("-")[-2:]
                return 1000*int(numbers[0]) + int(numbers[1])
            except ValueError:
                return x
        df.sort_values(by="Pipeline", key = lambda x: x.map(custom_sort),inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def sort_job(df: pd.DataFrame) -> pd.DataFrame:
        """ For sorting jobs so servers are grouped together and jobs are sorted by number. """
        def custom_sort(x):
            try:
                server, number = x.rsplit("-", 1)
                return (server, int(number))
            except ValueError:
                return (x, 0)
        return df.sort_values(by="Job ID", key=lambda x: x.map(custom_sort))

    def get_status(self) -> pd.DataFrame:
        """ Return the status of all pipelines as a DataFrame. """
        columns = ["Pipeline", "Sample ID", "Job ID", "Server Label"]
        result = self.get_from_db("pipelines", columns=", ".join([f"`{c}`" for c in columns]))
        self.status = self.sort_pipeline(pd.DataFrame(result, columns=columns))
        return self.status

    def get_queue(self) -> pd.DataFrame:
        """ Return all running and queued jobs as a DataFrame. """
        columns = ["Job ID", "Sample ID", "Status", "Server Label"]
        result = self.get_from_db("jobs", columns=", ".join([f"`{c}`" for c in columns]),
                                  where="`Status` IN ('q', 'qw', 'r', 'rd')")
        self.queue = self.sort_job(pd.DataFrame(result, columns=columns))
        return self.queue

    def get_queue_all(self) -> pd.DataFrame:
        """ Return all jobs as a DataFrame. """
        columns = ["Job ID", "Sample ID", "Status", "Server Label"]
        result = self.get_from_db("jobs", columns=", ".join([f"`{c}`" for c in columns]),
                                  where="`Status` IN ('q', 'qw', 'r', 'rd', 'c', 'cd')")
        self.queue_all = self.sort_job(pd.DataFrame(result, columns=columns))
        return self.queue_all

    def get_sample_capacity(
            self,
            sample: str,
            mode: Literal['areal','mass','nominal'],
            ignore_anode: bool = True
        ) -> float:
        """ Get the capacity of a sample in Ah based on the mode.

        Args:
            sample : str
                The sample ID to get the capacity for
            mode : str
                The mode to calculate the capacity. Must be 'areal', 'mass', or 'nominal'
                areal: calculate fromAnode/Cathode C-rate Definition Areal Capacity (mAh/cm2) and
                    Anode/Cathode Diameter (mm)
                mass: calculate from Anode/Cathode C-rate Definition Specific Capacity (mAh/g) and 
                    Anode/Cathode Active Material Weight (mg)
                nominal: use C-rate Definition Capacity (mAh)
            ignore_anode : bool, optional
                If True, only use the cathode capacity. Default is True.
        
        Returns:
            float: The capacity of the sample in Ah
        """
        with sqlite3.connect(self.db) as conn:
            cursor = conn.cursor()
            if mode == "mass":
                cursor.execute("""
                               SELECT 
                               `Anode C-rate Definition Specific Capacity (mAh/g)`, 
                               `Anode Active Material Weight (mg)`, 
                               `Anode Diameter (mm)`, 
                               `Cathode C-rate Definition Specific Capacity (mAh/g)`, 
                               `Cathode Active Material Weight (mg)`,
                               `Cathode Diameter (mm)`
                                FROM samples WHERE `Sample ID` = ?
                               """, (sample,))
            elif mode == "areal":
                cursor.execute("""
                               SELECT
                               `Anode C-rate Definition Areal Capacity (mAh/cm2)`,
                               `Anode Diameter (mm)`,
                               `Cathode C-rate Definition Areal Capacity (mAh/cm2)`,
                               `Cathode Diameter (mm)`
                               FROM samples WHERE `Sample ID` = ?
                               """, (sample,))
            elif mode == "nominal":
                cursor.execute("""
                               SELECT
                               `C-rate Definition Capacity (mAh)`
                               FROM samples WHERE `Sample ID` = ?
                               """, (sample,))
            else:
                raise ValueError(f"Mode '{mode}' not recognized. Must be 'mass' or 'areal'")
            result = cursor.fetchone()
        if result is None:
            raise ValueError(f"Sample '{sample}' not found in the database.")
        if mode == "mass":
            anode_capacity_mAh_g, anode_weight_mg, anode_diameter_mm, cathode_capacity_mAh_g, cathode_weight_mg, cathode_diameter_mm = result
            anode_frac_used = min(1,cathode_diameter_mm**2 / anode_diameter_mm**2)
            cathode_frac_used = min(1,anode_diameter_mm**2 / cathode_diameter_mm**2)
            anode_capacity_Ah = anode_frac_used * (anode_capacity_mAh_g * anode_weight_mg * 1e-6)
            cathode_capacity_Ah = cathode_frac_used * (cathode_capacity_mAh_g * cathode_weight_mg * 1e-6)
            if ignore_anode:
                capacity_Ah = cathode_capacity_Ah
            else:
                capacity_Ah = min(anode_capacity_Ah, cathode_capacity_Ah)
        elif mode == "areal":
            anode_capacity_mAh_cm2, anode_diameter_mm, cathode_capacity_mAh_cm2, cathode_diameter_mm = result
            anode_frac_used = min(1,cathode_diameter_mm**2 / anode_diameter_mm**2)
            cathode_frac_used = min(1,anode_diameter_mm**2 / cathode_diameter_mm**2)
            anode_capacity_Ah = (
                anode_frac_used * anode_capacity_mAh_cm2 * (anode_diameter_mm/2)**2 * 3.14159 * 1e-5
            )
            cathode_capacity_Ah = (
                cathode_frac_used * cathode_capacity_mAh_cm2 * (cathode_diameter_mm/2)**2 * 3.14159 * 1e-5
            )
            if ignore_anode:
                capacity_Ah = cathode_capacity_Ah
            else:
                capacity_Ah = min(anode_capacity_Ah, cathode_capacity_Ah)
        elif mode == "nominal":
            capacity_Ah = result[0] * 1e-3
        return capacity_Ah

    def load(self, sample: str, pipeline: str) -> str:
        """ Load a sample on a pipeline.

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
        result = self.get_from_db(
            "samples",
            columns="`Sample ID`",
            where=f"`Sample ID` = '{sample}'"
        )
        # Get pipeline and load
        result = self.get_from_db(
            "pipelines",
            columns="`Server Label`",
            where=f"`Pipeline` = '{pipeline}'"
        )
        server = next((server for server in self.servers if server.label == result[0][0]), None)
        print(f"Loading {sample} on server: {server.label}")
        output = server.load(sample, pipeline)
        return output

    def eject(self, pipeline: str) -> str:
        """ Eject a sample from a pipeline.
        
        Args:
            pipeline (str):
                The pipeline to eject the sample from, must exist in pipelines table of database
        Returns:
            The output from the server eject command as a string
        """
        # Find server associated with pipeline
        result = self.get_from_db(
            "pipelines",
            columns="`Server Label`",
            where=f"`Pipeline` = '{pipeline}'"
        )
        server = next((server for server in self.servers if server.label == result[0][0]), None)
        print(f"Ejecting {pipeline} on server: {server.label}")
        output = server.eject(pipeline)
        return output

    def ready(self, pipeline: str) -> str:
        """ Ready a pipeline for a new job.

        Args:
            pipeline (str):
                The pipeline to ready, must exist in pipelines table of database
        Returns:
            The output from the server ready command as a string
        """
        # find server with pipeline, if there is more than one throw an error
        result = self.get_from_db(
            "pipelines",
            columns="`Server Label`",
            where=f"`Pipeline` = '{pipeline}'"
        )
        server = next((server for server in self.servers if server.label == result[0][0]), None)
        print(f"Readying {pipeline} on server: {server.label}")
        output = server.ready(pipeline)
        return output

    def submit(
            self,
            sample: str,
            json_file: str | dict,
            capacity_Ah: float | Literal['areal','mass','nominal'],
            comment: str = ""
        ) -> None:
        """ Submit a job to a server.

        args:
            sample : str
                The sample ID to submit the job for, must exist in samples table of database
            json_file : str or dict
                A json file, json string, or dictionary with payload to submit to the server
            capacity_Ah : float or str
                The capacity of the sample in Ah, if 'areal', 'mass', or 'nominal', the capacity is
                calculated from the sample information
            comment : str, optional
                A comment to add to the job in the database
        """
        # Get the sample capacity
        if capacity_Ah in ["areal", "mass", "nominal"]:
            capacity_Ah = self.get_sample_capacity(sample, capacity_Ah)
        elif not isinstance(capacity_Ah, float):
            raise ValueError(f"Capacity {capacity_Ah} must be 'areal', 'mass', or a float in Ah.")
        if capacity_Ah > 0.05:
            raise ValueError(f"Capacity {capacity_Ah} too large - value must be in Ah, not mAh")

        # Find the server with the sample loaded, if there is more than one throw an error
        result = self.get_from_db(
            "pipelines",
            columns="`Server Label`",
            where=f"`Sample ID` = '{sample}'"
        )
        server = next((server for server in self.servers if server.label == result[0][0]), None)

        # Check if json_file is a string that could be a file path or a JSON string
        if isinstance(json_file, str):
            try:
                # Attempt to load json_file as JSON string
                payload = json.loads(json_file)
            except json.JSONDecodeError:
                # If it fails, assume json_file is a file path
                with open(json_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
        elif not isinstance(json_file, dict):
            raise ValueError("json_file must be a file path, a JSON string, or a dictionary")
        else: # If json_file is already a dictionary, use it directly
            payload = json_file

        print(f"Submitting job to {sample} with capacity {capacity_Ah:.5f} Ah")
        full_jobid, jobid, json_string = server.submit(sample, capacity_Ah, payload)

        # Update the job table in the database
        with sqlite3.connect(self.db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO jobs (`Job ID`, `Sample ID`, `Server Label`, `Job ID on Server`, "
                "`Submitted`, `Payload`, `Comment`) VALUES (?, ?, ?, ?, datetime('now'), ?, ?)",
                (full_jobid, sample, server.label, int(jobid), json_string, comment)
            )
            conn.commit()

        return

    def cancel(self, jobid: str) -> str:
        """ Cancel a job on a server.

        Args:
            jobid : str
                The job ID to cancel, must exist in jobs table of database
        Returns:
            The output from the server cancel command as a string
        """
        result = self.get_from_db(
            "jobs",
            columns="`Server Label`,`Job ID on Server`",
            where=f"`Job ID` = '{jobid}'"
        )
        server_label, jobid_on_server = result[0]
        server = next((server for server in self.servers if server.label == server_label), None)
        output = server.cancel(jobid_on_server)
        return output

    def snapshot(
            self,
            samp_or_jobid: str,
            get_raw: bool = False,
            mode: Literal["always","new_data","if_not_exists"] = "new_data"
        ) -> None:
        """
        Run snapshots of a sample or job, save the data locally as a json and hdf5, return the data 
        as a list of pandas DataFrame.

        Parameters
        ----------
        samp_or_jobid : str
            The sample ID or (cucumber) job ID to snapshot.
        get_raw : bool, optional
            If True, get raw data. If False, get processed data. Default is False.
        mode : str, optional
            When to make a new snapshot. Can be one of the following:
                - 'always': Force a snapshot even if job is already done and data is downloaded.
                - 'new_data': Snapshot if there is new data.
                - 'if_not_exists': Snapshot only if the file doesn't exist locally.
            Default is 'new_data'.
        """
        # Check if the input is a sample ID
        columns = ["Sample ID", "Job ID", "Server Label", "Job ID on Server", "Snapshot Status"]
        columns_sql = ", ".join([f"`{c}`" for c in columns])

        # Define the column names to check
        column_checks = ["Job ID", "Sample ID", "Job ID on Server"]

        for column_check in column_checks:
            result = self.get_from_db("jobs", columns=columns_sql, where=f"`{column_check}` = '{samp_or_jobid}'")
            if len(result) > 0:
                break
        if len(result) == 0:
            raise ValueError(f"Sample or job ID '{samp_or_jobid}' not found in the database.")

        for sampleid, jobid, server_name, jobid_on_server, snapshot_status in result:
            if not sampleid:
                print(f"Job {jobid} not associated with a sample, skipping.") # TODO should this update the db as well?
                print(f"sql results: {sampleid=}, {jobid=}, {server_name=}, {jobid_on_server=}, {snapshot_status=}")
                continue
            # Check if the snapshot should be skipped
            batchid = sampleid.rsplit("_", 1)[0]
            local_save_location = f"{self.config["Snapshots Folder Path"]}/{batchid}/{sampleid}"
            local_save_location_processed = f"{self.config["Processed Snapshots Folder Path"]}/{batchid}/{sampleid}"

            files_exist = os.path.exists(f"{local_save_location_processed}/snapshot.{jobid}.h5")
            if files_exist and mode != "always":
                if mode == "if_not_exists":
                    print(f"Snapshot {jobid} already exists, skipping.")
                    continue
                if mode == "new_data" and snapshot_status is not None and snapshot_status.startswith("c"):
                    print(f"Snapshot {jobid} already complete.")
                    continue

            # Otherwise snapshot the job
            server = next((server for server in self.servers if server.label == server_name), None)

            print(f"Snapshotting sample {sampleid} job {jobid}")
            try:
                snapshot_status = server.snapshot(jobid, jobid_on_server, local_save_location, get_raw)
                # Update the snapshot status in the database
                with sqlite3.connect(self.db) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE jobs SET `Snapshot Status` = ?, `Last Snapshot` = datetime('now') "
                        "WHERE `Job ID` = ?",
                        (snapshot_status, jobid)
                        )
                    print("Updating database")
                    conn.commit()
            except FileNotFoundError as e:
                warnings.warn(f"Error snapshotting {jobid}: {e}", RuntimeWarning)
                warnings.warn(
                    "Likely the job was cancelled before starting. "
                    "Setting `Snapshot Status` to 'ce' in the database."
                )
                with sqlite3.connect(self.db) as conn:
                    cursor = conn.cursor()
                    cursor.execute("UPDATE jobs SET `Snapshot Status` = 'ce' WHERE `Job ID` = ?", (jobid,))
                    conn.commit()
                continue
            except ValueError as e:
                warnings.warn(f"Error snapshotting {jobid}: {e}", RuntimeWarning)

            # Process the file and save to processed snapshots folder
            convert_tomato_json(
                f"{local_save_location}/snapshot.{jobid}.json",
                f"{local_save_location_processed}/snapshot.{jobid}.h5",
            )

        return

    def snapshot_all(
            self,
            sampleid_contains: str = "",
            mode: Literal["always","new_data","if_not_exists"] = "new_data",
        ) -> None:
        """ Snapshot all jobs in the database.
        
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
        assert mode in ["always", "new_data", "if_not_exists"]
        where = "`Status` IN ( 'c', 'r', 'rd', 'cd', 'ce')"
        if mode in ["new_data"]:
            where += " AND (`Snapshot Status` NOT LIKE 'c%' OR `Snapshot Status` IS NULL)"
        if sampleid_contains:
            where += f" AND `Sample ID` LIKE '%{sampleid_contains}%'"
        result = self.get_from_db("jobs", columns="`Job ID`",
                                  where=where)
        total_jobs = len(result)
        print(f"Snapshotting {total_jobs} jobs:")
        print([jobid for jobid, in result])
        t0 = time()
        for i, (jobid,) in enumerate(result):
            try:
                self.snapshot(jobid, mode=mode)
            except Exception as e:
                tb = traceback.format_exc()
                error_message = str(e) if str(e) else "An error occurred but no message was provided."
                warnings.warn(f"Error snapshotting {jobid}: {error_message}\n{tb}", RuntimeWarning)
            percent_done = (i + 1) / total_jobs * 100
            time_elapsed = time() - t0
            time_remaining = time_elapsed / (i + 1) * (total_jobs - i - 1)
            sleep(20) # to not overload the server
            print(f"{percent_done:.2f}% done, {int(time_remaining/60)} minutes remaining")
        return


if __name__ == "__main__":
    pass
