""" Copyright © 2024, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia

server_manager manages a database and tomato servers for battery cycling

This module contains the ServerManager class which communicates with multiple tomato
servers and manages a database of samples, pipelines and jobs from all servers.

Server manager can do all ketchup functions (load, submit, eject, ready, cancel,
snapshot) without the user having to know which server samples are on.

Jobs can be submitted with C-rates, and the capacity can be automatically
calculated based on the sample information in the database.

The server manager can also take snapshots of all jobs in the database, save the
data locally as a json and convert to a zipped json file. The data can then be
processed and plotted. See the daemon.py script for how to run this process 
automatically.
"""
import os
import warnings
import json
import sqlite3
from time import time, sleep
from datetime import datetime
import traceback
from typing import Literal, Tuple
import pandas as pd
import paramiko
from aurora_cycler_manager.cycler_servers import CyclerServer, TomatoServer
from aurora_cycler_manager.database_setup import create_config, create_database
from aurora_cycler_manager.analysis import convert_tomato_json, _run_from_sample

class ServerManager:
    """ The ServerManager: class is the only class in the server_manager module.

    Typical usage:

        # This will connect to servers and update the database
        # Add sample files to ./samples folder and they will be added to the database automatically
        sm = ServerManager()

        # Load a sample, submit a job, ready the pipeline
        sm.load("sample_id", "pipeline_name")
        sm.submit("sample_id", payload_dict, sample_capacity_Ah)
        sm.ready("pipeline_name")

        # Update the database to check status of jobs and pipelines
        sm.update_db()

        # Snapshot a job or sample to get the data
        sm.snapshot("sample_id")
    """

    def __init__(self):
        """ Initialize the server manager object.
        
        Reads configuration from './config.json' to connect to the database and servers.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, '..', 'config.json')
        
        if not os.path.exists(config_path):
            create_config()
            create_database()
            raise ValueError(
                "Config file or database not found. Created new config.json."
                "Please check the config file and restart the program, "
                "or change and rerun database_setup.py as needed."
            )
        with open(config_path, encoding = 'utf-8') as f:
            self.config = json.load(f)
        self.db = self.config["Database path"]

        # get the private key
        private_key_path = self.config.get(
            "SSH private key path",
            os.path.join(os.path.expanduser("~"), ".ssh", "id_rsa")
        )
        if not os.path.exists(private_key_path):
            raise FileNotFoundError(f"Private key file {private_key_path} not found.")
        self.private_key = paramiko.RSAKey.from_private_key_file(private_key_path)

        print("Creating cycler server objects")
        self.get_servers()
        print("Server manager initialised, consider updating database with update_db()")

    def get_servers(self) -> None:
        """ Create the cycler server objects from the config file. """
        server_list = self.config["Servers"]

        self.servers=[]
        for server_config in server_list:
            if server_config["server_type"] == "tomato":
                try:
                    self.servers.append(
                        TomatoServer(
                            server_config,
                            self.private_key,
                       ),
                    )
                except (OSError, ValueError, TimeoutError, paramiko.SSHException) as exc:
                    print(f"CRITICAL: Server {server_config['label']} could not be created, skipping")
                    print(f"Error: {exc}")
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
        # If csv is over 1 MB, do not insert
        if os.path.getsize(csv_file) > 1e6:
            warnings.warn(f"File {csv_file} is over 1 MB, skipping", RuntimeWarning)
        df = pd.read_csv(csv_file,delimiter=';')
        # If csv contains more than 1000 rows or 100 columns, do not insert
        if len(df) > 1000:
            warnings.warn(f"File {csv_file} contains more than 1000 rows, skipping", RuntimeWarning)

        # Load the config file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, '..', 'config.json')
        with open(config_path, encoding = 'utf-8') as f:
            self.config = json.load(f)
        column_config = self.config["Sample database"]

        # Create a dictionary for easy lookup of alternative names
        col_names = [col["Name"] for col in column_config]
        alt_name_dict = {
            alt_name: item["Name"] for item in column_config for alt_name in item["Alternative names"]
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

        # Check that timestamps are in the correct format
        for col in df.columns:
            if "Timestamp" in col:
                try:
                    pd.to_datetime(df[col], errors='raise', format='%Y-%m-%d %H:%M:%S')
                except ValueError:
                    try:
                        # Attempt conversion with a different format if the first fails
                        df[col] = pd.to_datetime(df[col], errors='raise', format='%d.%m.%Y %H:%M')
                        df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                    except ValueError as exc:
                        raise ValueError(
                            f"Timestamp column '{col}' in the sample file is not in the correct format. "
                            "Please use the format 'YYYY-MM-DD HH:MM:SS'."
                        ) from exc
                    
        # Calculate/overwrite certain columns
        # Active material masses
        required_columns = ["Anode mass (mg)", "Anode current collector mass (mg)", "Anode active material mass fraction"]
        if all(col in df.columns for col in required_columns):
            df["Anode active material mass (mg)"] = (
                (df["Anode mass (mg)"] - df["Anode current collector mass (mg)"])
                * df["Anode active material mass fraction"]
            )
        required_columns = ["Cathode mass (mg)", "Cathode current collector mass (mg)", "Cathode active material mass fraction"]
        if all(col in df.columns for col in required_columns):
            df["Cathode active material mass (mg)"] = (
                (df["Cathode mass (mg)"] - df["Cathode current collector mass (mg)"])
                * df["Cathode active material mass fraction"]
            )
        # Capacities
        required_columns = ["Anode active material mass (mg)", "Anode balancing specific capacity (mAh/g)"]
        if all(col in df.columns for col in required_columns):
            df["Anode balancing capacity (mAh)"] = (
                1e-3 * df["Anode active material mass (mg)"] * df["Anode balancing specific capacity (mAh/g)"]
            )
        required_columns = ["Cathode active material mass (mg)", "Cathode balancing specific capacity (mAh/g)"]
        if all(col in df.columns for col in required_columns):
            df["Cathode balancing capacity (mAh)"] = (
                1e-3 * df["Cathode active material mass (mg)"] * df["Cathode balancing specific capacity (mAh/g)"]
            )
        # N:P ratio overlap factor
        required_columns = ["Anode diameter (mm)", "Cathode diameter (mm)"]
        if all(col in df.columns for col in required_columns):
            df["N:P ratio overlap factor"] = (df["Cathode diameter (mm)"]**2 / df["Anode diameter (mm)"]**2).fillna(0)
        # Actual N:P ratio
        required_columns = ["Anode balancing capacity (mAh)", "Cathode balancing capacity (mAh)", "N:P ratio overlap factor"]
        if all(col in df.columns for col in required_columns):
            df["Actual N:P ratio"] = (
                df["Anode balancing capacity (mAh)"] * df["N:P ratio overlap factor"]
                / df["Cathode balancing capacity (mAh)"]
            )
        # Run ID - if column is missing or where it is empty, find from the sample ID
        if "Run ID" not in df.columns:
            df["Run ID"] = df["Sample ID"].apply(lambda x: _run_from_sample(x))
        else:
            df["Run ID"] = df["Run ID"].fillna(df["Sample ID"].apply(lambda x: _run_from_sample(x)))

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
        samples_folder = self.config["Samples folder path"]
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
            dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            label = server.label
            hostname = server.hostname
            if jobs:
                with sqlite3.connect(self.db) as conn:
                    cursor = conn.cursor()
                    for jobid_on_server, jobname, status, pipeline in zip(
                        jobs['jobid'], jobs['jobname'], jobs['status'], jobs['pipeline']
                        ):
                        # If pipeline is none, do not update (keep old value)
                        if pipeline is None:
                            cursor.execute(
                                "UPDATE jobs "
                                "SET `Status` = ?, `Jobname` = ?, `Server label` = ?, "
                                "`Server hostname` = ?, `Last checked` = ? "
                                "WHERE `Job ID` = ?",
                                (status, jobname, label, hostname, dt, f"{label}-{jobid_on_server}")
                            )
                        else:
                            cursor.execute(
                                "UPDATE jobs "
                                "SET `Status` = ?, `Pipeline` = ?, `Jobname` = ?, `Server label` = ?, "
                                "`Server Hostname` = ?, `Job ID on server` = ?, "
                                "`Last Checked` = ? "
                                "WHERE `Job ID` = ?",
                                (status, pipeline, jobname, label, hostname, jobid_on_server, dt, f"{label}-{jobid_on_server}")
                            )
                    conn.commit()

    def update_pipelines(self) -> None:
        """ Update the pipelines table in the database with the current status """
        for server in self.servers:
            status = server.get_pipelines()
            dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            label = server.label
            hostname = server.hostname
            if status:
                with sqlite3.connect(self.db) as conn:
                    cursor = conn.cursor()
                    for ready, pipeline, sampleid, jobid_on_server in zip(status['ready'],status['pipeline'], status['sampleid'], status['jobid']):
                        jobid = f"{label}-{jobid_on_server}" if jobid_on_server else None
                        cursor.execute(
                            "INSERT OR REPLACE INTO pipelines "
                            "(`Pipeline`, `Sample ID`, `Job ID`, `Ready`, `Last Checked`, "
                            "`Server label`, `Server Hostname`, `Job ID on server`) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (pipeline, sampleid, jobid, ready, dt, label, hostname, jobid_on_server)
                        )
                    conn.commit()

    def update_flags(self) -> None:
        """ Update the flags in the pipelines table from the results table. """
        with sqlite3.connect(self.db) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE pipelines SET `Flag` = NULL")
            cursor.execute("SELECT `Pipeline`, `Flag`, `Sample ID` FROM results")
            results = cursor.fetchall()
            for pipeline, flag, sampleid in results:
                cursor.execute(
                    "UPDATE pipelines SET `Flag` = ? WHERE `Pipeline` = ? AND `Sample ID` = ?",
                    (flag, pipeline, sampleid)
                )
            conn.commit()

    def update_db(self) -> None:
        """ Update all tables in the database """
        self.update_samples()
        self.update_pipelines()
        self.update_jobs()
        self.update_flags()
        self.update_all_payloads()

    def execute_sql(self, query, params=None):
        """ Execute a query on the database.

        Args:
            query : str
                The query to execute
            params : tuple, optional
                The parameters to pass to the query

        Returns:
            The result of the query as a list of tuples
        """
        commit_keywords = ['UPDATE', 'INSERT', 'DELETE', 'REPLACE', 'CREATE', 'DROP', 'ALTER']
        commit = any(keyword in query.upper() for keyword in commit_keywords)
        with sqlite3.connect(self.db) as conn:
            cursor = conn.cursor()
            if params is not None:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            if commit:
                conn.commit()
            result = cursor.fetchall()
        return result

    def find_server(self, label: str) -> CyclerServer:
        """Get the server object from the label."""
        if not label:
            msg = (
                "No server label found from query, there is probably a mistake in the query. "
                "E.g. if you are searching for a sample ID, the ID might be wrong."
            )
            raise ValueError(msg)
        server = next((server for server in self.servers if server.label == label), None)
        if not server:
            msg = (
                f"Server with label {label} not found. "
                "Either there is a mistake in the label name or you do not have access to the server."
            )
            raise ValueError(msg)
        return server

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

    def get_pipelines(self) -> pd.DataFrame:
        """ Return the status of all pipelines as a DataFrame. """
        columns = ["Pipeline", "Sample ID", "Job ID on server", "Server label"]
        result = self.execute_sql("SELECT `Pipeline`, `Sample ID`, `Job ID on server`, `Server label` FROM pipelines")
        return self.sort_pipeline(pd.DataFrame(result, columns=columns))

    def get_queue(self) -> pd.DataFrame:
        """ Return all running and queued jobs as a DataFrame. """
        columns = ["Job ID", "Sample ID", "Status", "Server label"]
        result = self.execute_sql("SELECT `Job ID`, `Sample ID`, `Status`, `Server label` FROM jobs WHERE `Status` IN ('q', 'qw', 'r', 'rd')")
        return self.sort_job(pd.DataFrame(result, columns=columns))

    def get_jobs(self) -> pd.DataFrame:
        """ Return all jobs as a DataFrame. """
        columns = ["Job ID", "Sample ID", "Status", "Server label"]
        result = self.execute_sql("SELECT `Job ID`, `Sample ID`, `Status`, `Server label` FROM jobs WHERE `Status` IN ('q', 'qw', 'r', 'rd', 'c', 'cd')")
        return  self.sort_job(pd.DataFrame(result, columns=columns))

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
                (sample,)
            )
        elif mode == "areal":
            result = self.execute_sql(
                "SELECT "
                "`Anode C-rate definition areal capacity (mAh/cm2)`, "
                "`Anode diameter (mm)`, "
                "`Cathode C-rate definition areal capacity (mAh/cm2)`, "
                "`Cathode diameter (mm)` "
                "FROM samples WHERE `Sample ID` = ?",
                (sample,)
            )
        elif mode == "nominal":
            result = self.execute_sql(
                "SELECT "
                "`C-rate definition capacity (mAh)` "
                "FROM samples WHERE `Sample ID` = ?",
                (sample,)
            )
        if result is None:
            raise ValueError(f"Sample '{sample}' not found in the database.")
        if mode == "mass":
            anode_capacity_mAh_g, anode_weight_mg, anode_diameter_mm, cathode_capacity_mAh_g, cathode_weight_mg, cathode_diameter_mm = result[0]
            anode_frac_used = min(1,cathode_diameter_mm**2 / anode_diameter_mm**2)
            cathode_frac_used = min(1,anode_diameter_mm**2 / cathode_diameter_mm**2)
            anode_capacity_Ah = anode_frac_used * (anode_capacity_mAh_g * anode_weight_mg * 1e-6)
            cathode_capacity_Ah = cathode_frac_used * (cathode_capacity_mAh_g * cathode_weight_mg * 1e-6)
            if ignore_anode:
                capacity_Ah = cathode_capacity_Ah
            else:
                capacity_Ah = min(anode_capacity_Ah, cathode_capacity_Ah)
        elif mode == "areal":
            anode_capacity_mAh_cm2, anode_diameter_mm, cathode_capacity_mAh_cm2, cathode_diameter_mm = result[0]
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
            capacity_Ah = result[0][0] * 1e-3
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
        result = self.execute_sql("SELECT `Sample ID` FROM samples WHERE `Sample ID` = ?", (sample,))
        # Get pipeline and load
        result = self.execute_sql("SELECT `Server label` FROM pipelines WHERE `Pipeline` = ?", (pipeline,))
        server = self.find_server(result[0][0])
        print(f"Loading {sample} on server: {server.label}")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = server.load(sample, pipeline)
            if w:
                for warning in w:
                    print(f"Warning raised: {warning.message}")
            else:
                # Update database preemtively
                self.execute_sql(
                    "UPDATE pipelines SET `Sample ID` = ? WHERE `Pipeline` = ?",
                    (sample, pipeline)
                )
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
        result = self.execute_sql("SELECT `Server label` FROM pipelines WHERE `Pipeline` = ?", (pipeline,))
        server = self.find_server(result[0][0])
        print(f"Ejecting {pipeline} on server: {server.label}")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = server.eject(pipeline)
            if w:
                for warning in w:
                    print(f"Warning raised: {warning.message}")
            else:
                # Update database preemtively
                self.execute_sql(
                    "UPDATE pipelines SET `Sample ID` = NULL, `Flag` = Null, `Ready` = 0 WHERE `Pipeline` = ?",
                    (pipeline,)
                )
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
        result = self.execute_sql("SELECT `Server label` FROM pipelines WHERE `Pipeline` = ?", (pipeline,))
        server = self.find_server(result[0][0])
        print(f"Readying {pipeline} on server: {server.label}")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = server.ready(pipeline)
            if w:
                for warning in w:
                    print(f"Warning raised: {warning.message}")
            else:
                # Update database preemtively
                self.execute_sql(
                    "UPDATE pipelines SET `Ready` = 1 WHERE `Pipeline` = ?",
                    (pipeline,)
                )
        return output
    
    def unready(self, pipeline: str) -> str:
        """ Unready a pipeline, only works if no job running, if job is running user must cancel.

        Args:
            pipeline (str):
                The pipeline to unready, must exist in pipelines table of database
        Returns:
            The output from the server unready command as a string
        """
        # Find server with pipeline, if there is more than one throw an error
        result = self.execute_sql("SELECT `Server label` FROM pipelines WHERE `Pipeline` = ?", (pipeline,))
        server = self.find_server(result[0][0])
        print(f"Unreadying {pipeline} on server: {server.label}")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = server.unready(pipeline)
            if w:
                for warning in w:
                    print(f"Warning raised: {warning.message}")
            else:
                # Update database preemtively
                self.execute_sql(
                    "UPDATE pipelines SET `Ready` = 0 WHERE `Pipeline` = ?",
                    (pipeline,)
                )
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
        result = self.execute_sql("SELECT `Server label` FROM pipelines WHERE `Sample ID` = ?", (sample,))
        server = self.find_server(result[0][0])

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
        dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Update the job table in the database
        self.execute_sql(
            "INSERT INTO jobs (`Job ID`, `Sample ID`, `Server label`, `Job ID on server`, "
            "`Submitted`, `Payload`, `Comment`) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (full_jobid, sample, server.label, int(jobid), dt, json_string, comment)
        )
        self.execute_sql(
            "UPDATE pipelines SET `Job ID on server` = ?, `Job ID` = ?, `Server Label` = ?, `Server Hostname` = ? WHERE `Sample ID` = ?",
            (int(jobid), full_jobid, server.label, server.hostname, sample)
        )

        return

    def cancel(self, jobid: str) -> str:
        """ Cancel a job on a server.

        Args:
            jobid : str
                The job ID to cancel, must exist in jobs table of database
        Returns:
            The output from the server cancel command as a string
        """
        result = self.execute_sql("SELECT `Server label`, `Job ID on server` FROM jobs WHERE `Job ID` = ?", (jobid,))
        server_label, jobid_on_server = result[0]
        server = self.find_server(server_label)
        output = server.cancel(jobid_on_server)
        # If no error, assume job is cancelled and update the database
        self.execute_sql(
            "UPDATE jobs SET `Status` = 'cd' WHERE `Job ID` = ?",
            (jobid,)
        )
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
            The sample ID or (aurora) job ID to snapshot.
        get_raw : bool, optional
            If True, get raw data. If False, get processed data. Default is False.
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
                "SELECT `Sample ID`, `Status`, `Job ID on server`, `Server label`, `Snapshot Status` "
                "FROM jobs WHERE `Sample ID` = ? ",
                (samp_or_jobid,)
            )
        else:  # it's a job ID
            result = self.execute_sql(
                "SELECT `Sample ID`, `Status`, `Job ID on server`, `Server label`, `Snapshot Status` "
                "FROM jobs WHERE `Job ID` = ?",
                (samp_or_jobid,)
            )
        if not result:
            raise ValueError(f"Sample or job ID '{samp_or_jobid}' not found in the database.")

        for sample_id, status, jobid_on_server, server_label, snapshot_status in result:
            jobid = f"{server_label}-{jobid_on_server}"
            if not sample_id: # TODO should this update the db as well?
                print(f"Job {server_label}-{jobid_on_server} has no sample, skipping.")
                continue
            # Check that sample is known
            if sample_id == "Unknown":
                print(f"Job {server_label}-{jobid_on_server} has no sample name or payload, skipping.")
                continue
            run_id = _run_from_sample(sample_id)

            local_save_location = f"{self.config["Snapshots folder path"]}/{run_id}/{sample_id}"
            local_save_location_processed = f"{self.config["Processed snapshots folder path"]}/{run_id}/{sample_id}"

            files_exist = os.path.exists(f"{local_save_location_processed}/snapshot.{jobid}.h5")
            if files_exist and mode != "always":
                if mode == "if_not_exists":
                    print(f"Snapshot {jobid} already exists, skipping.")
                    continue
                if mode == "new_data" and snapshot_status is not None and snapshot_status.startswith("c"):
                    print(f"Snapshot {jobid} already complete.")
                    continue

            # Check that the job has started
            if status in ['q', 'qw']:
                print(f"Job {jobid} is still queued, skipping snapshot.")
                continue

            # Otherwise snapshot the job
            server = self.find_server(server_label)

            print(f"Snapshotting sample {sample_id} job {jobid}")
            try:
                snapshot_status = server.snapshot(jobid, jobid_on_server, local_save_location, get_raw)
                dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                # Update the snapshot status in the database
                self.execute_sql(
                    "INSERT OR IGNORE INTO results "
                    "(`Sample ID`) VALUES (?)",
                    (sample_id,)
                )
                self.execute_sql(
                    "UPDATE results SET `Last snapshot` = ? WHERE `Sample ID` = ?",
                    (dt, sample_id)
                )
                self.execute_sql(
                    "UPDATE jobs SET `Snapshot status` = ?, `Last snapshot` = ? WHERE `Job ID` = ?",
                    (snapshot_status, dt, jobid)
                )
            except FileNotFoundError as e:
                warnings.warn(f"Error snapshotting {jobid}: {e}", RuntimeWarning)
                warnings.warn(
                    "Likely the job was cancelled before starting. "
                    "Setting `Snapshot Status` to 'ce' in the database."
                )
                self.execute_sql(
                    "UPDATE jobs SET `Snapshot status` = 'ce' WHERE `Job ID` = ?",
                    (jobid,)
                )
            except ValueError as e:
                warnings.warn(f"Error snapshotting {jobid}: {e}", RuntimeWarning)

            # Process the file and save to processed snapshots folder
            convert_tomato_json(
                f"{local_save_location}/snapshot.{jobid}.json",
                output_jsongz_file=f"{local_save_location_processed}/snapshot.{jobid}.json.gz",
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
        where += " AND `Sample ID` IS NOT 'Unknown'"
        if mode in ["new_data"]:
            where += " AND (`Snapshot status` NOT LIKE 'c%' OR `Snapshot status` IS NULL)"
        if sampleid_contains:
            where += f" AND `Sample ID` LIKE '%{sampleid_contains}%'"
        result = self.execute_sql("SELECT `Job ID` FROM jobs WHERE " + where)
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

    def get_last_data(self, samp_or_jobid: str) -> Tuple[str,dict]:
        """ Get the last data from a sample or job. 
        
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
                (samp_or_jobid,)
            )
        else:  # it's a job ID
            result = self.execute_sql(
                "SELECT `Job ID on server`, `Server label` FROM jobs WHERE `Job ID` = ?",
                (samp_or_jobid,)
            )
        if not result:
            raise ValueError(f"Job {samp_or_jobid} not found in the database")

        jobid_on_server, server_label = result[0]
        server = self.find_server(server_label)
        return server.get_last_data(jobid_on_server)

    def update_payload(self, jobid: str) -> None:
        """ Get the payload information from a job ID. """
        result = self.execute_sql("SELECT `Job ID on server`, `Server label` FROM jobs WHERE `Job ID` = ?", (jobid,))
        jobid_on_server, server_label = result[0]
        server = self.find_server(server_label)
        try:
            jobdata = server.get_job_data(jobid_on_server)
        except FileNotFoundError:
            print(f"Job data not found on remote PC for {jobid}")
            self.execute_sql(
                "UPDATE jobs SET `Payload` = ?, `Sample ID` = ? WHERE `Job ID` = ?",
                (json.dumps("Unknown"), "Unknown", jobid)
            )
            return
        payload = jobdata['payload']
        sampleid = jobdata['payload']['sample']['name']
        self.execute_sql(
            "UPDATE jobs SET `Payload` = ?, `Sample ID` = ? WHERE `Job ID` = ?",
            (json.dumps(payload), sampleid, jobid)
        )

    def update_all_payloads(self, force_retry = False) -> None:
        """ Update the payload information for all jobs in the database. """
        if force_retry:
            result = self.execute_sql("SELECT `Job ID` FROM jobs WHERE `Payload` IS NULL OR `Payload` = '\"Unknown\"'")
        else:
            result = self.execute_sql("SELECT `Job ID` FROM jobs WHERE `Payload` IS NULL")
        for jobid, in result:
            self.update_payload(jobid)
        return

if __name__ == "__main__":
    pass
