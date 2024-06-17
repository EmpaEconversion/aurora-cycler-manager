import os
import warnings
import json
import base64
import paramiko
from scp import SCPClient
import pandas as pd


class CyclerServer():
    def __init__(self, server_config, local_private_key):
        self.label = server_config["label"]
        self.hostname = server_config["hostname"]
        self.username = server_config["username"]
        self.server_type = server_config["server_type"]
        self.command_prefix = server_config["command_prefix"]
        self.tomato_scripts_path = server_config["tomato_scripts_path"]
        self.local_private_key = local_private_key
        self.last_status = None
        self.last_queue = None
        self.last_queue_all = None
        self.check_connection()

    def command(self, command):
        with paramiko.SSHClient() as ssh:
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.hostname, username=self.username, pkey=self.local_private_key)
            stdin, stdout, stderr = ssh.exec_command(self.command_prefix + " " + command)
            output = stdout.read().decode('utf-8')
            error = stderr.read().decode('utf-8')
        if error:
            if "Error" in error:
                print(f"Error running '{command}' on {self.label}")
                raise ValueError(error)
            elif error.startswith("WARNING"):
                warnings.warn(error, RuntimeWarning)
            else:
                print(f"Error running '{command}' on {self.label}")
                raise ValueError(error)
        return output
    
    def check_connection(self):
        test_phrase = "hellothere"
        output = self.command(f"echo {test_phrase}")
        if output != test_phrase+"\r\n":
            raise ValueError(f"Connection error, expected output '{repr(test_phrase+"\r\n")}', got '{repr(output)}'")
        print(f"Succesfully connected to {self.label}")
        return True

    def get_status(self):
        raise NotImplementedError

    def get_queue(self):
        raise NotImplementedError

class TomatoServer(CyclerServer):
    def __init__(self, server_config, local_private_key):
        super().__init__(server_config, local_private_key)
        self.save_location = "C:/tomato/cucumber_scratch"

    def eject(self, pipeline):
        output = self.command(f"{self.tomato_scripts_path}ketchup eject {pipeline}")
        return output

    def load(self, sample, pipeline):
        output = self.command(f"{self.tomato_scripts_path}ketchup load {sample} {pipeline}")
        return output

    def ready(self, pipeline):
        output = self.command(f"{self.tomato_scripts_path}ketchup ready {pipeline}")
        return output

    def submit(self, sample, capacity_Ah, json_file, send_file=False):
        # Check if json_file is a string that could be a file path or a JSON string
        if isinstance(json_file, str):
            try:
                # Attempt to load json_file as JSON string
                payload = json.loads(json_file)
            except json.JSONDecodeError:
                # If it fails, assume json_file is a file path
                with open(json_file, 'r') as f:
                    payload = json.load(f)
        elif isinstance(json_file, dict):
            # If json_file is already a dictionary, use it directly
            payload = json_file
        else:
            raise ValueError("json_file must be a file path, a JSON string, or a dictionary")

        # Add the sample name and capacity to the payload
        payload["sample"]["name"] = sample
        payload["sample"]["capacity"] = capacity_Ah
        # Convert the payload to a json string
        json_string = json.dumps(payload)
        # Change all other instances of $NAME to the sample name
        json_string = json_string.replace("$NAME", sample)

        if send_file: # Write the json string to a file, send it, run it on the server
            # Write file locally
            with open("temp.json", "w") as f:
                f.write(json_string)

            # Send file to server
            ssh = paramiko.SSHClient()
            ssh.load_system_host_keys()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.hostname, username=self.username, pkey=self.local_private_key)
            with SCPClient(ssh.get_transport()) as scp:
                scp.put("temp.json", f"{self.save_location}/temp.json")
            ssh.close()

            # Submit the file on the server
            output = self.command(f"{self.tomato_scripts_path}ketchup submit {self.save_location}/temp.json")

        else: # Encode the json string to base64 and submit it directly
            encoded_json_string = base64.b64encode(json_string.encode()).decode()
            output = self.command(f'{self.tomato_scripts_path}ketchup submit -J {encoded_json_string}')
        if "jobid: " in output:
            jobid = output.split("jobid: ")[1].split("\r\n")[0]
            print(f"Sample {sample} submitted on server {self.label} with jobid {jobid}")
            full_jobid = f"{self.label}-{jobid}"
            print(f"Full jobid: {full_jobid}")
            return full_jobid, jobid

        raise ValueError(f"Error submitting job: {output}")
        
    def cancel(self, job_id):
        output = self.command(f"{self.tomato_scripts_path}ketchup cancel {job_id}")
        return output

    def get_status(self):
        output = self.command(f"{self.tomato_scripts_path}ketchup status -J")
        status_dict = json.loads(output)
        self.last_status = status_dict
        return status_dict

    def get_queue(self):
        output = self.command(f"{self.tomato_scripts_path}ketchup status queue -J")
        queue_dict = json.loads(output)
        self.last_queue = queue_dict
        return queue_dict

    def get_queue_all(self):
        output = self.command(f"{self.tomato_scripts_path}ketchup status queue -v -J")
        queue_all_dict = json.loads(output)
        self.last_queue_all = queue_all_dict
        return queue_all_dict

    def snapshot(self, sampleid, jobid, jobid_on_server, get_raw=False):
        # Save a snapshot on the remote machine
        save_location = f"{self.save_location}/{jobid_on_server}"
        self.command(f"if (!(Test-Path \"{save_location}\")) {{ New-Item -ItemType Directory -Path \"{save_location}\" }}")
        output = self.command(f"{self.tomato_scripts_path}ketchup status -J {jobid_on_server}")
        json_output = json.loads(output)
        snapshot_status = json_output["status"][0]

        # Catch errors
        try:
            with warnings.catch_warnings(record=True) as w:
                self.command(f"cd {save_location} ; {self.tomato_scripts_path}ketchup snapshot {jobid_on_server}")
                for warning in w:
                    if "out-of-date version" in str(warning.message):
                        continue
                    elif "has been completed" in str(warning.message):
                        continue
                    else:
                        print(f"Warning: {warning.message}")
        except ValueError as e:
            emsg = str(e)
            if "AssertionError" in emsg and "os.path.isdir(jobdir)" in emsg:
                raise FileNotFoundError from e
            raise e

        # Get local directory to save the snapshot data
        batchid = sampleid.rsplit("_", 1)[0]
        local_directory = f"./snapshots/{batchid}/{sampleid}"
        if not os.path.exists(local_directory):
            os.makedirs(local_directory)

        # Use SCPClient to transfer the file from the remote machine
        ssh = paramiko.SSHClient()
        ssh.load_system_host_keys()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.hostname, username=self.username, pkey=self.local_private_key)
        with SCPClient(ssh.get_transport()) as scp:
            scp.get(
                f"{save_location}/snapshot.{jobid_on_server}.json",
                f"{local_directory}/snapshot.{jobid}.json",
            )
            if get_raw:
                print("Downloading snapshot raw data")
                scp.get(
                    f"{save_location}/snapshot.{jobid_on_server}.zip",
                    f"{local_directory}/snapshot.{jobid}.zip"
                )
        ssh.close()

        # Convert the snapshot data to hdf5
        snapshot_df = self.convert_data(f"{local_directory}/snapshot.{jobid}.json")

        return snapshot_df, snapshot_status
    
    def convert_data(self,snapshot_file):
        """Saves data as .h5 file and returns the DataFrame"""
        with open(snapshot_file) as f:
            input_dict = json.load(f)
        n_steps = len(input_dict["steps"])
        data = []
        technique_code = {"OCV":0,"CPLIMIT":1,"CALIMIT":2}
        for i in range(n_steps):
            step_data = input_dict["steps"][i]["data"]
            step_dict = {
                "uts" : [row["uts"] for row in step_data],
                "Ewe" : [row["raw"]["Ewe"]["n"] for row in step_data],
                "I": [row["raw"]["I"]["n"] if "I" in row["raw"] else 0 for row in step_data],
                "cycle_number": [row["raw"]["cycle number"] if "cycle number" in row["raw"] else -1 for row in step_data],
                "loop_number": [row["raw"]["loop number"] if "cycle number" in row["raw"] else -1 for row in step_data],
                "index" : [row["raw"]["index"] if "index" in row["raw"] else -1 for row in step_data],
                "technique" : [technique_code.get(row["raw"]["technique"], -1) if "technique" in row["raw"] else -1 for row in step_data],
            }
            data.append(pd.DataFrame(step_dict))
        hdf5_file = snapshot_file.replace(".json", ".h5")
        data = pd.concat(data, ignore_index=True)
        # Save the DataFrame to an HDF5 file
        data.to_hdf(hdf5_file, key='df', mode='w')
        return data


if __name__ == "__main__":
    pass
