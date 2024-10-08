<p align="center">
  <img src="https://github.com/user-attachments/assets/3cd5c5b3-0921-45e7-a2d4-4d9acdab894f" width="500" align="center" alt="Aurora robot tools">
</p>
</br>

Tools for managing a database and multiple Tomato battery cycler servers.

The database tracks all samples produced by the Aurora robot, all cycler channels and their status, all jobs that have been run on every sample, and the combined results for each sample.

### Jobs

The Aurora cycler manager can be used to do all Tomato Ketchup functions (load, submit, eject, ready, cancel, snapshot) from one place to multiple Tomato servers. Jobs can be submitted using C-rates, and can automatically calculate the current required based on measured electrode masses from the robot.

### Snapshots

Functions are available to snapshot all jobs, or only jobs that have new data recorded since the previous snapshot. There is also a script for harvesting data from jobs run directly on Biologic EC-lab, which builds on the mpr conversion capabilities of yadg. The snapshotting and harvesting is automated by running periodically with a daemon script.

### Analysis

Cycling data is converted to .hdf5 files with provenance tracked metadata. The voltage and current vs time data is analysed to extract per-cycle data such as charge and discharge capacities. There are also functions for plotting sample data.

Batches of samples can be defined in a .yaml file, which can then be merged into one data file and plotted together.

There is also a data visualiser based on Plotly Dash which allows for rapid and interactive viewing of data, as well as control of the cyclers through a graphical interface.

## Installation

Clone the repo, pip install requirements in requirements.txt, preferably in a virtual environment.

Run database_setup.py to create a default config file and database, then make the changes you want in the configuration.

To connect to a tomato server, Tomato (v0.2.3) must be configured on the remote PC and you must be authorised to ssh connect to the PC. Currently the script is only suitable for Windows PCs with either Command Prompt or Powershell default shells.

## Usage

Place output .csv files from the Aurora robot into the samples folder defined in the config file.

Either load samples, submit jobs and ready pipelines using Tomato directly, or write a script to use the functions in server_manager.py.

Run daemon.py to periodically update the database with the samples, job statuses, as well as periodically harvest data from the cyclers and run analysis. This can also harvest data from EC-lab directly if eclab_harvester.py is configured.

Run visualiser/app.py to view an interactive visualiser of the results for samples and batches of samples, and to control the cyclers.

## Contributors

- [Graham Kimbell](https://github.com/g-kimbell)

## Acknowledgements

This software was developed at the Materials for Energy Conversion Lab at the Swiss Federal Laboratories for Materials Science and Technology (Empa).
