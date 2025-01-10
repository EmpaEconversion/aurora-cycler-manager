<p align="center">
  <img src="https://github.com/user-attachments/assets/bd006861-ad54-4a85-937c-3f9458ec717c#gh-light-mode-only" width="500" align="center" alt="Aurora cycler manager">
  <img src="https://github.com/user-attachments/assets/a4bd3db5-5c16-4be8-9655-45e34b6d9e06#gh-dark-mode-only" width="500" align="center" alt="Aurora cycler manager">
</p>

</br>

Cycler management, data pipeline, and data visualisation for Empa's robotic battery lab.

- Track samples, experiments and results with a database.
- Sample data is imported from the Aurora battery assembly robot.
- Automatically harvest and analyse cycling data.
- Results in consistent, open format including metadata with provenance tracking and sample information.
- Reads data from `tomato` servers, Biologic's EC-lab, and Neware's BTS software running on different machines.
- Control experiments on `tomato` servers with a graphical interface.
- Convenient, in-depth data exploration using `Dash`-based webapp.

### Jobs

The Aurora cycler manager can be used to do all `tomato` functions (load, submit, eject, ready, cancel, snapshot) from one place to multiple `tomato` servers. Jobs can be submitted using C-rates, and can automatically calculate the current required based on measured electrode masses from the robot.

### Data harvesting

Functions are available to snapshot and download data from `tomato` servers. Harvesters are available to download new data from Biologic's EC-lab, converting from the closed .mpr filetype using `yadg`, and from Neware's BTS .xlsx reports or closed .ndax files using `NewareNDA`.

### Analysis

Full cycling data (voltage and current vs time) is converted to fast, efficient .h5 files with provenance tracked metadata. This data is analysed to extract per-cycle summary data such as charge and discharge capacities, stored alongside metadata in a .json file.

### Visualisation

A web-app based on `Plotly Dash` allows rapid, interactive viewing of data, as well as the ability to control experiments on tomato cyclers through the graphical interface.

## Installation

Clone the repo, pip install requirements in requirements.txt, preferably in a virtual environment.

If using "view-only":
- Run visualiser/app.py
- This will create a config.json file in the root directory
- In config.json fill in 'Shared config path' to point to the shared configuration file
- Run visualiser/app.py again to start viewing data

If you want to interact with cyclers, this works using OpenSSH:
- Generate a public/private key pair on your system with `ssh-keygen`
- Ensure your public key is authorized on the system running the cycler
- In config.json fill in 'SSH private key path' and 'Snapshots folder path'
- Snapshots folder path stores the raw data downloaded from cyclers which is processed. This data can be deleted any time.

If you are setting up the system and shared configuration file:
- Run database_setup.py to create a default shared config file and sqlite database
- Move the configuration and database anywhere - this is designed for use on a network drive
- Fill in the empty fields in the configuration file
- To connect and control `tomato` servers, `tomato v0.2.3` must be configured on the remote PC
- To harvest from EC-lab or Neware cyclers, set data to save/backup to some location and specify this location in the shared configuration file
- Run the daemon.py script to periodically download and analyse new data and update the database

## Usage

Place output .csv files from the Aurora robot into the samples folder defined in the configuration.

Either load samples, submit jobs and ready pipelines using `tomato` directly, or use the `Dash` app, or write a script to use the functions in server_manager.py.

Run visualiser/app.py to view an interactive visualiser of the results for samples and batches of samples, and to control `tomato` cyclers.

## Contributors

- [Graham Kimbell](https://github.com/g-kimbell)

## Acknowledgements

This software was developed at the Materials for Energy Conversion Lab at the Swiss Federal Laboratories for Materials Science and Technology (Empa).
