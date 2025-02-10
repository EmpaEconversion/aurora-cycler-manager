<p align="center">
  <img src="https://github.com/user-attachments/assets/33a4416a-3fae-4bb3-acce-3862bc87a4a6#gh-light-mode-only" width="500" align="center" alt="Aurora cycler manager">
  <img src="https://github.com/user-attachments/assets/95845ec0-e155-4e4f-95d2-ab1c992de940#gh-dark-mode-only" width="500" align="center" alt="Aurora cycler manager">
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

In a Python environment:

```
pip install git+https://github.com/EmpaEconversion/aurora-cycler-manager.git
```
After successfully installing, run and follow the instructions:
```
aurora-setup
```
To _view data from an existing set up_:
- Say yes to 'Connect to an existing configuration and database', then give the path to this folder.

To _interact with servers on an existing set up_:
- Interacting with servers (submitting jobs, harvesting data etc.) works with OpenSSH
- Generate a public/private key pair on your system with `ssh-keygen`
- Ensure your public key is authorized on the system running the cycler
- In config.json fill in 'SSH private key path' and 'Snapshots folder path'
- Snapshots folder path stores the raw data downloaded from cyclers which is processed. This data can be deleted any time.
- To connect and control `tomato` servers, `tomato v0.2.3` must be configured on the remote PC
- To harvest from EC-lab or Neware cyclers, set data to save/backup to some location and specify this location in the shared configuration file

To _create a new set up_: 
- Use `aurora-setup` to create a configuration and database - it is currently designed with network storage in mind, so other users can access data.
- Fill in the configuration file with details about e.g. tomato, Neware and EC-lab servers. Examples are left in the default config file.

## Usage

A web app allows users to view analysed data and see the status of samples, jobs, and cyclers, and submit jobs to cyclers if they have access. Run with:
```
aurora-app
```

To upload sample information to the database, place output .csv files from the Aurora robot into the samples folder defined in the configuration.

Hand made cells can also be added, a .csv must be created with the headers defined in the shared configuration.

Loading samples, submitting jobs etc. can be performed on `tomato` directly, or using the `aurora-app` GUI, or by writing a Python script to use the functions in server_manager.py.

With SSH access, automatic data harvesting and analysis is run using:
```
aurora-daemon
```

## Contributors

- [Graham Kimbell](https://github.com/g-kimbell)

## Acknowledgements

This software was developed at the Materials for Energy Conversion Lab at the Swiss Federal Laboratories for Materials Science and Technology (Empa).
