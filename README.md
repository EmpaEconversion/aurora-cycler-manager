<p align="center">
  <img src="https://github.com/user-attachments/assets/33a4416a-3fae-4bb3-acce-3862bc87a4a6#gh-light-mode-only" width="500" align="center" alt="Aurora cycler manager">
  <img src="https://github.com/user-attachments/assets/95845ec0-e155-4e4f-95d2-ab1c992de940#gh-dark-mode-only" width="500" align="center" alt="Aurora cycler manager">
</p>

</br>

Cycler management, data pipeline, and data visualisation for Empa's robotic battery lab.

- Tracks samples, experiments and results.
- Control Neware and Biologic cyclers on multiple machines from one place.
- Automatically collect and analyse cycling data.
- Results in consistent, open format including metadata with provenance tracking and sample information.
- Convenient cycler control and in-depth data exploration using `Dash`-based webapp.

### Jobs

Aurora cycler manager can be used to control and submit experiments to Biologic and Neware cyclers. Biologic cyclers are controlled through `tomato`, and Neware through a Python API.

Jobs can be either be submitted with a cycler-specific file (e.g. xml protocols for Neware).

Alternatively, a `unicycler` universal .json protocol can be used, which is converted to the appropriate format on submission. `unicycler` protocols can currently be converted to Neware .xml, `tomato` .json, and `PyBaMM` experiments. It is particularly useful for generating Neware .xml which can be difficult to define programmatically.

Experiments can use C-rates and the program will automatically calculate the current required based on the sample information in the database.

### Data harvesting

Data is automatically gathered from cyclers. Harvesters are also available to download data from Biologic's EC-Lab.

The program converts all incoming filetypes to one open standard - accepts Biologic .mpr, `tomato` .json, Neware .ndax, and Neware .xlsx. Raw time-series data is converted to a hdf5 file including provenance tracked metadata.

### Analysis

The time-series hdf5 data is analysed to extract per-cycle summary data such as charge and discharge capacities, stored alongside metadata in a .json file.

### Visualisation

A web-app based on `Plotly Dash` allows rapid, interactive viewing of time-series and per-cycle data, as well as the ability to control experiments on tomato cyclers through the graphical interface.

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
- To harvest from EC-Lab or Neware cyclers, set data to save/backup to some location and specify this location in the shared configuration file

To _create a new set up_: 
- Use `aurora-setup` to create a configuration and database - it is currently designed with network storage in mind, so other users can access data.
- Fill in the configuration file with details about e.g. tomato, Neware and EC-Lab servers. Examples are left in the default config file.

## Updating

From versions `0.5.x` you do not have to re-do any of the setup steps, just upgrade with pip:
```
pip install git+https://github.com/EmpaEconversion/aurora-cycler-manager.git --upgrade
```
If upgrading from earlier versions, first `pip uninstall aurora-cycler-manager` then follow the installation steps.

## Usage

A web app allows users to view analysed data and see the status of samples, jobs, and cyclers, and submit jobs to cyclers if they have access. Run with:
```
aurora-app
```

To upload sample information to the database, place output .json files from the Aurora robot into the samples folder defined in the configuration.

Hand-made cells can also be added, a .json must be created with the keys defined in the shared configuration.

Loading samples, submitting jobs etc. can be performed on `tomato` or Neware directly, or using the `aurora-app` GUI, or by writing a Python script to use the functions in server_manager.py.

With SSH access, automatic data harvesting and analysis is run using:
```
aurora-daemon
```

## Contributors

- [Graham Kimbell](https://github.com/g-kimbell)

## Acknowledgements

This software was developed at the Materials for Energy Conversion Lab at the Swiss Federal Laboratories for Materials Science and Technology (Empa), and supported through the EU's Horizon program under the IntelLiGent project (101069765) and the Swiss State Secretariat for Education, Research and Innovation (SERI) (22.00142). 🇪🇺🇨🇭
