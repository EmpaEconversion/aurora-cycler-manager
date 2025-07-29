<p align="center">
  <img src="https://github.com/user-attachments/assets/33a4416a-3fae-4bb3-acce-3862bc87a4a6#gh-light-mode-only" width="500" align="center" alt="Aurora cycler manager">
  <img src="https://github.com/user-attachments/assets/95845ec0-e155-4e4f-95d2-ab1c992de940#gh-dark-mode-only" width="500" align="center" alt="Aurora cycler manager">
</p>

<br>

Cycler control, data pipeline, and data visualisation from Empa's robotic battery lab.

- Tracks samples, experiments and results.
- Control Neware and Biologic cyclers on multiple machines from one place.
- Automatically collect and analyse cycling data.
- Results in consistent, open format including metadata with provenance tracking and sample information.
- Convenient cycler control and in-depth data exploration using `Dash`-based webapp.

### Jobs

Aurora cycler manager can be used to control and submit experiments to Biologic and Neware cyclers.

Jobs can be submitted with a cycler-specific file (e.g. .xml or .mps).

Alternatively, a `unicycler` universal .json protocol can be used, which is converted to the appropriate format on submission.

Experiments can use C-rates and the program will automatically calculate the current required based on the sample information in the database.

### Data harvesting

Data is automatically gathered from cyclers, all incoming files are converted to one open standard - accepts Biologic .mpr, Neware .ndax, Neware .xlsx, and tomato .json. Raw time-series data is converted to a hdf5 file including provenance tracked metadata.

Data is converted using [`NewareNDA`](https://github.com/d-cogswell/NewareNDA) and [`yadg`](https://github.com/dgbowl/yadg), processing the raw binary data directly. This is much faster and more space efficient than exporting to text or Excel formats from these cyclers.

### Analysis

The time-series hdf5 data is analysed to extract per-cycle summary data such as charge and discharge capacities, stored alongside metadata in a .json file.

### Visualisation

A web-app based on `Plotly Dash` allows rapid, interactive viewing of time-series and per-cycle data, as well as the ability to control experiments on tomato cyclers through the graphical interface.

## Installation

In a Python environment:

```
pip install aurora-cycler-manager
```

To _view data from an existing set up_:
```
aurora-setup connect --project-dir="path\to\your\setup"
```

To _interact with servers on an existing set up_:
- Interacting with servers (submitting jobs, harvesting data etc.) works with OpenSSH, servers must have OpenSSH installed and running
- Generate a public/private key pair on your system with `ssh-keygen`
- Copy your public key (usually in `%USERPROFILE%\.ssh\id_rsa.pub`) to the cycler server
- Add it to the server's `authorized_keys` file (usually in `C:\Users\username\.ssh\authorized_keys`)
- (optional) You can make changes to your user config, this is stored in your user folder e.g. /users/yourname/appdata/local/aurora_cycler_manager/
  - "SSH private key path" can be changed, if your key is not in a standard location
  - "Snapshots folder path" is where raw data downloaded from cyclers is stored, this can become very large

To _create a new set up_:
```
aurora-setup init --project-dir="path\to\your\setup"
```
- This generates subfolders within the directory, a database, and a configuation file
- Fill in the configuration file with details about e.g. Neware and EC-Lab servers, examples are left in the default config
- In `Servers`, the `server_type` must be `neware`, `biologic`, or `tomato`
- The `server_label` should be short and only letters and numbers, no special characters like `-_/\`
- The `shell_type` is the default shell when SSH-ing into the machine, it must be `cmd` or `powershell`
- To set up a `neware` server, follow the instructions from [`aurora-neware`](https://github.com/empaeconversion/aurora-neware)
- To set up a `biologic` server, follow the instructions from [`aurora-biologic`](https://github.com/empaeconversion/aurora-biologic)
- To set up a `tomato` server, follow instructions from [`tomato-0.2.3`](https://dgbowl.github.io/tomato/0.2.3/)
- If you change database columns in the shared configuration file, you can update the database with `aurora-setup update`
```
aurora-setup update
```
- Use the option `--force` if you want to permanetly delete columns and their data.

## Updating

Upgrade with pip, you do not have to redo any setup steps:
```
pip install aurora-cycler-manager --upgrade
```
If upgrading from earlier than 0.5.0, first `pip uninstall aurora-cycler-manager` then follow the installation steps.

## Usage

A web app allows users to view analysed data and see the status of samples, jobs, and cyclers, and submit jobs to cyclers if they have access. Run with:
```
aurora-app
```

- There are three tabs, samples plotting, batch plotting, and database.
- To upload sample information to the database, use the 'Add samples' button in the database tab, and select a .json file defining the cells.
- Hand-made cells can also be added, a .json must be created with the keys defined in the configuration file.
- Protocols can be created in database -> protocols.
- In database -> pipelines, load samples onto the pipelines, then submit a protocol.
- Loading samples, submitting jobs, analysing data etc. can also be run in Python scripts directly - see the example in `server_manager.py`.

With SSH access, automatic data harvesting and analysis is run using:
```
aurora-daemon
```

## Contributors

- [Graham Kimbell](https://github.com/g-kimbell)

## Acknowledgements

This software was developed at the Laboratory of Materials for Energy Conversion at Empa, the Swiss Federal Laboratories for Materials Science and Technology, and supported by funding from the [IntelLiGent](https://heuintelligent.eu/) project from the European Union’s research and innovation program under grant agreement No. 101069765, and from the Swiss State Secretariat for Education, Research, and Innovation (SERI) under contract No. 22.001422.

<img src="https://github.com/user-attachments/assets/373d30b2-a7a4-4158-a3d8-f76e3a45a508#gh-light-mode-only" height="100" alt="IntelLiGent logo">
<img src="https://github.com/user-attachments/assets/9d003d4f-af2f-497a-8560-d228cc93177c#gh-dark-mode-only" height="100" alt="IntelLiGent logo">&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://github.com/user-attachments/assets/1d32a635-703b-432c-9d42-02e07d94e9a9" height="100" alt="EU flag">&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://github.com/user-attachments/assets/cd410b39-5989-47e5-b502-594d9a8f5ae1" height="100" alt="Swiss secretariat">
