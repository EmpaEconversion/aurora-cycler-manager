<h1 align="center">
  <img src="https://github.com/user-attachments/assets/931d2c11-db04-41b6-995a-ef66ead759fc" width="500" align="center" alt="Aurora cycler manager">
</h1>

<br>

[![Docs](https://img.shields.io/badge/docs-gh--pages-blue.svg)](https://empaeconversion.github.io/aurora-cycler-manager/)
[![PyPI version](https://img.shields.io/pypi/v/aurora-cycler-manager.svg)](https://pypi.org/project/aurora-cycler-manager/)
[![License](https://img.shields.io/github/license/empaeconversion/aurora-cycler-manager?color=blue)](https://github.com/empaeconversion/aurora-cycler-manager/blob/main/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/aurora-cycler-manager.svg)](https://pypi.org/project/aurora-cycler-manager/)
[![Checks](https://img.shields.io/github/actions/workflow/status/empaeconversion/aurora-cycler-manager/CI.yml)](https://github.com/EmpaEconversion/aurora-cycler-manager/actions/workflows/CI.yml)
[![Coverage](https://img.shields.io/codecov/c/github/empaeconversion/aurora-cycler-manager)](https://app.codecov.io/gh/EmpaEconversion/aurora-cycler-manager)

[Documentation](https://empaeconversion.github.io/aurora-cycler-manager/)

Cycler control, data pipeline, and data visualisation from Empa's robotic battery lab.

- Track samples, experiments and results.
- Control Neware and Biologic cyclers on multiple machines from one place.
- Automatically collect and analyse cycling data.
- Conveniently control cyclers and explore data with a graphical web-app.

### Controlling cyclers

[`aurora-biologic`](https://github.com/EmpaEConversion/aurora-biologic) and [`aurora-neware`](https://github.com/EmpaEConversion/aurora-neware) provide a Python and command-line interface to query, start and stop cyclers, allowing programmatic and remote control.

Experiments can be submitted with a cycler-specific file (e.g. .xml or .mps), or an [`aurora-unicycler`](https://github.com/EmpaEConversion/aurora-unicycler) protocol, which is automatically converted to the appropriate format on submission.

Experiments can be defined with C-rates and without sample names - the program will automatically attach sample info and calculate the current required based on the sample information in the database.

### Data processing

Data is automatically gathered from cyclers, all incoming files are converted to one open standard - accepts Biologic .mpr, Neware .ndax, Neware .xlsx. Incoming data is converted to fast, open, and space-efficient parquet format.

Data is converted using [`fastnda`](https://github.com/g-kimbell/fastnda) and [`yadg`](https://github.com/dgbowl/yadg), processing the raw binary data directly. This is much faster and more space efficient than exporting to text or Excel formats from these cyclers.

Time-series data is automatically analysed to extract per-cycle and summary data.

### Visualisation

A web-app based on `Plotly Dash` allows rapid, interactive viewing of time-series and per-cycle data, as well as the ability to control experiments on cyclers through the graphical interface.

## Quickstart


Install with Python>=3.10:

```shell
pip install aurora-cycler-manager
```

Update an existing installation:
```shell
pip install aurora-cycler-manager --upgrade
```

Connect to an existing project:
```shell
aurora-setup connect --project-dir="path\to\your\setup"
```

Create a new project:
```shell
aurora-setup init --project-dir="path\to\your\setup"
```
You must then fill out the configuration file.

Start the app:
```
aurora-app
```

Start the daemon:
```
aurora-daemon
```

You interact with cyclers, you need password-less SSH access to the cycler PCs.
To control cyclers, [`aurora-biologic`](https://github.com/empaeconversion/aurora-biologic) or [`aurora-neware`](https://github.com/empaeconversion/aurora-neware) needs to be installed on the cycler PC.

Read the [documentation](https://empaeconversion.github.io/aurora-cycler-manager/) for more detail.

## Acknowledgements

This software was developed at the Laboratory of Materials for Energy Conversion at Empa, the Swiss Federal Laboratories for Materials Science and Technology, and supported by funding from the [IntelLiGent](https://heuintelligent.eu/) project from the European Union’s research and innovation program under grant agreement No. 101069765, and from the Swiss State Secretariat for Education, Research, and Innovation (SERI) under contract No. 22.001422.

<img src="https://github.com/user-attachments/assets/373d30b2-a7a4-4158-a3d8-f76e3a45a508#gh-light-mode-only" height="100" alt="IntelLiGent logo">
<img src="https://github.com/user-attachments/assets/9d003d4f-af2f-497a-8560-d228cc93177c#gh-dark-mode-only" height="100" alt="IntelLiGent logo">&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://github.com/user-attachments/assets/1d32a635-703b-432c-9d42-02e07d94e9a9" height="100" alt="EU flag">&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://github.com/user-attachments/assets/cd410b39-5989-47e5-b502-594d9a8f5ae1" height="100" alt="Swiss secretariat">
