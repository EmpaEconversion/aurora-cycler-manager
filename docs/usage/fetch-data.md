# Getting data

## Manually fetching data

In database -> pipelines, select your samples, and press 'Snapshot'. This downloads the latest raw data, parses it to an open format, analyses it together with any existing data, and updates the data folder.


## Automatically fetching data

```
aurora-daemon
```
This starts a process that updates the cycler status every 5 minutes, and fetches and analyses all new data overnight. Only one machine should be running the daemon.


## Manually uploading data

Datasets can be uploaded manually as a zip if they follow this structure:

```
dataset.zip/
├── sample_id_01/
│   ├── data_file.bdf.parquet
│   └── data_file.bdf.csv
├── sample_id_02/
│   └── data_file.mpr
└── sample_id_03/
    └── data_file.ndax
```
i.e. each folder name is a unique Sample ID, and the folder contains time-series data for that sample in BDF format, or Biologic .mpr, or Neware .ndax. The file names do not matter.
