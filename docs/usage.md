# Usage

## Starting the app

A web app allows users to view analysed data and see the status of samples, jobs, and cyclers, and submit jobs to cyclers if they have access. Run with:
```shell
aurora-app
```

There are three tabs, samples plotting, batch plotting, and database.

## Adding samples

To upload sample information to the database, use the 'Upload' button in the database tab, and select a .json file defining the cells.

The .json file should look like:
```python
[
    {
        "Sample ID": "my_cell_001",
        # Other keys that are columns in the database
    },
    {
        "Sample ID": "my_cell_002",
        # Other keys that are columns in the database
    }
    # etc.
]
```

## Defining an protocol

Go to database tab, then protocols.

Define a protocol - add techniques with the green (+) button at the bottom, select the technique type, fill in the parameters, and press update. Make sure you also fill in safety and measurement parameters.

If the save button is disabled and there is a warning symbol next to it, hover over the symbol to see the issue with the protocol.

Finally, click 'Save as', give it a name and save.


## Starting an experiment

Now you have a sample and a protocol. Go to database -> pipelines, click on a pipeline, click 'load' and select your sample.

(You need to also put the actual sample on the physical machine)

Now select the loaded sample, press 'Submit', choose a protocol and a way to define C-rate, then submit it and cross your fingers.


## Manually getting data

In database -> pipelines, select your samples, and press 'Snapshot'. This downloads the latest raw data, parses it to an open format, analyses it together with any existing data, and updates the data folder.

## Automatically getting data

```
aurora-daemon
```
This starts a process that updates the cycler status every 5 minutes, and fetches and analyses all new data overnight. Only one machine should be running the daemon.


## Using the Python interface

The Python package gives access to all the tools for interacting with cyclers and the database. See the API reference for more details, but for a short example:
```python
from aurora_cycler_manager.server_manager import ServerManager

sm = ServerManager()
sm.load("MPG2-1-1", "my_cell_001")  # puts the cell on a pipeline
sm.submit(
    "my_cell_001",
    "my/unicycler/protocol.json",  # a unicycler json file
    capacity_Ah = "mass",  # reads capacity from database
)
```
Once some data has been recorded:
```python
from aurora_cycler_manager.data_parse import SampleDataBundle
data = SampleDataBundle("my_cell_001")
print(data.cycling)  # Time-series data, polars df, e.g. 'V (V)'
print(data.eis)  # Frequency-domain, polars df
print(data.cycles_summary)  # Per-cycle data, polars df, e.g. 'Discharge capacity (mAh)'
print(data.overall_summary)  # Overall data, e.g. 'First formation efficiency (%)'
print(data.metadata)  # Sample metadata, e.g. 'N:P ratio'
# Now do some plotting or further analysis
```

## Single-user vs multi-user

The simplest set up is a single user installing the app locally and keeping all data locally.

You can also serve the app from a server machine which users then connect to. Users will only have access to the features through the GUI, and cannot directly access data or write scripts to control cyclers. You can set the host and port, and skip opening in a browser with e.g. `aurora-app --host="0.0.0.0" --port=8050 --no-browser"`. You must set up your firewalls to allow user to connect.

Multiple users can install `aurora-cycler-manager` and access the same project if it is in a network accessible location. Then multiple users are able to run complex scripts. It is possible to store an sqlite database on a network drive, but it can become slow if it the database gets large. It is better to set up postgresql.
