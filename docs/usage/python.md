# Using the Python interface

The Python package gives access to all the tools for interacting with cyclers and the database. See the API reference for more details. As a short example:
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

