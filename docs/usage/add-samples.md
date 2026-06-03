# Adding samples

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
