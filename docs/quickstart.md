# Quickstart

## Essentials

```shell
pip install aurora-cycler-manager   # install
aurora-setup init "path/to/project" # make a project
aurora-app                          # start the app
```

## Example

Create a virtual environment in a folder with e.g. `python -m venv` or `uv venv`:

![Create virtual environment](assets/tut1.webp)

Install aurora-cycler-manager with `pip install aurora-cycler-manager`:

![Install aurora-cycler-manager](assets/tut2.webp)

Create an aurora project with `aurora-setup init path/to/project` (`.` means current folder):

![Create an aurora project](assets/tut3.webp)

Start the app with `aurora-app`:

![Start the app](assets/tut4.webp)

This will open an empty app:

![Empty app](assets/app1.webp)

To load some example data, you can download `kiye_dataset_parquet.zip` from [Zenodo](https://zenodo.org/records/19107066). On the `aurora-app`, go to the **Database** tab, click the **Upload** button, drag and drop the `.zip` file, scroll to the bottom and press `Confirm`.

![Uploading data](assets/app2.webp)

The **Database** samples table should now show the uploaded samples.

To view the data, you can select rows in the samples table and press **View**, or go to the plotting tabs and select the samples in the menus on the left.

![Clicking view](assets/app3.webp)

![Viewing data](assets/app4.webp)

Here you can change the axis on the plot, and add more samples to compare. The **Batch Plotting** tab does not show full time-series data, but includes more options for coloring, styling, and aggregating data from larger numbers of samples, as well as showing correlation maps and plots for summary statistics of samples.
