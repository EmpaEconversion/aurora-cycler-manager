"""Copyright © 2024, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Useful functions for the visualiser app.
"""
import sqlite3
from typing import Union

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ArrayLike = Union[list, np.ndarray, pd.Series]

def get_sample_names(config: dict) -> list:
    """Get all sample IDs from the database."""
    db_path = config["Database path"]
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT `Sample ID` FROM samples")
        samples = cursor.fetchall()
    return [sample[0] for sample in samples]

def get_batch_names(config: dict) -> list:
    """Get all batch names from the database."""
    graph_config_path = config["Graph config path"]
    with open(graph_config_path) as f:
        graph_config = yaml.safe_load(f)
    return list(graph_config.keys())

def get_database(config: dict) -> dict:
    """Get all data from the database.

    Formatted for viewing in Dash AG Grid.
    """
    db_path = config["Database path"]
    unused_pipelines = config.get("Unused pipelines", [])
    pipeline_query = "SELECT * FROM pipelines WHERE " + " AND ".join([f"Pipeline NOT LIKE '{pattern}'" for pattern in unused_pipelines])
    db_data = {
        "samples": pd.read_sql_query("SELECT * FROM samples", sqlite3.connect(db_path)).to_dict("records"),
        "results": pd.read_sql_query("SELECT * FROM results", sqlite3.connect(db_path)).to_dict("records"),
        "jobs": pd.read_sql_query("SELECT * FROM jobs", sqlite3.connect(db_path)).to_dict("records"),
        "pipelines": pd.read_sql_query(pipeline_query, sqlite3.connect(db_path)).to_dict("records"),
    }
    db_columns = {
        "samples": [{"field" : col, "filter": True, "tooltipField": col} for col in db_data["samples"][0]],
        "results": [{"field" : col, "filter": True, "tooltipField": col} for col in db_data["results"][0]],
        "jobs": [{"field" : col, "filter": True, "tooltipField": col} for col in db_data["jobs"][0]],
        "pipelines": [{"field" : col, "filter": True, "tooltipField": col} for col in db_data["pipelines"][0]],
    }

    # Use custom comparator for pipeline column
    pipeline_field = next((col for col in db_columns["pipelines"] if col["field"] == "Pipeline"), None)
    if pipeline_field:
        pipeline_field["comparator"] = {"function": "pipelineComparatorCustom"}
        pipeline_field["sort"] = "asc"

    return {"data":db_data, "column_defs": db_columns}

def cramers_v(x: ArrayLike, y: ArrayLike) -> float:
    """Calculate Cramer's V for two categorical variables."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def anova_test(x: ArrayLike, y: ArrayLike) -> float:
    """ANOVA test between categorical and continuous variables."""
    categories = x.unique()
    groups = [y[x == category] for category in categories]
    f_stat, p_value = stats.f_oneway(*groups)
    return p_value

def correlation_ratio(categories: ArrayLike, measurements: ArrayLike) -> float:
    """Measure of the relationship between a categorical and numerical variable."""
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    eta = 0.0 if numerator == 0 else np.sqrt(numerator / denominator)
    return eta

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the correlation matrix for a DataFrame including categorical columns.

    For continuous-continuous use Pearson correlation
    For continuous-categorical use correlation ratio
    For categorical-categorical use Cramer's V.

    Args:
        df (pd.DataFrame): The DataFrame to calculate the correlation matrix for.

    """
    corr = pd.DataFrame(index=df.columns, columns=df.columns)
    # Calculate the correlation matrix
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 == col2:
                corr.loc[col1, col2] = 1.0
            elif pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                corr.loc[col1, col2] = df[[col1, col2]].corr().iloc[0, 1]
            elif pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_object_dtype(df[col2]):
                corr.loc[col1, col2] = correlation_ratio(df[col2], df[col1])
            elif pd.api.types.is_object_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                corr.loc[col1, col2] = correlation_ratio(df[col1], df[col2])
            elif pd.api.types.is_object_dtype(df[col1]) and pd.api.types.is_object_dtype(df[col2]):
                corr.loc[col1, col2] = cramers_v(df[col1], df[col2])
    return corr

def moving_average(x: ArrayLike, npoints: int = 11) -> np.ndarray:
    if npoints % 2 == 0:
        npoints += 1  # Ensure npoints is odd for a symmetric window
    window = np.ones(npoints) / npoints
    xav = np.convolve(x, window, mode="same")
    xav[:npoints // 2] = np.nan
    xav[-npoints // 2:] = np.nan
    return xav

def deriv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        dydx = np.zeros(len(y))
        dydx[0] = (y[1] - y[0]) / (x[1] - x[0])
        dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
        dydx[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])

    # for any 3 points where x direction changes sign set to nan
    mask = (x[1:-1] - x[:-2]) * (x[2:] - x[1:-1]) < 0
    dydx[1:-1][mask] = np.nan
    return dydx

def smoothed_derivative(
        x: np.ndarray,
        y: np.ndarray,
        npoints: int = 21,
    ) -> np.ndarray:
    x_smooth = moving_average(x, npoints)
    y_smooth = moving_average(y, npoints)
    dydx_smooth = deriv(x_smooth, y_smooth)
    dydx_smooth[deriv(x_smooth,np.arange(len(x_smooth))) < 0] *= -1
    dydx_smooth[abs(dydx_smooth) > 100] = np.nan
    return dydx_smooth
