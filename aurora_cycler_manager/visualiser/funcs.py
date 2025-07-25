"""Copyright © 2025, Empa.

Useful functions for the visualiser app.
"""

import sqlite3
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pytz import timezone
from scipy import stats

from aurora_cycler_manager.config import get_config

ArrayLike = list | np.ndarray | pd.Series

CONFIG = get_config()


def get_database() -> dict[str, Any]:
    """Get all data from the database.

    Formatted for viewing in Dash AG Grid.
    """
    db_path = CONFIG["Database path"]
    unused_pipelines = CONFIG.get("Unused pipelines", [])

    with sqlite3.connect(db_path) as conn:
        if unused_pipelines:
            not_like_conditions = " OR ".join(["Pipeline LIKE ?"] * len(unused_pipelines))
            pipelines_df = pd.read_sql_query(
                "SELECT * FROM pipelines WHERE NOT (" + not_like_conditions + ")",  # noqa: S608 - injection safe
                conn,
                params=unused_pipelines,
            )
        else:
            pipelines_df = pd.read_sql_query("SELECT * FROM pipelines", conn)
        samples_df = pd.read_sql_query("SELECT * FROM samples", conn)
        results_df = pd.read_sql_query("SELECT * FROM results", conn)
        jobs_df = pd.read_sql_query("SELECT * FROM jobs", conn)
    pipelines_df["Ready"] = pipelines_df["Ready"].astype(bool)
    db_data = {
        "samples": samples_df.to_dict("records"),
        "results": results_df.to_dict("records"),
        "jobs": jobs_df.to_dict("records"),
        "pipelines": pipelines_df.to_dict("records"),
    }
    db_columns = {
        "samples": [{"field": col, "filter": True, "tooltipField": col} for col in samples_df.columns],
        "results": [{"field": col, "filter": True, "tooltipField": col} for col in results_df.columns],
        "jobs": [{"field": col, "filter": True, "tooltipField": col} for col in jobs_df.columns],
        "pipelines": [{"field": col, "filter": True, "tooltipField": col} for col in pipelines_df.columns],
    }

    # Ready is boolean
    try:
        ready_field = next(col for col in db_columns["pipelines"] if col["field"] == "Ready")
        ready_field["cellDataType"] = "boolean"
    except StopIteration:
        pass

    # Use custom comparator for pipeline column
    with suppress(StopIteration):
        pipeline_field: dict[str, Any] = next(col for col in db_columns["pipelines"] if col["field"] == "Pipeline")
        pipeline_field["comparator"] = {"function": "pipelineComparatorCustom"}
        pipeline_field["sort"] = "asc"

    return {"data": db_data, "column_defs": db_columns}


def get_db_last_update() -> str:
    """Get the last update time of the database."""
    db_path = Path(CONFIG["Database path"])
    modified_uts = db_path.stat().st_mtime
    tz = timezone(CONFIG.get("Time zone", "Europe/Zurich"))
    modified_datetime = datetime.fromtimestamp(int(modified_uts), tz=tz)
    return modified_datetime.strftime("%Y-%m-%d %H:%M:%S %z")


def make_pipelines_comparable(pipelines: list[str | None]) -> list[str | None]:
    """Convert pipelines string to a comparable format.

    Important! This should always be consistent with the JavaScript function pipelineComparatorCustom.
    The JavaScript function is used in the AG Grid to display the pipelines in the correct order.
    This function is used when loading multiple samples to pipelines.
    """

    def convert_pipeline(pipeline: str | None) -> str | None:
        """Make single pipeline string comparable."""
        if pipeline is None:
            return None

        # Split the pipeline string by '-'
        parts = pipeline.split("-")
        # Iterate over the parts and pad numbers with zeros
        for i in range(len(parts)):
            if parts[i].isdigit():
                parts[i] = parts[i].zfill(3)

        # Join the parts back together with '-'
        pipeline = "-".join(parts)

        # Now split by "_" and put the first part at the end
        parts = pipeline.split("_")
        if len(parts) < 2:
            return parts[0]
        return "_".join(parts[1:]) + "_" + parts[0]

    return [convert_pipeline(p) for p in pipelines]


def cramers_v(x: ArrayLike, y: ArrayLike) -> float:
    """Calculate Cramer's V for two categorical variables."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    if min((kcorr - 1), (rcorr - 1)) == 0:
        return 0.0
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def anova_test(x: ArrayLike, y: ArrayLike) -> float:
    """ANOVA test between categorical and continuous variables."""
    categories = pd.Series(x).unique()
    groups = [y[x == category] for category in categories]
    f_stat, p_value = stats.f_oneway(*groups)
    return p_value


def correlation_ratio(categories: ArrayLike, measurements: ArrayLike) -> float:
    """Measure of the relationship between a categorical and numerical variable."""
    fcat, _ = pd.factorize(np.array(categories))
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(cat_num):
        cat_measures = measurements[fcat == i]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
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
