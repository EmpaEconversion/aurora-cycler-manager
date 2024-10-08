import os
import yaml
import numpy as np
import json
import sqlite3
import pandas as pd
from scipy import stats

def get_sample_names(config: dict) -> list:
    db_path = config['Database path']
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT `Sample ID` FROM samples")
        samples = cursor.fetchall()
    return [sample[0] for sample in samples]

def get_batch_names(config: dict) -> list:
    graph_config_path = config['Graph config path']
    with open(graph_config_path, 'r') as f:
        graph_config = yaml.safe_load(f)
    return list(graph_config.keys())

def get_database(config: dict) -> dict:
    db_path = config['Database path']
    unused_pipelines = config.get('Unused pipelines', [])
    pipeline_query = "SELECT * FROM pipelines WHERE " + " AND ".join([f"Pipeline NOT LIKE '{pattern}'" for pattern in unused_pipelines])
    db_data = {
        'samples': pd.read_sql_query("SELECT * FROM samples", sqlite3.connect(db_path)).to_dict("records"),
        'results': pd.read_sql_query("SELECT * FROM results", sqlite3.connect(db_path)).to_dict("records"),
        'jobs': pd.read_sql_query("SELECT * FROM jobs", sqlite3.connect(db_path)).to_dict("records"),
        'pipelines': pd.read_sql_query(pipeline_query, sqlite3.connect(db_path)).to_dict("records")
    }
    db_columns = {
        'samples': [{'field' : col, 'filter': True, 'tooltipField': col} for col in db_data['samples'][0].keys()],
        'results': [{'field' : col, 'filter': True, 'tooltipField': col} for col in db_data['results'][0].keys()],
        'jobs': [{'field' : col, 'filter': True, 'tooltipField': col} for col in db_data['jobs'][0].keys()],
        'pipelines': [{'field' : col, 'filter': True, 'tooltipField': col} for col in db_data['pipelines'][0].keys()],
    }
    return {'data':db_data, 'column_defs': db_columns}

def cramers_v(x, y):
    """ Calculate Cramer's V for two categorical variables. """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def anova_test(x, y):
    """ ANOVA test between categorical and continuous variables."""
    categories = x.unique()
    groups = [y[x == category] for category in categories]
    f_stat, p_value = stats.f_oneway(*groups)
    return p_value

def correlation_ratio(categories, measurements):
    """ Measure of the relationship between a categorical and numerical variable. """
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta

def correlation_matrix(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the correlation matrix for a DataFrame including categorical columns.
    For continuous-continuous use Pearson correlation
    For continuous-categorical use correlation ratio
    For categorical-categorical use Cramer's V

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

def moving_average(x, npoints=11):
    if npoints % 2 == 0:
        npoints += 1  # Ensure npoints is odd for a symmetric window
    window = np.ones(npoints) / npoints
    xav = np.convolve(x, window, mode='same')
    xav[:npoints // 2] = np.nan
    xav[-npoints // 2:] = np.nan
    return xav

def deriv(x, y):
    with np.errstate(divide='ignore'):
        dydx = np.zeros(len(y))
        dydx[0] = (y[1] - y[0]) / (x[1] - x[0])
        dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
        dydx[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])

    # for any 3 points where x direction changes sign set to nan
    mask = (x[1:-1] - x[:-2]) * (x[2:] - x[1:-1]) < 0
    dydx[1:-1][mask] = np.nan
    return dydx

def smoothed_derivative(x, y, npoints=21):
    x_smooth = moving_average(x, npoints)
    y_smooth = moving_average(y, npoints)
    dydx_smooth = deriv(x_smooth, y_smooth)
    dydx_smooth[deriv(x_smooth,np.arange(len(x_smooth))) < 0] *= -1
    dydx_smooth[abs(dydx_smooth) > 100] = np.nan
    return dydx_smooth
