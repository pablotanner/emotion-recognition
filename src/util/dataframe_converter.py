import numpy as np
import dask.dataframe as dd
import pandas as pd
import dask_cudf


# Used during gpu memory optimization attempts, not sure if it's useful


def convert_to_cudf_df(X_train, X_val, X_test, y_train, y_val, y_test, npartitions=10):
    """
    Convert NumPy arrays or Pandas DataFrames to Dask-cuDF DataFrames.

    Parameters:
    X_train, X_val, X_test: Features for training, validation, and testing.
    y_train, y_val, y_test: Labels for training, validation, and testing.
    npartitions: Number of partitions for the Dask DataFrames (default is 10).

    Returns:
    A dictionary containing the converted Dask-cuDF DataFrames.
    """

    def to_dask_cudf(df, npartitions):
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)
        ddf = dd.from_pandas(df, npartitions=npartitions)
        return dask_cudf.from_dask_dataframe(ddf)

    dask_cudf_data = {
        'X_train': to_dask_cudf(X_train, npartitions),
        'X_val': to_dask_cudf(X_val, npartitions),
        'X_test': to_dask_cudf(X_test, npartitions),
        'y_train': to_dask_cudf(y_train, npartitions),
        'y_val': to_dask_cudf(y_val, npartitions),
        'y_test': to_dask_cudf(y_test, npartitions)
    }

    return dask_cudf_data


def convert_and_save_to_disk(X_train, X_val, X_test, y_train, y_val, y_test, npartitions=10, path='./dask_data'):
    """
    Convert datasets to Dask DataFrames and save them to disk.

    Parameters:
    X_train, X_val, X_test: Features for training, validation, and testing.
    y_train, y_val, y_test: Labels for training, validation, and testing.
    npartitions: Number of partitions for the Dask DataFrames (default is 10).
    path: Directory path to save the Dask DataFrames.
    """

    def to_dask(df, npartitions):
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)
        return dd.from_pandas(df, npartitions=npartitions)

    # Convert to Dask DataFrames
    dask_data = {
        'X_train': to_dask(X_train, npartitions),
        'X_val': to_dask(X_val, npartitions),
        'X_test': to_dask(X_test, npartitions),
        'y_train': to_dask(y_train, npartitions),
        'y_val': to_dask(y_val, npartitions),
        'y_test': to_dask(y_test, npartitions)
    }

    # Save Dask DataFrames to disk
    for key, df in dask_data.items():
        df.to_parquet(f'{path}/{key}.parquet')

    return dask_data


def load_from_disk(path='./dask_data'):
    """
    Load Dask DataFrames from disk.

    Parameters:
    path: Directory path from where to load the Dask DataFrames.

    Returns:
    A dictionary containing the loaded Dask DataFrames.
    """
    dask_data = {
        'X_train': dd.read_parquet(f'{path}/X_train.parquet'),
        'X_val': dd.read_parquet(f'{path}/X_val.parquet'),
        'X_test': dd.read_parquet(f'{path}/X_test.parquet'),
        'y_train': dd.read_parquet(f'{path}/y_train.parquet'),
        'y_val': dd.read_parquet(f'{path}/y_val.parquet'),
        'y_test': dd.read_parquet(f'{path}/y_test.parquet')
    }

    return dask_data