import datetime
import numpy as np
import os
import pandas as pd
import shutil
import urllib.request
import zipfile


__all__ = [
    'fetch_ml_ratings',
]

VARIANTS = {
    '100k': {'filename': 'u.data', 'sep': '\t'},
    '1m': {'filename': 'ratings.dat', 'sep': r'::'},
    '10m': {'filename': 'ratings.dat', 'sep': r'::'},
    '20m': {'filename': 'ratings.csv', 'sep': ','}
}


def fetch_ml_ratings(data_dir_path=None, variant='20m', verbose=False):
    """Fetches MovieLens ratings dataset.

    Parameters
    ----------
    data_dir_path : str, default=None
        Explicit data directory path to MovieLens ratings file.
    variant : {'100k', '1m', '10m', '20m'}, default='20m'
        Movie lens dataset variant.
    verbose : bool, default=False
        Whether or not downloading and unzipping the dataset with verbose.

    Returns
    -------
    df : pandas.DataFrame
        The MovieLens ratings dataset.
    """
    if data_dir_path is None:
        data_dir_path = _get_data_dir_path(data_dir_path)
        dirname = 'ml-' + variant
        filename = VARIANTS[variant]['filename']
        csv_path = os.path.join(data_dir_path, dirname, filename)
        zip_path = os.path.join(data_dir_path, dirname) + '.zip'
        url = 'http://files.grouplens.org/datasets/movielens/ml-' + variant + \
              '.zip'
    else:
        csv_path = data_dir_path

    if os.path.exists(csv_path):
        # Return data loaded into a DataFrame
        df = _ml_ratings_csv_to_df(csv_path, variant)
        return df

    elif os.path.exists(zip_path):
        # Unzip file before calling back itself
        if verbose:
            print('Unzipping data...')

        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_dir_path)

        if variant == '10m':
            os.rename(os.path.join(data_dir_path, 'ml-10M100K'),
                      os.path.join(data_dir_path, dirname))

        os.remove(zip_path)

        return fetch_ml_ratings(variant=variant, verbose=verbose)

    else:
        # Download the ZIP file before calling back itself
        if verbose:
            print('Downloading data...')

        with urllib.request.urlopen(url) as r, open(zip_path, 'wb') as f:
            shutil.copyfileobj(r, f)

        return fetch_ml_ratings(variant=variant, verbose=verbose)


def _get_data_dir_path(data_dir_path=None):
    """Returns the path of the funk-svd data directory.

    This folder is used to store large datasets to avoid downloading them
    several times.
    By default the data dir is set to a folder named 'funk_svd_data' in the
    user home folder. Alternatively, it can be set by the `FUNK_SVD_DATA`
    environment variable or programmatically by giving an explicit
    `data_dir_path`.
    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_dir_path : str, default=None
        Explicit data directory path for large datasets.

    Returns
    -------
    data_dir_path: str
        Explicit data directory path for large datasets.

    """
    if data_dir_path is None:
        default = os.path.join('~', 'funk_svd_data')
        data_dir_path = os.environ.get('FUNK_SVD_DATA', default=default)
        data_dir_path = os.path.expanduser(data_dir_path)

    if not os.path.exists(data_dir_path):
        os.makedirs(data_dir_path)

    return data_dir_path


def _ml_ratings_csv_to_df(csv_path, variant):
    names = ['u_id', 'i_id', 'rating', 'timestamp']
    dtype = {'u_id': np.uint32, 'i_id': np.uint32, 'rating': np.float64}

    def date_parser(time):
        return datetime.datetime.fromtimestamp(float(time))

    df = pd.read_csv(csv_path, names=names, dtype=dtype, header=0,
                     sep=VARIANTS[variant]['sep'], parse_dates=['timestamp'],
                     date_parser=date_parser, engine='python')

    df.sort_values(by='timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
