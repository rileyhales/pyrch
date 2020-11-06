# conda install google-api-python-client google-auth-httplib2 google-auth-oauthlib pandas numpy
import numpy as np
import pandas as pd
import requests
from googleapiclient.discovery import build

__all__ = ['read_google_sheet', 'write_google_sheet', 'tmdb_find_movie', 'tmdb_data_for_id']


def read_google_sheet(service: build, sheet_id: str, sheet_range: str,
                      skip_cols: int or list or tuple = None, skip_rows: int or list or tuple = None,
                      columns: bool = True, indexed: bool = False) -> pd.DataFrame:
    """
    Reads a subset of a google sheet via the google sheets api and returns it converted to a pandas DataFrame

    Args:
        service (googleapiclient.discovery.build): a google api service python instance
        sheet_id (str): the identifier for the google sheet
        sheet_range (str): the absolute reference to the cells to be read. For example: "Sheet1!A:H" reads columns
            A -> H on the sheet names Sheet1 within the specified Google Sheet.
        skip_cols (int): an int or iterable of ints of the spreadsheet columns numbers to be ignored in the output
            (e.g. column A = 1, column B = 2, columns A through C = (1, 2, 3)
        skip_rows (int): an int or iterable of ints of the spreadsheet row numbers to be ignored in the output
        indexed (bool): Indicates whether the first column (after skip columns is applied) should be interpreted
            as the index in pandas
        columns (bool): Indicates whether the first row (after skip rows is applied) should be interpreted as
            labels for the data in the columns

    Returns:
        pd.DataFrame
    """
    # Call the Sheets API to get the spreadsheet data
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=sheet_id, range=sheet_range).execute()
    values = result.get('values', [])

    # make all the rows the same length
    length = max(map(len, values))
    array = np.array([xi + [''] * (length - len(xi)) for xi in values])

    # delete any rows or columns specified by the user
    if skip_rows is not None:
        array = np.delete(array, np.asarray(skip_rows) - 1, axis=0)  # subtract 1 -> sheets start at 1, np at 0
    if skip_cols is not None:
        array = np.delete(array, np.asarray(skip_cols) - 1, axis=1)

    # extract index and column labels if applicable
    idx = None
    col = None
    if indexed:
        idx = array[:, 0]
        array = np.delete(array, 0, axis=1)
    if columns:
        col = array[0]
        array = np.delete(array, 0, axis=0)
    if indexed and columns:
        idx = np.delete(idx, 0, axis=0)

    return pd.DataFrame(array, columns=col, index=idx)


def write_google_sheet(df: pd.DataFrame, service: build, sheet_id: str, sheet_range: str) -> build:
    """
    writes a pandas dataframe to a google sheet on the range specified

    Args:
        service (googleapiclient.discovery.build): a google api service python instance
        sheet_id (str): the identifier for the google sheet
        sheet_range (str): the absolute reference to the cells to be read. For example: "Sheet1!A:H" reads columns
            A -> H on the sheet names Sheet1 within the specified Google Sheet.
        df (pd.DataFrame): the pandas dataframe to be written to the google sheet

    Returns:

    """
    col = np.asarray(df.columns)
    col = col.reshape(1, len(col))
    body = df.to_numpy()
    body = {'values': np.vstack((col, body)).tolist()}
    return service.spreadsheets().values().update(
        spreadsheetId=sheet_id, range=sheet_range, valueInputOption='RAW', body=body).execute()


def tmdb_find_movie(movie: str, tmdb_api_token: str):
    """
    Search the tmdb api for movies by title

    Args:
        movie (str): the title of a movie
        tmdb_api_token (str): your tmdb v3 api token

    Returns:
        dict
    """
    url = 'https://api.themoviedb.org/3/search/movie?'
    params = {'query': movie, 'language': 'en-US', 'api_key': tmdb_api_token, }
    return requests.get(url, params).json()


def tmdb_data_for_id(tmdb_id: int, tmdb_api_token: str) -> dict:
    """
    Get additional information for a movie for which you already have the ID

    Args:
        tmdb_id (int): the ID for a movie on The Movie Database
        tmdb_api_token (str): your tmdb v3 api token

    Returns:
        dict
    """
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?"
    params = {'language': 'en-US', 'api_key': tmdb_api_token, }
    return requests.get(url, params).json()
