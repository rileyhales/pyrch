import datetime
from dateutil import relativedelta as durt

import pandas as pd

__all__ = ['time_offset', 'window_avg']


def time_offset(df: pd.DataFrame, label: str = '_timeoffset',
                years: int = 0, months: int = 0,
                days: int = 0, hours: int = 0, minutes: int = 0, seconds: int = 0) -> pd.DataFrame:
    """
    Make a duplicate dataframe where the datetime offset has been offset by given amount

    Recommended usage- make a time offset dataframe and then merge it with the original dataset

    Args:
        df: the pd.DataFrame to compute time offsets for
        label: a suffix to be applied to each column's name
        years: years to offset used by dateutil.relativedelta
        months: months to offset used by dateutil.relativedelta
        days: days to offset used by datetime.timedelta
        hours: hours to offset used by datetime.timedelta
        minutes: minutes to offset used by datetime.timedelta
        seconds: seconds to offset used by datetime.timedelta

    Returns:
        pd.DataFrame
    """
    df1 = df.copy()

    du_offset = durt.relativedelta(years=years, months=months)
    dt_offset = datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

    df1.index = [i + du_offset + dt_offset for i in df1.index]
    df1.columns = [f'{i}{label}' for i in df1.columns]
    return df1


def window_avg(df: pd.DataFrame, steps: int = 1, label: str = '_winavg',) -> pd.DataFrame:
    """
    Make a duplicate dataframe where the rows are rolling window averages.

    Args:
        df: the pd.DataFrame to compute window averages for
        steps: the number of steps (rows) to average including the row at the end of the window
        label: a suffix to be applied to each column's name

    Returns:
        pd.DataFrame
    """
    df1 = df.copy()
    df1 = df1.rolling(window=steps).mean()
    df1.dropna(inplace=True)
    df1.columns = [f'{i}{label}' for i in df1.columns]
    return df1
