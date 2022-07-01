import math

import numpy as np
import pandas as pd

__all__ = ['gumbel1', 'fdc', 'downstream', 'upstream', ]


def gumbel1(std: float, xbar: float, rp: int or float) -> float:
    """
    Solves the Gumbel Type I probability distribution function (pdf) = exp(-exp(-b)) where b is the covariate. Provide
    the standard deviation and mean of the list of annual maximum flows. Compare scipy.stats.gumbel_r

    Args:
        std (float): the standard deviation of the series
        xbar (float): the mean of the series
        rp (int or float): the return period in years

    Returns:
        float, the flow corresponding to the return period specified
    """
    # xbar = statistics.mean(year_max_flow_list)
    # std = statistics.stdev(year_max_flow_list, xbar=xbar)
    return -math.log(-math.log(1 - (1 / rp))) * std * .7797 + xbar - (.45 * std)


def fdc(a: list or np.array, steps: int = 500, exceed: bool = True) -> pd.DataFrame:
    """
    Generate the flow duration curve for a provided series of data

    Args:
        a (list or np.array): the flow values for which to compute the flow duration curve
        steps (int): the number of intervals between 0 and 100 for which to compute the exceedance probabilities
        exceed (bool): whether to show the exceedance probabilities (True) or non-exceedance probabilities (False)

    Returns:
        pd.DataFrame with 2 columns, probability and flow
    """
    percentiles = [round((100 / steps) * i, 5) for i in range(steps + 1)]
    flows = np.nanpercentile(a, percentiles)
    columns = ['probability', 'flow']
    if exceed:
        percentiles.reverse()
    return pd.DataFrame(np.transpose([percentiles, flows]), columns=columns)


def downstream(df: pd.DataFrame, target_id: int, id_col: str, next_col: str, order_col: str = None,
               same_order: bool = False, outlet_next_id: str or int = -1) -> tuple:
    """
    Traverse a stream network table containing a column of unique ID's, a column of the ID for the stream/basin
    downstream of that point, and, optionally, a column containing the stream order.

    Args:
        df (pd.DataFrame): a pandas DataFrame containing the id_col, next_col, and order_col if same_order is True
        target_id (int): the ID of the stream to begin the search from
        id_col (str): name of the DataFrame column which contains stream/basin ID's
        next_col (str): name of the DataFrame column which contains stream/basin ID's of the downstream segments
        order_col (str): name of the DataFrame column which contains stream orders
        same_order (bool): True limits searching to streams of the same order as the starting stream. False searches
            all streams until the outlet is found
        outlet_next_id (str or int): The placeholder value used as the downstream stream ID value at the outlet

    Returns:
        Tuple of stream ids in the order they come from the starting point.
    """
    downstream_ids = [target_id, ]

    df_ = df.copy()
    if same_order:
        df_ = df_[df_[order_col] == df_[df_[id_col] == target_id][order_col].values[0]]

    stream_row = df_[df_[id_col] == target_id]
    while stream_row[next_col].values[0] != outlet_next_id:
        downstream_ids.append(stream_row[next_col].values[0])
        stream_row = df_[df_[id_col] == stream_row[next_col].values[0]]
        if len(stream_row) == 0:
            break
    return tuple(downstream_ids)


def upstream(df: pd.DataFrame, target_id: int, id_col: str, next_col: str, order_col: str = None,
             same_order: bool = False) -> tuple:
    """
    Traverse a stream network table containing a column of unique ID's, a column of the ID for the stream/basin
    downstream of that point, and, optionally, a column containing the stream order.

    Args:
        df (pd.DataFrame): a pandas DataFrame containing the id_col, next_col, and order_col if same_order is True
        target_id (int): the ID of the stream to begin the search from
        id_col (str): name of the DataFrame column which contains stream/basin ID's
        next_col (str): name of the DataFrame column which contains stream/basin ID's of the downstream segments
        order_col (str): name of the DataFrame column which contains stream orders
        same_order (bool): True limits searching to streams of the same order as the starting stream. False searches
            all streams until the head of each branch is found

    Returns:
        Tuple of stream ids in the order they come from the starting point. If you chose same_order = False, the
        streams will appear in order on each upstream branch but the various branches will appear mixed in the tuple in
        the order they were encountered by the iterations.
    """
    df_ = df.copy()
    if same_order:
        df_ = df_[df_[order_col] == df_[df_[id_col] == target_id][order_col].values[0]]

    # start a list of the upstream ids
    upstream_ids = [target_id, ]
    upstream_rows = df_[df_[next_col] == target_id]

    while not upstream_rows.empty or len(upstream_rows) > 0:
        if len(upstream_rows) == 1:
            upstream_ids.append(upstream_rows[id_col].values[0])
            upstream_rows = df_[df_[next_col] == upstream_rows[id_col].values[0]]
        elif len(upstream_rows) > 1:
            for s_id in upstream_rows[id_col].values.tolist():
                upstream_ids += list(upstream(df_, s_id, id_col, next_col, order_col, same_order))
                upstream_rows = df_[df_[next_col] == s_id]
    return tuple(set(upstream_ids))


def manning_q(k: float, n: float, a: float, r: float, s: float) -> float:
    """
    Compute discharge with the manning equation given k, n, a, r and s.

    Args:
        k (float): unit coefficient 1 for SI or 1.49 for US Customary
        n (float): the Manning's n value for the stream
        a (float): the flow area of the stream
        r (float): the hydraulic radius of the stream
        s (float): the streambed slope

    Returns:
        float, the Manning's Q value for the stream
    """
    return (k / n) * a * r ** 2 / 3 * s ** 0.5


def manning_v(k: float, n: float, r: float, s: float) -> float:
    """
    Compute average cross section velocity with the manning equation given k, n, r and s.

    Args:
        k (float): unit coefficient 1 for SI or 1.49 for US Customary
        n (float): the Manning's n value for the stream
        r (float): the hydraulic radius of the stream
        s (float): the streambed slope

    Returns:
        float, the Manning's V value for the stream
    """
    return manning_q(k, n, 1, r, s)


class ChannelCalculator:
    """
    Given a cross section profile, computes several channel properties including the area, wetted perimeter, top width,
    froude number, hydraulic depth, and discharge

    Args:
        cross_section (pd.DataFrame): a DataFrame of cross section data with columns 'x', 'z', and optionally 'n'
        s (float): the streambed slope

    Returns:
        pd.DataFrame containing channel properties
    """
    cs: pd.DataFrame
    s: float
    min_z: float
    max_z: float

    def __init__(self, cross_section: pd.DataFrame, s: float):
        self.cs = cross_section.copy()
        self.slope = s
        self.min_z = self.cs['z'].min()
        self.max_z = self.cs['z'].max()

    def __repr__(self):
        return f'ChannelCalculator(cross_section={self.cs}, s={self.slope})'

    def __str__(self):
        return f'ChannelCalculator(cross_section={self.cs}, s={self.slope})'

    def q_given_y(self, y: float) -> float:
        """
        Compute the discharge given the channel depth.

        Args:
            y (float): the channel depth

        Returns:
            float, the discharge
        """
        return

    def y_given_q(self, q: float) -> float:
        """
        Compute the channel depth given the discharge.

        Args:
            q (float): the discharge

        Returns:
            float, the channel depth
        """
        return

    def rating_curves(self, steps: int = 25) -> pd.DataFrame:
        """
        Computes rating curves of depth vs discharge and other channel properties including area, wetted perimeter, top
        width, froude number, hydraulic depth, and discharge.

        Args:
            steps (int): the number of steps to use in the rating curve

        Returns:
            pd.DataFrame, the rating curves
        """
        return

    @staticmethod
    def _area_trap(l1, l2, w):
        return (l1 + l2) * w / 2

    @staticmethod
    def _hypot(a, b):
        return (a ** 2 + b ** 2) ** 0.5

    @staticmethod
    def _width(x2, x1):
        return x2 - x1

    @staticmethod
    def _froude(v, y):
        return v / (9.81 * y) ** 0.5

    @staticmethod
    def x_intercept(row):
        return row.z_next / ((row.z_next - row.z) / (row.x_next - row.x)) + row.x
