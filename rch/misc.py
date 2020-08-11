import math

import numpy as np
import pandas as pd

__all__ = ['interpolate_idw', 'gumbel_1', 'flow_duration_curve']


def interpolate_idw(a: np.array, loc: tuple, p: int = 1, r: int or float = None, nearest: int = None,
                    bound: int = None) -> float:
    """
    Computes the interpolated value at a specified location (loc) from an array of measured values (a). There are 3
    ways to limit the interpolation points considered.

    1. Use a search radius: only points within a certain radius of the location to be interpolated will be considered
    2. Choose a number of nearest neighbors to use.
    3. Use a number of points on each side that bound the interpolation location.

    All 3 may be applied but the most restrictive option will govern. For example, suppose you choose a radius of 10
    and to use the 5 nearest neighbors. If there are 100 points within the radius of 10, 95 of them will be ignored
    since you limited the interpolation to the nearest 5 points. Similarly, If you choose to take 5 points from each of
    the bounding quadrants and also the nearest 5 neighbors, then 15 of the points chosen because they bound the
    location will get ignored. This behavior could be intentional so all 3 variables can still be specified and applied.

    Args:
        a (np.array): a numpy array with 3 columns (x, y, value) and a row for each measurement
        loc (tuple): a tuple of (x, y) coordinates representing the location to get the interpolated values at
        p (int): an integer representing the power factor applied to the distances before inverting (usually 1, 2, 3)
        r (int or float): the radius, same length units as x & y values, to limit the value pairs in a. only points <=
            r distance away are used for the interpolation
        nearest (int): number of nearest points to include in the interpolation, if that many are available.
        bound (int): number of nearest points to include from the 4 bounding quadrants, if that many are available.

    Returns:
        float, the IDW interpolated value for the loc specified
    """
    # identify the x distance from location to the measurement points
    x = np.subtract(a[:, 0], loc[0])
    # identify the y distance from location to the measurement points
    y = np.subtract(a[:, 1], loc[1])
    # select the values column of the data
    val = a[:, 2]
    # compute the pythagorean distance (square root sum of the squares)
    dist = np.sqrt(np.add(np.multiply(x, x), np.multiply(y, y)))
    # filter the nearest number of points, limit by radius, and/or use bounding points (all use df sorted by distance)
    if r is not None or nearest is not None or bound is not None:
        b = pd.DataFrame({'dist': dist, 'val': val, 'x': x, 'y': y})
        b.sort_values('dist', inplace=True)
        if bound is not None:
            b = pd.concat((b.loc[(b['x'] >= 0) & (b['y'] > 0)].head(bound),
                           b.loc[(b['x'] >= 0) & (b['y'] < 0)].head(bound),
                           b.loc[(b['x'] < 0) & (b['y'] > 0)].head(bound),
                           b.loc[(b['x'] < 0) & (b['y'] < 0)].head(bound),))
        if nearest is not None:
            b = b.head(nearest)
        if r is not None:
            b = b[b['dist'] <= r]
        dist = b['dist'].values
        val = b['val'].values

    # raise distances to power (usually 1 or 2)
    dist = np.power(dist, p)
    # inverse the distances
    dist = np.divide(1, dist)

    return float(np.divide(np.sum(np.multiply(dist, val)), np.sum(dist)))


def gen_interpolation_grid(random: bool = False, n: int = 41):
    """
    Generates a symmetrical, square shaped grid centered at 0, 0 where the length of one size is n. N should be an odd
    number so that the grid can be symmetrical and include the x and y axis lines. This function will add 1 to the
    shape if you provide an even number.

    Args:
        random (bool): Whether the values are random or equal to x * y
        n (int): an odd integer defining the length of the square grid's edges

    Returns:
        pd.DateFrame
    """
    if n % 2 == 0:
        l = n + 1
    else:
        l = n
    x = np.asarray([[i] * l for i in range(-(l // 2), (l // 2) + 1)]).flatten()
    y = np.asarray(list(range(-(l // 2), (l // 2) + 1)) * l).flatten()
    if random:
        return pd.DataFrame({'x': x, 'y': y, 'v': np.random.randint(-(l // 2), l // 2, size=(l * l,))})
    else:
        return pd.DataFrame({'x': x, 'y': y, 'v': x * y})


def gumbel_1(std: float, xbar: float, rp: int or float) -> float:
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


def flow_duration_curve(a: list or np.array, steps: int = 500, exceedence: bool = True) -> pd.DataFrame:
    """
    Generate the flow duration curve for a provided series of data

    Args:
        a (list or np.array): the flow values for which to compute the flow duration curve
        steps (int): the number of intervals between 0 and 100 for which to compute the exceedence probabilities
        exceedence (bool): whether to show the exceedence probabilities (True) or non-exceedence probabilities (False)

    Returns:
        pd.DataFrame with 2 columns, probability and flow
    """
    percentiles = [round((100 / steps) * i, 5) for i in range(steps + 1)]
    flows = np.nanpercentile(a, percentiles)
    if exceedence:
        percentiles.reverse()
        columns = ['Exceedence Probability', 'Flow']
    else:
        columns = ['Non-Exceedence Probability', 'Flow']
    return pd.DataFrame(np.transpose([percentiles, flows]), columns=columns)
