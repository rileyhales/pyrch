import numpy as np
import pandas as pd

__all__ = ['interpolate_idw', 'resample_1d', 'resample_2d', ]


def interpolate_idw(a: np.array, loc: tuple,
                    p: int = 1, r: int or float = None, nearest: int = None, bound: int = None) -> float:
    """
    Computes the inverse distance weighted interpolated value at a specified location (loc) from an array of measured
    values (a). There are 3 ways to limit the interpolation points considered.

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


def gen_interpolation_grid(n: int = 41, random: bool = False) -> pd.DataFrame:
    """
    Generates a symmetrical, square shaped grid centered at 0, 0 where the length of one side is n. N should be an odd
    number so that the grid can be symmetrical and include the x and y axis lines. This function will add 1 to the
    shape if you provide an even number.

    Args:
        n (int): an odd integer defining the length of the square grid's edges
        random (bool): True -> randomly generated values, False -> values = x * y

    Returns:
        pd.DateFrame with an 'x', 'y', and 'v' (value) column.
    """
    size = n
    if size % 2 == 0:
        size += 1

    x = np.asarray([[i] * size for i in range(-(size // 2), (size // 2) + 1)]).flatten()
    y = np.asarray(list(range(-(size // 2), (size // 2) + 1)) * size).flatten()
    if random:
        return pd.DataFrame({'x': x, 'y': y, 'v': np.random.randint(-(size // 2), size // 2, size=(size * size,))})
    else:
        return pd.DataFrame({'x': x, 'y': y, 'v': x * y})


def resample_1d(a: np.array, f: int, stat: str = 'mean'):
    """
    Resamples the values in a 1D array by an integer factor using a specified stat type

    Args:
        a (np.array): A 1D array of numbers
        f (int): An integer resampling factor/ratio evenly divisible into len(a)
        stat (str): mean, max, min, median -> the method for aggregating the numeric values

    Returns:
        resampled np.array, ndim == 1,  shape == (a.shape[0] / f)
    """
    assert a.ndim == 1, "valid for 1D arrays only"
    assert a.shape[0] % f == 0, "length of axis=0 not evenly divisible by f0"
    assert stat in ('mean', 'max', 'min', 'median'), "choose either mean, max, min or median for aggregation stat"

    if stat == 'mean':
        return np.array([np.nanmean(i) for i in np.split(a, a.shape[0] / f)])
    elif stat == 'max':
        return np.array([np.nanmax(i) for i in np.split(a, a.shape[0] / f)])
    elif stat == 'min':
        return np.array([np.nanmin(i) for i in np.split(a, a.shape[0] / f)])
    elif stat == 'median':
        return np.array([np.nanmedian(i) for i in np.split(a, a.shape[0] / f)])


def resample_2d(a: np.array, f0: int, f1: int, stat: str = 'mean'):
    """
    Resamples the values in a 2D array by an integer factor for each axis using a specified stat type

    Args:
        a (np.array): A 2D array of numbers
        f0 (int): An integer resampling factor/ratio evenly divisible into a.shape[0] (along the row axis)
        f1 (int): An integer resampling factor/ratio evenly divisible into a.shape[1] (along the col axis)
        stat (str): mean, max, min, median -> the method for aggregating the numeric values

    Returns:
        resampled np.array, ndim == 2, shape == (a.shape[0] / f0, a.shape[1] / f1)
    """
    assert a.ndim == 2, "valid for 2D arrays only"
    assert a.shape[0] % f0 == 0, "length of axis=0 not evenly divisible by f0"
    assert a.shape[1] % f1 == 0, "length of axis=1 not evenly divisible by f1"
    assert stat in ('mean', 'max', 'min', 'median'), "choose either mean, max, min or median for aggregation stat"

    a0_segments = a.shape[0] / f0
    a1_segments = a.shape[1] / f1
    arr = []
    # split along first axis
    for b in np.split(a, a0_segments, axis=0):
        # split along second axis and aggregate values of each sub array
        if stat == 'mean':
            arr.append([np.nanmean(c) for c in np.split(b, a1_segments, axis=1)])
        elif stat == 'max':
            arr.append([np.nanmax(c) for c in np.split(b, a1_segments, axis=1)])
        elif stat == 'min':
            arr.append([np.nanmin(c) for c in np.split(b, a1_segments, axis=1)])
        elif stat == 'median':
            arr.append([np.nanmedian(c) for c in np.split(b, a1_segments, axis=1)])
    return np.array(arr)
