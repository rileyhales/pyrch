from multiprocessing import Pool

import numpy as np
import pandas as pd

__all__ = ['idw', 'idw_pt', 'idw_grid_iter', 'idw_grid_mp', 'idw_grid_vector',
           'xy_to_grid', 'xyv_to_grid', 'xyv_to_grid', 'random_xyv', 'uniform_xy_coords',
           'resample_1d', 'resample_2d', 'checkerboard', ]


def idw(a: np.array or pd.DataFrame,
        x: int or float,
        y: int or float,
        p: int = 1,
        r: int or float = None,
        n: int = None,
        bound: int = None, ) -> float:
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
        a (np.array): array or DataFrame with 3 columns (x, y, value) and a row for each measurement -> shape (n, 3)
        x (int or float): the x coordinate of the location to get the interpolated values at
        y (int or float): the y coordinate of the location to get the interpolated values at
        p (int): an integer representing the power factor applied to the distances before inverting (usually 1, 2, 3)
        r (int or float): the radius, same length units as x & y values, to limit the value pairs in a. only points <=
            r distance away are used for the interpolation
        n (int): number of the nearest points to include in the interpolation, if that many are available.
        bound (int): number of the nearest points to include from the 4 bounding quadrants, if that many are available.

    Returns:
        float, the IDW interpolated value for the loc specified
    """
    # identify the xs distance from location to the measurement points
    xs = np.subtract(a[:, 0], x)
    # identify the y distance from location to the measurement points
    ys = np.subtract(a[:, 1], y)
    # select the values column of the data
    vs = a[:, 2]
    # compute the pythagorean distance (square root sum of the squares)
    dist = np.sqrt(np.add(np.multiply(xs, xs), np.multiply(ys, ys)))
    # filter the nearest number of points, limit by radius, and/or use bounding points (all use df sorted by distance)
    if r is not None or n is not None or bound is not None:
        b = pd.DataFrame({'dist': dist, 'val': vs, 'x': xs, 'y': ys})
        b.sort_values('dist', inplace=True)
        if bound is not None:
            b = pd.concat((b.loc[(b['x'] >= 0) & (b['y'] > 0)].head(bound),
                           b.loc[(b['x'] >= 0) & (b['y'] < 0)].head(bound),
                           b.loc[(b['x'] < 0) & (b['y'] > 0)].head(bound),
                           b.loc[(b['x'] < 0) & (b['y'] < 0)].head(bound),))
        if n is not None:
            b = b.head(n)
        if r is not None:
            b = b[b['dist'] <= r]
        dist = b['dist'].values
        vs = b['val'].values

    # raise distances to power (usually 1 or 2)
    dist = np.power(dist, p)
    # inverse the distances
    dist = np.divide(1, dist)

    return float(np.divide(np.sum(np.multiply(dist, vs)), np.sum(dist)))


def idw_pt(a: np.array,
           x: int or float,
           y: int or float,
           p: int = 1,
           r: int or float = None,
           n: int = None,
           bound: int = None, ) -> tuple:
    """
    Computes the inverse distance weighted interpolated value at a specified (x, y)

    Args:
        a (np.array): a numpy array with 3 columns (x, y, value) and a row for each measurement
        x (int or float): the x coordinate of the location to get the interpolated values at
        y (int or float): the y coordinate of the location to get the interpolated values at
        p (int): an integer representing the power factor applied to the distances before inverting (usually 1, 2, 3)
        r (int or float): the radius, same length units as x & y values, to limit the value pairs in a. only points <=
            r distance away are used for the interpolation
        n (int): number of the nearest points to include in the interpolation, if that many are available.
        bound (int): number of the nearest points to include from the 4 bounding quadrants, if that many are available.

    Returns:
        tuple, (x, y, value) of the location and IDW interpolated value for the loc specified
    """
    return x, y, idw(a, x, y, p, r, n, bound)


def idw_grid_iter(a: np.array,
                  grid: np.array,
                  p: int = 1,
                  r: int or float = None,
                  n: int = None,
                  bound: int = None, ) -> np.array:
    """
    Computes the inverse distance weighted interpolated values on a regular grid of locations

    Args:
        a (np.array): a numpy array with 3 columns (x, y, value) and a row for each measurement
        grid (list): a list of tuples of the format (x: float, y: float) representing the locations to interpolate
        p (int): an integer representing the power factor applied to the distances before inverting (usually 1, 2, 3)
        r (int or float): the radius, same length units as x & y values, to limit the value pairs in a. only points <=
            r distance away are used for the interpolation
        n (int): number of the nearest points to include in the interpolation, if that many are available.
        bound (int): number of the nearest points to include from the 4 bounding quadrants, if that many are available.

    Returns:
        np.array: a 2D array with the interpolated values
    """
    return np.array([idw_pt(a, x, y, p, r, n, bound) for x, y in grid])


def idw_grid_mp(a: np.array,
                grid: np.array,
                p: int = 1,
                r: int or float = None,
                n: int = None,
                bound: int = None,
                n_processes: int = None, ) -> np.array:
    """
    Computes the inverse distance weighted interpolated values on a regular grid of locations using multiprocessing

    Args:
        a (np.array): a numpy array with 3 columns (x, y, value) and a row for each measurement
        grid (list): a list of tuples of the format (x: float, y: float) representing the locations to interpolate
        p (int): an integer representing the power factor applied to the distances before inverting (usually 1, 2, 3)
        r (int or float): the radius, same length units as x & y values, to limit the value pairs in a. only points <=
            r distance away are used for the interpolation
        n (int): number of the nearest points to include in the interpolation, if that many are available.
        bound (int): number of the nearest points to include from the 4 bounding quadrants, if that many are available.
        n_processes (int): number of processes to use for multiprocessing passed to Pool()

    Returns:
        np.array: a 2D array with the interpolated values
    """
    # create a pool of workers to compute the interpolated values
    with Pool(n_processes) as pool:
        # map the idw function to the list of locations to interpolate
        values = pool.starmap(idw_pt, [(a, x, y, p, r, n, bound) for x, y in grid])

    return np.array(values)


def idw_grid_vector(a: np.array,
                    x: np.array,
                    y: np.array,
                    p: int = 1, ) -> np.array:
    """
    Computes the inverse distance weighted interpolated value at a series of (x, y) coordinates from an array of point
    values (a). This is a vectorized version of the idw function that is faster than looping through each coordinate.
    The idw function is more flexible and can be used to limit the number of points used in the interpolation.

    Args:
        a: array or DataFrame with 3 columns (x, y, value) and a row for each measurement -> shape (n, 3)
        x: a 2d array of the x coordinate of every grid cell to be interpolated (e.g. np.meshgrid(x, y)[0])
        y: a 2d array of the y coordinate of every grid cell to be interpolated (e.g. np.meshgrid(x, y)[1])
        p: a positive integer representing the exponent applied to the distances before inverting (usually 1, 2, 3)

    Returns:
        np.array: a 2d array of the interpolated values
    """
    # mask (drop) rows where the value is NaN in case data have not been cleaned
    arr = a[~np.isnan(a[:, 2])]

    # broadcast a 2d array to a 3d array
    shape = (arr[:, 2].shape[0], y.shape[0], x.shape[1])
    xs = np.broadcast_to(x, shape) - arr[:, 0].reshape(-1, 1, 1)
    ys = np.broadcast_to(y, shape) - arr[:, 1].reshape(-1, 1, 1)
    vs = np.broadcast_to(arr[:, 2].reshape(-1, 1, 1), shape)

    dists = np.sqrt(np.add(np.multiply(xs, xs), np.multiply(ys, ys)))
    dists = np.power(dists, p)
    dists = np.divide(1, dists)

    return np.divide(np.sum(np.multiply(dists, vs), axis=0), np.sum(dists, axis=0))


def xy_to_grid(xy: np.array) -> np.array:
    """
    Converts a numpy array of (x, y) to a 2D grid of values sorted by decreasing y and increasing x

    Args:
        xy: a numpy array of (x, y) float values of shape (n_rows, 2)

    Returns:
        np.array: a 2D array of values sorted by decreasing y and increasing x
    """
    # todo
    # convert to array sorted by decreasing y value then by increasing x value
    arr = np.array(sorted(xy, key=lambda x: (-x[1], x[0])))

    # find the number of unique x and y values
    num_x = len(np.unique(xy[:, 0]))
    num_y = len(np.unique(xy[:, 1]))

    # reshape the array to a 2D array with the interpolated values
    return arr.reshape((num_y, num_x))


def xyv_to_grid(xyv: np.array) -> np.array:
    """
    Converts a numpy array of (x, y, value) to a 2D grid of values sorted by decreasing y and increasing x

    Args:
        xyv: a numpy array of (x, y, value) float values of shape (n_rows, 3)

    Returns:
        np.array: a 2D array of values sorted by decreasing y and increasing x
    """

    # convert to array sorted by decreasing y value then by increasing x value
    arr = np.array(sorted(xyv, key=lambda x: (-x[1], x[0])))

    # find the number of unique x and y values
    num_x = len(np.unique(xyv[:, 0]))
    num_y = len(np.unique(xyv[:, 1]))

    # select only the interpolated values
    arr = arr[:, 2]

    # reshape the array to a 2D array with the interpolated values
    return arr.reshape((num_y, num_x))


def uniform_xy_coords(x_min: int or float, x_max: int or float, y_min: int or float, y_max: int or float,
                      step: int or float) -> tuple[np.array, ...]:
    """
    Creates a uniform grid of coordinates from the specified min and max values and step size

    Args:
        x_min: the minimum x value of the grid to interpolate
        x_max: the maximum x value of the grid to interpolate
        y_min: the minimum y value of the grid to interpolate
        y_max: the maximum y value of the grid to interpolate
        step: the step size of the grid

    Returns:
        np.array: a 1D array with the x and y coordinates of the grid
    """
    # create the grid of locations to interpolate
    x = np.arange(x_min, x_max + step, step)
    y = np.arange(y_min, y_max + step, step)

    # reverse the y values, so they are sorted by decreasing value
    y = y[::-1]

    # make a list of (x, y) tuples for each coordinate pair
    return np.array([(i, j) for i in x for j in y]), x, y


def random_xyv(n: int = 41, random: bool = False) -> pd.DataFrame:
    """
    Generates a symmetrical, square shaped grid centered at 0, 0 where the length of one side is n. N should be an odd
    number so that the grid can be symmetrical and include the x and y-axis lines. This function will add 1 to the
    shape if you provide an even number.

    Args:
        n (int): an odd integer defining the length of the square grid's edges
        random (bool): True -> randomly generated values, False -> values = x * y

    Returns:
        pd.DateFrame with an 'x', 'y', and 'v' (value) column.
    """
    size = int(n)
    if size % 2 == 0:
        size += 1

    x = np.asarray([[i] * size for i in range(-(size // 2), (size // 2) + 1)]).flatten()
    y = np.asarray(list(range(-(size // 2), (size // 2) + 1)) * size).flatten()
    if random:
        return pd.DataFrame({'x': x, 'y': y, 'v': np.random.randint(-(size // 2), size // 2, size=(size * size,))})
    else:
        return pd.DataFrame({'x': x, 'y': y, 'v': x * y})


def resample_1d(a: np.array, f: int, stat: str = 'mean') -> np.array:
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


def resample_2d(a: np.array, f0: int, f1: int, stat: str = 'mean') -> np.array:
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


def checkerboard(y: int, x: int) -> np.array:
    """
    Generates an array which alternates 0 and 1 in a checkerboard pattern

    Args:
        y: the height of the array, len(axis=0)
        x: the width of the array, len(axis=1)

    Returns:
        np.array of shape (y, x)
    """
    a = []
    a1 = [i % 2 for i in range(x)]
    a2 = [(i + 1) % 2 for i in range(x)]
    for i in range(y // 2):
        a.append(a1)
        a.append(a2)
    if y % 2 == 1:
        a.append(a1)
    return np.array(a)
