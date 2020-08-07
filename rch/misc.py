import datetime
import math
import os

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr

__all__ = ['interpolate_idw', 'gumbel_1', 'flow_duration_curve', 'tif_to_nc']


def interpolate_idw(a: np.array, loc: tuple, p: int = 1, r: int or float = None, nearest: int = None, ) -> float:
    """
    Computes the interpolated value at a specified location (loc) from an array of measured values (a)

    Args:
        a (np.array): a numpy array with 3 columns (x, y, value) and a row for each measurement
        loc (tuple): a tuple of (x, y) coordinates representing the location to get the interpolated values at
        p (int): an integer representing the power factor applied to the distances before inverting (usually 1, 2, 3)
        r (int or float): the radius, same length units as x & y values, to limit the value pairs in a. only points <=
            r distance away are used for the interpolation
        nearest: the number of nearest points to consider as part of the interpolation

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
    # raise distances to power (usually 1 or 2)
    dist = np.power(dist, p)
    # filter the nearest number of points and/or limit by radius
    if r is not None or nearest is not None:
        b = pd.DataFrame({'dist': dist, 'val': val})
        if nearest is not None:
            b = b.sort_values('dist')
            b = b[b.index < nearest]
        if r is not None:
            b = b[b['dist'] <= r]
        dist = b['dist'].values
        val = b['val'].values

    # inverse the distances
    dist = np.divide(1, dist)

    return float(np.divide(np.sum(np.multiply(dist, val)), np.sum(dist)))


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


def tif_to_nc(tif: str, var: str, time: datetime.datetime, ext: str = 'nc4', dtype: str = 'i2', fill: int or str = 0,
              compress: bool = False, level: int = 9) -> None:
    """

    Args:
        tif (str): path to the tif to convert
        var (str): name to assign the netcdf variable where the geotiff information is stored
        time (datetime.datetime): the start time of the data in the tiff
        ext (str): the file extension to apply to the new file: either 'nc4' (default) or 'nc'
        dtype (str): the netcdf datatype of the variable to store in the new netcdf: default to i2. consult
            https://unidata.github.io/netcdf4-python/netCDF4/index.html
        fill (int or str): the fill value to apply when using a masked array in the new variable's data array
        compress (bool): True = compress the netcdf, False = do not compress the new file
        level (int): An integer between 1 and 9. 1 = least compression and 9 = most compression

    Returns:
        None
    """
    # read the tif with the xarray wrapper to rasterio for convenience in coding
    a = xr.open_rasterio(tif, 'r')
    shape = a.values.shape

    # create the new netcdf
    new_nc = nc.Dataset(f'./{os.path.splitext(tif)[0]}.{ext}', 'w')

    # create latitude dimension, variable, add values, metadata
    new_nc.createDimension('lat', shape[1])
    new_nc.createVariable('lat', 'f', ('lat',))
    new_nc['lat'].axis = "lat"
    new_nc['lat'][:] = a.y.values
    new_nc['lat'].units = "degrees_north"

    # create longitude dimension, variable, add values, metadata
    new_nc.createDimension('lon', shape[2])
    new_nc.createVariable('lon', 'f', ('lon',))
    new_nc['lon'][:] = a.x.values
    new_nc['lon'].axis = "lon"
    new_nc['lon'].units = "degrees_east"

    # create time dimension, variable, add values AND specify the units string (essential for thredds)
    new_nc.createDimension('time', 1)
    new_nc.createVariable('time', 'i2', ('time',))
    new_nc['time'].long_name = 'time'
    new_nc['time'].units = f'days since {time.strftime("%Y-%m-%d %X")}'
    new_nc['time'].calendar = 'standard'
    new_nc['time'].axis = 'T'

    # now create the variable which holds the tif's array (use a.values[0] because first dim is the band #)
    if compress:
        new_nc.createVariable(var, dtype, ('time', 'lat', 'lon'), fill_value=fill)
    else:
        new_nc.createVariable(var, dtype, ('time', 'lat', 'lon'), fill_value=fill, complevel=level, zlib=True)
    new_nc[var].axis = "lat lon"
    new_nc[var][:] = a.values

    # save and close the new_nc
    new_nc.sync()
    new_nc.close()
    return
