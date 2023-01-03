import affine
import geopandas as gpd
import numpy as np
import rasterio.features


def polygon_raster_mask(vector: gpd.GeoDataFrame,
                        aff: affine.Affine = None,
                        height: int = None,
                        width: int = None,
                        x: np.array = None,
                        y: np.array = None,
                        all_touched: bool = False,
                        invert: bool = True, ) -> np.array:
    """
    Creates a raster mask from a vector polygon. The mask is the same shape as the grid_shape and the grid_affine
    defines the pixel size and location.

    Args:
        vector (gpd.GeoDataFrame): A vector polygon
        aff (Affine): The affine transformation of the raster grid
        height (int): The height of the raster grid
        width (int): The width of the raster grid
        x (np.array): The x coordinates of the raster grid
        y (np.array): The y coordinates of the raster grid
        all_touched (bool): If True, all pixels touched by geometries will be included in the mask
        invert (bool): If True, the mask values will be inverted from default (1 = selected, 0 = not selected)

    Returns:
        np.array of shape described by aff or x and y
    """
    if aff is not None and height is not None and width is not None:
        grid_shape = (height, width)
    elif x is not None and y is not None:
        grid_shape = (len(y), len(x))
        aff = affine.Affine(np.abs(x[1] - x[0]), 0, x.min(), 0, -np.abs(y[1] - y[0]), y.max())
    else:
        raise ValueError('Either an affine transform or the grid\'s x and y cooredinates must be provided')

    # create a raster mask from the vector polygon
    return rasterio.features.geometry_mask(
        geometries=vector.geometry,
        out_shape=grid_shape,
        transform=aff,
        all_touched=all_touched,
        invert=invert
    )
