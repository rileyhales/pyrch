from shapely.geometry import Point

__all__ = ['point_to_xy', 'linestring_first_point', 'linestring_last_point']


def point_to_xy(row) -> tuple:
    """
    A function that can be applied to a GeoDataFrame to create columns of the x and y locations of a Point.
    All rows must contain Point geometry.

    Examples:
        # gdf: GeoDataFrame
        gdf[['x', 'y']] = gdf.apply(get_point_xy, axis=1)

    Args:
        row: the row of a GeoDataFrame from GeoPandas

    Returns:
        tuple (x, y)
    """
    return row.x, row.y


def linestring_first_point(row) -> Point:
    """
    A function that can be applied to a GeoDataFrame to extract the location of the first point of a LineString.
    All rows must contain LineString data.

    Examples:
        # gdf: GeoDataFrame
        gdf['line_first_point'] = gdf.apply(linestring_first_point, axis=1)

    Args:
        row: the row of a GeoDataFrame from GeoPandas

    Returns:
        shapely.geometry.Point
    """
    return Point(row.geometry.coords[0])


def linestring_last_point(row) -> Point:
    """
    A function that can be applied to a GeoDataFrame to extract the location of the last point of a LineString.
    All rows must contain LineString data.

    Examples:
        # gdf: GeoDataFrame
        gdf['line_last_point'] = gdf.apply(linestring_last_point, axis=1)

    Args:
        row: the row of a GeoDataFrame from GeoPandas

    Returns:
        shapely.geometry.Point
    """
    return Point(row.geometry.coords[-1])
