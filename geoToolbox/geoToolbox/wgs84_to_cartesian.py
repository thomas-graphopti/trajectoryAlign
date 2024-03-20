import pymap3d as pm
"""
step2: Cartesian to WGS84

At this step we realize the mutual transformation between Cartesian and WGS84. 

The projection is calculated from a fixed point in order to minimize the loss.

The input parameter includes the reference point,the Coordinate remains to be transformed 
and the alt value, which should be defaulted as zero if you do not have access to the data.
"""


def cartesian_to_wgs84(WGS84Reference, CartesianPosition):
    """
    :param WGS84Reference: reference coordinate[lat,lon]
    :param CartesianPosition: Cartesian [x,y] to be transformed
    :return:lat, lon, alt for WGS84
    """
    lat, lon, alt = pm.ned2geodetic(CartesianPosition[1], CartesianPosition[0],
                                    0, WGS84Reference[0], WGS84Reference[1], 0)
    return [lat, lon]


def wgs84_to_cartesian(WGS84Reference, WGS84Position):
    """
    :param WGS84Reference:
    :param WGS84Position:
    :return: Returning North,East,Down for Cartesian
    """
    north, east, down = pm.geodetic2ned(WGS84Position[0], WGS84Position[1], 0,
                                        WGS84Reference[0], WGS84Reference[1],
                                        0)
    return [east, north]
