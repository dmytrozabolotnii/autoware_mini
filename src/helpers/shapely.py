import math
import numpy as np
from shapely import Point, LineString, Polygon
from shapely.affinity import rotate
from helpers.geometry import project_vector


def convert_to_shapely_points_list(geometry):
    """
    Shapely intersection can result in different geometries.
    This functions converts the result to a list of points

    :param geometry: intersection result
    :return: list of shapely Point objects
    """

    # Convert intersection result (geometry) to a list of points
    if isinstance(geometry, Polygon):
        # skip last point because it is the same as the first
        intersection_points = [Point(coord) for coord in geometry.exterior.coords[:-1]]
    elif isinstance(geometry, Point):
        intersection_points = [geometry]
    elif isinstance(geometry, LineString):
        intersection_points = [Point(coord) for coord in geometry.coords]
    # in case of MultiPoint, MultiLineString, MultiPolygon, GeometryCollection
    else:
        intersection_points = []
        for geom in geometry.geoms:
            intersection_points.extend(convert_to_shapely_points_list(geom))

    return intersection_points


def get_polygon_width(polygon, heading_angle):
    """
    Get width of the polygon. Measured perpendicular to moving direction.
    :param polygon: shapely Polygon
    :param heading_angle: heading angle in degrees
    :return: width of the polygon
    """

    # rotate polygon to align with y axis, so the width will be in x direction
    angle = 90 - heading_angle
    rotated_polygon = rotate(polygon, angle, origin='centroid', use_radians=False)
    minx, miny, maxx, maxy = rotated_polygon.bounds
    width = maxx - minx
    return width

def get_path_heading(path, distance):
    """
    Get heading of the path at a given distance
    :param path: shapely LineString
    :param distance: distance along the path
    :return: heading angle in radians
    """

    object_location = path.interpolate(distance)
    track_point = path.interpolate(distance - 1.0)

    # get heading between two points
    path_heading = math.atan2(object_location.y - track_point.y, object_location.x - track_point.x)

    return path_heading


def transform_velocity_with_respect_to_path(path, object_distance, object_velocity):
    """
    Transform object velocity with respect to closest point on path
    :param path: shapely LineString
    :param object_distance: distance of object from the path
    :param object_velocity: velocity of object (Vector3)
    :return: transformed velocity
    """

    path_heading = get_path_heading(path, object_distance)
    speed = project_vector(path_heading, (object_velocity.x, object_velocity.y))

    return speed