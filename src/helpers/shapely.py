import shapely
import math

def get_polygon_width(polygon, heading_angle):
    """
    Get width of the polygon. Measured perpendicular to moving direction.
    :param polygon: shapely Polygon
    :param heading_angle: heading angle in radians
    :return: width of the polygon
    """

    # rotate polygon to align with y axis, so the width will be in x direction
    angle = math.pi/2 - heading_angle
    rotated_polygon = shapely.affinity.rotate(polygon, angle, origin='centroid', use_radians=True)
    minx, miny, maxx, maxy = rotated_polygon.bounds
    width = maxx - minx
    return width

def get_heading_at_distance_along_linestring(linestring, distance):

    point_after_object = linestring.interpolate(distance + 0.1)
    # if distance is negative it is measured from the end of the linestring in reverse direction
    point_before_object = linestring.interpolate(max(0, distance - 0.1))

    # get heading between two points
    path_heading = math.atan2(point_after_object.y - point_before_object.y, point_after_object.x - point_before_object.x)

    return path_heading