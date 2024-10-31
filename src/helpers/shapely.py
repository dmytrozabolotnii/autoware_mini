import math
from shapely.affinity import rotate


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

def calculate_cross_track_error(linsetring, position):
    """
    Calculate cross track error - calc distance from track and get the sign
    https://robotics.stackexchange.com/questions/22989/what-is-wrong-with-my-stanley-controller-for-car-steering-control

    :param linsetring: shapely linestring
    :param position: current position
    :return: cross track error
    """

    ego_distance_from_path_start = linsetring.project(position)

    # if distance is negative it is measured from the end of the linestring in reverse direction
    pos1 = linsetring.interpolate(max(0, ego_distance_from_path_start - 0.1))
    pos2 = linsetring.interpolate(ego_distance_from_path_start + 0.1)

    numerator = (pos2.x - pos1.x) * (pos1.y - position.y) - (pos1.x - position.x) * (pos2.y - pos1.y)
    denominator = math.sqrt((pos2.x - pos1.x) ** 2 + (pos2.y - pos1.y) ** 2)

    return numerator / denominator