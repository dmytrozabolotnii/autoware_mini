import shapely
import math

def get_polygon_width_and_prediction_origin(polygon, heading_angle):
    """
    Get width of the polygon and origin points for prediction center_front and center_center.
    :param polygon: shapely Polygon
    :param heading_angle: heading angle in radians
    :return: width of the polygon
    """

    # rotate polygon to align with y axis, so the width will be in x direction
    angle = math.pi/2 - heading_angle
    rotated_polygon = shapely.affinity.rotate(polygon, angle, origin='centroid', use_radians=True)
    minx, miny, maxx, maxy = rotated_polygon.bounds
    buffer_width = min(max((maxx - minx) / 2, 0.25), 2.0)

    # Calculate x and y coordinates in rotated coordinate system
    center_x = (minx + maxx) / 2
    front_y = maxy
    center_y = (miny + maxy) / 2

    # Reverse rotate the point to get the coordinates in the original coordinate system (correct orientation)
    center_front = rotate_point_around_origin(center_x, front_y, polygon.centroid, -angle)
    center_center = rotate_point_around_origin(center_x, center_y, polygon.centroid, -angle)

    return buffer_width, center_front, center_center


def rotate_point_around_origin(point_x, point_y, rotation_center, angle):
    """
    Rotate point around origin point and return new coordinates of the rotated point
    :param point_x: x coordinate of the point that needs to be rotated
    :param point_y: y coordinate of the point that needs to be rotated
    :param rotation_center: shapely Point, center of the rotation
    :param angle: angle in radians for rotation
    :return: shapely Point with coordinates of the rotatetd point
    """
    # Rotate the center point back to original coordinates
    relative_x = point_x - rotation_center.x
    relative_y = point_y - rotation_center.y

    # Apply inverse rotation (backwards rotation)
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    rotated_point_x = cos_angle * relative_x - sin_angle * relative_y + rotation_center.x
    rotated_point_y = sin_angle * relative_x + cos_angle * relative_y + rotation_center.y

    return shapely.Point(rotated_point_x, rotated_point_y)