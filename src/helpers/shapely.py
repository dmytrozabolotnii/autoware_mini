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

    center_front = rotate_point_back_to_original_coordinates(center_x, front_y, polygon.centroid, angle)
    center_center = rotate_point_back_to_original_coordinates(center_x, center_y, polygon.centroid, angle)

    return buffer_width, center_front, center_center


def rotate_point_back_to_original_coordinates(point_x, point_y, rotation_center, angle):
    """
    Rotate point back to original coordinates, by doing a reverse rotation.
    :param point_x: x coordinate of the point that needs to be rotated back
    :param point_y: y coordinate of the point that needs to be rotated back
    :param rotation_center: shapely Point, center of the rotation
    :param angle: angle in radians for reverse rotation
    :return: shapely Point, rotated point back to original coordinates
    """
    # Rotate the center point back to original coordinates
    rotated_center_x = point_x - rotation_center.x
    rotated_center_y = point_y - rotation_center.y

    # Apply inverse rotation (backwards rotation)
    cos_angle = math.cos(-angle)
    sin_angle = math.sin(-angle)
    origin_x = cos_angle * rotated_center_x - sin_angle * rotated_center_y + rotation_center.x
    origin_y = sin_angle * rotated_center_x + cos_angle * rotated_center_y + rotation_center.y

    return shapely.Point(origin_x, origin_y)