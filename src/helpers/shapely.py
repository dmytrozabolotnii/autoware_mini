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
