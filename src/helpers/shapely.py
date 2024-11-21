import shapely
from shapely import ops

def get_polygon_width(polygon, heading_angle):
    """
    Get width of the polygon. Measured perpendicular to moving direction.
    :param polygon: shapely Polygon
    :param heading_angle: heading angle in degrees
    :return: width of the polygon
    """

    # rotate polygon to align with y axis, so the width will be in x direction
    angle = 90 - heading_angle
    rotated_polygon = shapely.affinity.rotate(polygon, angle, origin='centroid', use_radians=False)
    minx, miny, maxx, maxy = rotated_polygon.bounds
    width = maxx - minx
    return width

def split_linestring_with_two_points(linestring, point1, point2):
    """
    Spilits linestring with two points and returns the middle segment
    :param linestring: shapely LineString
    :param point1: shapely Point
    :param point2: shapely Point
    :return: middle segment
    """

    split_points = shapely.MultiPoint([point1, point2])
    split_lines = ops.split(linestring, split_points.buffer(0.001))

    # Return the middle section
    for segment in split_lines.geoms:
        if segment.intersects(point1.buffer(0.01)) and segment.intersects(point2.buffer(0.01)):
            return segment
        
    return None
