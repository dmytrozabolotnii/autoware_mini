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

def split_linestring_with_two_lines(main_linestring, cutting_line1, cutting_line2):
    split_goems1 = ops.split(main_linestring, cutting_line1).geoms
    if len(split_goems1) < 2:
        return None
    
    split2 = ops.split(split_goems1[1], cutting_line2).geoms[0]

    return split2
