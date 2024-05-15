from shapely import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
from shapely.affinity import rotate


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
    elif isinstance(geometry, MultiPolygon):
        intersection_points = []
        for polygon in geometry.geoms:
            # skip last points of each polygon because it is the same as the first
            intersection_points.extend([Point(coord) for coord in polygon.exterior.coords[:-1]])
    elif isinstance(geometry, MultiPoint):
        intersection_points = [point for point in geometry.geoms]
    elif isinstance(geometry, MultiLineString):
        intersection_points = []
        for line in geometry.geoms:
            intersection_points.extend([Point(coord) for coord in line.coords])

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