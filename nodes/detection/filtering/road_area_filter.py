#!/usr/bin/env python3

import rospy
import json
import shapely
from autoware_mini.msg import DetectedObjectArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from localization.WGS84ToUTMTransformer import WGS84ToUTMTransformer
from helpers.geometry import get_distance_between_two_points_2d

class RoadAreaFilter:
    def __init__(self):

        # get parameters
        self.road_area_file = rospy.get_param("~road_area_file")
        self.filtering_method = rospy.get_param("~filtering_method")
        self.filtering_extent = rospy.get_param("~filtering_extent")
        self.map_extraction_distance = rospy.get_param("~map_extraction_distance")
        self.local_path_length = rospy.get_param("/planning/local_path_length")
        self.coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        self.utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        self.utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")

        if self.filtering_method not in ["centroid", "intersects", "within"]:
            raise ValueError(f"{rospy.get_name()} - 'filtering_method' must be one of 'centroid', 'intersects' or 'within', not '{self.filtering_method}'")

        self.current_location = None
        self.map_extraction_location = None

        # initialize coordinate_transformer
        if self.coordinate_transformer == "utm":
            self.transformer = WGS84ToUTMTransformer(False, self.utm_origin_lat, self.utm_origin_lon)
        self.easting, self.northing = self.transformer.transform_lat_lon(self.utm_origin_lat, self.utm_origin_lon, 0)

        rospy.loginfo("%s - loading road area from file %s", rospy.get_name(), self.road_area_file)

        # Read the GeoJSON file
        with open(self.road_area_file, 'r') as f:
            self.geojson_data = json.load(f)

        # detected objects publisher
        self.objects_pub = rospy.Publisher('detected_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)

        self.road_area_pub = rospy.Publisher('road_area', MarkerArray, queue_size=1, tcp_nodelay=True, latch=True)

        # Subscribers
        rospy.Subscriber('detected_objects_unfiltered', DetectedObjectArray, self.detected_objects_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)

        rospy.loginfo("%s - initialized", rospy.get_name())

    def get_road_area_markers(self, road_area):

        road_area_markers = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "Road area"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 0.9
        marker.color.g = 0.1
        marker.color.b = 0.1

        for geom in road_area.geoms:
            marker.points.extend(create_coords_for_line_list(geom.exterior.coords))
            for interior in geom.interiors:
                marker.points.extend(create_coords_for_line_list(interior.coords))

        road_area_markers.markers.append(marker)
        return road_area_markers

    def current_pose_callback(self, msg):

        point_utm_local = shapely.geometry.Point(msg.pose.position.x, msg.pose.position.y)
        point_utm = shapely.affinity.translate(point_utm_local, xoff=self.easting, yoff=self.northing)

        if self.map_extraction_location is None or get_distance_between_two_points_2d(self.map_extraction_location, msg.pose.position) >= (self.map_extraction_distance - self.local_path_length):
            self.map_extraction_location = msg.pose.position
            road_area = []
            for feature in self.geojson_data['features']:
                geometry = shapely.geometry.shape(feature['geometry'])
                if geometry.dwithin(point_utm, self.map_extraction_distance):
                    geometry = shapely.affinity.translate(geometry, xoff=-self.easting, yoff=-self.northing)
                    road_area.append(geometry)
            road_area = shapely.unary_union(road_area)
            shapely.prepare(road_area)

            # create inverted road area
            if self.filtering_method == "within":
                xmin, ymin, xmax, ymax = shapely.total_bounds(road_area)
                full_extent = shapely.box(xmin, ymin, xmax, ymax)
                not_road_area = full_extent.difference(road_area)
                shapely.prepare(not_road_area)
                self.not_road_area = not_road_area

            self.road_area = road_area
            self.road_area_pub.publish(self.get_road_area_markers(road_area))

        self.current_location = msg.pose.position

    def detected_objects_callback(self, msg):

        current_location = self.current_location
        if current_location is None:
            return
        local_extent = shapely.box(current_location.x - self.filtering_extent, current_location.y - self.filtering_extent, current_location.x + self.filtering_extent, current_location.y + self.filtering_extent)

        if self.filtering_method == "centroid" or self.filtering_method == "intersects":
            extracted_area = local_extent.intersection(self.road_area)
        elif self.filtering_method == "within":
            extracted_area = local_extent.intersection(self.not_road_area)
        shapely.prepare(extracted_area)

        # Create detected objects array
        detected_objects = DetectedObjectArray()
        detected_objects.header = msg.header

        for obj in msg.objects:
            if self.filtering_method == "centroid":
                obj_geom = shapely.Point(obj.pose.position.x, obj.pose.position.y)
            else:
                obj_geom = shapely.Polygon([(p.x, p.y) for p in obj.convex_hull.points])
            shapely.prepare(obj_geom)

            if self.filtering_method == "centroid" or self.filtering_method == "intersects":
                if obj_geom.intersects(extracted_area):
                    detected_objects.objects.append(obj)
            else:  # filtering_method == "within" / use intersects, but with area that is not road area
                if not obj_geom.intersects(extracted_area) and obj_geom.intersects(local_extent):
                    detected_objects.objects.append(obj)

        self.objects_pub.publish(detected_objects)

    def run(self):
        rospy.spin()

def create_coords_for_line_list(coords):
    line_list = []
    for i in range(len(coords)-1):
        line_list.append(Point(x=coords[i][0], y=coords[i][1], z=coords[i][2]))
        line_list.append(Point(x=coords[i+1][0], y=coords[i+1][1], z=coords[i+1][2]))
    return line_list

if __name__ == '__main__':
    rospy.init_node('road_area_filter', log_level=rospy.INFO)
    node = RoadAreaFilter()
    node.run()
