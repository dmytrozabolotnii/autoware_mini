#!/usr/bin/env python3

import rospy
import json
import numpy as np

from autoware_msgs.msg import DetectedObjectArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped

from shapely.geometry import shape
from shapely.affinity import translate
from shapely.ops import unary_union
from shapely import prepare, dwithin, box, Polygon, Point as ShapelyPoint
from localization.WGS84ToUTMTransformer import WGS84ToUTMTransformer

class RoadAreaFilter:
    def __init__(self):

        # get parameters
        self.road_area_file = rospy.get_param("~road_area_file")
        self.filtering_method = rospy.get_param("~filtering_method")
        self.within_filtering_extent = rospy.get_param("~within_filtering_extent")
        self.coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        self.utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        self.utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")

        assert self.filtering_method in ["centroid", "intersects", "within"], "filtering_method must be one of 'centroid', 'intersects', 'within'"

        self.current_location = None

        # initialize coordinate_transformer
        if self.coordinate_transformer == "utm":
            self.transformer = WGS84ToUTMTransformer(False, self.utm_origin_lat, self.utm_origin_lon)
        easting, northing = self.transformer.transform_lat_lon(self.utm_origin_lat, self.utm_origin_lon, 0)

        rospy.loginfo("%s - loading road area from file %s", rospy.get_name(), self.road_area_file)

        # Read the GeoJSON file
        with open(self.road_area_file, 'r') as f:
            geojson_data = json.load(f)

        road_area = []
        for feature in geojson_data['features']:
            geometry = shape(feature['geometry'])
            geometry = translate(geometry, xoff=-easting, yoff=-northing)
            prepare(geometry)
            road_area.append(geometry)
        self.road_area = np.array(road_area)

        # detected objects publisher
        self.objects_pub = rospy.Publisher('detected_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)

        self.road_area_pub = rospy.Publisher('road_area', MarkerArray, queue_size=1, tcp_nodelay=True, latch=True)
        self.road_area_pub.publish(self.get_road_area_markers())

        # Subscribers
        rospy.Subscriber('detected_objects_unfiltered', DetectedObjectArray, self.detected_objects_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)

        rospy.loginfo("%s - initialized", rospy.get_name())

    def get_road_area_markers(self):
        boundary = unary_union(self.road_area)
        geometry = []
        for geom in boundary.geoms:
            geometry.append(geom.exterior.coords)
            for interior in geom.interiors:
                geometry.append(interior.coords)

        road_area_markers = MarkerArray()
        for i, polygon in enumerate(geometry):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "Road area"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 0.9
            marker.color.g = 0.1
            marker.color.b = 0.1
    
            for x, y, z in polygon:
                p = Point(x, y, z)
                marker.points.append(p)

            road_area_markers.markers.append(marker)

        return road_area_markers

    def current_pose_callback(self, msg):
        self.current_location = ShapelyPoint(msg.pose.position.x, msg.pose.position.y)

    def detected_objects_callback(self, msg):
        current_location = self.current_location

        if current_location is None:
            return

        # get nearby road area blocks from within 200m
        nearby_road_areas = self.road_area[dwithin(self.road_area, current_location, 200)]
        if self.filtering_method == "within":
            extent = box(current_location.x - self.within_filtering_extent, current_location.y - self.within_filtering_extent, current_location.x + self.within_filtering_extent, current_location.y + self.within_filtering_extent)
            # create nearby area that is not road area, so we could use intersect (instead of within that is a lot slower)
            nearby_not_road_area = extent.difference(unary_union(nearby_road_areas))
            prepare(nearby_not_road_area)

        # Create array objects
        detected_objects = DetectedObjectArray()
        detected_objects.header = msg.header

        for obj in msg.objects:
            if self.filtering_method == "centroid":
                obj_geom = ShapelyPoint(obj.pose.position.x, obj.pose.position.y)
            else:
                obj_geom = Polygon([(p.x, p.y) for p in obj.convex_hull.polygon.points])
            prepare(obj_geom)

            if self.filtering_method == "centroid" or self.filtering_method == "intersects":
                if obj_geom.intersects(nearby_road_areas).any():
                    detected_objects.objects.append(obj)
            else:  # filtering_method == "within" / use intersects, but with area that is not road area
                if not obj_geom.intersects(nearby_not_road_area):
                    detected_objects.objects.append(obj)

        self.objects_pub.publish(detected_objects)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('road_area_filter', log_level=rospy.INFO)
    node = RoadAreaFilter()
    node.run()
