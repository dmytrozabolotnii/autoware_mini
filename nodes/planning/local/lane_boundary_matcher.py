#!/usr/bin/env python3

import rospy
import numpy as np
import shapely
import threading
from lanelet2.geometry import approximatedLength2d
from autoware_msgs.msg import Lane
from std_msgs.msg import Float32MultiArray, UInt32MultiArray, ColorRGBA
from geometry_msgs.msg import PoseStamped, Point
from helpers.path import Path

from visualization_msgs.msg import MarkerArray, Marker

from helpers.lanelet2 import load_lanelet2_map

class LaneBoundaryMatcher:

    def __init__(self):

        # parameters
        lanelet2_map_name = rospy.get_param("~lanelet2_map_name")

        self.lanelet2_map = load_lanelet2_map(lanelet2_map_name)

        # variables
        self.supercombo_lane_lines = None
        self.global_path_lanelet_ids = None
        self.lanelet_polygons = None
        self.lanelet_centerline_waypoints = None
        self.current_lanelet_idx = 0
        self.current_position = None
        self.new_global_path = None
        self.approximated_lanelet_lengths = None

        self.lock = threading.Lock()

        # publishers
        self.correction_pub = rospy.Publisher('lateral_position_correction', Lane, queue_size=1, tcp_nodelay=True)
        self.lanelet_bounds_pub = rospy.Publisher('lanelet_bounds_match', MarkerArray, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('/openpilot/lane_lines', Float32MultiArray, self.lane_line_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('global_path_lenelet_ids', UInt32MultiArray, self.global_path_callback, queue_size=None, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)

    def current_pose_callback(self, msg):
        self.current_position = shapely.Point(msg.pose.position.x, msg.pose.position.y)

    def lane_line_callback(self, msg):
        self.supercombo_lane_lines = np.array(msg.data).reshape(msg.layout.dim[0].size, msg.layout.dim[1].size, msg.layout.dim[2].size)

        if self.lanelet_polygons is None or self.current_position is None or self.new_global_path is None or self.approximated_lanelet_lengths is None:
            return
        
        with self.lock:
            current_position = self.current_position
            lanelet_polygons = self.lanelet_polygons
            global_path_lanelet_ids = self.global_path_lanelet_ids
            current_lanelet_idx = self.current_lanelet_idx
            approximated_lanelet_lengths = self.approximated_lanelet_lengths
            self.new_global_path = False
        
        # find the current lanelet
        while not lanelet_polygons[current_lanelet_idx].contains(current_position):
            if len(lanelet_polygons) <= current_lanelet_idx + 1:
                return
            
            current_lanelet_idx += 1

        right_lane_points = []
        left_lane_points = []
        idx = current_lanelet_idx
        total_length = 0

        # take left and right boundaries from lanelets that are 50 m or closer on the global path
        while total_length < 50:
            lanelet_id = global_path_lanelet_ids[idx]
            for point in self.lanelet2_map.laneletLayer.get(lanelet_id).rightBound:
                right_lane_points.append((point.x, point.y, point.z))

            for point in self.lanelet2_map.laneletLayer.get(lanelet_id).leftBound:
                left_lane_points.append((point.x, point.y, point.z))

            total_length += approximated_lanelet_lengths[lanelet_id]
            idx += 1

        self.publish_lanelet_bounds(right_lane_points, left_lane_points)


        if not self.new_global_path:
            self.current_lanelet_idx = current_lanelet_idx


    def global_path_callback(self, msg):
        self.new_global_path = True
        if len(msg.data) == 0:
            self.lanelet_polygons = None
            self.current_lanelet_idx = 0
            return
        
        self.global_path_lanelet_ids = msg.data

        lanelet_polys = []
        centerline_waypoints = []
        approximated_lanelet_lengths = {}

        # get all lanelets on the global path
        for lanelet_id in msg.data:
            lanelet = self.lanelet2_map.laneletLayer.get(lanelet_id)

            for point in lanelet.centerline:
                centerline_waypoints.append(shapely.Point(point.x, point.y))

            polygon = shapely.Polygon([(pt.x, pt.y) for pt in lanelet.polygon2d()])
            lanelet_polys.append(polygon)
            approximated_lanelet_lengths[lanelet_id] = approximatedLength2d(lanelet)

        self.lanelet_polygons = lanelet_polys
        self.current_lanelet_idx = 0
        self.approximated_lanelet_lengths = approximated_lanelet_lengths


    def publish_lanelet_bounds(self, right_lane_points, left_lane_points):
        # For debugging
        lanes_marker_array = MarkerArray()

        points = []
        for x, y, z in right_lane_points:
            point = Point(x=x,y=y, z=z)
            points.append(point)

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "right bound"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.color = ColorRGBA(0.6, 0.3, 0.0, 1.0)
        marker.points = points
        lanes_marker_array.markers.append(marker)

        points = []
        for x, y, z in left_lane_points:
            point = Point(x=x,y=y, z=z)
            points.append(point)

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "left bound"
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.color = ColorRGBA(0.6, 0.3, 0.0, 1.0)
        marker.points = points
        lanes_marker_array.markers.append(marker)
        
        self.lanelet_bounds_pub.publish(lanes_marker_array)


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('lane_boundary_matcher')
    node = LaneBoundaryMatcher()
    node.run()