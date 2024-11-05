#!/usr/bin/env python3

import rospy
import numpy as np
import shapely
from autoware_msgs.msg import Lane
from std_msgs.msg import Float32MultiArray, UInt32MultiArray
from geometry_msgs.msg import PoseStamped
from helpers.path import Path

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

        # publishers
        self.correction_pub = rospy.Publisher('lateral_position_correction', Lane, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('/openpilot/lane_lines', Float32MultiArray, self.lane_line_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('global_path_lenelet_ids', UInt32MultiArray, self.global_path_callback, queue_size=None, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)

    def lane_line_callback(self, msg):
        self.supercombo_lane_lines = np.array(msg.data).reshape(msg.layout.dim[0].size, msg.layout.dim[1].size, msg.layout.dim[2].size)


    def global_path_callback(self, msg):
        if len(msg.data) == 0:
            self.lanelet_polygons = None
            self.current_lanelet_idx = 0
            return
        
        self.global_path_lanelet_ids = msg.data

        lanelet_polys = []
        centerline_waypoints = []
        for lanelet_id in msg.data:
            lanelet = self.lanelet2_map.laneletLayer.get(lanelet_id)

            for point in lanelet.centerline:
                centerline_waypoints.append(shapely.Point(point.x, point.y))

            polygon = shapely.Polygon([(pt.x, pt.y) for pt in lanelet.polygon2d()])
            lanelet_polys.append(polygon)

        self.lanelet_polygons = lanelet_polys


    def current_pose_callback(self, msg):
        if self.lanelet_polygons is None:
            return
        
        current_location = shapely.Point(msg.pose.position.x, msg.pose.position.y)
        while not self.lanelet_polygons[self.current_lanelet_idx].contains(current_location):
            if len(self.lanelet_polygons) >= self.current_lanelet_idx + 1:
                return
            self.current_lanelet_idx += 1

        self.global_path_lanelet_ids[self.current_lanelet_idx]
        


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('lane_boundary_matcher')
    node = LaneBoundaryMatcher()
    node.run()