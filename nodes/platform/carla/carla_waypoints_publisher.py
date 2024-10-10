#!/usr/bin/env python3
#
# Copyright (c) 2023 Autonomous Driving Lab (ADL), University of Tartu.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
receive a path from carla_ros_waypoint_publisher and convert it to autoware
"""
import rospy
from autoware_msgs.msg import Lane
from autoware_msgs.msg import Waypoint
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from localization.SimulationToUTMTransformer import SimulationToUTMTransformer
from localization.UTMToSimulationTransformer import UTMToSimulationTransformer


class CarlaWaypointsPublisher():

    def __init__(self):

        # Node parameters
        self.speed_limit = rospy.get_param("speed_limit")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")

        # Internal parameters
        self.sim2utm_transformer = SimulationToUTMTransformer(use_custom_origin=use_custom_origin,
                                                              origin_lat=utm_origin_lat,
                                                              origin_lon=utm_origin_lon)
        self.utm2sim_transformer = UTMToSimulationTransformer(use_custom_origin=use_custom_origin,
                                                              origin_lat=utm_origin_lat,
                                                              origin_lon=utm_origin_lon)
        
        # Publishers
        self.waypoints_pub = rospy.Publisher('lane_change_global_path', Lane, queue_size=10, latch=True, tcp_nodelay=True)
        self.goal_publisher = rospy.Publisher('/carla/ego_vehicle/goal', PoseStamped, queue_size=10, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('/carla/ego_vehicle/waypoints', Path, self.path_callback, queue_size=None, tcp_nodelay=True)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=None, tcp_nodelay=True)

    def path_callback(self, data):
        """
        Callback for path. Convert it to Autoware LaneArray and publish it
        """
        msg = Lane()
        msg.header = data.header

        waypoints = []
        for pose in data.poses:
            pose.pose = self.sim2utm_transformer.transform_pose(pose.pose)
            waypoint = Waypoint(pose=pose)
            waypoint.twist.twist.linear.x = self.speed_limit / 3.6
            waypoints.append(waypoint)

        msg.waypoints = waypoints

        self.waypoints_pub.publish(msg)

    def goal_callback(self, msg):
        """
        Converts goal point simulation coordinates to UTM coordinates
        """
        goal_msg = PoseStamped()
        goal_msg.header = msg.header
        goal_msg.pose = self.utm2sim_transformer.transform_pose(msg.pose)

        self.goal_publisher.publish(goal_msg)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('carla_waypoints_publisher', log_level=rospy.INFO)
    node = CarlaWaypointsPublisher()
    node.run()
