#!/usr/bin/env python3
#
# Copyright (c) 2023 Autonomous Driving Lab (ADL), University of Tartu.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Convert initial pose to simulation pose.
    initialpose     (geometry_msgs::PoseWithCovarianceStamped)
    initialpose_sim (geometry_msgs::PoseWithCovarianceStamped)
"""

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from localization.UTMToSimulationTransformer import UTMToSimulationTransformer


class CarlaInitialPose:

    def __init__(self):

        # parameters
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")

        self.dropping_height = rospy.get_param("~dropping_height")

        self.initialpose_sim_pub = rospy.Publisher('/initialpose_sim', PoseWithCovarianceStamped, queue_size=1)

        self.initialpose_sub = rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.initialpose_callback, queue_size=1)

        self.utm2sim_transformer = UTMToSimulationTransformer(use_custom_origin=use_custom_origin,
                                                              origin_lat=utm_origin_lat,
                                                              origin_lon=utm_origin_lon)


    def initialpose_callback(self, msg):

        # Add dropping height to the z coordinate
        msg.pose.pose.position.z += self.dropping_height

        pose_sim = PoseWithCovarianceStamped()
        pose_sim.header = msg.header
        pose_sim.pose.pose = self.utm2sim_transformer.transform_pose(msg.pose.pose)

        self.initialpose_sim_pub.publish(pose_sim)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    # log_level set to errors only
    rospy.init_node('carla_initialpose', log_level=rospy.INFO)
    node = CarlaInitialPose()
    node.run()