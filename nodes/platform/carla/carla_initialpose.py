#!/usr/bin/env python3
#
# Copyright (c) 2023 Autonomous Driving Lab (ADL), University of Tartu.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Convert initial pose to simulation pose.
"""

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose
from localization.UTMToSimulationTransformer import UTMToSimulationTransformer


class CarlaInitialPose:

    def __init__(self):

        # parameters
        self.use_transformer = rospy.get_param("/carla_localization/use_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")

        self.dropping_height = rospy.get_param("~dropping_height")

        self.initialpose_sim_pub = rospy.Publisher('/carla/ego_vehicle/control/set_transform', Pose, queue_size=1)

        self.initialpose_sub = rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.initialpose_callback, queue_size=1)

        self.utm2sim_transformer = UTMToSimulationTransformer(use_custom_origin=use_custom_origin,
                                                              origin_lat=utm_origin_lat,
                                                              origin_lon=utm_origin_lon)


    def initialpose_callback(self, msg):

        if self.use_transformer:
            # Add dropping height to the z coordinate
            pose_sim = self.utm2sim_transformer.transform_pose(msg.pose.pose)
            pose_sim.position.z += self.dropping_height
        else:
            pose_sim = msg.pose.pose
            pose_sim.position.z += 2.0

        self.initialpose_sim_pub.publish(pose_sim)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    # log_level set to errors only
    rospy.init_node('carla_initialpose', log_level=rospy.INFO)
    node = CarlaInitialPose()
    node.run()