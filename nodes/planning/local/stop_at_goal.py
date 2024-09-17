#!/usr/bin/env python3

import rospy
import math
from autoware_msgs.msg import Lane
from sensor_msgs.msg import PointCloud2
from helpers.collision import CollisionPoints
from helpers.geometry import get_distance_between_two_points_2d

class StopAtGoal:

    def __init__(self):

        # parameters
        self.braking_safety_distance_goal = rospy.get_param("~braking_safety_distance_goal")

        # variables
        self.goal_point = None

        # publishers
        self.goal_point_pub = rospy.Publisher('goal_collision_points', PointCloud2, queue_size=1, latch=True, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('extracted_local_path', Lane, self.local_path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('smoothed_path', Lane, self.global_path_callback, queue_size=1, tcp_nodelay=True)

    def global_path_callback(self, msg):

        if len(msg.waypoints) > 0:
            # lasst point of the global path is goal point
            self.goal_point = msg.waypoints[-1].pose.pose.position
        else:
            self.goal_point = None

    def local_path_callback(self, msg):

        collision_points = CollisionPoints()
        goal_point = self.goal_point

        if goal_point is not None and len(msg.waypoints) > 0:
            # check if goal point is at the end of the local path
            if math.isclose(get_distance_between_two_points_2d(goal_point, msg.waypoints[-1].pose.pose.position), 0.0):
                # add goal point as collision point
                collision_points.add_point(goal_point.x, goal_point.y, goal_point.z, 0.0, 0.0, 0.0, self.braking_safety_distance_goal, CollisionPoints.GOAL_POINT)

        collision_points_msg = collision_points.create_message()
        collision_points_msg.header.frame_id = msg.header.frame_id
        self.goal_point_pub.publish(collision_points_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('stop_at_goal')
    node = StopAtGoal()
    node.run()