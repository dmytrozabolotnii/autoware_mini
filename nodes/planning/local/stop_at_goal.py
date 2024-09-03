#!/usr/bin/env python3

import rospy
import numpy as np
from ros_numpy import msgify
from autoware_msgs.msg import Lane
from sensor_msgs.msg import PointCloud2

class StopAtGoal:

    def __init__(self):

        # parameters
        self.braking_safety_distance_goal = rospy.get_param("~braking_safety_distance_goal")

        # publishers
        self.goal_point_pub = rospy.Publisher('collision_goal', PointCloud2, queue_size=1, latch=True, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('smoothed_path', Lane, self.path_callback, queue_size=None, tcp_nodelay=True)

    def path_callback(self, msg):

        # Create empty array for collision points
        dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('vx', np.float32),
            ('vy', np.float32),
            ('vz', np.float32),
            ('distance_to_stop', np.float32),
            ('category', np.int32)
        ])
        goal_point = np.array([], dtype=dtype)

        if len(msg.waypoints) == 0:
            collision_points = msgify(PointCloud2, goal_point)
        else:
            # Extract last point from the path
            last_point = msg.waypoints[-1]

            # Extract x, y, z coordinates
            x = last_point.pose.pose.position.x
            y = last_point.pose.pose.position.y
            z = last_point.pose.pose.position.z

            # Create goal point
            # TODO can't have category as string label - need to agree on the label coding?!?
            goal_point = np.append(goal_point, np.array([(x, y, z, 0.0, 0.0, 0.0, self.braking_safety_distance_goal, 0)], dtype=dtype))

        # publish clustered points message
        collision_points = msgify(PointCloud2, goal_point)
        collision_points.header.frame_id = msg.header.frame_id
        self.goal_point_pub.publish(collision_points)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('stop_at_goal')
    node = StopAtGoal()
    node.run()