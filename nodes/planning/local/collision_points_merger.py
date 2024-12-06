#!/usr/bin/env python3

import rospy
import numpy as np
from ros_numpy import msgify, numpify
import message_filters
import traceback
from sensor_msgs.msg import PointCloud2

class CollisionPointsMerger:

    def __init__(self):

        # parameters
        synchronization_method = rospy.get_param("~synchronization_method")
        synchronization_queue_size = rospy.get_param("~synchronization_queue_size")
        synchronization_slop = rospy.get_param("~synchronization_slop")

        # publishers
        self.collision_points_pub = rospy.Publisher('collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        collision_goal_sub = message_filters.Subscriber('goal_collision_points', PointCloud2, tcp_nodelay=True)
        collision_object_sub = message_filters.Subscriber('object_collision_points', PointCloud2, tcp_nodelay=True)
        collision_tfl_stopline_sub = message_filters.Subscriber('tfl_stopline_collision_points', PointCloud2, tcp_nodelay=True)
        collision_crosswalk_sub = message_filters.Subscriber('crosswalk_collision_points', PointCloud2, tcp_nodelay=True)
        collision_trajectory_sub = message_filters.Subscriber('trajectory_collision_points', PointCloud2, tcp_nodelay=True)
        collision_stop_line_sub = message_filters.Subscriber('stop_line_collision_points', PointCloud2, tcp_nodelay=True)

        if synchronization_method == "approximate":
            ts = message_filters.ApproximateTimeSynchronizer([collision_goal_sub, collision_object_sub, collision_tfl_stopline_sub, 
                                                              collision_crosswalk_sub, collision_stop_line_sub, collision_trajectory_sub], 
                                                              queue_size=synchronization_queue_size, slop=synchronization_slop)
        elif synchronization_method == "exact":
            ts = message_filters.TimeSynchronizer([collision_goal_sub, collision_object_sub, collision_tfl_stopline_sub, 
                                                   collision_crosswalk_sub, collision_stop_line_sub, collision_trajectory_sub], queue_size=2)
        else:
            raise ValueError(f"'{synchronization_method}' is not a known synchronization method")

        ts.registerCallback(self.collision_points_callback)

    def collision_points_callback(self, collision_goal_points_msg, collision_object_msg, collision_tfl_stopline_msg, collision_crosswalk_msg, collision_stop_line_msg, trajectory_collision_msg):
        try:
            # Convert the messages to numpy arrays
            collision_object_np = numpify(collision_object_msg)
            collision_tfl_stopline_np =  numpify(collision_tfl_stopline_msg)
            collision_goal_points_np = numpify(collision_goal_points_msg)
            collision_crosswalk_np = numpify(collision_crosswalk_msg)
            collision_stop_line_np = numpify(collision_stop_line_msg)
            collision_trajectory_np = numpify(trajectory_collision_msg)

            # Concatenate all the collision points
            collision_points_np = np.concatenate((collision_object_np, collision_tfl_stopline_np, collision_goal_points_np, collision_crosswalk_np, collision_stop_line_np, collision_trajectory_np))

            # Create a new PointCloud2 message
            collision_points_msg = msgify(PointCloud2, collision_points_np)
            collision_points_msg.header = collision_object_msg.header

            # Publish the merged collision points
            self.collision_points_pub.publish(collision_points_msg)

        except Exception as e:
            rospy.logerr_throttle(10, "%s - Exception in callback: %s", rospy.get_name(), traceback.format_exc())

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('collision_points_merger')
    node = CollisionPointsMerger()
    node.run()