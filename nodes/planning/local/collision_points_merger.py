#!/usr/bin/env python3

import rospy
import numpy as np
from ros_numpy import msgify, numpify
import message_filters
import traceback
from sensor_msgs.msg import PointCloud2

class CollisionPointsMerger:

    def __init__(self):

        # variables
        self.collision_goal_points = None

        # publishers
        self.collision_points_pub = rospy.Publisher('collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        collision_goal_sub = message_filters.Subscriber('goal_collision_points', PointCloud2, tcp_nodelay=True)
        collision_local_path_sub = message_filters.Subscriber('local_path_collision_points', PointCloud2, tcp_nodelay=True)
        collision_tfl_stopline_sub = message_filters.Subscriber('tfl_stopline_collision_points', PointCloud2, tcp_nodelay=True)

        ts = message_filters.TimeSynchronizer([collision_goal_sub, collision_local_path_sub, collision_tfl_stopline_sub], queue_size=4)
        ts.registerCallback(self.collision_points_callback)

    def collision_points_callback(self, collision_goal_points, collision_local_path, collision_tfl_stopline):
        try:
            collision_local_path_np = numpify(collision_local_path)
            collision_tfl_stopline_np =  numpify(collision_tfl_stopline)
            collision_goal_points_np = numpify(collision_goal_points)

            collision_points = np.concatenate((collision_local_path_np, collision_tfl_stopline_np, collision_goal_points_np))

            # Create a new PointCloud2 message
            merged_points_msg = msgify(PointCloud2, collision_points)
            merged_points_msg.header = collision_local_path.header

            # Publish the merged collision points
            self.collision_points_pub.publish(merged_points_msg)

        except Exception as e:
            rospy.logerr_throttle(10, "%s - Exception in callback: %s", rospy.get_name(), traceback.format_exc())

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('collision_points_merger')
    node = CollisionPointsMerger()
    node.run()