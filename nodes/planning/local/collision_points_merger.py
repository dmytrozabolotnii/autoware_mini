#!/usr/bin/env python3

import rospy
import numpy as np
from ros_numpy import msgify, numpify
import message_filters
from autoware_msgs.msg import Lane
from sensor_msgs.msg import PointCloud2

class CollisionPointsMerger:

    def __init__(self):

        # variables
        self.collision_goal_points = None

        # publishers
        self.collision_points_pub = rospy.Publisher('collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('collision_goal', PointCloud2, self.collision_goal_callback, queue_size=1, tcp_nodelay=True)

        collision_local_path_sub = message_filters.Subscriber('collision_local_path', PointCloud2, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        collision_tfl_stopline_sub = message_filters.Subscriber('collision_tfl_stopline', PointCloud2, queue_size=1, tcp_nodelay=True)

        ts = message_filters.ApproximateTimeSynchronizer([collision_local_path_sub, collision_tfl_stopline_sub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.collision_points_callback)


    def collision_goal_callback(self, msg):
        self.collision_goal_points = msg.data


    def collision_points_callback(self, collision_local_path, collision_tfl_stopline):

        if self.collision_goal_points is None:
            return

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

        # TODO if emty arrays, then from rviz it is not deleted?
        collision_local_path_np = np.frombuffer(collision_local_path.data, dtype=dtype)
        collision_tfl_stopline_np = np.frombuffer(collision_tfl_stopline.data, dtype=dtype)
        collision_goal_points_np = np.frombuffer(self.collision_goal_points, dtype=dtype)

        collision_points = np.concatenate((collision_local_path_np, collision_tfl_stopline_np, collision_goal_points_np))

        # Create a new PointCloud2 message
        merged_points_msg = msgify(PointCloud2, collision_points)
        merged_points_msg.header = collision_local_path.header

        # Publish the merged collision points
        self.collision_points_pub.publish(merged_points_msg)


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('collision_points_merger')
    node = CollisionPointsMerger()
    node.run()