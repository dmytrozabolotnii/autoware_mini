#!/usr/bin/env python3

import rospy
import numpy as np
from ros_numpy import msgify
from autoware_msgs.msg import Lane, DetectedObjectArray
from sensor_msgs.msg import PointCloud2

class LocalPathCollision:

    def __init__(self):


        # publishers
        self.local_path_collision_pub = rospy.Publisher('collision_local_path', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('extracted_local_path', Lane, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)

    def detected_objects_callback(self, msg):
        # TODO
        pass

    def path_callback(self, msg):

        # create empty numpy array for the stopline points
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
        loca_path_collision_points = np.array([], dtype=dtype)

        if len(msg.waypoints) == 0:
            collision_points = msgify(PointCloud2, loca_path_collision_points)
        else:
            # TODO - implement collision detection with final_objects
            collision_points = msgify(PointCloud2, loca_path_collision_points)

        collision_points.header.frame_id = msg.header.frame_id
        collision_points.header.stamp = msg.header.stamp
        self.local_path_collision_pub.publish(collision_points)


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('local_path_collision')
    node = LocalPathCollision()
    node.run()