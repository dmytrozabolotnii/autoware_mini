#!/usr/bin/env python3

import rospy
from autoware_msgs.msg import Lane
from sensor_msgs.msg import PointCloud2

class SpeedPlanner:

    def __init__(self):

        # parameters

        # variables

        # publishers
        self.local_path_pub = rospy.Publisher('local_path', Lane, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('collision_points', PointCloud2, self.collision_points_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('extracted_local_path', Lane, self.path_callback, queue_size=1, tcp_nodelay=True)

    def collision_points_callback(self, msg):
        # TODO
        pass

    def path_callback(self, msg):

        # publish local path
        self.local_path_pub.publish(msg)



    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('speed_planner')
    node = SpeedPlanner()
    node.run()