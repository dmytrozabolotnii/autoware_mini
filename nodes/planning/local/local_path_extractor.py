#!/usr/bin/env python3

import rospy
import threading
import traceback
import shapely
from autoware_mini.msg import Path
from geometry_msgs.msg import PoseStamped
from helpers.path import PathWrapper

class LocalPathExtractor:

    def __init__(self):

        # parameters
        self.publish_rate = rospy.get_param("~publish_rate")
        self.local_path_length = rospy.get_param("local_path_length")

        # variables
        self.current_pose = None
        self.global_path = None

        # publishers
        self.local_path_pub = rospy.Publisher('extracted_local_path', Path, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('global_path', Path, self.global_path_callback, queue_size=None, tcp_nodelay=True)

    def current_pose_callback(self, msg):
        self.current_pose = msg

    def global_path_callback(self, msg):
        if len(msg.waypoints) == 0:
            self.global_path = None
            rospy.loginfo("%s - Empty global path received", rospy.get_name())
        else:
            self.global_path = PathWrapper(msg.waypoints, distances=True)
            rospy.loginfo("%s - Global path received with %i waypoints", rospy.get_name(), len(self.global_path.waypoints))

    def extract_local_path(self):
        try:
            current_pose = self.current_pose
            global_path = self.global_path

            if current_pose is None:
                return

            local_path = Path()
            local_path.header = current_pose.header

            if global_path is None:
                self.local_path_pub.publish(local_path)
                return

            # TODO avoid jumping from one place to another on path - just finding the closest point is dangerous!
            # Example of global path overlapping with itself.
            current_position = shapely.Point(current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z)
            ego_distance_from_global_path_start = global_path.linestring.project(current_position)

            # extract local path using dstances
            local_path.waypoints = global_path.extract_waypoints(ego_distance_from_global_path_start, ego_distance_from_global_path_start + self.local_path_length)
            self.local_path_pub.publish(local_path)
        except Exception as e:
            rospy.logerr_throttle(10, "%s - Exception in callback: %s", rospy.get_name(), traceback.format_exc())

    def run(self):
        # start separate thread for spinning subcribers
        t = threading.Thread(target=rospy.spin)
        t.daemon = True # make sure Ctrl+C works
        t.start()

        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            self.extract_local_path()
            try:
                rate.sleep()
            except (rospy.ROSTimeMovedBackwardsException, rospy.exceptions.ROSInterruptException):
                pass

if __name__ == '__main__':
    rospy.init_node('local_path_extractor')
    node = LocalPathExtractor()
    node.run()