#!/usr/bin/env python3

import rospy
import threading
import traceback
from shapely.geometry import Point as ShapelyPoint
from autoware_msgs.msg import Lane
from geometry_msgs.msg import PoseStamped
from helpers.path import Path

class LocalPathExtractor:

    def __init__(self):

        # parameters
        self.publish_rate = rospy.get_param("~publish_rate")
        self.local_path_length = rospy.get_param("~local_path_length")

        # variables
        self.current_position = None
        self.global_path = None
        self.output_frame = None

        # publishers
        self.local_path_pub = rospy.Publisher('extracted_local_path', Lane, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('global_path', Lane, self.path_callback, queue_size=None, tcp_nodelay=True)

    def current_pose_callback(self, msg):
        self.current_position = ShapelyPoint(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

    def path_callback(self, msg):
        output_frame = msg.header.frame_id

        if len(msg.waypoints) == 0:
            global_path = None
            rospy.loginfo("%s - Empty global path received", rospy.get_name())
        else:
            global_path = Path(msg.waypoints)
            rospy.loginfo("%s - Global path received with %i waypoints", rospy.get_name(), len(global_path.waypoints))

        self.output_frame = output_frame
        self.global_path = global_path

    def extract_local_path(self):
        try:
            current_position = self.current_position
            global_path = self.global_path
            output_frame = self.output_frame

            local_path = Lane()
            local_path.header.frame_id = output_frame
            local_path.header.stamp = rospy.Time.now()

            if current_position is None or global_path is None:
                self.local_path_pub.publish(local_path)
                return

            # TODO avoid jumping from one place to another on path - just finding the closest point is dangerous!
            # Example of global path overlapping with itself.
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
            except rospy.ROSTimeMovedBackwardsException:
                pass

if __name__ == '__main__':
    rospy.init_node('local_path_extractor')
    node = LocalPathExtractor()
    node.run()