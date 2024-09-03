#!/usr/bin/env python3

import rospy
import math
import threading
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
        self.lock = threading.Lock()
        self.current_position = None
        self.global_path = None
        self.output_frame = None

        # publishers
        self.local_path_pub = rospy.Publisher('extracted_local_path', Lane, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('smoothed_path', Lane, self.path_callback, queue_size=None, tcp_nodelay=True)


    def current_pose_callback(self, msg):
        self.current_position = ShapelyPoint(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

    def path_callback(self, msg):
        
        if len(msg.waypoints) == 0:
            with self.lock:
                self.global_path = None
                self.output_frame = None
            rospy.loginfo("%s - Empty global path received", rospy.get_name())
        else:
            with self.lock:
                self.global_path = Path(msg.waypoints)
                self.output_frame = msg.header.frame_id
            rospy.loginfo("%s - Global path received with %i waypoints", rospy.get_name(), len(self.global_path.waypoints))


    def extract_local_path(self):
        with self.lock:
            current_position = self.current_position
            global_path = self.global_path
            output_frame = self.output_frame

        if current_position is None or global_path is None:
            self.publish_local_path_wp([], output_frame)
            return

        # # TODO how to avoid jumping from one place to another on path - just finding the closest point is dangerous!
        # # Example of global path overlapping with itself or ego doing the 90deg turn and cutting the corner!

        ego_distance_from_global_path_start = global_path.linestring.project(current_position)

        ### TODO not sure it actually helps
        # if current position is projected at the end of the global path - goal reached
        if math.isclose(ego_distance_from_global_path_start, global_path.linestring.length):
            self.publish_local_path_wp([], output_frame)
            return

        # find the index for the start and end of the local path
        local_path = Path(global_path.extract_waypoints(ego_distance_from_global_path_start, ego_distance_from_global_path_start + self.local_path_length, copy=True))
        self.publish_local_path_wp(local_path.waypoints, output_frame)

    def publish_local_path_wp(self, local_path_waypoints, output_frame):
        # create lane message
        lane = Lane()
        lane.header.frame_id = output_frame
        lane.header.stamp = rospy.Time.now()
        lane.waypoints = local_path_waypoints
        self.local_path_pub.publish(lane)

    def run(self):
        # start separate thread for spinning subcribers
        t = threading.Thread(target=rospy.spin)
        t.daemon = True # make sure Ctrl+C works
        t.start()

        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            self.extract_local_path()
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('local_path_extractor')
    node = LocalPathExtractor()
    node.run()