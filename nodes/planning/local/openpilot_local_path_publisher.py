#!/usr/bin/env python3

import rospy
import numpy as np
import tf2_ros
import threading
import shapely
from ros_numpy import numpify
from autoware_mini.msg import Path, Waypoint
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from helpers.path import PathWrapper
from helpers.geometry import get_heading_between_two_points

class OpenpilotLocalPathPublisher:

    def __init__(self):

        # parameters
        self.transform_timeout = rospy.get_param('~transform_timeout')

        # variables
        self.current_position = None
        self.global_path = None
        self.output_frame = None
        self.current_timestamp = None

        self.current_pose_lock = threading.Lock()
        self.global_path_lock = threading.Lock()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # publishers
        self.openpilt_local_path_pub = rospy.Publisher('openpilot_local_path', Path, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('global_path', Path, self.global_path_callback, queue_size=None, tcp_nodelay=True)
        rospy.Subscriber('/openpilot/position', Float32MultiArray, self.openpilot_position_callback, queue_size=1, tcp_nodelay=True)

    def current_pose_callback(self, msg):
        with self.current_pose_lock:
            self.current_position = shapely.Point(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
            self.current_timestamp = msg.header.stamp

    def global_path_callback(self, msg):
        output_frame = msg.header.frame_id

        if len(msg.waypoints) == 0:
            global_path = None
            rospy.loginfo("%s - Empty global path received", rospy.get_name())
        else:
            global_path = PathWrapper(msg.waypoints)
            rospy.loginfo("%s - Global path received with %i waypoints", rospy.get_name(), len(global_path.waypoints))

        with self.global_path_lock:
            self.output_frame = output_frame
            self.global_path = global_path

    def openpilot_position_callback(self, msg):
        openpilot_plan = float32_multiarray_to_numpy(msg).T

        with self.current_pose_lock:
            current_position = self.current_position
            current_timestamp = self.current_timestamp

        with self.global_path_lock:
            global_path = self.global_path
            output_frame = self.output_frame

        openpilot_local_path = Path()
        openpilot_local_path.header.frame_id = output_frame
        openpilot_local_path.header.stamp = rospy.Time.now()

        if current_position is None or global_path is None:
            self.openpilt_local_path_pub.publish(openpilot_local_path)
            return

        # fetch the transform from 'openpilot' frame to ouput frame
        try:
            transform = self.tf_buffer.lookup_transform(output_frame, "openpilot", current_timestamp, rospy.Duration(self.transform_timeout))
            tf_matrix = numpify(transform.transform)
        except (tf2_ros.TransformException, rospy.ROSTimeMovedBackwardsException) as e:
            rospy.logwarn("%s - %s", rospy.get_name(), e)
            return
        
        # transform openpilot plan to the given output frame
        openpilot_plan[:, 3] = 1 # replece the time dimension with ones to get homogeneous points
        openpilot_plan_homogeneous = openpilot_plan @ tf_matrix.T
        openpilot_plan = openpilot_plan_homogeneous[:, :3]

        waypoints = []
        for i in range(len(openpilot_plan)):
            if i == len(openpilot_plan) - 1:
                # use heading of previous point - last point of last lanelet has no following point 
                x, y, z = openpilot_plan[i]
                x_prev, y_prev, z_prev = openpilot_plan[i-1]
                waypoint = self.generate_waypoint(shapely.Point(x, y, z), previous_point=shapely.Point(x_prev, y_prev, z_prev))
                waypoints.append(waypoint)
            else:
                x, y, z = openpilot_plan[i]
                x_next, y_next, z_next = openpilot_plan[i+1]
                waypoint = self.generate_waypoint(shapely.Point(x, y, z), next_point=shapely.Point(x_next, y_next, z_next))
                waypoints.append(waypoint)

        openpilot_local_path.waypoints = waypoints
        self.openpilt_local_path_pub.publish(openpilot_local_path)

    def generate_waypoint(self, current_point, next_point=None, previous_point=None):
        if next_point is not None:
            heading = get_heading_between_two_points(current_point, next_point)
        else:
            heading = get_heading_between_two_points(previous_point, current_point)

        nearest_global_path_waypoint = self.global_path.get_nearest_waypoint(current_point)

        waypoint = Waypoint()
        waypoint.position.x = current_point.x
        waypoint.position.y = current_point.y
        waypoint.position.z = current_point.z
        waypoint.lanechange_state = 0
        waypoint.blinker_state = nearest_global_path_waypoint.blinker_state
        waypoint.heading = heading
        waypoint.speed = nearest_global_path_waypoint.speed
        waypoint.left_width = 1.2
        waypoint.right_width = 1.2

        return waypoint

    def run(self):
        rospy.spin()

def float32_multiarray_to_numpy(multiarray):
    dims = tuple(map(lambda x: x.size, multiarray.layout.dim))
    return np.array(multiarray.data, dtype=float).reshape(dims).astype(np.float32)

if __name__ == '__main__':
    rospy.init_node('openpilot_local_path_extractor')
    node = OpenpilotLocalPathPublisher()
    node.run()