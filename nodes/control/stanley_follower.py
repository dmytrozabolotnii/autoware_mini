#!/usr/bin/env python

import rospy
import tf
import math
import rospy
import message_filters
import numpy as np
from sklearn.neighbors import KDTree

from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose, PoseStamped,TwistStamped
from std_msgs.msg import ColorRGBA
from autoware_msgs.msg import LaneArray, VehicleCmd


class StanleyFollower:
    def __init__(self):

         # Parameters
        self.wheel_base = rospy.get_param("~wheel_base", 2.789)
        self.cte_gain = rospy.get_param("~cte_gain", 2.0) # gain for cross track error

        # Variables - init
        self.waypoint_tree = None
        self.waypoints = None
        self.last_wp_idx = 0
        self.target_velocity = 0.0
        self.l = 0
        self.r = 0

        # Subscribers
        self.path_sub = rospy.Subscriber('/path', LaneArray, self.path_callback)
        self.current_pose_sub = message_filters.Subscriber('/current_pose', PoseStamped)
        self.current_velocity_sub = message_filters.Subscriber('/current_velocity', TwistStamped)
        ts = message_filters.TimeSynchronizer([self.current_pose_sub, self.current_velocity_sub], queue_size=10)
        ts.registerCallback(self.current_status_callback)

        # Publishers
        self.stanley_rviz_pub = rospy.Publisher('follower_markers', MarkerArray, queue_size=1)
        self.vehicle_command_pub = rospy.Publisher('vehicle_cmd', VehicleCmd, queue_size=1)

        # output information to console
        rospy.loginfo("stanley_follower - wheel_base: " + str(self.wheel_base))
        rospy.loginfo("stanley_follower - cte_gain: " + str(self.cte_gain))
        rospy.loginfo("stanley_follower - initiliazed")


    def path_callback(self, path_msg):
        self.waypoints = path_msg.lanes[0].waypoints
        self.last_wp_idx = len(self.waypoints) - 1

        # create kd-tree for nearest neighbor search
        waypoints_xy = np.array([[w.pose.pose.position.x, w.pose.pose.position.y] for w in self.waypoints])
        self.waypoint_tree = KDTree(waypoints_xy)


    def current_status_callback(self, current_pose_msg, current_velocity_msg):

        if self.waypoint_tree is None:
            return

        self.current_pose = current_pose_msg.pose
        current_velocity = current_velocity_msg.twist.linear.x

        quaternion = (self.current_pose.orientation.x, self.current_pose.orientation.y, self.current_pose.orientation.z, self.current_pose.orientation.w)
        _, _, self.current_heading = tf.transformations.euler_from_quaternion(quaternion)
        
        # Find pose for the front wheel
        self.front_wheel_pose = self.get_front_wheel_pose()
        self.front_wheel_heading = self.get_heading_from_pose(self.front_wheel_pose)
        
        # calc cross track error and heading
        self.cross_track_error, self.track_heading, self.nearest_wp = self.calc_cross_track_error_and_heading(self.front_wheel_pose)

        # find heading error
        self.heading_error = self.track_heading - self.current_heading

        # TODO not actually used - cte already has +/- information?
        # self.min_path_yaw = math.atan2(self.nearest_wp.pose.pose.position.y - self.front_wheel_pose.position.y , self.nearest_wp.pose.pose.position.x - self.front_wheel_pose.position.x)
        # self.cross_yaw_error = self.min_path_yaw - self.current_heading

        # calc delta error
        self.delta_error = math.atan(self.cte_gain * self.cross_track_error / (current_velocity + 0.00001))
        self.steering_angle = self.heading_error + self.delta_error

        # TODO limit steering angle before output

        # get blinker information and target_velocity
        self.l, self.r = self.get_blinker_state(self.nearest_wp.wpstate.steering_state)
        self.target_velocity = self.nearest_wp.twist.twist.linear.x

        self.publish_stanley_rviz(self.front_wheel_pose, self.nearest_wp.pose.pose, self.heading_error)
        # TODO publish also debug output to another topic (e.g. /waypoint_follower/debug) - lateral error
        self.publish_vehicle_command()


    def publish_vehicle_command(self):
        vehicle_cmd = VehicleCmd()
        vehicle_cmd.header.stamp = rospy.Time.now()
        vehicle_cmd.header.frame_id = "/map"
        # blinkers
        vehicle_cmd.lamp_cmd.l = self.l
        vehicle_cmd.lamp_cmd.r = self.r
        # velocity and steering
        vehicle_cmd.ctrl_cmd.linear_velocity = self.target_velocity
        vehicle_cmd.ctrl_cmd.linear_acceleration = 0.0
        vehicle_cmd.ctrl_cmd.steering_angle = self.steering_angle
        self.vehicle_command_pub.publish(vehicle_cmd)


    def publish_stanley_rviz(self, front_pose, nearest_wp_pose, heading_error):
        
        marker_array = MarkerArray()

        # draws a line between current pose and nearest_wp point
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "nearest_wp distance"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.color = ColorRGBA(1.0, 0.0, 1.0, 1.0)
        marker.points = ([front_pose.position, nearest_wp_pose.position])
        marker_array.markers.append(marker)

        # label of angle alpha
        average_pose = Pose()
        average_pose.position.x = (front_pose.position.x + nearest_wp_pose.position.x) / 2
        average_pose.position.y = (front_pose.position.y + nearest_wp_pose.position.y) / 2
        average_pose.position.z = (front_pose.position.z + nearest_wp_pose.position.z) / 2

        marker_text = Marker()
        marker_text.header.frame_id = "/map"
        marker_text.header.stamp = rospy.Time.now()
        marker_text.ns = "heading_error"
        marker_text.id = 1
        marker_text.type = Marker.TEXT_VIEW_FACING
        marker_text.action = Marker.ADD
        marker_text.pose = average_pose
        marker_text.scale.z = 0.6
        marker_text.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
        marker_text.text = str(round(math.degrees(heading_error),1))
        marker_array.markers.append(marker_text)

        self.stanley_rviz_pub.publish(marker_array)

    def get_heading_from_pose(self, pose):
        quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        _, _, heading = tf.transformations.euler_from_quaternion(quaternion)
        return heading

    def get_front_wheel_pose(self):
        
        pose = Pose()
        pose.position.x = self.current_pose.position.x + self.wheel_base * math.cos(self.current_heading)
        pose.position.y = self.current_pose.position.y + self.wheel_base * math.sin(self.current_heading)
        pose.position.z = self.current_pose.position.z
        pose.orientation = self.current_pose.orientation

        return pose

    def get_blinker_state(self, steering_state):

        if steering_state == 1:     # left
            return 1, 0
        elif steering_state == 2:   # right
            return 0, 1
        else:                       # straight (no blinkers)
            return 0, 0

    def calc_cross_track_error_and_heading(self, front_wheel_pose):

        x_ego = front_wheel_pose.position.x
        y_ego = front_wheel_pose.position.y
        heading = 0.0
        cte = 0.0

        # find nearest wp distance and id
        d, idx = self.waypoint_tree.query([[self.front_wheel_pose.position.x, self.front_wheel_pose.position.y]], 1)
        idx = int(idx)

        x_nearest = self.waypoints[idx].pose.pose.position.x
        y_nearest = self.waypoints[idx].pose.pose.position.y

        # calc based on forward point
        if idx < self.last_wp_idx:
            x_front = self.waypoints[idx+1].pose.pose.position.x
            y_front = self.waypoints[idx+1].pose.pose.position.y

            cte = calc_dist_from_track(x_ego, y_ego, x_nearest, y_nearest, x_front, y_front)
            heading = math.atan2(y_front - y_nearest, x_front - x_nearest)

        # in case of last wp, calc based on backward point
        else:
            x_back = self.waypoints[idx-1].pose.pose.position.x
            y_back = self.waypoints[idx-1].pose.pose.position.y

            cte = calc_dist_from_track(x_ego, y_ego, x_back, y_back, x_nearest, y_nearest)
            heading = math.atan2(y_nearest - y_back, x_nearest - x_back)

        nearest_wp = self.waypoints[idx]

        return cte, heading, nearest_wp


    def run(self):
        rospy.spin()


def calc_dist_from_track(x_ego, y_ego, x1, y1, x2, y2):
    # calc distance from track
    # https://robotics.stackexchange.com/questions/22989/what-is-wrong-with-my-stanley-controller-for-car-steering-control

    numerator = (x2 - x1) * (y1 - y_ego) - (x1 - x_ego) * (y2 - y1)
    denominator = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return numerator / denominator



if __name__ == '__main__':
    rospy.init_node('stanley_follower', log_level=rospy.INFO)
    node = StanleyFollower()
    node.run()