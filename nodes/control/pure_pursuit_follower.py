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


class PurePursuitFollower:
    def __init__(self):

        # Parameters
        self.planning_time = rospy.get_param("~planning_time", 2.0)
        self.min_lookahead_distance = rospy.get_param("~min_lookahead_distance", 5.5)
        self.wheel_base = rospy.get_param("~wheel_base", 2.789)

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

        # Publishers TODO / publish to same topic as Stanley: "follower_markers"
        self.pure_pursuit_rviz_pub = rospy.Publisher('follower_markers', MarkerArray, queue_size=1)
        self.vehicle_command_pub = rospy.Publisher('vehicle_cmd', VehicleCmd, queue_size=1)

        # output information to console
        rospy.loginfo("pure_pursuit_follower - initiliazed")

    def path_callback(self, path_msg):
        self.waypoints = path_msg.lanes[0].waypoints
        self.last_wp_idx = len(self.waypoints) - 1

        # create kd-tree for nearest neighbor search
        waypoints_xy = np.array([[w.pose.pose.position.x, w.pose.pose.position.y] for w in self.waypoints])
        self.waypoint_tree = KDTree(waypoints_xy)


    def current_status_callback(self, current_pose_msg, current_velocity_msg):

        if self.waypoint_tree is None:
            return

        current_pose = current_pose_msg.pose
        current_velocity = current_velocity_msg.twist.linear.x

        d, nearest_wp_idx = self.waypoint_tree.query([[current_pose.position.x, current_pose.position.y]], 1)
        nearest_wp = self.waypoints[int(nearest_wp_idx)]

        # calc lookahead distance (velocity dependent)
        lookahead_distance = current_velocity * self.planning_time
        if lookahead_distance < self.min_lookahead_distance:
            lookahead_distance = self.min_lookahead_distance
        
        # TODO assume 1m distance between waypoints - currently OK, but need to make it more universal
        lookahead_wp_idx = int(nearest_wp_idx) + math.floor(lookahead_distance)

        if lookahead_wp_idx > self.last_wp_idx:
            lookahead_wp_idx = self.last_wp_idx
        lookahead_wp = self.waypoints[int(lookahead_wp_idx)]

        # find current pose heading
        quaternion = (current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w)
        _, _, current_heading = tf.transformations.euler_from_quaternion(quaternion)
        lookahead_heading = get_heading_from_two_positions(current_pose.position, lookahead_wp.pose.pose.position)
        alpha = lookahead_heading - current_heading

        curvature = 2 * math.sin(alpha) / lookahead_distance
        self.steering_angle = math.atan(self.wheel_base * curvature)

        # TODO - add limits to steering angle

        # calc cross track error - used only for debug output
        cross_track_error = self.calc_cross_track_error(current_pose, int(nearest_wp_idx))
        print("cross track error: %f, %f" % (cross_track_error, d))


        # get blinker information from nearest waypoint and target velocity from lookahead waypoint
        self.l, self.r = self.get_blinker_state(nearest_wp.wpstate.steering_state)
        self.target_velocity = lookahead_wp.twist.twist.linear.x

        self.publish_pure_pursuit_rviz(current_pose, lookahead_wp.pose.pose, alpha)
        # publish also debug output to another topic (e.g. /waypoint_follower/debug) - lateral error
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


    def publish_pure_pursuit_rviz(self, current_pose, lookahead_pose, alpha):
        
        marker_array = MarkerArray()

        # draws a line between current pose and lookahead point
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "Lookahead distance"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.color = ColorRGBA(1.0, 0.0, 1.0, 1.0)
        marker.points = ([current_pose.position, lookahead_pose.position])
        marker_array.markers.append(marker)

        # label of angle alpha
        average_pose = Pose()
        average_pose.position.x = (current_pose.position.x + lookahead_pose.position.x) / 2
        average_pose.position.y = (current_pose.position.y + lookahead_pose.position.y) / 2
        average_pose.position.z = (current_pose.position.z + lookahead_pose.position.z) / 2

        marker_text = Marker()
        marker_text.header.frame_id = "/map"
        marker_text.header.stamp = rospy.Time.now()
        marker_text.ns = "Angle alpha"
        marker_text.id = 1
        marker_text.type = Marker.TEXT_VIEW_FACING
        marker_text.action = Marker.ADD
        marker_text.pose = average_pose
        marker_text.scale.z = 0.6
        marker_text.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
        marker_text.text = str(round(math.degrees(alpha),1))
        marker_array.markers.append(marker_text)

        self.pure_pursuit_rviz_pub.publish(marker_array)


    def get_blinker_state(self, steering_state):

        if steering_state == 1:     # left
            return 1, 0
        elif steering_state == 2:   # right
            return 0, 1
        else:                       # straight (no blinkers)
            return 0, 0

    def calc_cross_track_error(self, current_pose, nearest_wp_idx):

        x_ego = current_pose.position.x
        y_ego = current_pose.position.y
        cte = 0.0

        # find nearest wp distance and id
        idx = nearest_wp_idx

        x_nearest = self.waypoints[idx].pose.pose.position.x
        y_nearest = self.waypoints[idx].pose.pose.position.y

        # calc based on forward point
        if idx < self.last_wp_idx:
            x_front = self.waypoints[idx+1].pose.pose.position.x
            y_front = self.waypoints[idx+1].pose.pose.position.y
            cte_front = calc_dist_from_track(x_ego, y_ego, x_nearest, y_nearest, x_front, y_front)
            cte = cte_front

        # calc based on backward point
        if idx > 0:
            x_back = self.waypoints[idx-1].pose.pose.position.x
            y_back = self.waypoints[idx-1].pose.pose.position.y
            cte_back = calc_dist_from_track(x_ego, y_ego, x_back, y_back, x_nearest, y_nearest)
            # select smaller error
            if abs(cte_back) < abs(cte):
                cte = cte_back

        return cte

    def run(self):
        rospy.spin()


def calc_dist_from_track(x_ego, y_ego, x1, y1, x2, y2):
    # calc distance from track
    # https://robotics.stackexchange.com/questions/22989/what-is-wrong-with-my-stanley-controller-for-car-steering-control

    numerator = (x2 - x1) * (y1 - y_ego) - (x1 - x_ego) * (y2 - y1)
    denominator = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return numerator / denominator

def get_heading_from_two_positions(position1, position2):
    # calc heading from two positions
    heading = math.atan2(position2.y - position1.y, position2.x - position1.x)

    return heading


if __name__ == '__main__':
    rospy.init_node('pure_pursuit_follower', log_level=rospy.INFO)
    node = PurePursuitFollower()
    node.run()