#!/usr/bin/env python3

import rospy
import math
from autoware_msgs.msg import Lane
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA, Header
from helpers.waypoints import get_point_and_orientation_on_path_within_distance

class LocalPathVisualizer:
    def __init__(self):

        # Parameters
        self.stopping_lateral_distance = rospy.get_param("stopping_lateral_distance")
        self.slowdown_lateral_distance = rospy.get_param("slowdown_lateral_distance")
        self.current_pose_to_car_front = rospy.get_param("current_pose_to_car_front")
        self.stopping_speed_limit = rospy.get_param("stopping_speed_limit")

        self.published_waypoints = 0

        # Publishers
        self.local_path_markers_pub = rospy.Publisher('local_path_markers', MarkerArray, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('local_path', Lane, self.local_path_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)

    def local_path_callback(self, lane):

        local_path_len = len(lane.waypoints)
        current_waypoints = 0
        # lane.cost is used to determine the stopping point distance from path start
        stopping_point_distance = lane.cost

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = lane.header.frame_id

        marker_array = MarkerArray()

        if local_path_len > 1:
            points = [waypoint.pose.pose.position for waypoint in lane.waypoints]
            color = ColorRGBA(0.2, 1.0, 0.2, 0.3)

            # local path with stopping_lateral_distance
            marker = Marker(header=header)
            marker.ns = "Stopping lateral distance"
            marker.type = marker.LINE_STRIP
            marker.action = marker.ADD
            marker.id = 0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 2*self.stopping_lateral_distance
            marker.color = color
            marker.points = points
            marker_array.markers.append(marker)

            # local path with slowdown_lateral_distance
            #marker = Marker(header=header)
            #marker.ns = "Slowdown lateral distance"
            #marker.type = marker.LINE_STRIP
            #marker.action = marker.ADD
            #marker.id = 0
            #marker.pose.orientation.w = 1.0
            #marker.scale.x = 2*self.slowdown_lateral_distance
            #marker.color = color
            #marker.points = points
            #marker_array.markers.append(marker)

            # velocity labels
            for i, waypoint in enumerate(lane.waypoints):
                marker = Marker(header=header)
                marker.ns = "Velocity label"
                marker.id = i
                marker.type = marker.TEXT_VIEW_FACING
                marker.action = marker.ADD
                marker.pose = waypoint.pose.pose
                marker.scale.z = 0.5
                marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
                marker.text = str(round(waypoint.twist.twist.linear.x * 3.6, 1))
                marker_array.markers.append(marker)

                current_waypoints = i
                # add only up to a first 0.0 velocity label
                if math.isclose(waypoint.twist.twist.linear.x, 0.0):
                    break

            if stopping_point_distance > 0.0:

                stop_position, stop_orientation = get_point_and_orientation_on_path_within_distance(lane.waypoints, stopping_point_distance)

                color = ColorRGBA(0.9, 0.9, 0.9, 0.2)           # white - obstcle affecting ego speed in slowdown area
                if lane.is_blocked:
                    color = ColorRGBA(1.0, 1.0, 0.0, 0.5)       # yellow - obstacle in stopping area
                    if lane.closest_object_velocity < self.stopping_speed_limit:
                        color = ColorRGBA(1.0, 0.0, 0.0, 0.5)   # red - obstacle in front and very slow

                # "Stopping point" - obstacle that currently causes the smallest target velocity
                marker = Marker(header=header)
                marker.ns = "Stopping point"
                marker.id = 0
                marker.type = marker.CUBE
                marker.action = marker.ADD
                marker.pose.position.x = stop_position.x
                marker.pose.position.y = stop_position.y
                marker.pose.position.z = stop_position.z + 1.0
                marker.pose.orientation = stop_orientation
                marker.scale.x = 0.3
                marker.scale.y = 5.0
                marker.scale.z = 2.5
                marker.color = color
                marker_array.markers.append(marker)
            else:
                marker = Marker(header=header)
                marker.ns = "Stopping point"
                marker.id = 0
                marker.action = marker.DELETE
                marker_array.markers.append(marker)

        # delete markers if local path not created
        if local_path_len == 0 and self.published_waypoints > 0:
            marker = Marker(header=header)
            marker.ns = "Stopping lateral distance"
            marker.id = 0
            marker.action = marker.DELETE
            marker_array.markers.append(marker)

            #marker = Marker(header=header)
            #marker.ns = "Slowdown lateral distance"
            #marker.id = 0
            #marker.action = marker.DELETE
            #marker_array.markers.append(marker)

            marker = Marker(header=header)
            marker.ns = "Stopping point"
            marker.id = 0
            marker.action = marker.DELETE
            marker_array.markers.append(marker)

        if self.published_waypoints > current_waypoints:
            # delete all markers if local path length decreased
            for i in range(current_waypoints + 1, self.published_waypoints + 1):
                marker = Marker(header=header)
                marker.ns = "Velocity label"
                marker.id = i
                marker.action = marker.DELETE
                marker_array.markers.append(marker)

        self.published_waypoints = current_waypoints

        self.local_path_markers_pub.publish(marker_array)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('local_path_visualizer', log_level=rospy.INFO)
    node = LocalPathVisualizer()
    node.run()