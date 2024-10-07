#!/usr/bin/env python3

import rospy
import math
import message_filters
import traceback
import shapely
import numpy as np
from ros_numpy import numpify
from autoware_msgs.msg import Lane
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from helpers.path import Path
from helpers.collision import CollisionPoints
from helpers.geometry import project_vector_to_heading, get_distance_between_two_points_2d

class SpeedPlanner:

    def __init__(self):

        # parameters
        self.current_pose_to_car_front = rospy.get_param("current_pose_to_car_front")
        self.default_deceleration = rospy.get_param("default_deceleration")
        self.braking_reaction_time = rospy.get_param("braking_reaction_time")

        # variables
        self.collision_points = None
        self.current_position = None
        self.current_speed = None

        # publishers
        self.local_path_pub = rospy.Publisher('local_path', Lane, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_velocity', TwistStamped, self.current_velocity_callback, queue_size=1, tcp_nodelay=True)

        collision_points_sub = message_filters.Subscriber('collision_points', PointCloud2, tcp_nodelay=True)
        local_path_sub = message_filters.Subscriber('extracted_local_path', Lane, tcp_nodelay=True)

        ts = message_filters.TimeSynchronizer([collision_points_sub, local_path_sub], queue_size=4)
        ts.registerCallback(self.collision_points_and_path_callback)

    def current_velocity_callback(self, msg):
        self.current_speed = msg.twist.linear.x

    def current_pose_callback(self, msg):
        self.current_position = shapely.Point(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

    def collision_points_and_path_callback(self, collision_points_msg, local_path_msg):
        try:
            collision_points = numpify(collision_points_msg)
            current_position = self.current_position
            current_speed = self.current_speed

            if current_speed is None or current_position is None:
                rospy.logwarn_throttle(3, "%s - current speed or position not received!", rospy.get_name())
                return

            if  len(local_path_msg.waypoints) == 0 or len(collision_points) == 0:
                # no local path or no collision points menas no alterations to the path
                self.local_path_pub.publish(local_path_msg)
                return

            closest_object_distance = 0.0
            closest_object_velocity = 0.0
            local_path_blocked = False
            stopping_point_distance = 0.0

            # create local path
            local_path = Path(local_path_msg.waypoints)
            ego_distance_from_local_path_start = local_path.linestring.project(current_position)

            # extract object distances, velocities and braking distances
            collision_points_shapely = [shapely.Point(x, y, z) for x, y, z, vx, vy, vz, distance_to_stop, category in collision_points]
            object_distances = np.array([local_path.linestring.project(point) for point in collision_points_shapely])
            collision_points_path_headings = [local_path.get_heading_at_distance(distance) for distance in object_distances]
            object_velocities = np.array([project_vector_to_heading(heading, Vector3(vx, vy, vz)) 
                                        for heading, (x, y, z, vx, vy, vz, distance_to_stop, category)
                                        in zip(collision_points_path_headings, collision_points)])
            object_braking_distances = collision_points['distance_to_stop']

            # calculate target velocity for every collision pont
            # 'abs' is used to turn negative speed of approaching cars into positive, so that target distance would be smaller and thus target_speed will be decreased
            target_distances = object_distances - self.current_pose_to_car_front - object_braking_distances - self.braking_reaction_time * np.abs(object_velocities)
            target_velocities = np.sqrt(np.maximum(0.0, np.maximum(0.0, object_velocities)**2 + 2 * self.default_deceleration * target_distances))

            # find the closest collision point
            min_value_index = np.argmin(target_velocities)
            closest_object_distance = object_distances[min_value_index] - ego_distance_from_local_path_start - self.current_pose_to_car_front
            closest_object_velocity = object_velocities[min_value_index]
            stopping_point_distance = object_distances[min_value_index] - object_braking_distances[min_value_index]
            collision_point_category = collision_points[min_value_index]["category"]
            if collision_point_category != CollisionPoints.GOAL_POINT:
                local_path_blocked = True

            # Recalculate target_velocity for all the waypoints using the closest object
            zero_speeds_onwards = False
            target_distance_object = target_distances[min_value_index]
            for i, wp in enumerate(local_path.waypoints):

                # once we get zero speed, keep it that way
                if zero_speeds_onwards:
                    wp.twist.twist.linear.x = 0.0
                    continue

                if i > 0:
                    target_distance_object -= get_distance_between_two_points_2d(local_path.waypoints[i-1].pose.pose.position, local_path.waypoints[i].pose.pose.position)
                target_velocity_object = np.sqrt(np.maximum(0.0, np.maximum(0.0, closest_object_velocity)**2 + 2 * self.default_deceleration * target_distance_object))

                # overwrite target velocity of wp
                wp.twist.twist.linear.x = min(target_velocity_object, wp.twist.twist.linear.x)

                # from stop point onwards all speeds are set to zero
                if math.isclose(wp.twist.twist.linear.x, 0.0):
                    zero_speeds_onwards = True

            # Update the lane message with the calculated values
            lane = Lane()
            lane.header = local_path_msg.header
            lane.waypoints = local_path.waypoints
            lane.closest_object_distance = closest_object_distance
            lane.closest_object_velocity = closest_object_velocity
            lane.is_blocked = local_path_blocked
            lane.cost = stopping_point_distance
            lane.increment = collision_point_category
            self.local_path_pub.publish(lane)

        except Exception as e:
            rospy.logerr_throttle(10, "%s - Exception in callback: %s", rospy.get_name(), traceback.format_exc())


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('speed_planner')
    node = SpeedPlanner()
    node.run()