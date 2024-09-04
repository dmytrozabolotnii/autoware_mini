#!/usr/bin/env python3

import rospy
import math
import numpy as np
from ros_numpy import numpify
from shapely.geometry import Polygon, LineString, Point as ShapelyPoint
from autoware_msgs.msg import Lane
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from helpers.path import Path
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
        rospy.Subscriber('collision_points', PointCloud2, self.collision_points_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('extracted_local_path', Lane, self.path_callback, queue_size=1, tcp_nodelay=True)

    def current_velocity_callback(self, msg):
        self.current_speed = msg.twist.linear.x

    def current_pose_callback(self, msg):
        self.current_position = ShapelyPoint(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

    def collision_points_callback(self, msg):
        self.collision_points = numpify(msg)

    def path_callback(self, msg):

        collision_points = self.collision_points
        current_position = self.current_position
        current_speed = self.current_speed

        if current_speed is None or current_position is None or collision_points is None:
            self.publish_local_path_wp([], msg.header)
            return

        if  len(msg.waypoints) == 0 or len(collision_points) == 0:
            # 0 length path or alterations from collision points are not possible
            self.local_path_pub.publish(msg)
            return

        closest_object_distance = 0.0
        closest_object_velocity = 0.0
        local_path_blocked = False
        stopping_point_distance = 0.0

        print("Collision points: ", collision_points)

        # START here
        local_path = Path(msg.waypoints)
        ego_distance_from_local_path_start = local_path.linestring.project(current_position)
        collision_points_shapely_points = [ShapelyPoint(x, y, z) for x, y, z, vx, vy, vz, distance_to_stop, category in collision_points]

        object_distances = [local_path.linestring.project(point) for point in collision_points_shapely_points]
        print("Object distances: ", object_distances)
        collision_points_path_headings = [local_path.get_heading_at_distance(distance) for distance in object_distances]
        # TODO verify object_speeds
        object_velocities = [project_vector_to_heading(heading, Vector3(vx, vy, vz))
                             for heading, (x, y, z, vx, vy, vz, distance_to_stop, category) 
                             in zip(collision_points_path_headings, collision_points)]
        print("Object speeds: ", object_velocities)
        
        # extract distance_to_stop from collision_points
        object_braking_distances = [distance_to_stop for x, y, z, vx, vy, vz, distance_to_stop, category in collision_points]
        print("Distances to stop: ", object_braking_distances)

        # TODO separate check for TFL stopline and other objects - should not exceed deceleration limit

        object_distances = np.array(object_distances)
        object_velocities = np.array(object_velocities)
        object_braking_distances = np.array(object_braking_distances)

        # calculate target velocity for every object to select min target velocity object
        target_distances = object_distances - ego_distance_from_local_path_start - self.current_pose_to_car_front - object_braking_distances - self.braking_reaction_time * np.abs(object_velocities)
        target_velocities = np.sqrt(np.maximum(0.0, np.maximum(0.0, object_velocities)**2 + 2 * self.default_deceleration * target_distances))

        # index of min target velocity object
        min_value_index = np.argmin(target_velocities)

        closest_object_distance = object_distances[min_value_index] - ego_distance_from_local_path_start - self.current_pose_to_car_front
        closest_object_velocity = object_velocities[min_value_index]
        stopping_point_distance = object_distances[min_value_index] - object_braking_distances[min_value_index]

        # Recalculate target_velocity and overwrite in the waypoints
        zero_speeds_onwards = False
        target_distance_obj = stopping_point_distance - self.current_pose_to_car_front - self.braking_reaction_time * np.abs(closest_object_velocity)
        for i, wp in enumerate(local_path.waypoints):

            # once we get zero speed, keep it that way
            if zero_speeds_onwards:
                wp.twist.twist.linear.x = 0.0
                continue

            if i > 0:
                target_distance_obj -= get_distance_between_two_points_2d(local_path.waypoints[i-1].pose.pose.position, local_path.waypoints[i].pose.pose.position)
            target_velocity_obj = np.sqrt(np.maximum(0.0, np.maximum(0.0, closest_object_velocity)**2 + 2 * self.default_deceleration * target_distance_obj))

            # overwrite target velocity of wp
            wp.twist.twist.linear.x = min(target_velocity_obj, wp.twist.twist.linear.x)

            # from stop point onwards all speeds are set to zero
            if math.isclose(wp.twist.twist.linear.x, 0.0):
                zero_speeds_onwards = True



        # Update the lane message with the calculated values
        self.publish_local_path_wp(local_path.waypoints, msg.header, closest_object_distance, closest_object_velocity, local_path_blocked, stopping_point_distance)


    def publish_local_path_wp(self, local_path_waypoints, header, closest_object_distance=0.0, closest_object_velocity=0.0, local_path_blocked=False, stopping_point_distance=0.0):
        # create lane message
        lane = Lane()
        lane.header = header
        lane.waypoints = local_path_waypoints
        lane.closest_object_distance = closest_object_distance
        lane.closest_object_velocity = closest_object_velocity
        lane.is_blocked = local_path_blocked
        lane.cost = stopping_point_distance
        self.local_path_pub.publish(lane)


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('speed_planner')
    node = SpeedPlanner()
    node.run()