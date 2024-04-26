#!/usr/bin/env python3

import rospy
import copy
import math
import threading
from tf2_ros import Buffer, TransformListener, TransformException
import numpy as np
from shapely.geometry import Polygon, LineString, Point as Point2d
from shapely import prepare
from scipy.interpolate import interp1d

from autoware_msgs.msg import Lane, DetectedObjectArray, TrafficLightResultArray, Waypoint
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3

from helpers.transform import transform_vector3
from helpers.lanelet2 import load_lanelet2_map, get_stoplines


class VelocityLocalPlanner:

    def __init__(self):

        # Parameters
        self.local_path_length = rospy.get_param("~local_path_length")
        self.transform_timeout = rospy.get_param("~transform_timeout")
        self.braking_safety_distance_obstacle = rospy.get_param("~braking_safety_distance_obstacle")
        self.braking_safety_distance_stopline = rospy.get_param("~braking_safety_distance_stopline")
        self.braking_safety_distance_goal = rospy.get_param("~braking_safety_distance_goal")
        self.braking_reaction_time = rospy.get_param("braking_reaction_time")
        self.stopping_lateral_distance = rospy.get_param("stopping_lateral_distance")
#        self.slowdown_lateral_distance = rospy.get_param("slowdown_lateral_distance")
        self.current_pose_to_car_front = rospy.get_param("current_pose_to_car_front")
        self.default_deceleration = rospy.get_param("default_deceleration")
        self.tfl_maximum_deceleration = rospy.get_param("~tfl_maximum_deceleration")

        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")
        lanelet2_map_name = rospy.get_param("~lanelet2_map_name")

        # Internal variables
        self.lock = threading.Lock()
        self.output_frame = None
        self.global_path_linestring = None
        self.global_path_waypoints = None
        self.global_path_distances = None
        self.distance_to_velocity_interpolator = None
        self.current_speed = None
        self.current_position = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        self.red_stoplines = {}

        lanelet2_map = load_lanelet2_map(lanelet2_map_name, coordinate_transformer, use_custom_origin, utm_origin_lat, utm_origin_lon)
        self.all_stoplines = get_stoplines(lanelet2_map)

        # Publishers
        self.local_path_pub = rospy.Publisher('local_path', Lane, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('smoothed_path', Lane, self.path_callback, queue_size=None, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_velocity', TwistStamped, self.current_velocity_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        rospy.Subscriber('/detection/traffic_light_status', TrafficLightResultArray, self.traffic_light_status_callback, queue_size=1, tcp_nodelay=True)


    def path_callback(self, msg):

        if len(msg.waypoints) == 0:
            global_path_linestring = None
            global_path_waypoints = None
            global_path_distances = None
            distance_to_velocity_interpolator = None
            rospy.loginfo("%s - Empty global path received", rospy.get_name())
        else:
            global_path_waypoints = msg.waypoints
            waypoints_xyz = np.array([(w.pose.pose.position.x, w.pose.pose.position.y, w.pose.pose.position.z) for w in global_path_waypoints])
            # convert waypoints to shapely linestring
            global_path_linestring = LineString(waypoints_xyz)
            prepare(global_path_linestring)

            # calculate distances between points, use only xy, and insert 0 at start of distances array
            global_path_distances = np.cumsum(np.sqrt(np.sum(np.diff(waypoints_xyz[:,:2], axis=0)**2, axis=1)))
            global_path_distances = np.insert(global_path_distances, 0, 0)

            # extract velocity at waypoints and create interpolator
            velocities = np.array([w.twist.twist.linear.x for w in msg.waypoints])
            distance_to_velocity_interpolator = interp1d(global_path_distances, velocities, kind='linear', bounds_error=False, fill_value=0.0)

            rospy.loginfo("%s - Global path received with %i waypoints", rospy.get_name(), len(msg.waypoints))

        with self.lock:
            self.output_frame = msg.header.frame_id
            self.global_path_linestring = global_path_linestring
            self.global_path_waypoints = global_path_waypoints
            self.global_path_distances = global_path_distances
            self.distance_to_velocity_interpolator = distance_to_velocity_interpolator


    def current_velocity_callback(self, msg):
        # save current velocity
        self.current_speed = msg.twist.linear.x


    def current_pose_callback(self, msg):
        # save current pose
        self.current_position = Point2d(msg.pose.position.x, msg.pose.position.y)


    def traffic_light_status_callback(self, msg):

        red_stoplines = {}
        for result in msg.results:
            if result.recognition_result == 0:
                red_stoplines[result.lane_id] = result.recognition_result

        self.red_stoplines = red_stoplines


    def detected_objects_callback(self, msg):
        with self.lock:
            output_frame = self.output_frame
            global_path_linestring = self.global_path_linestring
            global_path_waypoints = self.global_path_waypoints
            global_path_distances = self.global_path_distances
            distance_to_velocity_interpolator = self.distance_to_velocity_interpolator

        red_stoplines = self.red_stoplines
        current_position = self.current_position
        current_speed = self.current_speed

        # if global path, current pose or current_speed is None, publish empty local path, which stops the vehicle
        if global_path_linestring is None or global_path_waypoints is None or global_path_distances is None or distance_to_velocity_interpolator is None or current_position is None or current_speed is None or output_frame is None:
            self.publish_local_path_wp([], msg.header.stamp, output_frame)
            return

        # TODO how to avoid jumping from one place to another on path - just finding the closest point is dangerous!
        # Example of global path overlapping with itself or ego doing the 90deg turn and cutting the corner!
        d_ego_from_path_start = global_path_linestring.project(current_position)
        # Calculate the map speed at the current position as target velocity
        target_velocity = float(distance_to_velocity_interpolator(d_ego_from_path_start))

        # extract local path, if None is returned publish empty local path
        local_path_linestring, local_path_waypoints = self.extract_local_path(global_path_linestring, global_path_waypoints, global_path_distances, d_ego_from_path_start, self.local_path_length)
        prepare(local_path_linestring)
        if local_path_linestring is None:
            self.publish_local_path_wp([], msg.header.stamp, self.output_frame)
            return

        # initialize closest object distance and velocity
        closest_object_distance = 0.0
        closest_object_velocity = 0.0
        stopping_point_distance = 0.0
        local_path_blocked = False
        # collect objects (closest point from each object, path end point, stopline points)
        object_distances = []
        object_velocities = []
        object_braking_distances = []

        if len(msg.objects) > 0:
            # fetch the transform from the object frame to the base_link frame to align the speed with ego vehicle
            try:
                transform = self.tf_buffer.lookup_transform("base_link", msg.header.frame_id, msg.header.stamp, rospy.Duration(self.transform_timeout))
            except (TransformException, rospy.ROSTimeMovedBackwardsException) as e:
                rospy.logwarn("%s - unable to transform object speed to base frame, using speed 0: %s", rospy.get_name(), e)
                transform = None

        # create buffer around local path
        local_path_buffer = local_path_linestring.buffer(self.stopping_lateral_distance, cap_style="flat")
        prepare(local_path_buffer)

        # 1. ADD DETECTED OBJECTS AS OBSTACLES
        for object in msg.objects:
            # get the convex hulls and store as shapely polygons
            object_polygon = Polygon([(p.x, p.y) for p in object.convex_hull.polygon.points])
            # chek if object polygon intersects with local path buffer
            if local_path_buffer.intersects(object_polygon):
                local_path_blocked = True
                intersection_points = object_polygon.intersection(local_path_buffer)

                # calc distance for all intersection points and take the minimum
                object_distance = min([local_path_linestring.project(Point2d(coords)) for coords in intersection_points.exterior.coords[:-1]])

                # project object velocity to base_link frame to get longitudinal speed
                # in case there is no transform assume the object is not moving
                if transform is not None:
                    velocity = transform_vector3(object.velocity.linear, transform)
                else:
                    velocity = Vector3()

                object_distances.append(object_distance)
                object_velocities.append(velocity.x)
                object_braking_distances.append(self.braking_safety_distance_obstacle)

        # 2. ADD RED STOPLINES AS OBSTACLES
        red_stoplines_linestrings = [self.all_stoplines[stopline_id] for stopline_id in red_stoplines]
        for stopline_ls in red_stoplines_linestrings:
            if stopline_ls.intersects(local_path_linestring):
                intersection_point = local_path_linestring.intersection(stopline_ls)
                # calc distance for all intersection points
                d_to_stopline = local_path_linestring.project(Point2d(intersection_point.x, intersection_point.y))
                deceleration = (current_speed**2) / (2 * d_to_stopline)
                # check if deceleration is within the limits
                if 0 <= deceleration and deceleration <= self.tfl_maximum_deceleration:
                    object_distances.append(d_to_stopline)
                    object_velocities.append(0)
                    object_braking_distances.append(self.braking_safety_distance_stopline)
                else:
                    rospy.logwarn_throttle(3, "%s - ignore red traffic light, deceleration: %f", rospy.get_name(), deceleration)

        # 3. ADD GOAL POINT AS OBSTACLE
        # Add last wp as goal point to stop the car before it
        if global_path_linestring.coords[-1] == local_path_linestring.coords[-1]:
            object_distances.append(local_path_linestring.length)
            object_velocities.append(0)
            object_braking_distances.append(self.braking_safety_distance_goal)


        if len(object_distances) > 0:
            object_distances = np.array(object_distances)
            object_velocities = np.array(object_velocities)
            object_braking_distances = np.array(object_braking_distances)

            target_distances = object_distances - self.current_pose_to_car_front - object_braking_distances - self.braking_reaction_time * np.abs(object_velocities)
            target_velocities = np.sqrt(np.maximum(0.0, np.maximum(0.0, object_velocities)**2 + 2 * self.default_deceleration * target_distances))

            # index of min target velocity object
            min_value_index = np.argmin(target_velocities)

            closest_object_distance = object_distances[min_value_index] - self.current_pose_to_car_front
            closest_object_velocity = object_velocities[min_value_index]
            stopping_point_distance = object_distances[min_value_index] - object_braking_distances[min_value_index] 

            target_velocity = min(target_velocity, target_velocities[min_value_index])

            # Recalculate target_velocity and overwrite in the waypoints
            zero_speeds_onwards = False
            d_between_wp_cumulative = 0.0
            for i, wp in enumerate(local_path_waypoints):

                # once we get zero speed, keep it that way
                if zero_speeds_onwards:
                    wp.twist.twist.linear.x = 0.0
                    continue

                if i > 0:
                    d_between_wp_cumulative += np.hypot(local_path_waypoints[i].pose.pose.position.x - local_path_waypoints[i-1].pose.pose.position.x, 
                                                        local_path_waypoints[i].pose.pose.position.y - local_path_waypoints[i-1].pose.pose.position.y)
                target_distance_obj = closest_object_distance - object_braking_distances[min_value_index] - d_between_wp_cumulative - self.braking_reaction_time * np.abs(closest_object_velocity)
                target_velocity_obj = np.sqrt(np.maximum(0.0, np.maximum(0.0, closest_object_velocity)**2 + 2 * self.default_deceleration * target_distance_obj))
                target_velocity_obj = min(target_velocity_obj, target_velocity)

                # overwrite target velocity of wp
                wp.twist.twist.linear.x = target_velocity_obj

                # from stop point onwards all speeds are set to zero
                if math.isclose(wp.twist.twist.linear.x, 0.0):
                    zero_speeds_onwards = True

        self.publish_local_path_wp(local_path_waypoints, msg.header.stamp, output_frame, closest_object_distance, closest_object_velocity, local_path_blocked, stopping_point_distance)

    def publish_local_path_wp(self, local_path_waypoints, stamp, output_frame, closest_object_distance=0.0, closest_object_velocity=0.0, local_path_blocked=False, stopping_point_distance=0.0):
        # create lane message
        lane = Lane()
        lane.header.frame_id = output_frame
        lane.header.stamp = stamp
        lane.waypoints = local_path_waypoints
        lane.closest_object_distance = closest_object_distance
        lane.closest_object_velocity = closest_object_velocity
        lane.is_blocked = local_path_blocked
        lane.cost = stopping_point_distance
        self.local_path_pub.publish(lane)


    def extract_local_path(self, global_path_linestring, global_path_waypoints, global_path_distances, d_ego_from_path_start, local_path_length):

        # current position is projected at the end of the global path - goal reached
        if math.isclose(d_ego_from_path_start, global_path_linestring.length):
            return None, None

        # find index where distances are higher than ego_d_on_global_path
        index_start = max(np.argmax(global_path_distances >= d_ego_from_path_start) - 1, 0)
        index_end = np.argmax(global_path_distances >= d_ego_from_path_start + local_path_length)

        # if end point of local_path is past the end of the global path (returns 0) then take index of last point
        if index_end == 0:
            index_end = len(global_path_linestring.coords)

        local_path_linestring = LineString(global_path_linestring.coords[index_start:index_end])
        # deepcopy only the necessary part (other parts are not changed and are shared with global path waypoints)
        local_path_waypoints = []
        for waypoint in global_path_waypoints[index_start:index_end]:
            new_waypoint = Waypoint(pose = copy.copy(waypoint.pose), wpstate = copy.copy(waypoint.wpstate))
            new_waypoint.twist.twist.linear.x = copy.deepcopy(waypoint.twist.twist.linear.x)
            local_path_waypoints.append(new_waypoint)

        return local_path_linestring, local_path_waypoints

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('velocity_local_planner')
    node = VelocityLocalPlanner()
    node.run()