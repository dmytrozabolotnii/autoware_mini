#!/usr/bin/env python3

import rospy
import math
import shapely
import numpy as np

from autoware_mini.msg import Path, DetectedObjectArray
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import PointCloud2
from tf2_ros import TransformListener, Buffer, TransformException

from helpers.geometry import get_heading_from_vector, get_angle_between_two_headings, get_vector_norm_3d
from helpers.detection import calculate_time_to_destination
from helpers.collision import CollisionPoints
from helpers.path import PathWrapper

class TrajectoryCollisionChecker:

    def __init__(self):

        # parameters
        self.safety_box_width = rospy.get_param("safety_box_width")
        self.safety_box_length = rospy.get_param("safety_box_length")
        self.braking_safety_distance_trajectory = rospy.get_param("~braking_safety_distance_trajectory")
        self.heading_alignment_limit = rospy.get_param("~heading_alignment_limit")
        self.use_object_width = rospy.get_param("use_object_width")
        self.wp_buffer_distance = rospy.get_param("~wp_buffer_distance")
        self.safety_time_ego_front = rospy.get_param("~safety_time_ego_front")
        self.safety_time_ego_rear = rospy.get_param("~safety_time_ego_rear")

        # variables
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        self.detected_objects = None
        self.current_velocity = None
        self.current_acceleration = 0
        # publishers
        self.local_path_collision_pub = rospy.Publisher('trajectory_collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('/detection/predicted_objects_map', DetectedObjectArray, self.predicted_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        rospy.Subscriber('extracted_local_path', Path, self.local_path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_velocity', TwistStamped, self.current_velocity_callback, queue_size=1, tcp_nodelay=True)

    def predicted_objects_callback(self, msg):
        self.detected_objects = msg.objects

    def current_velocity_callback(self, msg):
        self.current_velocity = msg.twist.linear.x

    def local_path_callback(self, msg):

        detected_objects = self.detected_objects
        current_velocity = self.current_velocity
        current_acceleration = self.current_acceleration

        if detected_objects is None or current_velocity is None:
            rospy.logwarn_throttle(3, "%s - detected objects or current velocity not received!", rospy.get_name())
            return

        collision_points = CollisionPoints()

        if len(msg.waypoints) > 0:
            local_path = PathWrapper(msg.waypoints, distances=True)
            local_path_buffer = local_path.linestring.buffer(self.safety_box_width / 2, cap_style="flat")
            shapely.prepare(local_path_buffer)

            # get the car_front and projct to local_path
            try:
                transform = self.tf_buffer.lookup_transform(msg.header.frame_id, "car_front", msg.header.stamp, rospy.Duration(0.06))
            except (TransformException, rospy.ROSTimeMovedBackwardsException) as e:
                rospy.logwarn("%s - %s", rospy.get_name(), e)
                return
            car_front = shapely.Point(transform.transform.translation.x, transform.transform.translation.y)
            car_front_distance_from_local_path_start = local_path.linestring.project(car_front)

            for obj in detected_objects:
                for path in obj.candidate_trajectories.paths:

                    trajectory = PathWrapper(path.waypoints)
                    trajectory_to_check = trajectory.linestring

                    if self.use_object_width:
                        trajectory_to_check = trajectory_to_check.buffer(path.waypoints[0].left_width, cap_style="flat")

                    if local_path_buffer.intersects(trajectory_to_check):
                        trajectory_intersection_result = trajectory_to_check.intersection(local_path_buffer)
                        trajectory_intersection_points = shapely.get_coordinates(trajectory_intersection_result)

                        # calculate trajectory intersection distances for ego vehicle and object
                        trajectory_distance_from_local_path_start_min = float('inf')
                        trajectory_distance_from_local_path_start_max = 0.0
                        for x, y in trajectory_intersection_points:
                            distance = local_path.linestring.project(shapely.Point(x, y))
                            trajectory_distance_from_local_path_start_min = min(trajectory_distance_from_local_path_start_min, distance)
                            trajectory_distance_from_local_path_start_max = max(trajectory_distance_from_local_path_start_max, distance)


                        object_current_location = shapely.Point(obj.position.x, obj.position.y)
                        object_distance_from_local_path_start = local_path.linestring.project(object_current_location)

                        # Ignore trajectories from behind
                        if math.isclose(trajectory_distance_from_local_path_start_min, 0.0, abs_tol=0.001) and math.isclose(object_distance_from_local_path_start, 0.0, abs_tol=0.001):
                            continue

                        object_current_heading = get_heading_from_vector(obj.velocity)
                        object_local_path_heading = local_path.get_heading_at_distance(object_distance_from_local_path_start)
                        heading_difference = math.degrees(get_angle_between_two_headings(object_current_heading, object_local_path_heading))
                        object_polygon = shapely.Polygon([(p.x, p.y) for p in obj.convex_hull.points])

                        # Ignore object trajectories that are on our path and with similar heading - must be in front of us
                        if local_path_buffer.intersects(object_polygon) and heading_difference < self.heading_alignment_limit:
                            continue
                        
                        # Extract INTERSECTION AREA: distances on local_path and extract points
                        trajectory_distance_from_local_path_start_min = max(trajectory_distance_from_local_path_start_min - self.wp_buffer_distance, 0.0)
                        trajectory_distance_from_local_path_start_max += self.wp_buffer_distance
                        collision_area_distances = np.array(local_path.extract_distances(trajectory_distance_from_local_path_start_min, trajectory_distance_from_local_path_start_max))
                        collision_area_points = np.array(local_path.extract_points(trajectory_distance_from_local_path_start_min, trajectory_distance_from_local_path_start_max))

                        # EGO distances, arrival and leaving times
                        collision_distance_from_ego_front = collision_area_distances - car_front_distance_from_local_path_start
                        ego_arrival_times = calculate_time_to_destination(current_velocity, current_acceleration, collision_distance_from_ego_front)
                        ego_leaving_times = calculate_time_to_destination(current_velocity, current_acceleration, collision_distance_from_ego_front + self.safety_box_length)
                        ego_arrival_times -= self.safety_time_ego_front
                        ego_leaving_times += self.safety_time_ego_rear

                        # OBJECT distances, arrival and leaving times
                        obj_velocity = get_vector_norm_3d(obj.velocity)
                        obj_acceleration = get_vector_norm_3d(obj.acceleration)
                        collision_distance_from_obj_front = np.array([trajectory.linestring.project(p) for p in collision_area_points])
                        obj_arrival_times = calculate_time_to_destination(obj_velocity, obj_acceleration, collision_distance_from_obj_front)
                        obj_leaving_times = calculate_time_to_destination(obj_velocity, obj_acceleration, collision_distance_from_obj_front + obj.dimensions.x)

                        # FIND COLLISION AREA
                        collision_mask = ((ego_arrival_times <= obj_leaving_times) & (ego_leaving_times >= obj_arrival_times))
                        collision_area_points = collision_area_points[collision_mask]
                        collision_area_points = [(p.x, p.y) for p in collision_area_points]

                        if heading_difference < self.heading_alignment_limit:
                            # objects with similar heading - add collision points with object's velocity
                            collision_points.add_intersection_points(collision_area_points,
                                                                    z = obj.position.z,
                                                                    vx = obj.velocity.x,
                                                                    vy = obj.velocity.y,
                                                                    vz = obj.velocity.z,
                                                                    distance_to_stop = self.braking_safety_distance_trajectory,
                                                                    category = CollisionPoints.MERGING_TRAJECTORY)
                        else:
                            # objects intersecting at angle, add with 0 velocity
                            collision_points.add_intersection_points(collision_area_points,
                                    z = obj.position.z,
                                    vx = 0,
                                    vy = 0,
                                    vz = 0,
                                    distance_to_stop = self.braking_safety_distance_trajectory,
                                    category = CollisionPoints.COLLIDING_TRAJECTORY)

        collision_points_msg = collision_points.create_message()
        collision_points_msg.header = msg.header
        self.local_path_collision_pub.publish(collision_points_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('trajectory_collision_checker')
    node = TrajectoryCollisionChecker()
    node.run()