#!/usr/bin/env python3

import rospy
import math
import shapely
from autoware_mini.msg import Path, DetectedObjectArray
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import PointCloud2
from helpers.geometry import get_heading_from_vector, get_angle_between_two_headings
from helpers.collision import CollisionPoints
from helpers.path import PathWrapper

class TrajectoryCollisionChecker:

    def __init__(self):

        # parameters
        self.stopping_lateral_distance = rospy.get_param("stopping_lateral_distance")
        self.braking_safety_distance_obstacle = rospy.get_param("~braking_safety_distance_obstacle")
        self.heading_alignment_limit = rospy.get_param("~heading_alignment_limit")
        self.use_object_width = rospy.get_param("use_object_width")
        self.stopped_speed_limit = rospy.get_param("stopped_speed_limit")

        # variables
        self.detected_objects = None

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

        if detected_objects is None:
            rospy.logwarn_throttle(3, "%s - detected objects not received!", rospy.get_name())
            return

        collision_points = CollisionPoints()

        local_path = PathWrapper(msg.waypoints)
        local_path_buffer = local_path.linestring.buffer(self.stopping_lateral_distance, cap_style="flat")
        shapely.prepare(local_path_buffer)


        if current_velocity > self.stopped_speed_limit:
            for obj in detected_objects:

                if len(obj.candidate_trajectories.paths) > 0:
                    for path in obj.candidate_trajectories.paths:

                        trajectory_to_check = PathWrapper(path.waypoints).linestring

                        if self.use_object_width:
                            trajectory_to_check = trajectory_to_check.buffer(path.waypoints[0].left_width, cap_style="flat")

                        if local_path_buffer.intersects(trajectory_to_check):
                            trajectory_intersection_result = trajectory_to_check.intersection(local_path_buffer)
                            trajectory_intersection_points = shapely.get_coordinates(trajectory_intersection_result)
                            trajectory_intersection_distance = min([local_path.linestring.project(shapely.Point(x, y)) for x, y in trajectory_intersection_points])

                            object_current_location = shapely.Point(obj.position.x, obj.position.y)
                            object_distance_from_local_path_start = local_path.linestring.project(object_current_location)

                            # Ignore trajectories from behind
                            if math.isclose(trajectory_intersection_distance, 0.0, abs_tol=0.001) and math.isclose(object_distance_from_local_path_start, 0.0, abs_tol=0.001):
                                continue

                            object_polygon = shapely.Polygon([(p.x, p.y) for p in obj.convex_hull.points])
                            object_current_heading = get_heading_from_vector(obj.velocity)
                            object_local_path_heading = local_path.get_heading_at_distance(object_distance_from_local_path_start)
                            heading_difference = math.degrees(get_angle_between_two_headings(object_current_heading, object_local_path_heading))

                            # Ignore object trajectories that are on our path and with similar heading - must be in front of us
                            if local_path_buffer.intersects(object_polygon) and heading_difference < self.heading_alignment_limit:
                                continue

                            if heading_difference < self.heading_alignment_limit:
                                # objects with similar heading - add collision points with object's velocity
                                collision_points.add_intersection_points(trajectory_intersection_points,
                                                                        z = obj.position.z,
                                                                        vx = obj.velocity.x,
                                                                        vy = obj.velocity.y,
                                                                        vz = obj.velocity.z,
                                                                        distance_to_stop = self.braking_safety_distance_obstacle,
                                                                        category = CollisionPoints.MERGING_TRAJECTORY)
                            else:
                                # objects intersecting at angle, add with 0 velocity
                                collision_points.add_intersection_points(trajectory_intersection_points,
                                        z = obj.position.z,
                                        vx = 0,
                                        vy = 0,
                                        vz = 0,
                                        distance_to_stop = self.braking_safety_distance_obstacle,
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