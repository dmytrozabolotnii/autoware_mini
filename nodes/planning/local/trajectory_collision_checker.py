#!/usr/bin/env python3

import rospy
import math
import shapely
import numpy as np
from autoware_msgs.msg import Lane, DetectedObjectArray
from sensor_msgs.msg import PointCloud2
from helpers.geometry import get_heading_from_vector, get_angle_between_two_headings
from helpers.collision import CollisionPoints
from helpers.lanelet2 import load_lanelet2_map, get_stop_lines_using_subtype
from helpers.path import Path

class TrajectoryCollisionChecker:

    def __init__(self):

        # parameters
        self.stopping_lateral_distance = rospy.get_param("stopping_lateral_distance")
        self.stopping_speed_limit = rospy.get_param("stopping_speed_limit")
        self.braking_safety_distance_yield = rospy.get_param("~braking_safety_distance_yield")
        self.braking_safety_distance_obstacle = rospy.get_param("~braking_safety_distance_obstacle")
        self.yielding_distance_limit = rospy.get_param("~yielding_distance_limit")
        self.heading_alignment_limit = rospy.get_param("~heading_alignment_limit")
        lanelet2_map_name = rospy.get_param("~lanelet2_map_name")

        # variables
        self.detected_objects = None
        self.yield_lines_on_global_path = None

        lanelet2_map = load_lanelet2_map(lanelet2_map_name)
        self.yield_lines = get_stop_lines_using_subtype(lanelet2_map, subtype=["yield", "yield_stop"])

        # publishers
        self.local_path_collision_pub = rospy.Publisher('trajectory_collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('extracted_local_path', Lane, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('global_path', Lane, self.global_path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/predicted_objects_map', DetectedObjectArray, self.predicted_objects_map_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)

    def predicted_objects_map_callback(self, msg):
        self.detected_objects = msg.objects

    def global_path_callback(self, msg):
        global_path_linestring = shapely.LineString([(waypoint.pose.pose.position.x, waypoint.pose.pose.position.y) for waypoint in msg.waypoints])
        global_path_linestring = global_path_linestring.simplify(0.01)
        shapely.prepare(global_path_linestring)

        yield_lines_on_global_path = []
        for id, yield_line in self.yield_lines.items():
            if yield_line.intersects(global_path_linestring):
                yield_lines_on_global_path.append(yield_line)

        self.yield_lines_on_global_path = yield_lines_on_global_path

    def path_callback(self, msg):

        detected_objects = self.detected_objects

        if detected_objects is None:
            rospy.logwarn_throttle(3, "%s - detected objects not received!", rospy.get_name())
            return

        collision_points = CollisionPoints()

        if len(msg.waypoints) > 0 and len(detected_objects) > 0:
            local_path = Path(msg.waypoints)
            local_path_buffer = local_path.linestring.buffer(self.stopping_lateral_distance, cap_style="flat")
            shapely.prepare(local_path_buffer)

            # find if there are any yiled_lines on local_path and select the closest one
            yield_line_distance = np.inf
            yield_line_point = None
            for yield_line in self.yield_lines_on_global_path:
                if yield_line.intersects(local_path.linestring):
                    yield_line_intersection_result = yield_line.intersection(local_path.linestring)
                    assert isinstance(yield_line_intersection_result, shapely.geometry.Point), "local_path and yield_line intersection is not shapely Point!"
                    distance = local_path.linestring.project(yield_line_intersection_result)
                    if distance < yield_line_distance:
                        yield_line_distance = distance
                        yield_line_point = yield_line_intersection_result

            for obj in detected_objects:
                if len(obj.candidate_trajectories.lanes) > 0:

                    for trajectory in obj.candidate_trajectories.lanes:
                        trajectory_linestring = shapely.LineString([(p.pose.pose.position.x, p.pose.pose.position.y, p.pose.pose.position.z) for p in trajectory.waypoints])
                        shapely.prepare(trajectory_linestring)

                        if local_path_buffer.intersects(trajectory_linestring):
                            trajectory_intersection_result = trajectory_linestring.intersection(local_path_buffer)
                            trajectory_intersection_points = shapely.get_coordinates(trajectory_intersection_result)
                            trajectory_intersection_distance = min([local_path.linestring.project(shapely.Point(x, y)) for x, y in trajectory_intersection_points])

                            object_current_heading = get_heading_from_vector(obj.velocity.linear)
                            object_current_location = shapely.Point(obj.pose.position.x, obj.pose.position.y)
                            object_distance_from_local_path_start = local_path.linestring.project(object_current_location)
                            object_local_path_heading = local_path.get_heading_at_distance(object_distance_from_local_path_start)
                            heading_difference = math.degrees(get_angle_between_two_headings(object_current_heading, object_local_path_heading))

                            # HACK to ignore trajectories from behind
                            if math.isclose(trajectory_intersection_distance, 0.0, abs_tol=0.001):
                                continue

                            # 1. CHECK YIELDING
                            #    - trajectory_intersection after yiled line within 40m
                            #    - ignore objects that align with the local_path (for example car in front)
                            if yield_line_point is not None \
                                and yield_line_distance < trajectory_intersection_distance and trajectory_intersection_distance - yield_line_distance < self.yielding_distance_limit \
                                and heading_difference > self.heading_alignment_limit:

                                collision_points.add_point(x = yield_line_point.x,
                                                           y = yield_line_point.y,
                                                           z = obj.pose.position.z,
                                                           vx = 0.0,
                                                           vy = 0.0, 
                                                           vz = 0.0,
                                                           distance_to_stop = self.braking_safety_distance_yield,
                                                           category = CollisionPoints.YIELDING_TRAJECTORY)
                                # do not check for colliding trajectory if yielding
                                continue

                            # 2. CHECK COLLISION only the ones that are not included for yielding
                            # TODO implement time based collision checking and HACK should be removed
                            # HACK ignore trajectories from object in front of ego having similar heading with local path, so that 0 velocity could be used
                            if heading_difference < self.heading_alignment_limit and local_path_buffer.intersects(object_current_location):
                                continue
                            collision_points.add_intersection_points(trajectory_intersection_points,
                                                                    z = obj.pose.position.z,
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