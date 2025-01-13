#!/usr/bin/env python3

import math
import rospy
import shapely
import shapely.ops
from autoware_mini.msg import Path, DetectedObjectArray
from sensor_msgs.msg import PointCloud2
from tf2_ros import TransformListener, Buffer, TransformException
from helpers.geometry import get_vector_norm_3d, get_heading_from_vector, get_heading_between_two_points, get_angle_between_two_headings, get_minimum_angle_between_two_lines
from helpers.collision import CollisionPoints
from helpers.path import PathWrapper
from helpers.lanelet2 import load_lanelet2_map, get_crosswalks
from helpers.shapely import get_polygon_width

class PedestrianCrosswalkChecker:

    def __init__(self):

        # parameters
        self.stopping_lateral_distance = rospy.get_param("stopping_lateral_distance")
        self.stopped_speed_limit = rospy.get_param("stopped_speed_limit")
        self.braking_safety_distance_crosswalk = rospy.get_param("~braking_safety_distance_crosswalk")
        self.crossing_angle_max_limit = rospy.get_param("~crossing_angle_max_limit")
        self.use_object_width = rospy.get_param("use_object_width")
        lanelet2_map_name = rospy.get_param("~lanelet2_map_name")

        # variables
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        self.detected_objects = None
        self.crosswalks_on_global_path = None

        # load lanelet2 map
        lanelet2_map = load_lanelet2_map(lanelet2_map_name)
        self.crosswalks = self.prepare_crosswalks(get_crosswalks(lanelet2_map))

        # publishers
        self.crosswalk_collision_pub = rospy.Publisher('crosswalk_collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('global_path', Path, self.global_path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('extracted_local_path', Path, self.local_path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/predicted_objects', DetectedObjectArray, self.predicted_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)

    def predicted_objects_callback(self, msg):
        self.detected_objects = msg.objects

    def global_path_callback(self, msg):
        global_path = PathWrapper(msg.waypoints, distances=False)
        global_path.linestring = global_path.linestring.simplify(0.01)
        shapely.prepare(global_path.linestring)

        crosswalks_on_global_path = []
        for crosswalk in self.crosswalks:
            if crosswalk['polygon'].intersects(global_path.linestring):
                crosswalk['intersection_points'] = shapely.get_coordinates(global_path.linestring.intersection(crosswalk['polygon']))
                crosswalks_on_global_path.append(crosswalk)

        self.crosswalks_on_global_path = crosswalks_on_global_path

    def local_path_callback(self, msg):
        detected_objects = self.detected_objects
        crosswalks_on_global_path = self.crosswalks_on_global_path

        if crosswalks_on_global_path is None:
            rospy.logwarn_throttle(3, "%s - global path not received!", rospy.get_name())
            return

        if detected_objects is None:
            rospy.logwarn_throttle(3, "%s - detected objects not received!", rospy.get_name())
            return

        collision_points = CollisionPoints()
        if len(msg.waypoints) > 0 and len(self.crosswalks) > 0 and len(detected_objects) > 0:
            local_path = PathWrapper(msg.waypoints, distances=False)
            local_path_buffer = local_path.linestring.buffer(self.stopping_lateral_distance, cap_style="flat")
            shapely.prepare(local_path_buffer)

            # get the car_front and projct to local_path
            try:
                transform = self.tf_buffer.lookup_transform(msg.header.frame_id, "car_front", msg.header.stamp, rospy.Duration(0.06))
            except (TransformException, rospy.ROSTimeMovedBackwardsException) as e:
                rospy.logwarn("%s - %s", rospy.get_name(), e)
                return
            car_front = shapely.Point(transform.transform.translation.x, transform.transform.translation.y)
            car_front_distance_from_path_start = local_path.linestring.project(car_front)
            local_path_from_car_front = shapely.ops.substring(local_path.linestring, car_front_distance_from_path_start, local_path.linestring.length)

            # extract crosswalks that ego vehicle has not reached yet and that intersect with local path
            crosswalks_on_local_path = []
            for crosswalk in crosswalks_on_global_path:
                if crosswalk['polygon'].intersects(local_path_from_car_front):
                    crosswalks_on_local_path.append(crosswalk)

            if len(crosswalks_on_local_path) > 0:
                for obj in detected_objects:
                    object_speed = get_vector_norm_3d(obj.velocity)
                    # ignore objects that are not moving
                    if object_speed < self.stopped_speed_limit:
                        continue
                    object_centroid = shapely.Point(obj.position.x, obj.position.y)
                    object_distance_from_local_path_start = local_path.linestring.project(object_centroid)
                    # ignore objects behind the ego vehicle
                    if math.isclose(object_distance_from_local_path_start, 0.0, abs_tol=0.001):
                        continue
                    object_polygon = shapely.Polygon([(p.x, p.y) for p in obj.convex_hull.points])
                    object_heading = get_heading_from_vector(obj.velocity)
                    object_projection_on_path = local_path.linestring.interpolate(object_distance_from_local_path_start)
                    object_projection_on_path_heading = get_heading_between_two_points(object_centroid, object_projection_on_path)
                    object_path_approach_angle = math.degrees(get_angle_between_two_headings(object_heading, object_projection_on_path_heading))
                    if self.use_object_width:
                        object_width = get_polygon_width(object_polygon, object_heading)

                    for crosswalk in crosswalks_on_local_path[:]:

                        # INTERSECTING OBJECTS
                        if crosswalk['polygon'].intersects(object_polygon):
                            # objects on crosswalk approaching local path or have crossed it and departing, but still within the local path buffer
                            if object_path_approach_angle < self.crossing_angle_max_limit or \
                                (180 - object_path_approach_angle < self.crossing_angle_max_limit and local_path_buffer.intersects(object_polygon)):
                                collision_points.add_intersection_points(crosswalk['intersection_points'], z=obj.position.z, vx=0, vy=0, vz=0, distance_to_stop=self.braking_safety_distance_crosswalk, category=CollisionPoints.OBJECT_ON_CROSSWALK)
                                crosswalks_on_local_path.remove(crosswalk)

                        # NON-INTERSECTING OBJECTS - CONSIDER TRAJECTORIES
                        elif len(obj.candidate_trajectories.paths) > 0:
                            for path in obj.candidate_trajectories.paths:
                                trajectory = PathWrapper(path.waypoints, distances=False)
                                trajectory_to_check = trajectory.linestring

                                if self.use_object_width:
                                    trajectory_to_check = trajectory.linestring.buffer(object_width / 2, cap_style="flat")

                                if crosswalk['polygon'].intersects(trajectory_to_check):
                                    intersection_points = shapely.get_coordinates(crosswalk['polygon'].intersection(trajectory_to_check))
                                    closest_distance_to_object = float('inf')
                                    closest_intersection_point = None
                                    for p in intersection_points:
                                        distance = trajectory.linestring.project(shapely.Point(p))
                                        if distance < closest_distance_to_object:
                                            closest_distance_to_object = distance
                                            closest_intersection_point = p

                                    closest_intersection_point = shapely.Point(closest_intersection_point)
                                    trajectory_heading_at_closest_intersection = trajectory.get_heading_at_distance(closest_distance_to_object)
                                    # find heading from the closest intersection point to its projection on local_path
                                    closest_intersection_distance_from_local_path_start = local_path.linestring.project(closest_intersection_point)
                                    closest_intersection_on_path = local_path.linestring.interpolate(closest_intersection_distance_from_local_path_start)
                                    closest_intersection_on_path_heading = get_heading_between_two_points(closest_intersection_point, closest_intersection_on_path)
                                    closest_intersection_path_min_angle = math.degrees(get_minimum_angle_between_two_lines(trajectory_heading_at_closest_intersection, closest_intersection_on_path_heading))

                                    if closest_intersection_path_min_angle < self.crossing_angle_max_limit:
                                        collision_points.add_intersection_points(crosswalk['intersection_points'], z=obj.position.z, vx=0, vy=0, vz=0, distance_to_stop=self.braking_safety_distance_crosswalk, category=CollisionPoints.TRAJECTORY_ON_CROSSWALK)
                                        crosswalks_on_local_path.remove(crosswalk)
                                        break

                    # Exit early if all crosswalks are processed
                    if len(crosswalks_on_local_path) == 0:
                        break

        collision_points_msg = collision_points.create_message()
        collision_points_msg.header = msg.header
        self.crosswalk_collision_pub.publish(collision_points_msg)

    def prepare_crosswalks(self, crosswalks_in):
        crosswalks_out = []
        for crosswalk in crosswalks_in:
            
            polygon = shapely.Polygon([(p.x, p.y) for p in crosswalk.polygon2d()])
            shapely.prepare(polygon)

            crosswalks_out.append({
                'polygon': polygon
            })
        return crosswalks_out

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('pedestrian_crosswalk_checker')
    node = PedestrianCrosswalkChecker()
    node.run()