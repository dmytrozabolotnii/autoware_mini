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
from helpers.lanelet2 import load_lanelet2_map, get_crosswalks
from helpers.shapely import get_polygon_width
from helpers.path import PathWrapper

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

        global_path_linestring = shapely.LineString([(waypoint.position.x, waypoint.position.y) for waypoint in msg.waypoints])
        global_path_linestring = global_path_linestring.simplify(0.01)
        shapely.prepare(global_path_linestring)

        crosswalks_on_global_path = []
        for crosswalk in self.crosswalks:
            if crosswalk['polygon'].intersects(global_path_linestring):
                crosswalk['intersection_points'] = shapely.get_coordinates(global_path_linestring.intersection(crosswalk['polygon']))
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
            local_path_linestring = shapely.LineString([(waypoint.position.x, waypoint.position.y) for waypoint in msg.waypoints])
            shapely.prepare(local_path_linestring)
            local_path_buffer = local_path_linestring.buffer(self.stopping_lateral_distance, cap_style="flat")
            shapely.prepare(local_path_buffer)

            # get the transform 'car_front' location point(0,0,0) to the map frame
            try:
                transform = self.tf_buffer.lookup_transform(msg.header.frame_id, "car_front", msg.header.stamp, rospy.Duration(0.06))
            except (TransformException, rospy.ROSTimeMovedBackwardsException) as e:
                rospy.logwarn("%s - %s", rospy.get_name(), e)
                return
            car_front = shapely.Point(transform.transform.translation.x, transform.transform.translation.y)
            car_front_distance_from_path_start = local_path_linestring.project(car_front)
            linestring_up_to_car_front = shapely.ops.substring(local_path_linestring, 0, car_front_distance_from_path_start)

            # extract crosswalks that intersect with local path
            crosswalks_on_local_path = []
            for crosswalk in crosswalks_on_global_path:
                if crosswalk['polygon'].intersects(local_path_linestring) and not crosswalk['polygon'].intersects(linestring_up_to_car_front):
                    crosswalks_on_local_path.append(crosswalk)

            if len(crosswalks_on_local_path) > 0:
                for obj in detected_objects:
                    object_speed = get_vector_norm_3d(obj.velocity)
                    # ignore objects that are not moving
                    if object_speed < self.stopped_speed_limit:
                        continue
                    object_centroid = shapely.Point(obj.position.x, obj.position.y)
                    object_distance_from_local_path_start = local_path_linestring.project(object_centroid)
                    # ignore objects behind the ego vehicle
                    if math.isclose(object_distance_from_local_path_start, 0.0, abs_tol=0.001):
                        continue
                    object_polygon = shapely.Polygon([(p.x, p.y) for p in obj.convex_hull.points])
                    object_heading = get_heading_from_vector(obj.velocity)
                    object_projection_on_path = local_path_linestring.interpolate(object_distance_from_local_path_start)
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
                                trajectory = PathWrapper(path.waypoints)
                                trajectory_to_check = trajectory.linestring

                                if self.use_object_width:
                                    trajectory_to_check = trajectory.linestring.buffer(object_width / 2, cap_style="flat")
                                    shapely.prepare(trajectory_to_check)

                                if trajectory_to_check.intersects(crosswalk['polygon']):
                                    # find closest point along the object's trajectory to the crosswalk and get the heading from there!
                                    intersection_points = shapely.get_coordinates(trajectory_to_check.intersection(crosswalk['polygon']))
                                    closest_point_to_object = min([trajectory.linestring.project(shapely.Point(x, y)) for x, y in intersection_points])
                                    trajectory_heading = trajectory.get_heading_at_distance(closest_point_to_object)
                                    if math.degrees(get_minimum_angle_between_two_lines(crosswalk['heading'], trajectory_heading)) < self.crossing_angle_max_limit:
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
                'polygon': polygon,
                'heading': math.atan2(crosswalk.centerline[-1].y - crosswalk.centerline[0].y, crosswalk.centerline[-1].x - crosswalk.centerline[0].x),
            })
        return crosswalks_out

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('pedestrian_crosswalk_checker')
    node = PedestrianCrosswalkChecker()
    node.run()