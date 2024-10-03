#!/usr/bin/env python3

import math
import rospy
from shapely.geometry import Polygon, LineString, Point as ShapelyPoint
from shapely import prepare
from autoware_msgs.msg import Lane, DetectedObjectArray
from sensor_msgs.msg import PointCloud2
from helpers.geometry import get_vector_norm_3d, get_heading_from_vector, get_heading_between_two_points, get_angle_between_two_headings, get_minimum_angle_between_two_lines
from helpers.collision import CollisionPoints
from helpers.lanelet2 import load_lanelet2_map, get_crosswalks
from helpers.shapely import convert_to_shapely_points_list, get_polygon_width
from helpers.path import Path

class PedestrianCrosswalkChecker:

    def __init__(self):

        # parameters
        self.stopping_lateral_distance = rospy.get_param("stopping_lateral_distance")
        self.stopping_speed_limit = rospy.get_param("stopping_speed_limit")
        self.braking_safety_distance_crosswalk = rospy.get_param("~braking_safety_distance_crosswalk")
        self.crossing_angle_max_limit = rospy.get_param("~crossing_angle_max_limit")
        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")
        lanelet2_map_name = rospy.get_param("~lanelet2_map_name")

        # variables
        self.detected_objects = None
        self.crosswalks_on_global_path = None

        # load lanelet2 map
        lanelet2_map = load_lanelet2_map(lanelet2_map_name, coordinate_transformer, use_custom_origin, utm_origin_lat, utm_origin_lon)
        self.crosswalks = self.prepare_crosswalks(get_crosswalks(lanelet2_map))

        # publishers
        self.crosswalk_collision_pub = rospy.Publisher('crosswalk_collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('lanelet2_global_path', Lane, self.global_path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('extracted_local_path', Lane, self.local_path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)

    def detected_objects_callback(self, msg):
        self.detected_objects = msg.objects

    def global_path_callback(self, msg):

        global_path_linestring = LineString([(waypoint.pose.pose.position.x, waypoint.pose.pose.position.y) for waypoint in msg.waypoints])
        prepare(global_path_linestring)

        crosswalks_on_global_path = []
        for crosswalk in self.crosswalks:
            if crosswalk['polygon'].intersects(global_path_linestring):
                crosswalk['intersection_points'] = convert_to_shapely_points_list(global_path_linestring.intersection(crosswalk['polygon']))
                crosswalks_on_global_path.append(crosswalk)

        self.crosswalks_on_global_path = crosswalks_on_global_path

    def local_path_callback(self, msg):
        detected_objects = self.detected_objects
        crosswalks_on_global_path = self.crosswalks_on_global_path

        if detected_objects is None or crosswalks_on_global_path is None:
            rospy.logwarn_throttle(3, "%s - detected objects or crosswalks are not received!", rospy.get_name())
            return

        collision_points = CollisionPoints()
        if len(msg.waypoints) > 0 and len(self.crosswalks) > 0 and len(detected_objects) > 0:

            local_path_linestring = LineString([(waypoint.pose.pose.position.x, waypoint.pose.pose.position.y) for waypoint in msg.waypoints])
            prepare(local_path_linestring)
            local_path_buffer = local_path_linestring.buffer(self.stopping_lateral_distance, cap_style="flat")
            prepare(local_path_buffer)

            # extract crosswalks that intersect with local path
            crosswalks_on_local_path = []
            for crosswalk in crosswalks_on_global_path:
                if crosswalk['polygon'].intersects(local_path_linestring):
                    crosswalks_on_local_path.append(crosswalk)

            if len(crosswalks_on_local_path) > 0:
                for obj in detected_objects:
                    object_centroid = ShapelyPoint(obj.pose.position.x, obj.pose.position.y)
                    object_distance_from_local_path_start = local_path_linestring.project(object_centroid)
                    # ignore objects behind the ego vehicle
                    if math.isclose(object_distance_from_local_path_start, 0.0):
                        continue
                    object_speed = get_vector_norm_3d(obj.velocity.linear)
                    object_polygon = Polygon([(p.x, p.y) for p in obj.convex_hull.polygon.points])
                    object_heading = get_heading_from_vector(obj.velocity.linear)
                    object_projection_on_path = local_path_linestring.interpolate(object_distance_from_local_path_start)
                    object_projection_on_path_heading = get_heading_between_two_points(object_centroid, object_projection_on_path)
                    object_path_approach_angle = math.degrees(get_angle_between_two_headings(object_heading, object_projection_on_path_heading))

                    for crosswalk in crosswalks_on_local_path[:]:

                        # INTERSECTING OBJECTS
                        if object_polygon.intersects(crosswalk['polygon']):
                            if object_speed < self.stopping_speed_limit or object_path_approach_angle < self.crossing_angle_max_limit or \
                                (180 - object_path_approach_angle < self.crossing_angle_max_limit and local_path_buffer.intersects(object_polygon)):
                                collision_points.add_intersection_points(crosswalk['intersection_points'], z=obj.pose.position.z, vx=0, vy=0, vz=0, distance_to_stop=self.braking_safety_distance_crosswalk, category=CollisionPoints.OBJECT_ON_CROSSWALK)
                                crosswalks_on_local_path.remove(crosswalk)
                                if object_speed < self.stopping_speed_limit:
                                    break  # Stop checking other crosswalks for this object
                        # NON-INTERSECTING OBJECTS - CONSIDER TRAJECTORIES
                        else:
                            object_width = get_polygon_width(object_polygon, object_heading)
                            for lane in obj.candidate_trajectories.lanes:
                                trajectory = Path(lane.waypoints)
                                trajectory_buffer = trajectory.linestring.buffer(object_width / 2, cap_style="flat")
                                prepare(trajectory_buffer)
                                if trajectory_buffer.intersects(crosswalk['polygon']):
                                    # find closest point along the object'ss trajectory to the crosswalk and get the heading from there!
                                    intersection_points = convert_to_shapely_points_list(trajectory_buffer.intersection(crosswalk['polygon']))
                                    closest_point_to_object = min([trajectory.linestring.project(point) for point in intersection_points])
                                    trajectory_heading = trajectory.get_heading_at_distance(closest_point_to_object)
                                    if math.degrees(get_minimum_angle_between_two_lines(crosswalk['heading'], trajectory_heading)) < self.crossing_angle_max_limit:
                                        collision_points.add_intersection_points(crosswalk['intersection_points'], z=obj.pose.position.z, vx=0, vy=0, vz=0, distance_to_stop=self.braking_safety_distance_crosswalk, category=CollisionPoints.TRAJECTORY_ON_CROSSWALK)
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
            
            polygon = Polygon([(p.x, p.y) for p in crosswalk.polygon2d()])
            prepare(polygon)

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