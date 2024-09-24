#!/usr/bin/env python3

import math
import rospy
from shapely.geometry import Polygon, LineString, Point as ShapelyPoint
from shapely import prepare
from autoware_msgs.msg import Lane, DetectedObjectArray
from sensor_msgs.msg import PointCloud2
from helpers.geometry import get_vector_norm_3d, get_heading_from_vector, get_heading_between_two_points, get_angle_between_two_headings
from helpers.collision import CollisionPoints
from helpers.lanelet2 import load_lanelet2_map, get_crosswalks
from helpers.shapely import convert_to_shapely_points_list, get_polygon_width

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

        # load lanelet2 map
        lanelet2_map = load_lanelet2_map(lanelet2_map_name, coordinate_transformer, use_custom_origin, utm_origin_lat, utm_origin_lon)
        self.crosswalks = self.prepare_crosswalks(get_crosswalks(lanelet2_map))

        # publishers
        self.crosswalk_collision_pub = rospy.Publisher('crosswalk_collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('extracted_local_path', Lane, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)

    def detected_objects_callback(self, msg):
        self.detected_objects = msg.objects

    def path_callback(self, msg):
        detected_objects = self.detected_objects

        if detected_objects is None:
            rospy.logwarn_throttle(3, "%s - detected objects are not received!", rospy.get_name())
            return

        collision_points = CollisionPoints()
        if len(msg.waypoints) > 0 and len(self.crosswalks) > 0 and len(detected_objects) > 0:

            local_path_linestring = LineString([(waypoint.pose.pose.position.x, waypoint.pose.pose.position.y) for waypoint in msg.waypoints])
            prepare(local_path_linestring)

            # extract crosswalks that intersect with local path
            crosswalks_on_local_path = []
            for crosswalk_id, crosswalk in self.crosswalks.items():
                if crosswalk['polygon'].intersects(local_path_linestring):
                    crosswalks_on_local_path.append(crosswalk_id)

            if len(crosswalks_on_local_path) > 0:
                for object in detected_objects:
                    object_polygon = Polygon([(p.x, p.y) for p in object.convex_hull.polygon.points])
                    object_centroid = ShapelyPoint(object.pose.position.x, object.pose.position.y)
                    object_speed = get_vector_norm_3d(object.velocity.linear)
                    object_heading = get_heading_from_vector(object.velocity.linear)
                    object_width = get_polygon_width(object_polygon, object_heading)
                    object_projection_on_path = local_path_linestring.interpolate(local_path_linestring.project(object_centroid))
                    object_projection_on_path_heading = get_heading_between_two_points(object_centroid, object_projection_on_path)
                    object_path_approach_angle = math.degrees(get_angle_between_two_headings(object_heading, object_projection_on_path_heading))

                    for crosswalk_id in crosswalks_on_local_path[:]:
                        crosswalk_polygon = self.crosswalks[crosswalk_id]['polygon']

                        # INTERSECTING OBJECTS
                        if object_polygon.intersects(crosswalk_polygon):
                            if object_speed < self.stopping_speed_limit or object_path_approach_angle < self.crossing_angle_max_limit:
                                collision_points.add_intersection_points(self.crosswalks[crosswalk_id]['points'], z=object.pose.position.z, vx=0, vy=0, vz=0, distance_to_stop=self.braking_safety_distance_crosswalk, category=CollisionPoints.OBJECT_ON_CROSSWALK)
                                crosswalks_on_local_path.remove(crosswalk_id)
                                if object_speed < self.stopping_speed_limit:
                                    break  # Stop checking other crosswalks for this object
                        # NON-INTERSECTING OBJECTS - CONSIDER TRAJECTORIES
                        else:
                            # consider them crossing if they are within the crossing angle limit
                            if object_speed >= self.stopping_speed_limit and object_path_approach_angle < self.crossing_angle_max_limit:
                                for lane in object.candidate_trajectories.lanes:
                                    trajectory = LineString([(wp.pose.pose.position.x, wp.pose.pose.position.y) for wp in lane.waypoints])
                                    trajectory_buffer = trajectory.buffer(object_width / 2, cap_style="flat")
                                    prepare(trajectory_buffer)
                                    if trajectory_buffer.intersects(crosswalk_polygon):
                                        collision_points.add_intersection_points(self.crosswalks[crosswalk_id]['points'], z=object.pose.position.z, vx=0, vy=0, vz=0, distance_to_stop=self.braking_safety_distance_crosswalk, category=CollisionPoints.TRAJECTORY_ON_CROSSWALK)
                                        crosswalks_on_local_path.remove(crosswalk_id)
                                        break

                    # Exit early if all crosswalks are processed
                    if len(crosswalks_on_local_path) == 0:
                        break

        collision_points_msg = collision_points.create_message()
        collision_points_msg.header = msg.header
        self.crosswalk_collision_pub.publish(collision_points_msg)

    def prepare_crosswalks(self, crosswalks_in):
        crosswalks_out = {}
        for crosswalk in crosswalks_in:
            
            polygon = Polygon([(p.x, p.y) for p in crosswalk.polygon2d()])
            prepare(polygon)

            crosswalks_out[crosswalk.id] = {
                'polygon': polygon,
                'heading': math.atan2(crosswalk.centerline[-1].y - crosswalk.centerline[0].y, crosswalk.centerline[-1].x - crosswalk.centerline[0].x),
                'points': convert_to_shapely_points_list(polygon)
            }
        return crosswalks_out

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('pedestrian_crosswalk_checker')
    node = PedestrianCrosswalkChecker()
    node.run()