#!/usr/bin/env python3

import rospy
import math
import shapely
from autoware_msgs.msg import Lane, DetectedObjectArray
from sensor_msgs.msg import PointCloud2
from helpers.geometry import get_vector_norm_3d
from helpers.path import Path
from helpers.collision import CollisionPoints
from helpers.shapely import get_polygon_width

class TrajectoryCollisionChecker:

    def __init__(self):

        # parameters
        self.stopping_lateral_distance = rospy.get_param("stopping_lateral_distance")
        self.stopping_speed_limit = rospy.get_param("stopping_speed_limit")
        self.braking_safety_distance_obstacle = rospy.get_param("~braking_safety_distance_obstacle")

        # variables
        self.detected_objects = None

        # publishers
        self.local_path_collision_pub = rospy.Publisher('trajectory_collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('extracted_local_path', Lane, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/predicted_objects_map', DetectedObjectArray, self.detected_objects_map_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)

    def detected_objects_map_callback(self, msg):
        self.detected_objects = msg.objects

    def path_callback(self, msg):

        detected_objects = self.detected_objects
        collision_points = CollisionPoints()
        
        if detected_objects is None:
            rospy.logwarn_throttle(3, "%s - detected objects not received!", rospy.get_name())
            return

        if len(msg.waypoints) > 0 and len(detected_objects) > 0:
            local_path_linestring = shapely.LineString([(waypoint.pose.pose.position.x, waypoint.pose.pose.position.y) for waypoint in msg.waypoints])
            local_path_buffer = local_path_linestring.buffer(self.stopping_lateral_distance, cap_style="flat")
            shapely.prepare(local_path_buffer)

            for object in detected_objects:
                if len(object.candidate_trajectories.lanes) > 0:

                    for trajectory in object.candidate_trajectories.lanes:
                        trajectory_linestring = shapely.LineString([(p.pose.pose.position.x, p.pose.pose.position.y, p.pose.pose.position.z) for p in trajectory.waypoints])
                        shapely.prepare(trajectory_linestring)

                        if local_path_buffer.intersects(trajectory_linestring):
                            intersection_result = trajectory_linestring.intersection(local_path_buffer)
                            intersection_points = shapely.get_coordinates(intersection_result)

                            # TODO simple hack to ignore trajectories from behind
                            collision_distance = min([local_path_linestring.project(shapely.Point(x, y)) for x, y in intersection_points])
                            if math.isclose(collision_distance, 0.0, abs_tol=0.001):
                                continue

                            # TODO currently assigning to trajectory intersection points the speed vectors from the object!
                            collision_points.add_intersection_points(intersection_points,
                                                                    z = object.pose.position.z,
                                                                    vx = object.velocity.linear.x,
                                                                    vy = object.velocity.linear.y,
                                                                    vz = object.velocity.linear.z,
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