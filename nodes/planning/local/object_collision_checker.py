#!/usr/bin/env python3

import rospy
import shapely
from autoware_msgs.msg import Lane, DetectedObjectArray
from sensor_msgs.msg import PointCloud2
from helpers.geometry import get_vector_norm_3d
from helpers.collision import CollisionPoints

class ObjectCollisionChecker:

    def __init__(self):

        # parameters
        self.stopping_lateral_distance = rospy.get_param("stopping_lateral_distance")
        self.stopping_speed_limit = rospy.get_param("stopping_speed_limit")
        self.braking_safety_distance_obstacle = rospy.get_param("~braking_safety_distance_obstacle")

        # variables
        self.detected_objects = None

        # publishers
        self.local_path_collision_pub = rospy.Publisher('object_collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('extracted_local_path', Lane, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/tracked_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)

    def detected_objects_callback(self, msg):
        self.detected_objects = msg.objects

    def path_callback(self, msg):

        detected_objects = self.detected_objects
        collision_points = CollisionPoints()

        if detected_objects is None:
            rospy.logwarn_throttle(3, "%s - detected objects not received!", rospy.get_name())
            return

        if len(msg.waypoints) > 0 and len(detected_objects) > 0:
            local_path_linestring = shapely.LineString([(waypoint.pose.pose.position.x, waypoint.pose.pose.position.y) for waypoint in msg.waypoints])

            # create buffer around local path
            local_path_buffer = local_path_linestring.buffer(self.stopping_lateral_distance, cap_style="flat")
            shapely.prepare(local_path_buffer)

            for object in detected_objects:
                # get the convex hulls and store as shapely polygons
                object_polygon = shapely.Polygon([(p.x, p.y) for p in object.convex_hull.polygon.points])

                if local_path_buffer.intersects(object_polygon):
                    intersection_result = object_polygon.intersection(local_path_buffer)
                    intersection_points = shapely.get_coordinates(intersection_result)
                    object_speed = get_vector_norm_3d(object.velocity.linear)

                    collision_points.add_intersection_points(intersection_points,
                                                            z = object.pose.position.z,
                                                            vx = object.velocity.linear.x,
                                                            vy = object.velocity.linear.y,
                                                            vz = object.velocity.linear.z,
                                                            distance_to_stop = self.braking_safety_distance_obstacle,
                                                            category = CollisionPoints.STOPPED_OBSTACLE_ON_PATH if object_speed < self.stopping_speed_limit else CollisionPoints.MOVING_OBSTACLE_ON_PATH)

        collision_points_msg = collision_points.create_message()
        collision_points_msg.header = msg.header
        self.local_path_collision_pub.publish(collision_points_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('object_collision_checker')
    node = ObjectCollisionChecker()
    node.run()