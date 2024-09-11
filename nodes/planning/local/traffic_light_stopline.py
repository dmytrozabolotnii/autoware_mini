#!/usr/bin/env python3

import rospy
from shapely.geometry import Point as ShapelyPoint
from autoware_msgs.msg import Lane, TrafficLightResultArray
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import PointCloud2
from helpers.path import Path
from helpers.collision import CollisionPoints, CAT_TRAFFIC_LIGHT_STOPLINE
from helpers.lanelet2 import load_lanelet2_map, get_stoplines

class TrafficLightStopline:

    def __init__(self):

        # parameters
        self.braking_safety_distance_stopline = rospy.get_param("~braking_safety_distance_stopline")
        self.tfl_force_stop_speed_limit = rospy.get_param("~tfl_force_stop_speed_limit")
        self.tfl_maximum_deceleration = rospy.get_param("~tfl_maximum_deceleration")
        self.current_pose_to_car_front = rospy.get_param("current_pose_to_car_front")
        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")
        lanelet2_map_name = rospy.get_param("~lanelet2_map_name")

        # variables
        self.stopline_statuses = {}
        self.current_position = None
        self.current_speed = None

        # publishers
        self.traffic_light_stopline_pub = rospy.Publisher('collision_tfl_stopline', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_velocity', TwistStamped, self.current_velocity_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('extracted_local_path', Lane, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/traffic_light_status', TrafficLightResultArray, self.traffic_light_status_callback, queue_size=1, tcp_nodelay=True)

        lanelet2_map = load_lanelet2_map(lanelet2_map_name, coordinate_transformer, use_custom_origin, utm_origin_lat, utm_origin_lon)
        self.all_stoplines = get_stoplines(lanelet2_map)

    def current_pose_callback(self, msg):
        self.current_position = ShapelyPoint(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

    def current_velocity_callback(self, msg):
        self.current_speed = msg.twist.linear.x

    def traffic_light_status_callback(self, msg):
        stopline_statuses = {}
        for result in msg.results:
            stopline_statuses[result.lane_id] = result.recognition_result
        
        self.stopline_statuses = stopline_statuses

    def path_callback(self, msg):

        current_position = self.current_position
        current_speed = self.current_speed

        if current_speed is None or current_position is None:
            rospy.logwarn_throttle(3, "%s - current speed or position not received!", rospy.get_name())
            return

        stopline_statuses = self.stopline_statuses
        collision_points = CollisionPoints()

        if len(msg.waypoints) > 0 and len(stopline_statuses) > 0:
            local_path = Path(msg.waypoints)
            ego_distance_from_local_path_start = local_path.linestring.project(current_position)

            for stopline_id, stopline_linestring in self.all_stoplines.items():
                # if RED and intersects with local path
                if stopline_id in stopline_statuses and stopline_statuses[stopline_id] == 0 and stopline_linestring.intersects(local_path.linestring):
                    intersection_point = local_path.linestring.intersection(stopline_linestring)
                    assert isinstance(intersection_point, ShapelyPoint), "Stop line and local path intersection point is not a ShapelyPoint"
                    
                    # check deceleration
                    distance_to_stopline = local_path.linestring.project(intersection_point)
                    distance_for_deceleration = distance_to_stopline - ego_distance_from_local_path_start - self.current_pose_to_car_front
                    deceleration = (current_speed**2) / (2 * distance_for_deceleration)
                    # base_link has not crossed the stopline and velocity is below tfl_force_stop_speed_limit or deceleration is less than maximum allowed deceleration
                    if distance_to_stopline > 0 and current_speed < self.tfl_force_stop_speed_limit / 3.6 or 0 <= deceleration <= self.tfl_maximum_deceleration:
                        x, y, z = intersection_point.x, intersection_point.y, intersection_point.z
                        collision_points.add_point(x, y, z, 0.0, 0.0, 0.0, self.braking_safety_distance_stopline, CAT_TRAFFIC_LIGHT_STOPLINE)

        collision_points_msg = collision_points.create_message()
        collision_points_msg.header = msg.header
        self.traffic_light_stopline_pub.publish(collision_points_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('traffic_light_stopline')
    node = TrafficLightStopline()
    node.run()