#!/usr/bin/env python3

import rospy
import numpy as np
from ros_numpy import msgify
from shapely.geometry import Point as ShapelyPoint
from autoware_msgs.msg import Lane, TrafficLightResultArray
from sensor_msgs.msg import PointCloud2
from helpers.path import Path
from helpers.lanelet2 import load_lanelet2_map, get_stoplines

class TrafficLightStopline:

    def __init__(self):

        # parameters
        self.braking_safety_distance_stopline = rospy.get_param("~braking_safety_distance_stopline")
        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")
        lanelet2_map_name = rospy.get_param("~lanelet2_map_name")

        # variables
        self.stopline_statuses = {}

        # publishers
        self.traffic_light_stopline_pub = rospy.Publisher('collision_tfl_stopline', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('extracted_local_path', Lane, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/traffic_light_status', TrafficLightResultArray, self.traffic_light_status_callback, queue_size=1, tcp_nodelay=True)

        lanelet2_map = load_lanelet2_map(lanelet2_map_name, coordinate_transformer, use_custom_origin, utm_origin_lat, utm_origin_lon)
        self.all_stoplines = get_stoplines(lanelet2_map)


    def traffic_light_status_callback(self, msg):
        stopline_statuses = {}
        for result in msg.results:
            stopline_statuses[result.lane_id] = result.recognition_result
        
        self.stopline_statuses = stopline_statuses

    def path_callback(self, msg):

        stopline_statuses = self.stopline_statuses

        # create empty numpy array for the stopline points
        dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('vx', np.float32),
            ('vy', np.float32),
            ('vz', np.float32),
            ('distance_to_stop', np.float32),
            ('category', np.int32)
        ])
        stopline_points = np.array([], dtype=dtype)


        if len(msg.waypoints) == 0 or len(stopline_statuses) == 0:
            collision_points = msgify(PointCloud2, stopline_points)
        else:
            local_path = Path(msg.waypoints)

            for stopline_id, stopline_linestring in self.all_stoplines.items():
                # if RED and intersects with local path
                if stopline_id in stopline_statuses and stopline_statuses[stopline_id] == 0 and stopline_linestring.intersects(local_path.linestring):
                    intersection_point = local_path.linestring.intersection(stopline_linestring)
                    assert isinstance(intersection_point, ShapelyPoint), "Stop line and local path intersection point is not a ShapelyPoint"
                    x, y, z = intersection_point.x, intersection_point.y, intersection_point.z

                    # append point to stopline_points as new row
                    # TODO last field category !?!?  1 - TFL stopline
                    stopline_point = np.array([(x, y, z, 0.0, 0.0, 0.0, self.braking_safety_distance_stopline, 1)], dtype=dtype)
                    stopline_points = np.append(stopline_points, stopline_point)

            collision_points = msgify(PointCloud2, stopline_points)

        collision_points.header = msg.header
        self.traffic_light_stopline_pub.publish(collision_points)


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('traffic_light_stopline')
    node = TrafficLightStopline()
    node.run()