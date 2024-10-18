#!/usr/bin/env python3

import rospy
import shapely
import numpy as np
from helpers.lanelet2 import load_lanelet2_map, get_stop_lines_using_subtype
from helpers.collision import CollisionPoints
from std_msgs.msg import Bool
from autoware_msgs.msg import Lane
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Empty, EmptyResponse

class AutomaticStopBehavior:

    def __init__(self):

        # parameters
        lanelet2_map_name = rospy.get_param("~lanelet2_map_name")
        self.braking_safety_distance_stop_line = rospy.get_param("~braking_safety_distance_stop_line")

        # variables
        self.stop_lines_on_global_path = None
        self.current_closest_stop_line_id = None
        self.remove_stop_line_id = None
        self.timer = rospy.Time.now()

        lanelet2_map = load_lanelet2_map(lanelet2_map_name)
        self.stop_lines = get_stop_lines_using_subtype(lanelet2_map, subtype=["yield_stop"])

        # publishers
        self.remove_stop_pub = rospy.Publisher('remove_stop', Bool, queue_size=1, tcp_nodelay=True)
        self.stop_line_collision_pub = rospy.Publisher('stop_line_collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('lanelet2_global_path', Lane, self.global_path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('extracted_local_path', Lane, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('remove_stop', Bool, self.remove_stop_callback, queue_size=1, tcp_nodelay=True)

        # Services
        rospy.Service('call_remove_stop', Empty, self.call_remove_stop_callback)


    def global_path_callback(self, msg):
        global_path_linestring = shapely.LineString([(waypoint.pose.pose.position.x, waypoint.pose.pose.position.y) for waypoint in msg.waypoints])
        shapely.prepare(global_path_linestring)

        stop_lines_on_global_path = {}
        for id, stop_line in self.stop_lines.items():
            if stop_line.intersects(global_path_linestring):
                stop_lines_on_global_path[id] = stop_line

        self.stop_lines_on_global_path = stop_lines_on_global_path

    def path_callback(self, msg):


        stop_lines_on_global_path = self.stop_lines_on_global_path
        collision_points = CollisionPoints()

        if stop_lines_on_global_path is None:
            return

        local_path_linestring = shapely.LineString([(waypoint.pose.pose.position.x, waypoint.pose.pose.position.y) for waypoint in msg.waypoints])
        shapely.prepare(local_path_linestring)

        stop_line_distance = np.inf
        stop_line_id = None

        for id, stop_line in stop_lines_on_global_path.items():
            if stop_line.intersects(local_path_linestring):
                stop_line_intersection_result = stop_line.intersection(local_path_linestring)
                # assert stop_line_intersection_result.geom_type == shapely.Point, "Stop line intersection with local_path is not a shapely Point"

                # if not "remove point" then add to collision points
                if id != self.remove_stop_line_id:
                    collision_points.add_point(x = stop_line_intersection_result.x,
                                                y = stop_line_intersection_result.y,
                                                z = 50,                     # TODO
                                                vx = 0.0,
                                                vy = 0.0, 
                                                vz = 0.0,
                                                distance_to_stop = self.braking_safety_distance_stop_line,
                                                category = CollisionPoints.STOP_LINE_FORCED_STOP)

                # find closest stop line
                distance = local_path_linestring.project(stop_line_intersection_result)
                if distance < stop_line_distance:
                    stop_line_distance = distance
                    stop_line_id = id

        self.current_closest_stop_line_id = stop_line_id

        # set remove_stop_line_id to None if it current closest stopline changes or the the timer has expired
        if self.remove_stop_line_id != None and (self.current_closest_stop_line_id != self.remove_stop_line_id or self.timer + rospy.Duration(10) < rospy.Time.now()):
            self.remove_stop_line_id = None

        collision_points_msg = collision_points.create_message()
        collision_points_msg.header = msg.header
        self.stop_line_collision_pub.publish(collision_points_msg)

    def remove_stop_callback(self, msg):
        # reset timer and set current stop line id as the one to be removed
        self.timer = rospy.Time.now()
        self.remove_stop_line_id = self.current_closest_stop_line_id
        rospy.loginfo("Removed forced stop for stopline id %s for 10 seconds", self.remove_stop_line_id)

    # service call to simulate button press from rviz
    def call_remove_stop_callback(self, msg):
        self.remove_stop_pub.publish(Bool(True))
        return EmptyResponse()

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('automatic_stop_behavior')
    node = AutomaticStopBehavior()
    node.run()