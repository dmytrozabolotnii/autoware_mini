#!/usr/bin/env python3

import rospy
import numpy as np
import tf2_ros
import shapely
import threading
from lanelet2.geometry import approximatedLength2d
from std_msgs.msg import Float32MultiArray, UInt32MultiArray, ColorRGBA
from geometry_msgs.msg import PoseStamped, Point, TransformStamped, Quaternion, Pose
from visualization_msgs.msg import MarkerArray, Marker
from ros_numpy import numpify, msgify
from tf.transformations import quaternion_from_euler, quaternion_matrix

from helpers.lanelet2 import load_lanelet2_map
from helpers.transform import transform_point
from helpers.geometry import get_heading_from_orientation, get_point_using_heading_and_distance, split_linestring_by_point_and_heading

class LaneBoundaryMatcher:

    def __init__(self):

        # parameters
        lanelet2_map_name = rospy.get_param("~lanelet2_map_name")

        self.only_lateral_correction = True
        self.lanelet2_map = load_lanelet2_map(lanelet2_map_name)
        if self.only_lateral_correction:
            self.cutoff_length = 8
            self.lookahead_distance = 3
        else:
            self.cutoff_length = 50
            self.lookahead_distance = 42.1875
        self.correction_trshold = 1.2
        self.transform_timeout = 0.06
        self.base_link_openpilot_dist = 2.41

        # variables
        self.global_path_lanelet_ids = None
        self.lanelet_polygons = None
        self.lanelet_centerline_waypoints = None
        self.current_lanelet_idx = 0
        self.current_position = None
        self.current_heading = None
        self.current_timestamp = None
        self.new_global_path = None
        self.approximated_lanelet_lengths = None

        self.localization_corrections = {'x':0, 'y':0, 'z':0, 'roll':0, 'pitch':0, 'yaw':0}
        self.transform_matrix = None
        
        self.lock = threading.Lock()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.publish_base_link_correction_tf(**self.localization_corrections)

        # publishers
        self.current_pose_pub = rospy.Publisher('current_pose', PoseStamped, queue_size=1, tcp_nodelay=True)
        self.lanelet_bounds_pub = rospy.Publisher('lanelet_bounds_match', MarkerArray, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('/openpilot/lane_lines', Float32MultiArray, self.lane_line_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/planning/global_path_lenelet_ids', UInt32MultiArray, self.global_path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('current_pose_gnss', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)

    def current_pose_callback(self, msg):
        self.current_position = shapely.Point(msg.pose.position.x, msg.pose.position.y, msg.pose.position.y)
        self.current_heading = get_heading_from_orientation(msg.pose.orientation)
        self.current_timestamp = msg.header.stamp

        current_pose_matrix = numpify(msg.pose)
        corrected_current_pose_matrix = self.transform_matrix.dot(current_pose_matrix)

        corrected_current_pose = PoseStamped()
        corrected_current_pose.header = msg.header
        corrected_current_pose.pose = msgify(Pose, corrected_current_pose_matrix)
        self.current_pose_pub.publish(corrected_current_pose)
        

    def lane_line_callback(self, msg):
        supercombo_lane_lines = np.array(msg.data).reshape(msg.layout.dim[0].size, msg.layout.dim[1].size, msg.layout.dim[2].size)

        if self.lanelet_polygons is None or self.current_position is None or self.new_global_path is None or self.approximated_lanelet_lengths is None:
            return
        
        with self.lock:
            current_position = self.current_position
            current_heading = self.current_heading
            current_timestamp = self.current_timestamp
            lanelet_polygons = self.lanelet_polygons
            global_path_lanelet_ids = self.global_path_lanelet_ids
            current_lanelet_idx = self.current_lanelet_idx
            approximated_lanelet_lengths = self.approximated_lanelet_lengths
            self.new_global_path = False

        try:
            transform_openpilot = self.tf_buffer.lookup_transform("openpilot", "base_link", current_timestamp, rospy.Duration(self.transform_timeout))
            transform_map = self.tf_buffer.lookup_transform("map", "base_link_gnss", current_timestamp, rospy.Duration(self.transform_timeout))
        except (tf2_ros.TransformException, rospy.ROSTimeMovedBackwardsException) as e:
            rospy.logwarn("%s - %s", rospy.get_name(), e)
            return
        
        # find the current lanelet
        while not lanelet_polygons[current_lanelet_idx].contains(current_position):
            if len(lanelet_polygons) <= current_lanelet_idx + 1:
                return
            
            current_lanelet_idx += 1

        right_lane_points = []
        left_lane_points = []
        idx = current_lanelet_idx
        current_centerline = shapely.LineString([(point.x, point.y) for point in self.lanelet2_map.laneletLayer.get(global_path_lanelet_ids[idx]).centerline])
        total_length = -current_centerline.project(current_position)

        # take close left and right boundaries from lanelets
        while total_length < self.cutoff_length:
            lanelet_id = global_path_lanelet_ids[idx]
            for point in self.lanelet2_map.laneletLayer.get(lanelet_id).rightBound:
                right_lane_points.append((point.x, point.y, point.z))

            for point in self.lanelet2_map.laneletLayer.get(lanelet_id).leftBound:
                left_lane_points.append((point.x, point.y, point.z))

            total_length += approximated_lanelet_lengths[lanelet_id]
            idx += 1


        openpilot_point = get_point_using_heading_and_distance(current_position, current_heading, self.base_link_openpilot_dist)

        # trim the start of lane boundaries
        right_lane_bound = split_linestring_by_point_and_heading(shapely.LineString(right_lane_points), openpilot_point, current_heading)
        left_lane_bound = split_linestring_by_point_and_heading(shapely.LineString(left_lane_points), openpilot_point, current_heading)

        # endpoint of lane boundary matching
        lookahead_point = get_point_using_heading_and_distance(current_position, current_heading, self.lookahead_distance + self.base_link_openpilot_dist)

        # trim the end of lane boundaries
        right_lane_bound = split_linestring_by_point_and_heading(right_lane_bound, lookahead_point, current_heading - np.pi)
        left_lane_bound = split_linestring_by_point_and_heading(left_lane_bound, lookahead_point, current_heading - np.pi)


        right_lane_bound_openpilot = []
        left_lane_bound_openpilot = []
        for x, y, z in right_lane_points:
            point = transform_point(Point(x, y, z), transform_map)
            right_lane_bound_openpilot.append((point.x, point.y, point.z))
        for x, y, z in left_lane_points:
            point = transform_point(Point(x, y, z), transform_map)
            left_lane_bound_openpilot.append((point.x, point.y, point.z))

        supercombo_lane_points = []
        for i in range(1, 4):
            lane_points = []
            for x, y, z, t in supercombo_lane_lines[i].T:
                if x > self.lookahead_distance:
                    break
                point = transform_point(Point(x, y, z), transform_openpilot)
                lane_points.append((point.x, point.y, point.z))
                
            supercombo_lane_points.append(lane_points)

        avg_y_diff_left, avg_y_diff_right, avg_z_diff_left, avg_z_diff_right = self.find_average_distance(supercombo_lane_points, 
                                                                   shapely.LineString(left_lane_bound_openpilot), 
                                                                   shapely.LineString(right_lane_bound_openpilot))
        
        if avg_y_diff_left > self.correction_trshold or avg_y_diff_right > self.correction_trshold:
            return
        

        self.localization_corrections['y'] = (avg_y_diff_left + avg_y_diff_right) / 2
        self.publish_base_link_correction_tf(**self.localization_corrections)
        
        #print("LEFT", avg_y_diff_left, avg_z_diff_left)
        #print("RIGHT", avg_y_diff_right, avg_z_diff_right)

        """
        # Visualization for debugging
        lanes_marker_array = MarkerArray()
        marker1, marker2 = self.publish_lanelet_bounds(shapely.LineString(supercombo_lane_points[1]), shapely.LineString(supercombo_lane_points[0]), True)
        lanes_marker_array.markers.append(marker1)
        lanes_marker_array.markers.append(marker2)

        marker1, marker2 = self.publish_lanelet_bounds(right_lane_bound, left_lane_bound)
        lanes_marker_array.markers.append(marker1)
        lanes_marker_array.markers.append(marker2)
        
        self.lanelet_bounds_pub.publish(lanes_marker_array)

        """
        if not self.new_global_path:
            self.current_lanelet_idx = current_lanelet_idx
        


    def global_path_callback(self, msg):
        self.new_global_path = True
        if len(msg.data) == 0:
            self.lanelet_polygons = None
            self.current_lanelet_idx = 0
            return
        
        self.global_path_lanelet_ids = msg.data

        lanelet_polys = []
        centerline_waypoints = []
        approximated_lanelet_lengths = {}

        # get all lanelets on the global path
        for lanelet_id in msg.data:
            lanelet = self.lanelet2_map.laneletLayer.get(lanelet_id)

            for point in lanelet.centerline:
                centerline_waypoints.append(shapely.Point(point.x, point.y))

            polygon = shapely.Polygon([(pt.x, pt.y) for pt in lanelet.polygon2d()])
            lanelet_polys.append(polygon)
            approximated_lanelet_lengths[lanelet_id] = approximatedLength2d(lanelet)

        self.lanelet_polygons = lanelet_polys
        self.current_lanelet_idx = 0
        self.approximated_lanelet_lengths = approximated_lanelet_lengths

    def find_average_distance(self, supercombo_lane_points, left_lane_bound, right_lane_bound):
        left_y_diffs = []
        left_z_diffs = [] 
        for x, y, z in supercombo_lane_points[0]:
            point = left_lane_bound.intersection(shapely.LineString([(x, -10), (x, 10)]))
            left_y_diffs.append(point.y - y)
            left_z_diffs.append(point.z - z)

        right_y_diffs = []
        right_z_diffs = []
        for x, y, z in supercombo_lane_points[1]:
            point = right_lane_bound.intersection(shapely.LineString([(x, -10), (x, 10)]))
            right_y_diffs.append(point.y - y)
            right_z_diffs.append(point.z - z)


        #left_boundary_distances = [shapely.Point(x,y,z).distance(left_lane_bound) for x, y, z in supercombo_lane_points[0]]
        #right_boundary_distances = [shapely.Point(x,y,z).distance(right_lane_bound) for x, y, z in supercombo_lane_points[1]]

        return np.mean(left_y_diffs), np.mean(right_y_diffs), np.mean(left_z_diffs), np.mean(right_z_diffs)


    def publish_lanelet_bounds(self, right_lane_bound, left_lane_bound, supercombo=False):
        # For debugging

        if supercombo:
            color = ColorRGBA(0.0, 1.0, 0.7, 1.0)
            id_start = 0
        else:
            color = ColorRGBA(0.6, 0.3, 0.0, 1.0)
            id_start = 2

        points = []
        for x, y, z in right_lane_bound.coords:
            point = Point(x=x,y=y, z=z)
            points.append(point)

        marker1 = Marker()
        marker1.header.frame_id = "map"
        marker1.header.stamp = rospy.Time.now()
        marker1.ns = "right bound"
        marker1.id = id_start
        marker1.type = Marker.LINE_STRIP
        marker1.action = Marker.ADD
        marker1.pose.orientation.w = 1.0
        marker1.scale.x = 0.1
        marker1.color = color
        marker1.points = points

        points = []
        for x, y, z in left_lane_bound.coords:
            point = Point(x=x,y=y, z=z)
            points.append(point)

        marker2 = Marker()
        marker2.header.frame_id = "map"
        marker2.header.stamp = rospy.Time.now()
        marker2.ns = "left bound"
        marker2.id = id_start + 1
        marker2.type = Marker.LINE_STRIP
        marker2.action = Marker.ADD
        marker2.pose.orientation.w = 1.0
        marker2.scale.x = 0.1
        marker2.color = color
        marker2.points = points

        return marker1, marker2
    
    def publish_base_link_correction_tf(self, x, y, z, roll, pitch, yaw):
        t = TransformStamped()

        x_q, y_q, z_q, w_q = quaternion_from_euler(roll, pitch, yaw, axes='rxyz')
        orientation = Quaternion(x_q, y_q, z_q, w_q)

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "base_link_gnss"
        t.child_frame_id = "base_link"

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation = orientation

        self.tf_broadcaster.sendTransform(t)

        matrix = quaternion_matrix([x_q, y_q, z_q, w_q])
        matrix[0, 3] = x
        matrix[1, 3] = y
        matrix[2, 3] = z

        self.transform_matrix = matrix

        
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('lane_boundary_matcher')
    node = LaneBoundaryMatcher()
    node.run()