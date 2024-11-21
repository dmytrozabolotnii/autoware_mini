#!/usr/bin/env python3

import scipy.optimize
import rospy
import numpy as np
import tf2_ros
import shapely
import threading
import scipy
import lanelet2.geometry as ll2geometry
from std_msgs.msg import Float32MultiArray, UInt32MultiArray, ColorRGBA
from geometry_msgs.msg import PoseStamped, Point, TransformStamped, Quaternion, Pose
from visualization_msgs.msg import MarkerArray, Marker
from ros_numpy import numpify, msgify
from tf.transformations import quaternion_from_euler, quaternion_matrix

from helpers.lanelet2 import load_lanelet2_map
from helpers.geometry import get_heading_from_orientation
from helpers.shapely import split_linestring_with_two_lines

class StreamingRollingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.window = np.zeros(window_size)  # Pre-allocate a fixed-size window
        self.index = 0  # Index to track the circular buffer
        self.count = 1  # Number of elements added so far, initialization at 1 to avoid division by 0
        self.total = 0  # Running total of the window
    
    def add(self, value):
        # Subtract the value that is being replaced
        self.total -= self.window[self.index]
        
        # Add the new value to the total and update the window
        self.window[self.index] = value
        self.total += value
        
        # Update the circular index and count
        self.index = (self.index + 1) % self.window_size
        self.count = min(self.count + 1, self.window_size)
        
    def get(self):
        # Return the current rolling average
        return self.total / self.count
        
class LaneBoundaryMatcher:

    def __init__(self):

        # parameters
        lanelet2_map_name = rospy.get_param("~lanelet2_map_name")
        self.only_lateral_correction = rospy.get_param("~only_lateral_correction")

        self.lanelet2_map = load_lanelet2_map(lanelet2_map_name)
        if self.only_lateral_correction:
            self.cutoff_length = 8
            self.lookahead_distance = 3
        else:
            self.cutoff_length = 50
            self.lookahead_distance = 42.1875

        self.y_correction_treshold = rospy.get_param("~y_correction_treshold")
        self.z_correction_treshold = rospy.get_param("~z_correction_treshold")
        self.yaw_correction_treshold = rospy.get_param("~yaw_correction_treshold")
        self.transform_timeout = rospy.get_param("~transform_timeout")
        self.window_size = rospy.get_param("~window_size")

        # variables
        self.global_path_lanelet_ids = None
        self.lanelet_polygons = None
        self.lanelet_centerline_waypoints = None
        self.current_lanelet_idx = 0
        self.current_position = None
        self.current_timestamp = None
        self.new_global_path = None
        self.approximated_lanelet_lengths = None

        self.localization_corrections = {c : StreamingRollingAverage(self.window_size) for c in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']}
        self.localization_corrections['stamp'] = None
        self.transform_matrix = np.eye(4)
        
        self.lock = threading.Lock()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.publish_base_link_correction_tf(self.localization_corrections, init=True)

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
            current_timestamp = self.current_timestamp
            lanelet_polygons = self.lanelet_polygons
            global_path_lanelet_ids = self.global_path_lanelet_ids
            current_lanelet_idx = self.current_lanelet_idx
            approximated_lanelet_lengths = self.approximated_lanelet_lengths
            self.new_global_path = False

        try:
            transform_openpilot = self.tf_buffer.lookup_transform("base_link", "openpilot", current_timestamp, rospy.Duration(self.transform_timeout))
            tf_matrix_openpilot = numpify(transform_openpilot.transform)
            transform_map = self.tf_buffer.lookup_transform("base_link_gnss", "map", current_timestamp, rospy.Duration(self.transform_timeout))
            tf_matrix_map = numpify(transform_map.transform)
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

        # transform lane boundary points from map to base_link_gnss frame
        homogeneous_right_lane_points = np.hstack((np.array(right_lane_points), np.ones((len(right_lane_points), 1))))
        right_lane_bound_bl_homogeneous = homogeneous_right_lane_points @ tf_matrix_map.T
        right_lane_bound_bl = right_lane_bound_bl_homogeneous[:, :3]

        homogeneous_left_lane_points = np.hstack((np.array(left_lane_points), np.ones((len(left_lane_points), 1))))
        left_lane_bound_bl_homogeneous = homogeneous_left_lane_points @ tf_matrix_map.T
        left_lane_bound_bl = left_lane_bound_bl_homogeneous[:, :3]

        # transfrom openpilot predicted lane lines to base_link frame
        center_supercombo_lane_lines = np.transpose(supercombo_lane_lines, (0, 2, 1))[1:3, :, :] # transpose axis 1 and 2
        flattened_supercombo_lane_lines = center_supercombo_lane_lines.reshape(-1, 4) # flatten to (2*n_points, 4)
        trimmed_supercombo_lane_lines = flattened_supercombo_lane_lines[flattened_supercombo_lane_lines[:, 0] <= self.lookahead_distance] # filter out rows where x > lookahead_distance
        trimmed_supercombo_lane_lines[:, 3] = 1 # replece the time dimension with ones to get homogeneous points
        supercombo_lane_points_homogeneous = trimmed_supercombo_lane_lines @ tf_matrix_openpilot.T # do the transform
        supercombo_lane_points = supercombo_lane_points_homogeneous[:, :3].reshape(2, supercombo_lane_points_homogeneous.shape[0] // 2, 3) # convert back to 3d matrix with 3d points

        splitter_line1 = shapely.LineString([(supercombo_lane_points[0][0][0], -10), (supercombo_lane_points[0][0][0], 10)])
        splitter_line2 = shapely.LineString([(supercombo_lane_points[0][-1][0], -10), (supercombo_lane_points[0][-1][0], 10)])

        if self.only_lateral_correction:
            # calculate the average difference for right and left boundaries
            differences = self.find_average_distance(supercombo_lane_points, 
                                                    shapely.LineString(left_lane_bound_bl), 
                                                    shapely.LineString(right_lane_bound_bl))
            
            if differences is None:
                self.publish_empty_bounds()
                return
            
            avg_y_diff_left, avg_y_diff_right, avg_z_diff_left, avg_z_diff_right = differences

            # if the difference between map and openpilot lane boundaries is too big then don't use the correction
            if avg_y_diff_left > self.y_correction_treshold or avg_y_diff_right > self.y_correction_treshold:
                self.publish_empty_bounds()
                return

            if avg_z_diff_left > self.z_correction_treshold or avg_z_diff_right > self.z_correction_treshold:
                self.publish_empty_bounds()
                return
            
            # update the localization correction and publish the updated transform
            self.localization_corrections['y'].add((avg_y_diff_left + avg_y_diff_right) / 2)
            self.localization_corrections['z'].add((avg_z_diff_left + avg_z_diff_right) / 2)


        else:
            # trim the start of lane boundaries
            trimmed_right_lane_bound_bl = split_linestring_with_two_lines(shapely.LineString(right_lane_bound_bl), splitter_line1, splitter_line2)
            trimmed_left_lane_bound_bl = split_linestring_with_two_lines(shapely.LineString(left_lane_bound_bl), splitter_line1, splitter_line2)

            if trimmed_right_lane_bound_bl is None or trimmed_left_lane_bound_bl is None:
                self.publish_empty_bounds()
                return


            result = scipy.optimize.minimize(self.objective_function, np.array([0, 0, 0]), method='Nelder-Mead', args=(trimmed_left_lane_bound_bl, trimmed_right_lane_bound_bl, supercombo_lane_points_homogeneous))
            y, z, yaw = result.x

            # if the calculated correction is too big then don't use the correction
            if (abs(self.localization_corrections['y'].get() - y) > self.y_correction_treshold or 
                abs(self.localization_corrections['z'].get() - z) > self.z_correction_treshold or 
                abs(self.localization_corrections['yaw'].get() - yaw) > self.yaw_correction_treshold):
                self.publish_empty_bounds()
                return

            # update the localization correction and publish the updated transform
            self.localization_corrections['y'].add(y)
            self.localization_corrections['z'].add(z)
            self.localization_corrections['yaw'].add(yaw)

        self.localization_corrections['stamp'] = current_timestamp
        self.publish_base_link_correction_tf(self.localization_corrections)

        # Visualization for debugging
        lanes_marker_array = MarkerArray()
        marker1, marker2 = self.publish_lanelet_bounds(shapely.LineString(supercombo_lane_points[1]), shapely.LineString(supercombo_lane_points[0]), True)
        lanes_marker_array.markers.append(marker1)
        lanes_marker_array.markers.append(marker2)

        if self.only_lateral_correction:
            marker1, marker2 = self.publish_lanelet_bounds(shapely.LineString(right_lane_bound_bl), shapely.LineString(left_lane_bound_bl))
        else:
            marker1, marker2 = self.publish_lanelet_bounds(trimmed_right_lane_bound_bl, trimmed_left_lane_bound_bl)

        lanes_marker_array.markers.append(marker1)
        lanes_marker_array.markers.append(marker2)
        
        self.lanelet_bounds_pub.publish(lanes_marker_array)

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
            approximated_lanelet_lengths[lanelet_id] = ll2geometry.approximatedLength2d(lanelet)

        self.lanelet_polygons = lanelet_polys
        self.current_lanelet_idx = 0
        self.approximated_lanelet_lengths = approximated_lanelet_lengths

    def find_average_distance(self, supercombo_lane_points, left_lane_bound, right_lane_bound):
        left_y_diffs = []
        left_z_diffs = [] 
        for x, y, z in supercombo_lane_points[0]:
            point = left_lane_bound.intersection(shapely.LineString([(x, -10), (x, 10)]))
            if point.is_empty:
                return None
            left_y_diffs.append(point.y - y)
            left_z_diffs.append(point.z - z)

        right_y_diffs = []
        right_z_diffs = []
        for x, y, z in supercombo_lane_points[1]:
            point = right_lane_bound.intersection(shapely.LineString([(x, -10), (x, 10)]))
            if point.is_empty:
                return None
            right_y_diffs.append(point.y - y)
            right_z_diffs.append(point.z - z)

        return np.mean(left_y_diffs), np.mean(right_y_diffs), np.mean(left_z_diffs), np.mean(right_z_diffs)
    
    def objective_function(self, input_values, left_lane_map, right_lane_map, openpilot_lane_lines):
        lateral_correction, height_correction, yaw_correction = input_values
        x_q, y_q, z_q, w_q = quaternion_from_euler(0, 0, yaw_correction, axes='rxyz')

        matrix = quaternion_matrix([x_q, y_q, z_q, w_q])
        matrix[1, 3] = lateral_correction
        matrix[2, 3] = height_correction

        corrected_openpilot_lane_lines_h = openpilot_lane_lines @ matrix.T
        corrected_openpilot_lane_lines = corrected_openpilot_lane_lines_h[:, :3].reshape(2, corrected_openpilot_lane_lines_h.shape[0] // 2, 3)

        left_lane_openpilot = shapely.LineString(corrected_openpilot_lane_lines[0])
        right_lane_openpilot = shapely.LineString(corrected_openpilot_lane_lines[1])

        return shapely.hausdorff_distance(left_lane_openpilot, left_lane_map) + shapely.hausdorff_distance(right_lane_openpilot, right_lane_map)


    def publish_lanelet_bounds(self, right_lane_bound, left_lane_bound, supercombo=False):
        # For debugging

        if supercombo:
            color = ColorRGBA(0.0, 1.0, 0.7, 1.0)
            id_start = 0
            frame_id = "base_link"
        else:
            color = ColorRGBA(0.6, 0.3, 0.0, 1.0)
            id_start = 2
            frame_id = "base_link_gnss"

        points = []
        for x, y, z in right_lane_bound.coords:
            point = Point(x=x,y=y, z=z)
            points.append(point)

        marker1 = Marker()
        marker1.header.frame_id = frame_id
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
        marker2.header.frame_id = frame_id
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
    
    def publish_empty_bounds(self):
        lanes_marker_array = MarkerArray()

        for supercombo in [True, False]:

            if supercombo:
                color = ColorRGBA(0.0, 1.0, 0.7, 1.0)
                id_start = 0
                frame_id = "base_link"
            else:
                color = ColorRGBA(0.6, 0.3, 0.0, 1.0)
                id_start = 2
                frame_id = "base_link_gnss"


            marker1 = Marker()
            marker1.header.frame_id = frame_id
            marker1.header.stamp = rospy.Time.now()
            marker1.ns = "right bound"
            marker1.id = id_start
            marker1.type = Marker.LINE_STRIP
            marker1.action = Marker.ADD
            marker1.pose.orientation.w = 1.0
            marker1.scale.x = 0.1
            marker1.color = color
            marker1.points = []

            marker2 = Marker()
            marker2.header.frame_id = frame_id
            marker2.header.stamp = rospy.Time.now()
            marker2.ns = "left bound"
            marker2.id = id_start + 1
            marker2.type = Marker.LINE_STRIP
            marker2.action = Marker.ADD
            marker2.pose.orientation.w = 1.0
            marker2.scale.x = 0.1
            marker2.color = color
            marker2.points = []

            lanes_marker_array.markers.append(marker1)
            lanes_marker_array.markers.append(marker2)

        self.lanelet_bounds_pub.publish(lanes_marker_array)

    
    def publish_base_link_correction_tf(self, localization_corrections, init=False):
        t = TransformStamped()

        x_q, y_q, z_q, w_q = quaternion_from_euler(localization_corrections["roll"].get(), 
                                                   localization_corrections["pitch"].get(), 
                                                   localization_corrections["yaw"].get(), axes='rxyz')
        orientation = Quaternion(x_q, y_q, z_q, w_q)

        if init:
            t.header.stamp = rospy.Time.now()
        else:
            t.header.stamp = localization_corrections["stamp"]
        t.header.frame_id = "base_link_gnss"
        t.child_frame_id = "base_link"

        t.transform.translation.x = localization_corrections["x"].get()
        t.transform.translation.y = localization_corrections["y"].get()
        t.transform.translation.z = localization_corrections["z"].get()
        t.transform.rotation = orientation

        matrix = quaternion_matrix([x_q, y_q, z_q, w_q])
        matrix[0, 3] = localization_corrections["x"].get()
        matrix[1, 3] = localization_corrections["y"].get()
        matrix[2, 3] = localization_corrections["z"].get()

        if init:
            static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
            static_tf_broadcaster.sendTransform(t)
            self.transform_matrix = matrix
        else:
            if not np.allclose(self.transform_matrix, matrix):
                self.tf_broadcaster.sendTransform(t)
                self.transform_matrix = matrix

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('lane_boundary_matcher')
    node = LaneBoundaryMatcher()
    node.run()