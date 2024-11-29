#!/usr/bin/env python3

import threading
import numpy as np
import shapely
import shapely.ops as shpops
import scipy
import scipy.optimize
import rospy
import tf2_ros
from ros_numpy import numpify, msgify
from tf.transformations import euler_matrix
from std_msgs.msg import Float32MultiArray, ColorRGBA
from autoware_mini.msg import Path
from geometry_msgs.msg import PoseStamped, Point, TransformStamped, Pose
from visualization_msgs.msg import MarkerArray, Marker

from helpers.transform import transform_point
from helpers.geometry import get_orientation_from_heading

class LaneBoundaryMatcher:

    def __init__(self):

        # parameters
        self.only_lateral_correction = rospy.get_param("~only_lateral_correction")
        self.lookahead_distance = rospy.get_param("~lookahead_distance")
        self.y_correction_treshold = rospy.get_param("~y_correction_treshold")
        self.z_correction_treshold = rospy.get_param("~z_correction_treshold")
        self.yaw_correction_treshold = rospy.get_param("~yaw_correction_treshold")
        self.transform_timeout = rospy.get_param("~transform_timeout")
        self.alpha = rospy.get_param("~alpha")

        # variables
        self.global_path_left_boundary = None
        self.global_path_right_boundary = None
        self.current_timestamp = None

        self.y_correction = 0
        self.z_correction = 0
        self.yaw_correction = 0
        self.correction_stamp = None
        self.transform_matrix = np.eye(4)
        
        self.current_pose_lock = threading.Lock()
        self.global_path_lock = threading.Lock()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.publish_base_link_correction_tf(init=True)

        # publishers
        self.current_pose_pub = rospy.Publisher('current_pose', PoseStamped, queue_size=1, tcp_nodelay=True)
        self.lane_bound_markers_pub = rospy.Publisher('lane_boundary_matcher_markers', MarkerArray, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('/openpilot/lane_lines', Float32MultiArray, self.lane_line_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/planning/lanelet2_global_path', Path, self.global_path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('current_pose_gnss', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)

    def update_correction(self, new_value, old_value):
        return self.alpha * new_value + (1 - self.alpha) * old_value

    def current_pose_callback(self, msg):
        with self.current_pose_lock:
            self.current_timestamp = msg.header.stamp

        current_pose_matrix = numpify(msg.pose)
        corrected_current_pose_matrix = self.transform_matrix.dot(current_pose_matrix)

        msg.pose = msgify(Pose, corrected_current_pose_matrix)
        self.current_pose_pub.publish(msg)

    def lane_line_callback(self, msg):
        openpilot_lane_boundaries = float32_multiarray_to_numpy(msg)
        
        with self.current_pose_lock:
            current_timestamp = self.current_timestamp

        with self.global_path_lock:
            global_path_left_boundary = self.global_path_left_boundary
            global_path_right_boundary = self.global_path_right_boundary

        if global_path_left_boundary is None or global_path_right_boundary is None:
            return

        # Fetch transforms
        try:
            transform_openpilot_bl = self.tf_buffer.lookup_transform("base_link", "openpilot", current_timestamp, rospy.Duration(self.transform_timeout))
            tf_matrix_openpilot_bl = numpify(transform_openpilot_bl.transform)
            transform_map_bl = self.tf_buffer.lookup_transform("base_link_gnss", "map", current_timestamp, rospy.Duration(self.transform_timeout))
            tf_matrix_map_bl = numpify(transform_map_bl.transform)
            transform_openpilot_map = self.tf_buffer.lookup_transform("map", "openpilot", current_timestamp, rospy.Duration(self.transform_timeout))
        except (tf2_ros.TransformException, rospy.ROSTimeMovedBackwardsException) as e:
            rospy.logwarn("%s - %s", rospy.get_name(), e)
            return
        
        current_pos = transform_point(Point(0, 0, 0), transform_openpilot_map)
        
        ##################################################################
        # Transform openpilot lane boundaries and create linestrings
        ##################################################################

        # transfrom openpilot predicted lane boundaries to base_link frame
        center_openpilot_lane_boundaries = np.transpose(openpilot_lane_boundaries, (0, 2, 1))[1:3, :, :] # transpose axis 1 and 2
        flattened_openpilot_lane_boundaries = center_openpilot_lane_boundaries.reshape(-1, 4) # flatten to (2*n_points, 4)
        flattened_openpilot_lane_boundaries[:, 3] = 1 # replece the time dimension with ones to get homogeneous points
        homogeneous_openpilot_lane_boundaries = flattened_openpilot_lane_boundaries @ tf_matrix_openpilot_bl.T # do the transform
        openpilot_lane_boundary_points = homogeneous_openpilot_lane_boundaries[:, :3].reshape(2, -1, 3) # convert back to 3d matrix with 3d points

        # trim openpilot lane boundaries
        openpilot_left_lane_boundary, openpilot_right_lane_boundary = shapely.linestrings(openpilot_lane_boundary_points)
        openpilot_left_lane_boundary = shpops.substring(openpilot_left_lane_boundary, 0, self.lookahead_distance)
        openpilot_right_lane_boundary = shpops.substring(openpilot_right_lane_boundary, 0, self.lookahead_distance)
        
        ##################################################################
        # Transform map lane boundaries
        ##################################################################

        # find the distance of current position
        left_cur_pos_dist = global_path_left_boundary.project(shapely.Point(current_pos.x, current_pos.y, current_pos.z))
        right_cur_pos_dist = global_path_right_boundary.project(shapely.Point(current_pos.x, current_pos.y, current_pos.z))
        
        # cut out the relevant sections from the global path boundaries 
        trimmed_left_map_lane_boundary = shpops.substring(global_path_left_boundary, left_cur_pos_dist, left_cur_pos_dist + self.lookahead_distance)
        trimmed_right_map_lane_boundary = shpops.substring(global_path_right_boundary, right_cur_pos_dist, right_cur_pos_dist + self.lookahead_distance)

        # transform map lane boundary points from map to base_link_gnss frame
        homogeneous_left_lane_points = np.hstack((np.array(trimmed_left_map_lane_boundary.coords), np.ones((len(trimmed_left_map_lane_boundary.coords), 1))))
        map_left_lane_boundary_homogeneous = homogeneous_left_lane_points @ tf_matrix_map_bl.T
        map_left_lane_boundary = map_left_lane_boundary_homogeneous[:, :3]
        map_left_lane_boundary = shapely.LineString(map_left_lane_boundary)

        homogeneous_right_lane_points = np.hstack((np.array(trimmed_right_map_lane_boundary.coords), np.ones((len(trimmed_right_map_lane_boundary.coords), 1))))
        map_right_lane_boundary_homogeneous = homogeneous_right_lane_points @ tf_matrix_map_bl.T
        map_right_lane_boundary = map_right_lane_boundary_homogeneous[:, :3]
        map_right_lane_boundary = shapely.LineString(map_right_lane_boundary)

        ##################################################################
        # Perform matching
        ##################################################################
    
        no_correction = False
        if self.only_lateral_correction:
            # calculate the average difference for right and left boundaries
            differences = self.find_average_distance(map_left_lane_boundary, map_right_lane_boundary, 
                                                     openpilot_left_lane_boundary, openpilot_right_lane_boundary)
            
            if differences is None:
                y, z = 0, 0
                no_correction = True
            else:
                avg_y_diff_left, avg_y_diff_right, avg_z_diff_left, avg_z_diff_right = differences

                y = (avg_y_diff_left + avg_y_diff_right) / 2
                z = (avg_z_diff_left + avg_z_diff_right) / 2

                # if the difference between map and openpilot lane boundaries is too big then don't use the correction
                if y > self.y_correction_treshold or z > self.z_correction_treshold:
                    y, z = 0, 0
                    no_correction = True
            
            # use exponential moving average to smooth coordinate corrections
            self.y_correction = self.update_correction(y, self.y_correction)
            self.z_correction = self.update_correction(z, self.z_correction)

        else:
            result = scipy.optimize.minimize(self.objective_function, np.array([0, 0, 0]), method='Nelder-Mead', 
                                             args=(map_left_lane_boundary, map_right_lane_boundary, 
                                                   openpilot_left_lane_boundary, openpilot_right_lane_boundary))
            y, z, yaw = result.x

            # if the calculated correction is too big then don't use the correction
            if (abs(self.y_correction - y) > self.y_correction_treshold or 
                abs(self.z_correction - z) > self.z_correction_treshold or 
                abs(self.yaw_correction - yaw) > self.yaw_correction_treshold):
                y, z, yaw = 0, 0, 0
                no_correction = True

            # use exponential moving average to smooth coordinate corrections 
            self.y_correction = self.update_correction(y, self.y_correction)
            self.z_correction = self.update_correction(z, self.z_correction)
            self.yaw_correction = self.update_correction(yaw, self.yaw_correction)

        self.correction_stamp = current_timestamp
        self.publish_base_link_correction_tf()

        ##################################################################
        # Visualization
        ##################################################################

        lanes_marker_array = MarkerArray()
        marker1 = self.get_lane_boundary_marker(openpilot_right_lane_boundary, "right", no_correction, True)
        marker2 = self.get_lane_boundary_marker(openpilot_left_lane_boundary, "left", no_correction, True)
        lanes_marker_array.markers.append(marker1)
        lanes_marker_array.markers.append(marker2)

        marker3 = self.get_lane_boundary_marker(map_right_lane_boundary, "right", no_correction)
        marker4 = self.get_lane_boundary_marker(map_left_lane_boundary, "left", no_correction)
        lanes_marker_array.markers.append(marker3)
        lanes_marker_array.markers.append(marker4)
        
        self.lane_bound_markers_pub.publish(lanes_marker_array)

        
    def global_path_callback(self, msg):
        if len(msg.waypoints) == 0:
            return
        
        three_point_lines = []
        left_offsets = []
        right_offsets = []

        # create three-point linestring segments for every waypoint
        for i in range(len(msg.waypoints)):
            left_offsets.append(msg.waypoints[i].left_width)
            right_offsets.append(-msg.waypoints[i].right_width)

            # the first and last waypoints cannot be in the middle
            if i == 0:
                i += 1
            elif i == len(msg.waypoints) - 1:
                i -= 1

            three_point_lines.append([[msg.waypoints[i-1].position.x, msg.waypoints[i-1].position.y],
                                    [msg.waypoints[i].position.x, msg.waypoints[i].position.y],
                                    [msg.waypoints[i+1].position.x, msg.waypoints[i+1].position.y]])

        three_point_linestrings = shapely.linestrings(three_point_lines)
        
        left_offset_lines = shapely.offset_curve(three_point_linestrings, left_offsets)
        right_offset_lines = shapely.offset_curve(three_point_linestrings, right_offsets)

        assert len(three_point_linestrings) == len(left_offsets) == len(right_offsets)

        left_boundary_coords = []
        right_boundary_coords = []
        for i in range(len(three_point_linestrings)):
            z_coord = msg.waypoints[i].position.z
            if i == 0: # use the first point of the first segment as the first lane boundary point 
                left_offset_point = left_offset_lines[0].coords[0]
                right_offset_point = right_offset_lines[0].coords[0]
            elif i == len(three_point_linestrings) - 1:  # use the third point of the last segment as the last lane boundary point 
                left_offset_point = left_offset_lines[-1].coords[2]
                right_offset_point = right_offset_lines[-1].coords[2]
            else: # take the second point from every other segment
                left_offset_point = left_offset_lines[i].coords[1]
                right_offset_point = right_offset_lines[i].coords[1]

            left_boundary_coords.append((left_offset_point[0], left_offset_point[1], z_coord))
            right_boundary_coords.append((right_offset_point[0], right_offset_point[1], z_coord))

        left_boundary, right_boundary = shapely.linestrings([left_boundary_coords, right_boundary_coords])
        shapely.prepare(left_boundary)
        shapely.prepare(right_boundary)

        with self.global_path_lock:
            self.global_path_left_boundary = left_boundary
            self.global_path_right_boundary = right_boundary

    def find_average_distance(self, map_left_lane_boundary, map_right_lane_boundary, openpilot_left_lane_boundary, openpilot_right_lane_boundary):
        left_y_diffs = []
        left_z_diffs = [] 
        for x, y, z in openpilot_left_lane_boundary.coords:
            point = map_left_lane_boundary.interpolate(x)
            left_y_diffs.append(y - point.y)
            left_z_diffs.append(z - point.z)

        right_y_diffs = []
        right_z_diffs = []
        for x, y, z in openpilot_right_lane_boundary.coords:
            point = map_right_lane_boundary.interpolate(x)
            right_y_diffs.append(y - point.y)
            right_z_diffs.append(z - point.z)

        return np.mean(left_y_diffs), np.mean(right_y_diffs), np.mean(left_z_diffs), np.mean(right_z_diffs)
    
    def objective_function(self, input_values, map_left_lane_boundary, map_right_lane_boundary, openpilot_left_lane_boundary, openpilot_right_lane_boundary):
        lateral_correction, height_correction, yaw_correction = input_values

        matrix = euler_matrix(0, 0, yaw_correction)
        matrix[1, 3] = lateral_correction
        matrix[2, 3] = height_correction

        map_left_lane_boundary_h = np.hstack((np.array(map_left_lane_boundary.coords), np.ones((len(map_left_lane_boundary.coords), 1))))
        map_right_lane_boundary_h = np.hstack((np.array(map_right_lane_boundary.coords), np.ones((len(map_right_lane_boundary.coords), 1))))

        corrected_map_left_lane_bound_h = map_left_lane_boundary_h @ matrix.T
        corrected_map_left_lane_bound = corrected_map_left_lane_bound_h[:, :3]

        corrected_map_right_lane_bound_h = map_right_lane_boundary_h @ matrix.T
        corrected_map_right_lane_bound = corrected_map_right_lane_bound_h[:, :3]

        corrected_left_map_lane_boundary = shapely.LineString(corrected_map_left_lane_bound)
        corrected_right_map_lane_boundary = shapely.LineString(corrected_map_right_lane_bound)

        return shapely.hausdorff_distance(openpilot_left_lane_boundary, corrected_left_map_lane_boundary) + shapely.hausdorff_distance(openpilot_right_lane_boundary, corrected_right_map_lane_boundary)

    def get_lane_boundary_marker(self, lane_boundary, side, no_correction, openpilot=False):
        if openpilot:
            if no_correction:
                color = ColorRGBA(0.5, 0.8, 0.7, 1.0)
            else:
                color = ColorRGBA(0.0, 1.0, 0.7, 1.0)
            if side == "right":
                id_start = 0
            else:
                id_start = 1
            frame_id = "base_link"
        else:
            if no_correction:
                color = ColorRGBA(0.6, 0.5, 0.4, 1.0)
            else:
                color = ColorRGBA(0.6, 0.3, 0.0, 1.0)
            if side == "right":
                id_start = 2
            else:
                id_start = 3
            frame_id = "base_link_gnss"

        points = []
        for x, y, z in lane_boundary.coords:
            point = Point(x=x,y=y, z=z)
            points.append(point)

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = f"{side} bound"
        marker.id = id_start
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.color = color
        marker.points = points

        return marker

    def publish_base_link_correction_tf(self, init=False):
        t = TransformStamped()

        y_correction = self.y_correction
        z_correction = self.z_correction
        yaw_correction = self.yaw_correction
        correction_stamp = self.correction_stamp

        if init:
            t.header.stamp = rospy.Time.now()
        else:
            t.header.stamp = correction_stamp
        t.header.frame_id = "base_link_gnss"
        t.child_frame_id = "base_link"

        t.transform.translation.x = 0
        t.transform.translation.y = y_correction
        t.transform.translation.z = z_correction
        t.transform.rotation = get_orientation_from_heading(yaw_correction)

        matrix = euler_matrix(0, 0, yaw_correction)
        matrix[0, 3] = 0
        matrix[1, 3] = y_correction
        matrix[2, 3] = z_correction

        if init:
            static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
            static_tf_broadcaster.sendTransform(t)
            self.transform_matrix = matrix
        else:
            self.tf_broadcaster.sendTransform(t)
            self.transform_matrix = matrix

    def run(self):
        rospy.spin()
    
def float32_multiarray_to_numpy(multiarray):
    dims = tuple(map(lambda x: x.size, multiarray.layout.dim))
    return np.array(multiarray.data, dtype=float).reshape(dims).astype(np.float32)

if __name__ == '__main__':
    rospy.init_node('lane_boundary_matcher')
    node = LaneBoundaryMatcher()
    node.run()