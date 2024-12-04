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
from tf.transformations import compose_matrix
from std_msgs.msg import ColorRGBA
from autoware_mini.msg import Path
from vehicle_platform.msg import Float32MultiArrayStamped
from geometry_msgs.msg import Point, PoseStamped, Transform, TransformStamped
from visualization_msgs.msg import MarkerArray, Marker

from helpers.path import PathWrapper
from helpers.transform import transform_point

class LaneBoundaryMatcher:

    def __init__(self):

        # parameters
        self.only_lateral_correction = rospy.get_param("~only_lateral_correction")
        self.enable_height_correction = rospy.get_param("~enable_height_correction")
        self.lookahead_distance = rospy.get_param("~lookahead_distance")
        self.x_correction_treshold = rospy.get_param("~x_correction_treshold")
        self.y_correction_treshold = rospy.get_param("~y_correction_treshold")
        self.yaw_correction_treshold = rospy.get_param("~yaw_correction_treshold")
        self.transform_timeout = rospy.get_param("~transform_timeout")
        self.alpha = rospy.get_param("~alpha")

        # variables
        self.current_pose = None
        self.global_path = None
        self.global_path_left_boundary = None
        self.global_path_right_boundary = None

        self.x_correction = 0
        self.y_correction = 0
        self.z_correction = 0
        self.yaw_correction = 0
        self.transform_matrix = np.eye(4)
        
        self.global_path_lock = threading.Lock()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.publish_base_link_correction_tf()

        # publishers
        self.lane_bound_markers_pub = rospy.Publisher('lane_boundary_matcher_markers', MarkerArray, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/openpilot/lane_lines', Float32MultiArrayStamped, self.lane_line_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/planning/lanelet2_global_path', Path, self.global_path_callback, queue_size=1, tcp_nodelay=True)

    def calculate_updated_correction(self, new_value, old_value):
        return self.alpha * new_value + (1 - self.alpha) * old_value
    
    def current_pose_callback(self, msg):
        self.current_pose = msg.pose.position

    def lane_line_callback(self, msg):
        openpilot_lane_boundaries = float32_multiarray_to_numpy(msg)

        with self.global_path_lock:
            global_path = self.global_path
            global_path_left_boundary = self.global_path_left_boundary
            global_path_right_boundary = self.global_path_right_boundary

        current_pose = self.current_pose

        if global_path_left_boundary is None or global_path_right_boundary is None or current_pose is None:
            return

        # Fetch transforms
        try:
            transform = self.tf_buffer.lookup_transform("map_corrected", "openpilot", msg.header.stamp, rospy.Duration(self.transform_timeout))
            tf_matrix = numpify(transform.transform)
            transform_footprint = self.tf_buffer.lookup_transform("map_corrected", "base_footprint", msg.header.stamp, rospy.Duration(self.transform_timeout))
        except (tf2_ros.TransformException, rospy.ROSTimeMovedBackwardsException) as e:
            rospy.logwarn("%s - %s", rospy.get_name(), e)
            return
        
        current_pose_openpilot = transform_point(Point(0, 0, 0), transform)

        if self.enable_height_correction and global_path is not None:
            current_pose_footprint = transform_point(Point(0, 0, 0), transform_footprint)
            current_pos_dist = global_path.linestring.project(shapely.Point(current_pose_footprint.x, current_pose_footprint.y, current_pose_footprint.z))
            self.z_correction = global_path.get_elevation_at_distance(current_pos_dist) - current_pose_footprint.z
        
        ##################################################################
        # Transform openpilot lane boundaries and create linestrings
        ##################################################################

        # transfrom openpilot predicted lane boundaries to base_link frame
        center_openpilot_lane_boundaries = np.transpose(openpilot_lane_boundaries, (0, 2, 1))[1:3, :, :] # transpose axis 1 and 2
        flattened_openpilot_lane_boundaries = center_openpilot_lane_boundaries.reshape(-1, 4) # flatten to (2*n_points, 4)
        flattened_openpilot_lane_boundaries[:, 3] = 1 # replece the time dimension with ones to get homogeneous points
        homogeneous_openpilot_lane_boundaries = flattened_openpilot_lane_boundaries @ tf_matrix.T # do the transform
        openpilot_lane_boundary_points = homogeneous_openpilot_lane_boundaries[:, :3].reshape(2, -1, 3) # convert back to 3d matrix with 3d points

        # trim openpilot lane boundaries
        openpilot_left_lane_boundary, openpilot_right_lane_boundary = shapely.linestrings(openpilot_lane_boundary_points)
        openpilot_left_lane_boundary = shpops.substring(openpilot_left_lane_boundary, 0, self.lookahead_distance)
        openpilot_right_lane_boundary = shpops.substring(openpilot_right_lane_boundary, 0, self.lookahead_distance)
        
        ##################################################################
        # Trim map lane boundaries
        ##################################################################

        # find the distance of current position
        left_cur_pos_dist = global_path_left_boundary.project(shapely.Point(current_pose_openpilot.x, current_pose_openpilot.y, current_pose_openpilot.z))
        right_cur_pos_dist = global_path_right_boundary.project(shapely.Point(current_pose_openpilot.x, current_pose_openpilot.y, current_pose_openpilot.z))
        
        # cut out the relevant sections from the global path boundaries 
        map_left_lane_boundary = shpops.substring(global_path_left_boundary, left_cur_pos_dist, left_cur_pos_dist + self.lookahead_distance)
        map_right_lane_boundary = shpops.substring(global_path_right_boundary, right_cur_pos_dist, right_cur_pos_dist + self.lookahead_distance)

        ##################################################################
        # Perform matching
        ##################################################################
    
        no_correction = False
        if self.only_lateral_correction:
            # calculate the average difference for right and left boundaries
            x, y = self.find_average_distance(map_left_lane_boundary, map_right_lane_boundary, 
                                                openpilot_left_lane_boundary, openpilot_right_lane_boundary)
            yaw = 0
        else:
            # use optimizer to find the best match between map boundaries and openpilot boundaries
            result = scipy.optimize.minimize(self.objective_function, np.array([0, 0, 0]), method='Nelder-Mead', 
                                             args=(current_pose, map_left_lane_boundary, map_right_lane_boundary, 
                                                   openpilot_left_lane_boundary, openpilot_right_lane_boundary))
            x, y, yaw = result.x


        # if the difference between map and openpilot lane boundaries is too big then don't use the correction
        if abs(x) > self.x_correction_treshold or abs(y) > self.y_correction_treshold or abs(yaw) > self.yaw_correction_treshold:
            x, y, yaw = 0, 0, 0
            no_correction = True
        
        # use exponential moving average to smooth coordinate corrections 
        self.x_correction = self.calculate_updated_correction(x, self.x_correction)
        self.y_correction = self.calculate_updated_correction(y, self.y_correction)
        self.yaw_correction = self.calculate_updated_correction(yaw, self.yaw_correction)

        self.publish_base_link_correction_tf(correction_stamp=msg.header.stamp)

        ##################################################################
        # Visualization
        ##################################################################

        lanes_marker_array = MarkerArray()
        marker1 = self.get_lane_boundary_marker(openpilot_right_lane_boundary, "right", 0, "map_corrected", 
                                                ColorRGBA(0.5, 0.8, 0.7, 1.0) if no_correction else ColorRGBA(0.0, 1.0, 0.7, 1.0))
        marker2 = self.get_lane_boundary_marker(openpilot_left_lane_boundary, "left", 1, "map_corrected", 
                                                ColorRGBA(0.5, 0.8, 0.7, 1.0) if no_correction else ColorRGBA(0.0, 1.0, 0.7, 1.0))
        lanes_marker_array.markers.append(marker1)
        lanes_marker_array.markers.append(marker2)

        marker3 = self.get_lane_boundary_marker(map_right_lane_boundary, "right", 2, "map", 
                                                ColorRGBA(0.6, 0.5, 0.4, 1.0) if no_correction else ColorRGBA(0.6, 0.3, 0.0, 1.0))
        marker4 = self.get_lane_boundary_marker(map_left_lane_boundary, "left", 3, "map", 
                                                ColorRGBA(0.6, 0.5, 0.4, 1.0) if no_correction else ColorRGBA(0.6, 0.3, 0.0, 1.0))
        lanes_marker_array.markers.append(marker3)
        lanes_marker_array.markers.append(marker4)
        
        self.lane_bound_markers_pub.publish(lanes_marker_array)

    def global_path_callback(self, msg):
        if len(msg.waypoints) == 0:
            with self.global_path_lock:
                self.global_path_left_boundary = None
                self.global_path_right_boundary = None
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
            self.global_path = PathWrapper(msg.waypoints)
            self.global_path_left_boundary = left_boundary
            self.global_path_right_boundary = right_boundary

    def find_average_distance(self, map_left_lane_boundary, map_right_lane_boundary, openpilot_left_lane_boundary, openpilot_right_lane_boundary):
        left_x_diffs, left_y_diffs = [], []
        right_x_diffs, right_y_diffs = [], []

        dists = np.linspace(0, self.lookahead_distance, len(openpilot_left_lane_boundary.coords))

        for i in range(len(dists)):
            left_point = map_left_lane_boundary.interpolate(dists[i])
            right_point = map_right_lane_boundary.interpolate(dists[i])

            left_x, left_y, left_z = openpilot_left_lane_boundary.coords[i]
            right_x, right_y, right_z = openpilot_right_lane_boundary.coords[i]

            left_x_diffs.append(left_point.x - left_x)
            left_y_diffs.append(left_point.y - left_y)

            right_x_diffs.append(right_point.x - right_x)
            right_y_diffs.append(right_point.y - right_y)
            

        x_diffs = (np.array(left_x_diffs) + np.array(right_x_diffs)) / 2
        y_diffs = (np.array(left_y_diffs) + np.array(right_y_diffs)) / 2
        
        return np.mean(x_diffs), np.mean(y_diffs)
    
    def objective_function(self, input_values, current_pose, map_left_lane_boundary, map_right_lane_boundary, openpilot_left_lane_boundary, openpilot_right_lane_boundary):
        x_correction, y_correction, yaw_correction = input_values

        # Steps for correcting the car localization error:
        # 1. Move map origin to base_link
        # 2. Add yaw correction
        # 3. Move map origin back to (0,0,0) and add corrections for x and y coordinates
        matrix = compose_matrix(translate=[-current_pose.x, -current_pose.y, -current_pose.z], angles=[0, 0, yaw_correction])

        matrix[0, 3] += current_pose.x + x_correction
        matrix[1, 3] += current_pose.y + y_correction
        matrix[2, 3] += current_pose.z # correction for z-axis won't be calculated

        openpilot_left_lane_boundary_h = np.hstack((np.array(openpilot_left_lane_boundary.coords), np.ones((len(openpilot_left_lane_boundary.coords), 1))))
        openpilot_right_lane_boundary_h = np.hstack((np.array(openpilot_right_lane_boundary.coords), np.ones((len(openpilot_right_lane_boundary.coords), 1))))

        openpilot_left_lane_bound_map_fr_h = openpilot_left_lane_boundary_h @ matrix.T
        openpilot_left_lane_bound_map_fr = openpilot_left_lane_bound_map_fr_h[:, :3]

        openpilot_right_lane_bound_map_fr_h = openpilot_right_lane_boundary_h @ matrix.T
        openpilot_right_lane_bound_map_fr = openpilot_right_lane_bound_map_fr_h[:, :3]

        openpilot_left_map_lane_boundary_map_fr = shapely.LineString(openpilot_left_lane_bound_map_fr)
        openpilot_right_map_lane_boundary_map_fr = shapely.LineString(openpilot_right_lane_bound_map_fr)

        return shapely.hausdorff_distance(map_left_lane_boundary, openpilot_left_map_lane_boundary_map_fr) + shapely.hausdorff_distance(map_right_lane_boundary, openpilot_right_map_lane_boundary_map_fr)

    def get_lane_boundary_marker(self, lane_boundary, side, marker_id, frame_id, marker_color):
        points = []
        for x, y, z in lane_boundary.coords:
            point = Point(x=x,y=y, z=z)
            points.append(point)

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = f"{side} bound"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.color = marker_color
        marker.points = points

        return marker

    def publish_base_link_correction_tf(self, correction_stamp=None):
        t = TransformStamped()

        x_correction = self.x_correction
        y_correction = self.y_correction
        z_correction = self.z_correction
        yaw_correction = self.yaw_correction

        if correction_stamp is None:
            t.header.stamp = rospy.Time.now()
        else:
            t.header.stamp = correction_stamp
        t.header.frame_id = "map"
        t.child_frame_id = "map_corrected"

        matrix = compose_matrix(translate=[x_correction, y_correction, z_correction], angles=[0, 0, yaw_correction])
        t.transform = msgify(Transform, matrix)
        self.transform_matrix = matrix

        if correction_stamp is None:
            static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
            static_tf_broadcaster.sendTransform(t)
        else:
            self.tf_broadcaster.sendTransform(t)

    def run(self):
        rospy.spin()
    
def float32_multiarray_to_numpy(multiarray):
    dims = tuple(map(lambda x: x.size, multiarray.layout.dim))
    data = multiarray.data[multiarray.layout.data_offset:]
    return np.array(data, dtype=np.float32).reshape(dims)

if __name__ == '__main__':
    rospy.init_node('lane_boundary_matcher')
    node = LaneBoundaryMatcher()
    node.run()