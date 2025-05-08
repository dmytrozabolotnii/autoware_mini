#!/usr/bin/env python3

import rospy
import numpy as np
import cupy as cp
import message_filters

from tf2_ros import TransformListener, Buffer
from sensor_msgs.msg import PointCloud2
from ros_numpy import numpify, msgify

from helpers.naive_ground_removal import NaiveGroundRemoval


class PointsPreprocessor:
    def __init__(self):

        self.use_lidar_center = rospy.get_param("~use_lidar_center")
        self.use_lidar_front = rospy.get_param("~use_lidar_front")

        outer_min_x = rospy.get_param('~outer_min_x')
        outer_max_x = rospy.get_param('~outer_max_x')
        outer_min_y = rospy.get_param('~outer_min_y')
        outer_max_y = rospy.get_param('~outer_max_y')
        outer_min_z = rospy.get_param('~outer_min_z')
        outer_max_z = rospy.get_param('~outer_max_z')

        inner_min_x = rospy.get_param('~inner_min_x')
        inner_max_x = rospy.get_param('~inner_max_x')
        inner_min_y = rospy.get_param('~inner_min_y')
        inner_max_y = rospy.get_param('~inner_max_y')
        inner_min_z = rospy.get_param('~inner_min_z')
        inner_max_z = rospy.get_param('~inner_max_z')

        ground_removal_algorithm = rospy.get_param("~ground_removal")
        cell_size = rospy.get_param('~cell_size')
        tolerance = rospy.get_param('~tolerance')
        filter = rospy.get_param('~filter')
        filter_size = rospy.get_param('~filter_size')
        filter_iterations = rospy.get_param('~filter_iterations')

        self.outer_min = np.array([outer_min_x, outer_min_y, outer_min_z])
        self.outer_max = np.array([outer_max_x, outer_max_y, outer_max_z])
        self.inner_min = np.array([inner_min_x, inner_min_y, inner_min_z])
        self.inner_max = np.array([inner_max_x, inner_max_y, inner_max_z])

        if ground_removal_algorithm == "naive":
            self.ground_removal = NaiveGroundRemoval(outer_min_x, outer_max_x, outer_min_y, outer_max_y, cell_size, tolerance, 
                                                     filter, filter_size, filter_iterations)
        else:
            raise ValueError(f"{rospy.get_name()} - 'ground_removal' must be one of 'naive' or 'jcp', not '{ground_removal_algorithm}'")

        # TF buffer setup
        tf_buffer = Buffer()
        tf_listener = TransformListener(tf_buffer)
        # Allow time for tf buffer to fill
        rospy.sleep(0.5)

        # Static transform, fetch once
        transform = tf_buffer.lookup_transform("lidar_center", "lidar_front/os_sensor", rospy.Time(0))
        self.transfrom_matrix = numpify(transform.transform).astype(np.float32).T
        self.transfrom_matrix_gpu = cp.asarray(self.transfrom_matrix).astype(cp.float32) # Use GPU acceleration

        # Warmup GPU with dummy data
        for _ in range(3):
            points_gpu = cp.random.rand(131072, 4).astype(cp.float32) * 10
            nan_mask = np.random.rand(*points_gpu.shape) < 10. # Randomly set 10% of elements to NaN
            points_gpu[nan_mask] = np.nan
            self.remove_nan_rows_gpu(points_gpu)
            self.transform_to_lidar_center_gpu(points_gpu)
            self.crop_and_filter_gpu(points_gpu[:, :3])
            self.ground_removal.remove_ground(points_gpu[:, :3])

        # Publisher
        self.points_processed_pub = rospy.Publisher('points_processed', PointCloud2, queue_size=1, tcp_nodelay=True)

        if self.use_lidar_center and self.use_lidar_front:
            # Subscribe to both clouds using message_filters
            self.points1_sub = message_filters.Subscriber("points1", PointCloud2)
            self.points2_sub = message_filters.Subscriber("points2", PointCloud2)

            self.sync = message_filters.ApproximateTimeSynchronizer([self.points1_sub, self.points2_sub], queue_size=4, slop=0.1)
            self.sync.registerCallback(self.synced_pointcloud_callback)
        else:
            # Only one cloud, regular subscriber
            rospy.Subscriber("points1", PointCloud2, self.pointcloud_callback)

        rospy.loginfo("%s - initialized", rospy.get_name())

    def synced_pointcloud_callback(self, points1_msg, points2_msg):
        points1_array = numpify(points1_msg)
        points2_array = numpify(points2_msg)

        points1 = np.stack([points1_array['x'], points1_array['y'], points1_array['z']], axis=-1).reshape(-1, 3)
        points1 = cp.asarray(points1)

        untransformed_points2 = np.stack([points2_array['x'], points2_array['y'], points2_array['z']], axis=-1).reshape(-1, 3)
        untransformed_points2 = cp.asarray(untransformed_points2)

        # Transform lidar front points to lidar center frame
        untransformed_points2 = self.remove_nan_rows_gpu(untransformed_points2)
        untransformed_points2 = cp.concatenate((untransformed_points2, cp.ones((untransformed_points2.shape[0], 1))), axis=-1) # Homogeneous coordinates
        points2 = self.transform_to_lidar_center_gpu(untransformed_points2)

        # Concatenate poionts
        points_concatenated = cp.concatenate((points1, points2), axis=0)

        # Filter points
        points_filtered = self.crop_and_filter_gpu(points_concatenated)

        # Remove ground points
        points_no_ground = self.ground_removal.remove_ground(points_filtered)

        # Publish points
        self.publish_points(cp.asnumpy(points_no_ground).astype(np.float32), points1_msg.header)

    def pointcloud_callback(self, msg):
        points_array = numpify(msg)

        points = np.stack([points_array['x'], points_array['y'], points_array['z']], axis=-1).reshape(-1, 3)
        points = cp.asarray(points)

        # Transform lidar front points to lidar center frame if incoming points are from lidar front
        if self.use_lidar_front:
            untransformed_points = self.remove_nan_rows_gpu(points)
            untransformed_points = cp.concatenate((untransformed_points, cp.ones((untransformed_points.shape[0], 1))), axis=-1) # Homogeneous coordinates
            points = self.transform_to_lidar_center_gpu(untransformed_points)

        # Filter points
        points_filtered = self.crop_and_filter_gpu(points)

        # Remove ground points
        points_no_ground = self.ground_removal.remove_ground(points_filtered)

        # Publish points
        self.publish_points(cp.asnumpy(points_no_ground).astype(np.float32), msg.header)

    def crop_and_filter_gpu(self, points):
        """
        Transform points from source_frame to target_frame with GPU acceleration.
        Input points should be in homogeneous coordinates (x, y, z, 1).
        """
        points_gpu = cp.asarray(points)
        outer_min_gpu = cp.asarray(self.outer_min)
        outer_max_gpu = cp.asarray(self.outer_max)
        inner_min_gpu = cp.asarray(self.inner_min)
        inner_max_gpu = cp.asarray(self.inner_max)
        in_outer = cp.all((points_gpu >= outer_min_gpu) & (points_gpu <= outer_max_gpu), axis=1)
        in_inner = cp.all((points_gpu >= inner_min_gpu) & (points_gpu <= inner_max_gpu), axis=1)
        mask = in_outer & (~in_inner)
        return points_gpu[mask]
    
    def remove_nan_rows_gpu(self, points_gpu):

        # Create a boolean mask where rows with no NaN values are marked as True
        mask = cp.all(~cp.isnan(points_gpu), axis=1)
        
        # Use the mask to filter out rows with NaN values
        return points_gpu[mask]
    
    def transform_to_lidar_center_gpu(self, points):
        """
        Transform points from lidar_front frame to lidar_center frame with GPU acceleration.
        Input points should be in homogeneous coordinates (x, y, z, 1).
        """
        
        points_gpu = cp.asarray(points).astype(cp.float32)
        transformed_points_gpu = cp.matmul(points_gpu, self.transfrom_matrix_gpu)

        return transformed_points_gpu[:, :3]
    
    def publish_points(self, points, header):
        """
        Publish points as PointCloud2 message.
        """
        dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
        points_data = points.view(dtype).reshape(-1)
        points_msg = msgify(PointCloud2, points_data)
        points_msg.header.stamp = header.stamp
        points_msg.header.frame_id = "lidar_center"
        self.points_processed_pub.publish(points_msg)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('points_preprocessor', log_level=rospy.INFO)
    node = PointsPreprocessor()
    node.run()
