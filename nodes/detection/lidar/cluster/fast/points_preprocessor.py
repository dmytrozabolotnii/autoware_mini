#!/usr/bin/env python3

import rospy
import numpy as np
import cupy as cp
import message_filters

from tf2_ros import TransformListener, Buffer, TransformException
from sensor_msgs.msg import PointCloud2
from cuml.cluster import DBSCAN
from ros_numpy import numpify, msgify

from helpers.naive_ground_detector import NaiveGroundDetectorFast

import time

class PointsPreprocessor:
    def __init__(self):

        points_topics = rospy.get_param("~points_topics")
        self.output_frame = rospy.get_param("~output_frame")
        self.transform_timeout = rospy.get_param('~transform_timeout')

        # Crop box parameters
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

        # Ground detection parameters
        cell_size = rospy.get_param('~cell_size')
        tolerance = rospy.get_param('~tolerance')
        filter_size = rospy.get_param('~filter_size')
        filter_iterations = rospy.get_param('~filter_iterations')

        # Voxel grid filter parameters
        self.voxel_grid_filter_leaf_size = rospy.get_param('~voxel_grid_filter_leaf_size')

        # Clustering parameters
        self.cluster_epsilon = rospy.get_param('~cluster_epsilon')
        self.cluster_min_size = rospy.get_param('~cluster_min_size')
        self.cluster_in_2d = rospy.get_param('~cluster_in_2d')

        self.outer_min_gpu = cp.array([outer_min_x, outer_min_y, outer_min_z])
        self.outer_max_gpu = cp.array([outer_max_x, outer_max_y, outer_max_z])
        self.inner_min_gpu = cp.array([inner_min_x, inner_min_y, inner_min_z])
        self.inner_max_gpu = cp.array([inner_max_x, inner_max_y, inner_max_z])

        self.ground_detector = NaiveGroundDetectorFast(outer_min_x, outer_max_x, outer_min_y, outer_max_y, cell_size, tolerance, filter_size, filter_iterations)
        
        self.clusterer = DBSCAN(eps=self.cluster_epsilon, min_samples=self.cluster_min_size)

        # TF buffer setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        # Warmup GPU with dummy data
        for _ in range(3):
            points_gpu = cp.random.rand(131072, 4).astype(cp.float32) * 100
            transform_matrix_gpu= cp.array([[0.6492562, 0.7605143, 0.00918187, 0.], [-0.76056415, 0.64925057, 0.00399486, 0.], 
                                            [-0.00292319, -0.00957709, 0.9999499, 0.], [ 0.8559, 0.0642, -0.4051, 1.]]).astype(cp.float32)
            nan_mask = cp.random.rand(*points_gpu.shape) < .1 # Randomly set 10% of elements to NaN
            points_gpu[nan_mask] = cp.nan
            mask = cp.all(~cp.isnan(points_gpu), axis=1)
            points_gpu[mask]
            cp.matmul(points_gpu, transform_matrix_gpu)[:, :3]
            self.ground_detector.detect_ground(points_gpu[:, :3])
            self.voxel_grid_filter_gpu(points_gpu[:, :3], self.voxel_grid_filter_leaf_size)

        # Publisher
        self.points_clustered_pub = rospy.Publisher('points_clustered', PointCloud2, queue_size=1, tcp_nodelay=True)
        self.points_ground_pub = rospy.Publisher('points_ground', PointCloud2, queue_size=1, tcp_nodelay=True)
        self.points_no_ground_pub = rospy.Publisher('points_no_ground', PointCloud2, queue_size=1, tcp_nodelay=True)

        subscribers = []
        for topic in points_topics:
            subscribers.append(message_filters.Subscriber(topic, PointCloud2, queue_size=1, buff_size=2**24, tcp_nodelay=True))

        if not subscribers:
            raise ValueError("No topics to subscribe to.")

        ts = message_filters.ApproximateTimeSynchronizer(subscribers, queue_size=4, slop=0.1)
        ts.registerCallback(self.synced_pointcloud_callback)

        self.transforms = [None] * len(subscribers)

        rospy.loginfo("%s - initialized", rospy.get_name())
        self.totals = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.count = 0

    def synced_pointcloud_callback(self, *msgs):
        t0 = time.perf_counter()
        pointclouds = []
        for i, msg in enumerate(msgs):
            points_array = numpify(msg)
            if msg.header.frame_id == self.output_frame:
                points = np.stack([points_array['x'], points_array['y'], points_array['z']], axis=-1).reshape(-1, 3)
                points = cp.asarray(points)

                t1 = time.perf_counter()

                mask = cp.all(~cp.isnan(points), axis=1)
                points = points[mask]
                pointclouds.append(points)
            
            else:
                # Static transforms, fetch only once
                if self.transforms[i] is None:
                    try:
                        transform = self.tf_buffer.lookup_transform(self.output_frame, msg.header.frame_id, rospy.Time(0), rospy.Duration(self.transform_timeout))
                    except (TransformException, rospy.ROSTimeMovedBackwardsException) as e:
                        rospy.logwarn("%s - %s", rospy.get_name(), e)
                        return
                
                    transfrom_matrix = numpify(transform.transform).T
                    transfrom_matrix_gpu = cp.asarray(transfrom_matrix).astype(cp.float32) # Use GPU acceleration
                    self.transforms[i] = transfrom_matrix_gpu

                untransformed_points = np.stack([points_array['x'], points_array['y'], points_array['z'], points_array['z']], axis=-1).reshape(-1, 4)
                untransformed_points[:, 3] = 1.0 # Add homogeneous coordinate
                untransformed_points = cp.asarray(untransformed_points)

                t1 = time.perf_counter()
                
                mask = cp.all(~cp.isnan(untransformed_points), axis=1)
                untransformed_points = untransformed_points[mask]
                # Transform points to output frame
                points = cp.matmul(untransformed_points, self.transforms[i])
                pointclouds.append(points[:, :3])

        t2 = time.perf_counter()

        # Concatenate poionts
        points_concatenated = cp.concatenate(pointclouds, axis=0)

        t3 = time.perf_counter()

        # Filter points
        in_outer = cp.all((points_concatenated >= self.outer_min_gpu) & (points_concatenated <= self.outer_max_gpu), axis=1)
        in_inner = cp.all((points_concatenated >= self.inner_min_gpu) & (points_concatenated <= self.inner_max_gpu), axis=1)
        mask = in_outer & (~in_inner)
        points_filtered = points_concatenated[mask]

        t4 = time.perf_counter()

        # Remove ground points
        ground_mask = self.ground_detector.detect_ground(points_filtered)
        points_no_ground = points_filtered[~ground_mask]

        t5 = time.perf_counter()
        
        # Downsample points
        points_downsampled = self.voxel_grid_filter_gpu(points_no_ground, self.voxel_grid_filter_leaf_size)

        t6 = time.perf_counter()

        points_downsampled = cp.asnumpy(points_downsampled).astype(np.float32)

        # Cluster points
        labels = self.clusterer.fit_predict(points_downsampled[:, :2] if self.cluster_in_2d else points_downsampled)
        valid_labels = labels[labels != -1] # remove noise label (-1)

        counts = np.bincount(valid_labels)
        valid_labels = np.where(counts >= self.cluster_min_size)[0]

        filter_mask = np.isin(labels, valid_labels)

        cluster_labels = labels[filter_mask]
        points_clustered = points_downsampled[filter_mask]

        t7 = time.perf_counter()

        # Publish points
        self.publish_points(points_clustered, cluster_labels, msgs[0].header.stamp, self.points_clustered_pub)

        t8 = time.perf_counter()

        self.totals[0] += (t8 - t0)*1000
        self.totals[1] += (t1 - t0)*1000
        self.totals[2] += (t2 - t1)*1000
        self.totals[3] += (t3 - t2)*1000
        self.totals[4] += (t4 - t3)*1000
        self.totals[5] += (t5 - t4)*1000
        self.totals[6] += (t6 - t5)*1000
        self.totals[7] += (t7 - t6)*1000
        self.totals[8] += (t8 - t7)*1000
        self.count += 1

        print(f"PREPROCESSOR: Total time: {self.totals[0] / self.count:.2f} | Unstructured time: {self.totals[1] / self.count:.2f} | Transform time: {self.totals[2] / self.count:.2f} | Concatenate time: {self.totals[3] / self.count:.2f} | Crop time: {self.totals[4] / self.count:.2f} | Ground removal time: {self.totals[5] / self.count:.2f} | Downsampling time: {self.totals[6] / self.count:.2f} | Clustering time: {self.totals[7] / self.count:.2f} | Publishing time: {self.totals[8] / self.count:.2f}")

        if self.points_ground_pub.get_num_connections() > 0:
            points_ground = points_filtered[ground_mask]
            points_ground = cp.asnumpy(points_ground).astype(np.float32)
            self.publish_points(points_ground, None, msgs[0].header.stamp, self.points_ground_pub)

        if self.points_no_ground_pub.get_num_connections() > 0:
            points_no_ground = cp.asnumpy(points_no_ground).astype(np.float32)
            self.publish_points(points_no_ground, None, msgs[0].header.stamp, self.points_no_ground_pub)
    
    def voxel_grid_filter_gpu(self, points, voxel_size):
        """
        Voxel grid downsampling with GPU acceleration.
        Args:
            points: Nx3 cupy array (x, y, z)
            voxel_size: tuple or float, e.g., (0.1, 0.1, 0.1)
        Returns:
            Downsampled points (as cupy array).
        """
        if isinstance(voxel_size, float):
            voxel_size = (voxel_size, voxel_size, voxel_size)

        # Compute voxel indices
        voxel_indices = cp.floor(points / cp.asarray(voxel_size)).astype(cp.int32)

        # Hash voxel indices into scalar keys
        # Assumption: coordinates are within reasonable bounds (e.g., [-1000, 1000])
        hash_scale = cp.array([73856093, 19349669, 83492791], dtype=cp.int64)  # large primes
        voxel_hashes = cp.sum(voxel_indices.astype(cp.int64) * hash_scale, axis=1)

        # Unique voxel hashes and corresponding first indices
        _, unique_indices = cp.unique(voxel_hashes, return_index=True)
        
        # Select representative points (first in voxel)
        downsampled = points[unique_indices]
    
        return downsampled
    
    def publish_points(self, points, labels, stamp, publisher):
        """
        Publish points as PointCloud2 message.
        """
        dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
        if labels is not None:
            dtype.append(('label', np.int32))

        data = np.empty(points.shape[0], dtype=dtype)
        data['x'] = points[:, 0]
        data['y'] = points[:, 1]
        data['z'] = points[:, 2]
        if labels is not None:
            data['label'] = labels
        
        points_msg = msgify(PointCloud2, data)
        points_msg.header.stamp = stamp
        points_msg.header.frame_id = self.output_frame
        publisher.publish(points_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('points_preprocessor', log_level=rospy.INFO)
    node = PointsPreprocessor()
    node.run()
