#!/usr/bin/env python3

import rospy
import math
import numpy as np
import cv2

from tf2_ros import TransformListener, Buffer, TransformException
from ros_numpy import numpify

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point32
from std_msgs.msg import ColorRGBA
from autoware_mini.msg import DetectedObjectArray, DetectedObject

BLUE = ColorRGBA(0.0, 0.0, 1.0, 0.5)
DEG2RAD = math.pi / 180.0

class ClusterDetectorFast:
    def __init__(self):
        self.cluster_epsilon = rospy.get_param('~cluster_epsilon')
        self.cluster_min_size = rospy.get_param('~cluster_min_size')
        self.cluster_in_2d = rospy.get_param('~cluster_in_2d')
        self.bounding_box_type = rospy.get_param('~bounding_box_type')
        self.output_frame = rospy.get_param('/detection/output_frame')
        self.transform_timeout = rospy.get_param('~transform_timeout')

        try:
            from cuml.cluster import DBSCAN
            self.clusterer = DBSCAN(eps=self.cluster_epsilon, min_samples=self.cluster_min_size)
            rospy.loginfo("Using DBSCAN from cuML")
            # Warm up the GPU
            for _ in range(3):
                points = np.random.rand(1000, 3).astype(np.float32)
                labels = self.clusterer.fit_predict(points[:, :2] if self.cluster_in_2d else points)
        except ImportError:
            try:
                from sklearnex.cluster import DBSCAN
                self.clusterer = DBSCAN(eps=self.cluster_epsilon, min_samples=self.cluster_min_size, algorithm='auto')
                rospy.loginfo("Using DBSCAN from Intel® Extension for Scikit-learn")
            except ImportError:
                from sklearn.cluster import DBSCAN
                self.clusterer = DBSCAN(eps=self.cluster_epsilon, min_samples=self.cluster_min_size, algorithm='ball_tree')
                rospy.loginfo("Using DBSCAN from Scikit-learn")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.objects_pub = rospy.Publisher('detected_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('points_processed', PointCloud2, self.points_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

        rospy.loginfo("%s - initialized", rospy.get_name())

    def points_callback(self, msg):
        data = numpify(msg)

        # convert point cloud into ndarray, take only xyz coordinates
        points = np.stack([data['x'], data['y'], data['z']], axis=-1).reshape(-1, 3)
        points = points.astype(np.float32)

        # get labels for clusters
        labels = self.clusterer.fit_predict(points[:, :2] if self.cluster_in_2d else points)

        # concatenate points with labels
        points_labeled = np.hstack((points, labels.reshape(-1, 1)))

        # filter out noise points
        points_labeled = points_labeled[labels != -1]
        rospy.logdebug("%s - %d points, %d clusters", rospy.get_name(), len(points), np.max(labels) + 1)

        # if target frame does not match the header frame
        if msg.header.frame_id != self.output_frame:
            # fetch transform for target frame
            try:
                transform = self.tf_buffer.lookup_transform(self.output_frame, msg.header.frame_id, msg.header.stamp, rospy.Duration(self.transform_timeout))
            except (TransformException, rospy.ROSTimeMovedBackwardsException) as e:
                rospy.logwarn("%s - %s", rospy.get_name(), e)
                return
            tf_matrix = numpify(transform.transform).astype(np.float32)
            # make copy of points
            points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=-1) # Homogeneous coordinates
            # transform points to target frame
            points = points.dot(tf_matrix.T).astype(np.float32)

        # create detected objects
        objects = DetectedObjectArray()
        objects.header.stamp = msg.header.stamp
        objects.header.frame_id = self.output_frame

        if len(labels) == 0:
            num_clusters = 0
        else:
            num_clusters = np.max(labels) + 1

        for i in range(num_clusters):
            # filter points for this cluster
            idx = np.nonzero(labels == i)[0] #np.where(labels == i)[0]

            # ignore clusters smaller than certain size
            if len(idx) < self.cluster_min_size:
                continue

            # fetch points for this cluster
            points3d = points[idx,:3]
            points2d = np.ascontiguousarray(points3d[:,:2])

            if self.bounding_box_type == 'axis_aligned':
                # calculate centroid and dimensions
                maxs = np.max(points3d, axis=0)
                mins = np.min(points3d, axis=0)
                center_x, center_y, center_z = np.mean(points3d, axis=0)
                dim_x, dim_y, dim_z = maxs - mins
                min_z = mins[2]

                # always pointing forward
                heading = 0.0
            elif self.bounding_box_type == 'min_area':
                # calculate minimum area bounding box
                (center_x, center_y), (dim_x, dim_y), heading_angle = cv2.minAreaRect(points2d)

                # convert degrees to radians for heading angle
                heading = heading_angle * DEG2RAD

                # calculate height and vertical position
                z_points = points3d[:,2]
                max_z = float(z_points.max()) # native Python floats are faster with scalars
                min_z = float(z_points.min())

                dim_z = max_z - min_z
                center_z = (max_z + min_z) / 2.0
                
            else:
                assert False, "wrong bounding_box_type: " + self.bounding_box_type

            # create DetectedObject
            object = DetectedObject()
            object.id = i
            object.label = "unknown"
            object.color = BLUE
            object.valid = True
            object.position.x = center_x
            object.position.y = center_y
            object.position.z = center_z
            object.heading = heading
            object.dimensions.x = dim_x
            object.dimensions.y = dim_y
            object.dimensions.z = dim_z
            object.position_reliable = True
            object.velocity_reliable = False
            object.acceleration_reliable = False
            
            hull_points = cv2.convexHull(points2d)[:,0,:]

            object.convex_hull.points = [Point32(x, y, min_z) for x, y in hull_points]
            objects.objects.append(object)

        # publish detected objects message
        self.objects_pub.publish(objects)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('cluster_detector_fast', log_level=rospy.INFO)
    node = ClusterDetectorFast()
    node.run()
