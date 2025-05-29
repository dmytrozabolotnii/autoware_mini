#!/usr/bin/env python3

import rospy
import math
import numpy as np
import cv2

from tf2_ros import TransformListener, Buffer, TransformException
from ros_numpy import numpify

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
from autoware_mini.msg import DetectedObjectArray, DetectedObject
from geometry_msgs.msg import Point32

from autoware_mini.geometry import get_orientation_from_heading

BLUE = ColorRGBA(0.0, 0.0, 1.0, 0.5)

class ClusterDetectorFast:
    def __init__(self):
        self.bounding_box_type = rospy.get_param('~bounding_box_type')
        self.output_frame = rospy.get_param('/detection/output_frame')
        self.transform_timeout = rospy.get_param('~transform_timeout')

        if self.bounding_box_type not in ["axis_aligned", "min_area"]:
            raise ValueError(f"{rospy.get_name()} - 'bounding_box_type' must be one of 'axis_aligned' or 'min_area', not '{self.bounding_box_type}'")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.objects_pub = rospy.Publisher('detected_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('points_clustered', PointCloud2, self.points_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

        rospy.loginfo("%s - initialized", rospy.get_name())
        self.totals = [0, 0, 0, 0, 0, 0, 0]
        self.count = 0

    def points_callback(self, msg):
        t0 = time.perf_counter()
        data = numpify(msg)

        t1 = time.perf_counter()

        # convert point cloud into ndarray, take only xyz coordinates
        points_homogeneous = np.stack([data['x'], data['y'], data['z'], data['z']], axis=-1).reshape(-1, 4).astype(np.float32)
        points_homogeneous[:, 3] = 1.0  # Add homogeneous coordinate
        labels = data['label']

        t2 = time.perf_counter()

        # if target frame does not match the header frame
        if msg.header.frame_id != self.output_frame:
            # fetch transform for target frame
            try:
                transform = self.tf_buffer.lookup_transform(self.output_frame, msg.header.frame_id, msg.header.stamp, rospy.Duration(self.transform_timeout))
            except (TransformException, rospy.ROSTimeMovedBackwardsException) as e:
                rospy.logwarn("%s - %s", rospy.get_name(), e)
                return
            tf_matrix = numpify(transform.transform).astype(np.float32)
            # transform points to target frame
            points_homogeneous = points_homogeneous.dot(tf_matrix.T)

        t3 = time.perf_counter()
        
        # create detected objects
        objects = DetectedObjectArray()
        objects.header.stamp = msg.header.stamp
        objects.header.frame_id = self.output_frame

        # sort points by labels
        sorted_indices = np.argsort(labels)
        sorted_labels = labels[sorted_indices]
        sorted_points = points_homogeneous[sorted_indices]

        # get the split indices
        unique_labels, label_starts, label_counts = np.unique(sorted_labels, return_index=True, return_counts=True)
        label_ends = label_starts + label_counts

        for label, start, end in zip(unique_labels, label_starts, label_ends):
            # fetch points for this cluster
            points3d = sorted_points[start:end,:3]
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
                heading = math.radians(heading_angle)

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
            object.id = label
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

        t4 = time.perf_counter()
        self.totals[0] += (t4 - t0)*1000
        self.totals[1] += (t1 - t0)*1000
        self.totals[2] += (t2 - t1)*1000
        self.totals[3] += (t3 - t2)*1000
        self.totals[4] += (t4 - t3)*1000
        self.count += 1
        print(f"CLUSTER DETECTOR: Total time: {self.totals[0] / self.count:.2f} | Numpify time: {self.totals[1] / self.count:.2f} | Unstructured time: {self.totals[2] / self.count:.2f} | Transform time: {self.totals[3] / self.count:.2f} | Publishing time: {self.totals[4] / self.count:.2f}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('cluster_detector_fast', log_level=rospy.INFO)
    node = ClusterDetectorFast()
    node.run()
