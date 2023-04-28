#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division

import rospy
import math
import numpy as np
import cv2
import tf
from tf.transformations import quaternion_multiply, quaternion_from_euler, euler_from_quaternion
from ros_numpy import numpify
from geometry_msgs.msg import Quaternion, Point, PolygonStamped, Pose
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
from autoware_msgs.msg import DetectedObjectArray, DetectedObject

import onnxruntime
from scipy.ndimage.filters import maximum_filter

LIGHT_BLUE = ColorRGBA(0.5, 0.5, 1.0, 0.8)

## Model params
BEV_HEIGHT = 608  # number of rows of the bev image
BEV_WIDTH = 608  # number of columns of the bev image
DOWN_RATIO = 4
NUM_CLASSES = 3  # number of classes to detect
CLASS_NAMES = {0: "pedestrian", 1: "car", 2: "cyclist"}

class SFADetector:
    def __init__(self):
        rospy.loginfo(self.__class__.__name__ + " - Initializing")
        # Params
        self.onnx_path = rospy.get_param("~onnx_path", './kilynuaron_model.onnx')  # path of the trained model

        self.only_front = rospy.get_param("~only_front",False)  # only front detections. If False, both front and back detections enabled
        self.MIN_FRONT_X = rospy.get_param("~maxFrontX", 0)  # minimum distance on lidar +ve x-axis to process points at (front_detections)
        self.MAX_FRONT_X = rospy.get_param("~maxFrontX", 50)  # maximum distance on lidar +ve x-axis to process points at (back detections)
        self.MIN_BACK_X = rospy.get_param("~minBackX", -50)  # maxmimum distance on lidar -ve x-axis to process points at
        self.MAX_BACK_X = rospy.get_param("~maxBackX", 0)  # minimum distance on lidar -ve x-axis to process points at
        self.MIN_Y = rospy.get_param("~minY", -25)  # maxmimum distance on lidar -ve y-axis to process points at
        self.MAX_Y = rospy.get_param("~maxY", 25)  # maxmimum distance on lidar +ve y-axis to process points at
        self.MIN_Z = rospy.get_param("~minZ", -2.73)  # maxmimum distance on lidar -ve z-axis to process points at
        self.MAX_Z = rospy.get_param("~maxZ", 1.27)  # maxmimum distance on lidar +ve z-axis to process points at
        self.DISCRETIZATION = (self.MAX_FRONT_X - self.MIN_FRONT_X) / BEV_HEIGHT  # 3D world self.DISCRETIZATION to 2D image: distance encoded by 1 pixel
        self.BOUND_SIZE_X = self.MAX_FRONT_X - self.MIN_FRONT_X
        self.BOUND_SIZE_Y = self.MAX_Y - self.MIN_Y

        self.peak_thresh = rospy.get_param("~peak_thresh", 0.2)  # score filter
        self.K = rospy.get_param("~K", 50)  # number of top scoring detections to process
        self.output_frame = rospy.get_param("~output_frame", 'map')  # transform detected objects from lidar frame to this frame

        self.transform_timeout = rospy.get_param('~transform_timeout', 0.05)

        self.model = self.load_onnx(self.onnx_path) # get onnx model

        # transform listener
        self.tf_listener = tf.TransformListener()

        # Subscribers and Publishers
        self.detected_object_array_pub = rospy.Publisher('detected_objects', DetectedObjectArray, queue_size=1)
        rospy.Subscriber('points_raw', PointCloud2, self.pointcloud_callback, queue_size=1)

    def load_onnx(self, onnx_path):

        model = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        rospy.loginfo(self.__class__.__name__ + "- Loaded ONNX model file")

        return model

    def pointcloud_callback(self, pointcloud):
        """

        pointcloud: raw lidar data points
        return: None - publish autoware DetectedObjects
        """
        self.tf_listener.waitForTransform(self.output_frame, pointcloud.header.frame_id, pointcloud.header.stamp, rospy.Duration(self.transform_timeout))
        # get the transform (translation and rotation) from lidar_frame to output frame at the time when the poinctloud msg was published
        trans, tf_rot = self.tf_listener.lookupTransform(self.output_frame, pointcloud.header.frame_id, pointcloud.header.stamp)
        # convert the looked up transform to a 4x4 homogenous transformation matrix.
        tf_matrix = self.tf_listener.fromTranslationRotation(trans, tf_rot)
        # Unpack pointcloud2 msg ype to numpy array
        pcd_array = numpify(pointcloud)

        # Reshape the array into shape -> (num_points, 4). 4 corresponds to the fields -> x,y,z,intensity
        points = np.zeros((pcd_array.shape[0], 4))
        points[:, 0] = pcd_array['x']
        points[:, 1] = pcd_array['y']
        points[:, 2] = pcd_array['z']
        points[:, 3] = pcd_array['intensity']

        detected_objects_array = DetectedObjectArray()
        detected_objects_array.header.stamp = pointcloud.header.stamp
        detected_objects_array.header.frame_id = self.output_frame

        final_detections = self.detect(points, only_front=self.only_front)
        detected_objects_array.objects += self.generate_autoware_objects(final_detections, pointcloud.header, tf_matrix, tf_rot, len(detected_objects_array.objects))
        self.detected_object_array_pub.publish(detected_objects_array)

    def detect(self, points, only_front):

        if only_front:
            # process only detection from front FOV and return detections
            front_lidar = self.get_filtered_lidar(points)
            front_bev_map = self.make_bev_map(front_lidar)
            front_detections, front_bevmap = self.do_detection(front_bev_map, is_front=True)
            front_final_dets = self.convert_det_to_real_values(front_detections)

            return front_final_dets
        else:
            # process detections from both front and back FOVs and return detections
            #front
            front_lidar = self.get_filtered_lidar(points, is_front=True)
            front_bev_map = self.make_bev_map(front_lidar)
            front_detections, front_bevmap = self.do_detection(front_bev_map, is_front=True)
            front_final_dets = self.convert_det_to_real_values(front_detections)

            # back
            back_lidar = self.get_filtered_lidar(points, is_front=False)
            back_bev_map = self.make_bev_map(back_lidar)
            back_detections, back_bevmap, = self.do_detection(back_bev_map, is_front=False)
            back_final_dets = self.convert_det_to_real_values(back_detections)
            if back_final_dets.shape[0] != 0:
                back_final_dets[:, 1] *= -1
                back_final_dets[:, 2] *= -1

            # if both returning both front and back detections concatenate them along rows
            return np.concatenate((front_final_dets, back_final_dets), axis=0)

    def get_filtered_lidar(self, lidar, is_front=True):

        if is_front:
            # Remove the point out of range x,y,z
            mask = (lidar[:, 0] >= self.MIN_FRONT_X) & (lidar[:, 0] <= self.MAX_FRONT_X) & \
                    (lidar[:, 1] >= self.MIN_Y) & (lidar[:, 1] <= self.MAX_Y) & \
                    (lidar[:, 2] >= self.MIN_Z) & (lidar[:, 2] <= self.MAX_Z) \

        else:
            mask = (lidar[:, 0] >= self.MIN_BACK_X) & (lidar[:, 0] <= self.MAX_BACK_X) & \
                    (lidar[:, 1] >= self.MIN_Y) & (lidar[:, 1] <= self.MAX_Y) & \
                    (lidar[:, 2] >= self.MIN_Z) & (lidar[:, 2] <= self.MAX_Z) \

        lidar = lidar[mask]
        lidar[:, 2] -= self.MIN_Z

        return lidar

    def make_bev_map(self, pointcloud):
        height = BEV_HEIGHT + 1
        width = BEV_WIDTH + 1

        # Discretize Feature Map
        pointcloud = np.copy(pointcloud)
        pointcloud[:, 0] = np.int_(np.floor(pointcloud[:, 0] / self.DISCRETIZATION))
        pointcloud[:, 1] = np.int_(np.floor(pointcloud[:, 1] / self.DISCRETIZATION) + width / 2)

        # sort-3times
        sorted_indices = np.lexsort((-pointcloud[:, 2], pointcloud[:, 1], pointcloud[:, 0]))
        pointcloud = pointcloud[sorted_indices]
        _, unique_indices, unique_counts = np.unique(pointcloud[:, 0:2], axis=0, return_index=True, return_counts=True)
        pointcloud_top = pointcloud[unique_indices]

        # Height Map, Intensity Map & Density Map
        height_map = np.zeros((height, width))
        intensity_map = np.zeros((height, width))
        density_map = np.zeros((height, width))

        # some important problem is image coordinate is (y,x), not (x,y)
        max_height = float(np.abs(self.MAX_Z - self.MIN_Z))
        height_map[np.int_(pointcloud_top[:, 0]), np.int_(pointcloud_top[:, 1])] = pointcloud_top[:, 2] / max_height

        normalized_counts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64))
        intensity_map[np.int_(pointcloud_top[:, 0]), np.int_(pointcloud_top[:, 1])] = pointcloud_top[:, 3]
        density_map[np.int_(pointcloud_top[:, 0]), np.int_(pointcloud_top[:, 1])] = normalized_counts

        rgb_map = np.zeros((3, height - 1, width - 1))
        rgb_map[2, :, :] = density_map[:BEV_HEIGHT, :BEV_WIDTH]  # r_map
        rgb_map[1, :, :] = height_map[:BEV_HEIGHT, :BEV_WIDTH]  # g_map
        rgb_map[0, :, :] = intensity_map[:BEV_HEIGHT, :BEV_WIDTH]  # b_map

        return rgb_map

    def do_detection(self, bevmap, is_front):
        if not is_front:
            bevmap = np.flip(bevmap, (1, 2))

        input_bev_maps = np.expand_dims(bevmap, axis=0).astype(np.float32)
        onnx_outputs = self.model.run([], {"input":input_bev_maps})
        dict_keys = ['hm_cen', 'cen_offset', 'direction', 'z_coor', 'dim']
        outputs = {}
        for onnx_output, dict_key in zip(onnx_outputs, dict_keys):
            outputs[dict_key] = onnx_output

        outputs['hm_cen'] = self._sigmoid(outputs['hm_cen'])
        outputs['cen_offset'] = self._sigmoid(outputs['cen_offset'])

        # detections size (batch_size, K, 10)
        detections = self.decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'], outputs['dim'], self.K)
        detections = self.post_processing(detections)

        return detections[0], bevmap

    def _sigmoid(self, x):
        # Apply the sigmoid function element-wise to the input array
        sig_x = 1 / (1 + np.exp(-x))
        # Clamp the output values to be within the range [1e-4, 1 - 1e-4]
        sig_x = np.clip(sig_x, a_min=1e-4, a_max=1 - 1e-4)

        return sig_x

    def decode(self, hm_cen, cen_offset, direction, z_coor, dim, K=40):
        batch_size, num_classes, height, width = hm_cen.shape
        hm_cen = self._nms(hm_cen)
        scores, inds, clses, ys, xs = self._topk(hm_cen, K=K)
        if cen_offset is not None:
            cen_offset = self.transpose_and_gather_feat(cen_offset, inds)
            cen_offset = cen_offset.reshape(batch_size, K, 2)
            xs = xs.reshape(batch_size, K, 1) + cen_offset[:, :, 0:1]
            ys = ys.reshape(batch_size, K, 1) + cen_offset[:, :, 1:2]
        else:
            xs = xs.reshape(batch_size, K, 1) + 0.5
            ys = ys.reshape(batch_size, K, 1) + 0.5

        direction = self.transpose_and_gather_feat(direction, inds)
        direction = direction.reshape(batch_size, K, 2)
        z_coor = self.transpose_and_gather_feat(z_coor, inds)
        z_coor = z_coor.reshape(batch_size, K, 1)
        dim = self.transpose_and_gather_feat(dim, inds)
        dim = dim.reshape(batch_size, K, 3)
        clses = clses.reshape(batch_size, K, 1).astype(np.float32)
        scores = scores.reshape(batch_size, K, 1)
        # (scores x 1, ys x 1, xs x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
        # (scores-0:1, ys-1:2, xs-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
        # detections: [batch_size, K, 10]
        detections = np.concatenate([scores, xs, ys, z_coor, dim, direction, clses], axis=2)

        return detections

    def _nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2
        # Compute max pooling operation using SciPy
        hmax = maximum_filter(heat, size=kernel, mode='constant', cval=-np.inf)
        # Compute keep mask
        keep = (hmax == heat).astype(np.float32)
        # Apply keep mask to input tensor
        return heat * keep

    def _topk(self, scores, K=40):
        batch, cat, height, width = scores.shape
        scores_reshaped = scores.reshape(batch, cat, -1)
        topk_scores, topk_inds = self.topk(scores_reshaped, K)

        topk_inds = topk_inds % (height * width)
        topk_ys = np.floor_divide(topk_inds, width).astype(np.float32)
        topk_xs = (topk_inds % width).astype(np.float32)

        topk_scores_reshaped = topk_scores.reshape(batch, -1)
        topk_score, topk_ind = self.topk(topk_scores_reshaped, K)
        topk_clses = np.floor_divide(topk_ind, K).astype(np.int32)

        topk_inds = self.gather_feat(topk_inds.reshape(batch, -1, 1), topk_ind).reshape(batch, K)
        topk_ys = self.gather_feat(topk_ys.reshape(batch, -1, 1), topk_ind).reshape(batch, K)
        topk_xs = self.gather_feat(topk_xs.reshape(batch, -1, 1), topk_ind).reshape(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def topk(self,scores, K, dim=-1):
        # Compute the number of dimensions in the input array
        ndim = scores.ndim
        # Convert the specified dimension to a positive index
        if dim < 0:
            dim = ndim + dim
        # Compute the top K values and indices along the specified dimension
        topk_inds = np.argpartition(-scores, K, axis=dim)
        if ndim == 1:
            topk_inds = topk_inds[:K]
        else:
            topk_inds = topk_inds[(slice(None),) * dim + (slice(None, K),)]
        topk_vals = np.take_along_axis(scores, topk_inds, axis=dim)
        # Sort the top K values and indices along the specified dimension
        sort_inds = np.argsort(-topk_vals, axis=dim)
        topk_inds = np.take_along_axis(topk_inds, sort_inds, axis=dim)
        topk_vals = np.take_along_axis(topk_vals, sort_inds, axis=dim)
        # Return the top K values and indices
        return topk_vals, topk_inds

    def gather_feat(self, feat, ind, mask=None):
        dim = feat.shape[2]
        ind = np.repeat(ind[..., np.newaxis], dim, axis=2)
        feat = np.take_along_axis(feat, ind, axis=1)
        if mask is not None:
            mask = np.repeat(mask[..., np.newaxis], dim, axis=2)
            feat = feat[mask]
            feat = feat.reshape(-1, dim)
        return feat

    def transpose_and_gather_feat(self, feat, ind):
        feat = feat.transpose(0, 2, 3, 1)
        # make sure the array has a contiguous memory layout
        feat = np.ascontiguousarray(feat)
        feat = feat.reshape(feat.shape[0], -1, feat.shape[3])
        feat = self.gather_feat(feat, ind)
        return feat

    def post_processing(self,detections):
        """
        :param detections: [batch_size, K, 10]
        # (scores x 1, xs x 1, ys x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
        # (scores-0:1, xs-1:2, ys-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
        :return:
        """
        ret = []
        for i in range(detections.shape[0]):
            top_preds = {}
            classes = detections[i, :, -1]
            for j in range(NUM_CLASSES):
                inds = (classes == j)
                # x, y, z, h, w, l, yaw
                top_preds[j] = np.concatenate([
                    detections[i, inds, 0:1],
                    detections[i, inds, 1:2] * DOWN_RATIO,
                    detections[i, inds, 2:3] * DOWN_RATIO,
                    detections[i, inds, 3:4],
                    detections[i, inds, 4:5],
                    detections[i, inds, 5:6] / self.BOUND_SIZE_Y * BEV_WIDTH,
                    detections[i, inds, 6:7] / self.BOUND_SIZE_X * BEV_HEIGHT,
                    self.get_yaw(detections[i, inds, 7:9]).astype(np.float32)], axis=1)
                # Filter by peak_thresh
                if len(top_preds[j]) > 0:
                    keep_inds = (top_preds[j][:, 0] > self.peak_thresh)
                    top_preds[j] = top_preds[j][keep_inds]
            ret.append(top_preds)

        return ret

    def get_yaw(self, direction):
        return np.arctan2(direction[:, 0:1], direction[:, 1:2])

    def convert_det_to_real_values(self, detections, add_score=False):
        final_detections = []
        for cls_id in range(NUM_CLASSES):
            if len(detections[cls_id]) > 0:
                for det in detections[cls_id]:
                    # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                    _score, _x, _y, _z, _h, _w, _l, _yaw = det
                    _yaw = -_yaw
                    x = _y / BEV_HEIGHT * self.BOUND_SIZE_X + self.MIN_FRONT_X
                    y = _x / BEV_WIDTH * self.BOUND_SIZE_Y + self.MIN_Y
                    z = _z + self.MIN_Z
                    w = _w / BEV_WIDTH * self.BOUND_SIZE_Y
                    l = _l / BEV_HEIGHT * self.BOUND_SIZE_X
                    if add_score:
                        final_detections.append([cls_id, x, y, z, _h, w, l, _yaw, _score])
                    else:
                        final_detections.append([cls_id, x, y, z, _h, w, l, _yaw])

        return np.array(final_detections)

    def generate_autoware_objects(self, detections, header, tf_matrix, tf_rot, base=0):

        """
        Generate Autoware DetectedObject from Detections
        :param detections: SFA detections
        :param header: time stamp corresponding to poinctloud msg
        :param tf_matrix: 4x4 homogenous transformation matrix to go from lidar frame to output frame
        :param tf_rot: quaternion representing rotation to go from lidar frame to output frame
        :return: AutowareDetectedObject
        """
        detected_objects_list = []
        for i, (cls_id, x, y, z, height, width, length, yaw) in enumerate(detections):

            detected_object = DetectedObject()
            detected_object.id = i + base
            detected_object.header.frame_id = self.output_frame
            detected_object.header.stamp = header.stamp
            detected_object.label = CLASS_NAMES[cls_id]
            detected_object.color = LIGHT_BLUE
            detected_object.valid = True

            position_in_lidar = np.array([x, y, z, 1])
            orientation_in_lidar = quaternion_from_euler(0, 0, yaw)

            detected_object.pose = self.transform_pose(position_in_lidar, orientation_in_lidar, tf_matrix, tf_rot)
            detected_object.pose_reliable = True

            # object dimensions
            detected_object.dimensions.x = length
            detected_object.dimensions.y = width
            detected_object.dimensions.z = height
            # Populate convex hull
            detected_object.convex_hull = self.produce_hull(detected_object.pose, detected_object.dimensions, header.stamp)

            detected_objects_list.append(detected_object)

        return detected_objects_list

    def transform_pose(self, position, orientation, tf_matrix, tf_rot):

        # create Pose object.
        transformed_pose = Pose()
        # transform input pose's position using the transformation matrix
        transformed_x, transformed_y, transformed_z, _ = np.dot(tf_matrix, position)
        transformed_pose.position = Point(transformed_x, transformed_y, transformed_z)

        # transform the objects' orientation quaternion to output frame by multiplying it with the transform quaternion.
        # Note: it's not an ordinary element-wise multiplication
        x, y, z, w = quaternion_multiply(tf_rot, orientation)
        transformed_pose.orientation = Quaternion(x, y, z, w)

        return transformed_pose

    def produce_hull(self, obj_pose, obj_dims, stamp):

        """
        Produce convex hull for an object given its pose and dimensions
        :param obj_pose: geometry_msgs/Pose. Position and orientation of object
        :param obj_dims: Vector3 - length, width and height of object
        :param vella_stamp: Time stamp at which the lidar pointcloud was created
        :return: geometry_msgs/PolygonStamped
        """
        convex_hull = PolygonStamped()
        convex_hull.header.frame_id = self.output_frame
        convex_hull.header.stamp = stamp

        # Taken from Autoware mini
        # compute heading angle from object's orientation
        _, _, heading = euler_from_quaternion(
            (obj_pose.orientation.x, obj_pose.orientation.y, obj_pose.orientation.z, obj_pose.orientation.w))

        # use cv2.boxPoints to get a rotated rectangle given the angle
        points = cv2.boxPoints((
            (obj_pose.position.x, obj_pose.position.y),
            (obj_dims.x, obj_dims.y),
            math.degrees(heading)
        ))
        convex_hull.polygon.points = [Point(x, y, obj_pose.position.z) for x, y in points]

        return convex_hull

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('SFADetector', log_level=rospy.INFO)
    node = SFADetector()
    node.run()