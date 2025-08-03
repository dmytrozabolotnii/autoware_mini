#!/usr/bin/env python3
import numpy as np
import rospy
import tf2_ros
import message_filters
import sys
from image_geometry import PinholeCameraModel

from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo
from autoware_mini.msg import DetectedObjectArray
from autoware_mini.msg import Float32MultiArrayStamped
from ros_numpy import numpify
from geometry_msgs.msg import Pose, Vector3

from autoware_mini.box_matcher_3d_2d import BoxMatcher3DTo2D
from autoware_mini.detection import get_3d_bbox
from autoware_mini.transform import transform_pose, transform_vector3
from autoware_mini.head_pose_filter import HeadPoseFilter

class CameraHeadPoseClassifier:
    def __init__(self):
        # Parameters
        self.transform_timeout = rospy.get_param("~transform_timeout")
        box_matcher_iou_threshold = rospy.get_param("~box_matcher_iou_threshold")
        cameras = [rospy.get_param(f"~camera{i}") for i in range(1, 6) if rospy.has_param(f"~camera{i}")]

        # Head pose filter parameters
        filter_window_size = rospy.get_param("~filter_window_size", 5)
        filter_weight_factor = rospy.get_param("~filter_weight_factor", 0.8)

        # Initialize head pose temporal filter
        self.head_pose_filter = HeadPoseFilter(window_size=filter_window_size,
                                              weight_factor=filter_weight_factor)

        self.camera_intrinsics = {}
        self.head_pose_data = {}
        
        self.box_matcher = BoxMatcher3DTo2D(box_matcher_iou_threshold)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publishers
        self.head_pose_objects_pub = rospy.Publisher('head_pose_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)

        # Subscribers
        detected_objects_sub = message_filters.Subscriber('detected_objects', DetectedObjectArray, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        ts_subscribers = [detected_objects_sub]

        for camera in cameras:
            if camera == "":
                continue
            rospy.Subscriber(f'/{camera}/camera_info', CameraInfo, self.camera_info_callback, queue_size=1, tcp_nodelay=True)
            ts_subscribers.append(message_filters.Subscriber(f'camera/{camera}/head_pose_detections', Float32MultiArrayStamped, queue_size=1, buff_size=2**26, tcp_nodelay=True))

        ts = message_filters.ApproximateTimeSynchronizer(ts_subscribers, queue_size=2*len(ts_subscribers), slop=0.08)
        ts.registerCallback(self.detected_objects_callback)

    def camera_info_callback(self, camera_info_msg):
        if camera_info_msg.header.frame_id not in self.camera_intrinsics:
            camera_model = PinholeCameraModel()
            camera_model.fromCameraInfo(camera_info_msg)
            self.camera_intrinsics[camera_info_msg.header.frame_id] = camera_model.fullIntrinsicMatrix()

    def detected_objects_callback(self, det_objects_msg, *camera_det_msgs):
        detected_objects = det_objects_msg.objects
        if len(detected_objects) == 0:
            return
        
        # Get 3D bounding boxes and first head pose detections dictionary
        bboxes_3d = []
        num_head_pose_detections = {}
        for obj in detected_objects:
            bbox_3d = get_3d_bbox((obj.center.x, obj.center.y, obj.center.z),
                                  (obj.dimensions.x, obj.dimensions.y, obj.dimensions.z),
                                  obj.heading)
            bboxes_3d.append(bbox_3d)
            num_head_pose_detections[obj.id] = 0  # Initialize count for each object ID

        bboxes_3d = np.array(bboxes_3d)
        

        for cam_det_msg in camera_det_msgs:
            cam_frame_id = cam_det_msg.header.frame_id
            
            # Get camera intrinsics
            if cam_frame_id in self.camera_intrinsics:
                camera_intrinsics = self.camera_intrinsics[cam_frame_id]
                # rospy.loginfo("%s - Camera intrinsics for %s: \n%s",
                #              rospy.get_name(), cam_frame_id, camera_intrinsics)
            else:
                rospy.logwarn("%s - %s", rospy.get_name(), f"No intrinsics found for camera {cam_frame_id}")
                continue

            # Extract transform from map to camera (for projecting 3D boxes to 2D)
            try:
                map_to_cam_transform = self.tf_buffer.lookup_transform(cam_frame_id, det_objects_msg.header.frame_id,
                                                        det_objects_msg.header.stamp, rospy.Duration(self.transform_timeout))
                map_to_cam_transform_matrix = numpify(map_to_cam_transform.transform)


            except (tf2_ros.TransformException, rospy.ROSTimeMovedBackwardsException) as e:
                rospy.logwarn("%s - %s", rospy.get_name(), e)
                continue

            # Extract transform from camera to map (for transforming head pose to map frame)
            try:
                cam_to_map_transform = self.tf_buffer.lookup_transform(det_objects_msg.header.frame_id, cam_frame_id,
                                                        det_objects_msg.header.stamp, rospy.Duration(self.transform_timeout))
            except (tf2_ros.TransformException, rospy.ROSTimeMovedBackwardsException) as e:
                rospy.logwarn("%s - %s", rospy.get_name(), e)
                continue
            
            # Parse head pose detections
            head_pose_detections = float32_multiarray_to_numpy(cam_det_msg)
            
            if len(head_pose_detections) == 0:
                continue


            # Extract person bounding boxes and rotation matrices
            bboxes_2d = head_pose_detections[:, :4].astype(int)  # [x1, y1, x2, y2]

            # The next 9 values represent the flattened 3x3 rotation matrix
            rotation_matrices = head_pose_detections[:, 4:13].reshape(-1, 3, 3)  # Reshape to 3x3 matrices

            # Perform Hungarian matching between 3D boxes and 2D boxes using map_to_cam_transform
            try:
                matches, projected_bboxes_2d, kept_3d_boxes = self.box_matcher.match_3d_2d_boxes(bboxes_3d, bboxes_2d, map_to_cam_transform_matrix, camera_intrinsics)

            except Exception as e:
                rospy.logerr("%s - Error during box matching: %s", rospy.get_name(), str(e))
                continue

            # Process matched boxes
            for match in matches:
                i_3d, i_2d = match
                obj_idx = kept_3d_boxes[i_3d]
                obj = detected_objects[obj_idx]

                # Get the rotation matrix in camera frame
                rotation_matrix_camera = rotation_matrices[i_2d]

                # Apply temporal filtering using the object ID for consistent tracking
                filtered_rotation_matrix_camera = self.head_pose_filter.update(obj.id, rotation_matrix_camera)

                # Extract the head direction vector in camera frame (Z-axis is forward direction)
                head_dir_camera_vector = Vector3()
                head_dir_camera_vector.x = float(filtered_rotation_matrix_camera[0, 2])
                head_dir_camera_vector.y = float(filtered_rotation_matrix_camera[1, 2])
                head_dir_camera_vector.z = float(filtered_rotation_matrix_camera[2, 2])

                # Use transform helper function to transform from camera to map frame
                head_dir_map_vector = transform_vector3(head_dir_camera_vector, cam_to_map_transform)

                # Normalize the head direction vector
                head_dir_norm = np.sqrt(
                    head_dir_map_vector.x**2 +
                    head_dir_map_vector.y**2 +
                    head_dir_map_vector.z**2
                )

                if head_dir_norm > 0:
                    head_dir_map_vector.x /= head_dir_norm
                    head_dir_map_vector.y /= head_dir_norm
                    head_dir_map_vector.z /= head_dir_norm

                # Calculate 2D heading in the x-y plane from head direction vector
                head_heading = np.arctan2(head_dir_map_vector.y, head_dir_map_vector.x)

                # Update the object's heading to match the head pose
                if num_head_pose_detections[obj.id] == 0:
                    obj.heading = float(head_heading)
                    num_head_pose_detections[obj.id] = num_head_pose_detections[obj.id] + 1
                else:
                    # If this is not the first head pose detection, average the heading
                    obj.heading = (obj.heading * num_head_pose_detections[obj.id] + float(head_heading)) / (num_head_pose_detections[obj.id] + 1)
                    num_head_pose_detections[obj.id] = num_head_pose_detections[obj.id] + 1

                # Set the label to indicate this is a pedestrian with head pose
                obj.label = "pedestrian_with_head_pose"

        # Publish objects with head pose data
        self.head_pose_objects_pub.publish(det_objects_msg)

    def run(self):
        rospy.spin()

def float32_multiarray_to_numpy(multiarray):
    dims = tuple(map(lambda x: x.size, multiarray.layout.dim))
    data = multiarray.data[multiarray.layout.data_offset:]
    return np.array(data, dtype=np.float32).reshape(dims)

if __name__ == '__main__':
    rospy.init_node('camera_head_pose_classifier', log_level=rospy.INFO)
    node = CameraHeadPoseClassifier()
    node.run()
