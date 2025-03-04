#!/usr/bin/env python3
import numpy as np
import rospy
import tf2_ros
import message_filters
from image_geometry import PinholeCameraModel
from ast import literal_eval

from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo
from autoware_mini.msg import DetectedObjectArray
from vehicle_platform.msg import Float32MultiArrayStamped
from ros_numpy import numpify

from helpers.box_matcher_3d_2d import BoxMatcher3DTo2D
from helpers.detection import get_3d_bbox

class CameraObjectClassifier:
    def __init__(self):
        # Parameters
        self.transform_timeout = rospy.get_param("~transform_timeout")
        box_matcher_iou_threshold = rospy.get_param("~box_matcher_iou_threshold")
        cameras = [rospy.get_param(f"~camera{i}") for i in range(1, 6) if rospy.has_param(f"~camera{i}")]

        self.yolo_class_map = None
        self.camera_intrinsics = {}
        self.classified_objects_labels = {}
        
        self.box_matcher = BoxMatcher3DTo2D(box_matcher_iou_threshold)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publishers
        self.classified_objects_pub = rospy.Publisher('classified_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('camera/yolo_class_map', String, self.yolo_class_map_callback, queue_size=1, tcp_nodelay=True)
        
        detected_objects_sub = message_filters.Subscriber('detected_objects', DetectedObjectArray, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        ts_subscribers = [detected_objects_sub]

        for camera in cameras:
            if camera == "":
                continue
            rospy.Subscriber(f'/{camera}/camera_info', CameraInfo, self.camera_info_callback, queue_size=1, tcp_nodelay=True)
            ts_subscribers.append(message_filters.Subscriber(f'camera/{camera}/camera_detections', Float32MultiArrayStamped, queue_size=1, buff_size=2**26, tcp_nodelay=True))

        ts = message_filters.ApproximateTimeSynchronizer(ts_subscribers, queue_size=2*len(ts_subscribers), slop=0.08)
        ts.registerCallback(self.detected_objects_callback)

    def camera_info_callback(self, camera_info_msg):
        if camera_info_msg.header.frame_id not in self.camera_intrinsics:
            camera_model = PinholeCameraModel()
            camera_model.fromCameraInfo(camera_info_msg)
            self.camera_intrinsics[camera_info_msg.header.frame_id] = camera_model.fullIntrinsicMatrix()

    def yolo_class_map_callback(self, msg):
        if self.yolo_class_map is None:
            self.yolo_class_map = literal_eval(msg.data)

    def detected_objects_callback(self, det_objects_msg, *camera_det_msgs):
        if self.yolo_class_map is None:
            rospy.logwarn_throttle(10, "%s - No YOLO class map received", rospy.get_name())
            return
        
        detected_objects = det_objects_msg.objects
        if len(detected_objects) == 0:
            return
        
        # Get 3D bounding boxes
        bboxes_3d = []
        seen_ids = set()
        for obj in detected_objects:
            bbox_3d = get_3d_bbox((obj.position.x, obj.position.y, obj.position.z), 
                                  (obj.dimensions.x, obj.dimensions.y, obj.dimensions.z),
                                  obj.heading)
            bboxes_3d.append(bbox_3d)

            # Add previously predicted labels to objects
            if obj.id in self.classified_objects_labels:
                obj.label = self.classified_objects_labels[obj.id]
                seen_ids.add(obj.id)

        # Remove ids belonging to objects that weren't detected anymore
        for obj_id in list(self.classified_objects_labels.keys()):
            if obj_id not in seen_ids:
                del self.classified_objects_labels[obj_id]

        bboxes_3d = np.array(bboxes_3d)
        
        for cam_det_msg in camera_det_msgs:
            cam_frame_id = cam_det_msg.header.frame_id
            
            # Get camera intrinsics
            if cam_frame_id in self.camera_intrinsics:
                camera_intrinsics = self.camera_intrinsics[cam_frame_id]
            else:
                rospy.logwarn("%s - %s", rospy.get_name(), f"No intrinsics found for camera {cam_frame_id}")
                continue

            # Extract transform
            try:
                transform = self.tf_buffer.lookup_transform(cam_frame_id, det_objects_msg.header.frame_id, 
                                                            det_objects_msg.header.stamp, rospy.Duration(self.transform_timeout))
                transform_matrix = numpify(transform.transform)
            except (tf2_ros.TransformException, rospy.ROSTimeMovedBackwardsException) as e:
                rospy.logwarn("%s - %s", rospy.get_name(), e)
                continue
            
            # Parse YOLO detections
            camera_detections = float32_multiarray_to_numpy(cam_det_msg)

            bboxes_2d = camera_detections[:, :4].astype(int)
            classes = camera_detections[:, 4].astype(int)

            # Perform Hungarian matching between 3D boxes and 2D boxes
            matches, projected_bboxes_2d, kept_3d_boxes = self.box_matcher.match_3d_2d_boxes(bboxes_3d, bboxes_2d, transform_matrix, camera_intrinsics)

            # Get matched 3d boxes and change the label of detected objects
            for box_match in matches:
                i_3d, i_2d = box_match
                obj = detected_objects[kept_3d_boxes[i_3d]]
                obj.label = self.yolo_class_map[classes[i_2d]]
                self.classified_objects_labels[obj.id] = obj.label
        
        self.classified_objects_pub.publish(det_objects_msg)

    def run(self):
        rospy.spin()

def float32_multiarray_to_numpy(multiarray):
    dims = tuple(map(lambda x: x.size, multiarray.layout.dim))
    data = multiarray.data[multiarray.layout.data_offset:]
    return np.array(data, dtype=np.float32).reshape(dims)

if __name__ == '__main__':
    rospy.init_node('camera_object_classifier', log_level=rospy.INFO)
    node = CameraObjectClassifier()
    node.run()