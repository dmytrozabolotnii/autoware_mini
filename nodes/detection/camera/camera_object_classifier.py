#!/usr/bin/env python3
import time
import cv2
import numpy as np
import rospy
import tf2_ros
import message_filters
from image_geometry import PinholeCameraModel

from sensor_msgs.msg import CameraInfo, Image
from autoware_mini.msg import DetectedObjectArray
from ros_numpy import numpify

from cv_bridge import CvBridge
from helpers.yolo_models import Yolo11Model
from helpers.box_matcher_3d_2d import BoxMatcher3DTo2D
from helpers.detection import get_3d_bbox

class CameraObjectClassifier:
    def __init__(self):
        # Parameters
        onnx_path = rospy.get_param("~onnx_path")

        yolo_confidence_threshold = 0.4
        box_matcher_iou_threshold = 0.1
        self.transform_timeout = 0.03

        self.camera_model = None
        self.classified_objects_labels = {}
        
        self.bridge = CvBridge()
        self.yolo_model = Yolo11Model(onnx_path, confidence_threshold=yolo_confidence_threshold)
        self.box_matcher = BoxMatcher3DTo2D(box_matcher_iou_threshold)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publishers
        self.classified_objects_sub = rospy.Publisher('classified_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)
        self.yolo_detections_pub = rospy.Publisher('yolo_detections', Image, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('camera_info', CameraInfo, self.camera_info_callback, queue_size=1, tcp_nodelay=True)
        image_raw_sub = message_filters.Subscriber('image_raw', Image, queue_size=1, buff_size=2**26, tcp_nodelay=True)
        detected_objects_sub = message_filters.Subscriber('detected_objects', DetectedObjectArray, queue_size=1, buff_size=2**20, tcp_nodelay=True)

        ts = message_filters.ApproximateTimeSynchronizer([image_raw_sub, detected_objects_sub], queue_size=4, slop=0.05)
        ts.registerCallback(self.detected_objects_callback)

    def camera_info_callback(self, camera_info_msg):
        if self.camera_model is None:
            self.camera_model = PinholeCameraModel()
        self.camera_model.fromCameraInfo(camera_info_msg)
        camera_intrinsics = self.camera_model.fullIntrinsicMatrix()
        self.box_matcher.camera_intrinsics = camera_intrinsics

    def detected_objects_callback(self, image_msg, det_objects_msg):
        if self.camera_model is None:
            rospy.logwarn_throttle(10, "%s - No camera model received, skipping image", rospy.get_name())
            return
        
        detected_objects = det_objects_msg.objects

        if len(detected_objects) == 0:
            return

        # Extract image
        image = self.bridge.imgmsg_to_cv2(image_msg,  desired_encoding='rgb8')

        # Detect objects from image
        bboxes_2d, classes, scores = self.yolo_model.predict(image)

        # Extract transform
        try:
            transform = self.tf_buffer.lookup_transform(image_msg.header.frame_id, det_objects_msg.header.frame_id, image_msg.header.stamp, rospy.Duration(self.transform_timeout))
            transform_matrix = numpify(transform.transform)
        except (tf2_ros.TransformException, rospy.ROSTimeMovedBackwardsException) as e:
            rospy.logwarn("%s - %s", rospy.get_name(), e)
            return

        # Get 3D bounding boxes
        bboxes_3d = []
        for obj in detected_objects:
            bbox_3d = get_3d_bbox((obj.position.x, obj.position.y, obj.position.z), 
                                  (obj.dimensions.x, obj.dimensions.y, obj.dimensions.z),
                                  obj.heading)
            bboxes_3d.append(bbox_3d)
            if obj.id in self.classified_objects_labels:
                obj.label = self.classified_objects_labels[obj.id]

        # Perform Hungarian matching between 3D boxes and 2D boxes
        matches, projected_bboxes_2d, kept_3d_boxes = self.box_matcher.match_3d_2d_boxes(np.array(bboxes_3d), bboxes_2d, transform_matrix)

        # Get matched 3d boxes and change the label of detected objects
        matched_projected_bboxes_2d = []
        for box_match in matches:
            i_3d, i_2d = box_match
            matched_projected_bboxes_2d.append(projected_bboxes_2d[i_3d])
            obj = detected_objects[kept_3d_boxes[i_3d]]
            obj.label = self.yolo_model.class_name_map[classes[i_2d]]
            self.classified_objects_labels[obj.id] = obj.label

        self.publish_bbox_image(image, matched_projected_bboxes_2d, bboxes_2d, classes, scores, image_msg.header.stamp)
        self.classified_objects_sub.publish(det_objects_msg)

    def publish_bbox_image(self, image, projected_boxes, boxes, classes, scores, image_time_stamp):
        # add boxes and labels to image
        if len(boxes) > 0:
            img_size = image.shape
            # Add projected 3d bounding boxes to image
            for x1, y1, x2, y2 in projected_boxes:
                cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), int(max(img_size) * 0.001))

            # Add 2d bounding boxes, labels and scores to image 
            for cl, score, (x1, y1, x2, y2) in zip(classes, scores, boxes):
                label = self.yolo_model.class_name_map[cl]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), int(max(img_size) * 0.001))

                font_scale = int(max(img_size) * 0.0009)
                font_tickness = max(1, int(max(img_size) * 0.001))
                cv2.putText(image, f"{label}({score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,255), font_tickness, cv2.LINE_AA)
                
        #image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        img_msg = self.bridge.cv2_to_imgmsg(image, encoding='rgb8')
        
        img_msg.header.stamp = image_time_stamp
        self.yolo_detections_pub.publish(img_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('camera_object_classifier', log_level=rospy.INFO)
    node = CameraObjectClassifier()
    node.run()