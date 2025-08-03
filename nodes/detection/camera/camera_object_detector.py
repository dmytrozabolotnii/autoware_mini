#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from ast import literal_eval

from sensor_msgs.msg import Image
from std_msgs.msg import MultiArrayDimension, String
from autoware_mini.msg import Float32MultiArrayStamped

from cv_bridge import CvBridge
from autoware_mini.yolo_models import Yolo11Model

class CameraObjectDetector:
    def __init__(self):

        # Parameters
        onnx_path = rospy.get_param("~onnx_path")
        yolo_confidence_threshold = rospy.get_param("yolo_confidence_threshold")

        self.yolo_class_map = None
        
        self.bridge = CvBridge()

        # YOLO model
        self.yolo_model = Yolo11Model(onnx_path, confidence_threshold=yolo_confidence_threshold)

        # Publishers
        self.camera_detections_pub = rospy.Publisher('camera_detections', Float32MultiArrayStamped, queue_size=1, tcp_nodelay=True)
        self.camera_detections_vis_pub = rospy.Publisher('camera_detections_visualizer', Image, queue_size=1, tcp_nodelay=True)
        self.yolo_class_map_pub = rospy.Publisher('yolo_class_map', String, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('image_raw', Image, self.image_callback, queue_size=1, buff_size=2**26, tcp_nodelay=True)

    def image_callback(self, image_msg):
        # Extract image
        image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')

        # Detect objects from image
        bboxes_2d, classes, scores = self.yolo_model.predict(image)

        if len(bboxes_2d) == 0 or len(classes) == 0 or len(scores) == 0:
            # No detected objects
            yolo_predictions = np.empty((0,6), dtype=np.float32)

        else:
            bboxes_2d_f32 = bboxes_2d.astype(np.float32)
            classes_f32 = classes.astype(np.float32)
            scores_f32 = scores.astype(np.float32)

            yolo_predictions = np.hstack((bboxes_2d_f32, classes_f32[:, np.newaxis], scores_f32[:, np.newaxis]))
        
        # Create an array for YOLO detections
        yolo_pred_array = Float32MultiArrayStamped()
        yolo_pred_array.header.stamp = image_msg.header.stamp
        yolo_pred_array.header.frame_id = image_msg.header.frame_id
        yolo_pred_array.layout.dim = [MultiArrayDimension(f'dim{i}', yolo_predictions.shape[i],
                                yolo_predictions.shape[i] * yolo_predictions.dtype.itemsize) for i in range(yolo_predictions.ndim)]
        yolo_pred_array.data = yolo_predictions.flatten()

        # Publish YOLO detections as an array
        self.camera_detections_pub.publish(yolo_pred_array)

        # Publish YOLO class map
        self.yolo_class_map_pub.publish(String(self.yolo_model.class_names))

        if self.yolo_class_map is None:
            self.yolo_class_map = literal_eval(self.yolo_model.class_names)

        # Publish image visualizing YOLO detections
        self.publish_bbox_image(image, bboxes_2d, classes, scores, image_msg.header)

    def publish_bbox_image(self, image, boxes, classes, scores, image_header):
        if len(boxes) > 0:
            img_size = image.shape

            # Add 2D bounding boxes, labels and scores to image 
            for cl, score, (x1, y1, x2, y2) in zip(classes, scores, boxes):
                label = literal_eval(self.yolo_model.class_names)[cl]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), int(max(img_size) * 0.001))
            
                font_scale = int(max(img_size) * 0.0009)
                font_tickness = max(1, int(max(img_size) * 0.001))
                cv2.putText(image, f"{label}({score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,255), font_tickness, cv2.LINE_AA)
                
        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        img_msg = self.bridge.cv2_to_imgmsg(image, encoding='rgb8')
        
        img_msg.header.stamp = image_header.stamp
        img_msg.header.frame_id = image_header.frame_id
        self.camera_detections_vis_pub.publish(img_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('camera_object_detector', log_level=rospy.INFO)
    node = CameraObjectDetector()
    node.run()
