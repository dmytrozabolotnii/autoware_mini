#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from ast import literal_eval
import collections

from sensor_msgs.msg import Image
from std_msgs.msg import MultiArrayDimension, String
from autoware_mini.msg import Float32MultiArrayStamped

from cv_bridge import CvBridge
from autoware_mini.yolo_models import Yolo11Model
from head_pose_utils.head_pose_utils import compute_euler_angles_from_rotation_matrices, draw_axis
from head_pose_utils.head_pose_model import RepNet6D
from autoware_mini.head_pose_filter import HeadPoseFilter
import torch
from torchvision import transforms
import time

FACE_FRACTION = 0.25  # Fraction of the person bounding box height to use for head detection

class CameraHeadDetectorYolo:
    def __init__(self):
        # Parameters
        yolo_onnx_path = rospy.get_param("~yolo_onnx_path")
        head_pose_model_path = rospy.get_param("~head_pose_model_path")
        self.confidence_threshold = rospy.get_param("~confidence_threshold", 0.5)
        self.person_class_id = rospy.get_param("~person_class_id", 0)  # YOLO class ID for person
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        rospy.loginfo(f"Head detector using device: {self.device}")

        self.bridge = CvBridge()

        # YOLO model for object detection
        try:
            self.yolo_model = Yolo11Model(yolo_onnx_path, confidence_threshold=self.confidence_threshold)
            rospy.loginfo("YOLO model initialized successfully")
        except Exception as e:
            rospy.logerr(f"Failed to initialize YOLO model: {e}")
            raise

        # Head Pose model setup
        try:
            self.model = RepNet6D(backbone_name='RepVGG-B1g4',
                            backbone_file='',
                            deploy=True,
                            pretrained=False)

            # Load the model state
            saved_state_dict = torch.load(head_pose_model_path, map_location=self.device)
            if 'model_state_dict' in saved_state_dict:
                self.model.load_state_dict(saved_state_dict['model_state_dict'])
            else:
                self.model.load_state_dict(saved_state_dict)

            self.model.to(self.device)
            self.model.eval()
            rospy.loginfo("Head pose model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load head pose model: {e}")
            raise

        # Image transformation for the model
        self.transformations = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Publishers
        self.head_pose_pub = rospy.Publisher('head_pose_detections', Float32MultiArrayStamped, queue_size=1, tcp_nodelay=True)
        self.head_pose_vis_pub = rospy.Publisher('head_pose_visualizer', Image, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('image_raw', Image, self.image_callback, queue_size=1, buff_size=2**26, tcp_nodelay=True)

        # Head pose filter initialization
        self.head_pose_filter = HeadPoseFilter()

    def extract_heads_from_persons(self, bboxes, classes, scores):
        """Extract head regions from detected persons using heuristic approach"""
        heads = []

        for i, (bbox, cls, score) in enumerate(zip(bboxes, classes, scores)):
            # Only process person detections
            if int(cls) == self.person_class_id and score >= self.confidence_threshold:
                x1, y1, x2, y2 = [int(b) for b in bbox]

                # Estimate head region (top 25% of person bounding box)
                head_height = int((y2 - y1) * FACE_FRACTION)
                head = {
                    'left': x1,
                    'top': y1,
                    'width': x2 - x1,
                    'height': head_height,
                    'confidence': float(score)
                }
                heads.append(head)

        return heads

    def image_callback(self, image_msg):
        # Extract image
        image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
        original_image = image.copy()

        # Detect objects using YOLO
        try:
            bboxes_2d, classes, scores = self.yolo_model.predict(image)

            # Extract heads from person detections
            heads = self.extract_heads_from_persons(bboxes_2d, classes, scores)

            rospy.loginfo_throttle_identical(3, f"Detected {len(heads)} heads")
        except Exception as e:
            rospy.logerr(f"Error in head detection: {e}")
            heads = []

        if len(heads) == 0:
            # No heads detected
            head_poses = np.empty((0, 10), dtype=np.float32)  # [x, y, w, h, rotation_matrix[0...5]]
        else:
            head_poses = []
            for head in heads:
                x = head['left']
                y = head['top']
                w = head['width']
                h = head['height']

                # Extract face ROI with margin
                margin = int(min(w, h) * 0.2)
                face_roi = image[max(0, y-margin):min(image.shape[0], y+h+margin),
                                 max(0, x-margin):min(image.shape[1], x+w+margin)]

                if face_roi.size == 0:
                    continue

                # Prepare face for the model
                try:
                    # Transform the image for the model
                    face_tensor = self.transformations(face_roi).unsqueeze(0).to(self.device)

                    # Get head pose
                    with torch.no_grad():
                        rotation_matrix = self.model(face_tensor)

                    # Convert rotation matrix to numpy array (full 3x3 matrix)
                    rotation_matrix_np = rotation_matrix[0].cpu().numpy()

                    # Flatten the 3x3 rotation matrix for transmission (9 values)
                    rotation_matrix_flat = rotation_matrix_np.flatten()

                    # Store person box and full rotation matrix (x1, y, x2, y2, rotation_matrix[0...8]) for matching
                    head_pose = np.array([x, y, x + w, y + h * (1 / FACE_FRACTION), *rotation_matrix_flat], dtype=np.float32)
                    head_poses.append(head_pose)

                    # For visualization purposes, still compute Euler angles
                    euler = compute_euler_angles_from_rotation_matrices(rotation_matrix) * 180/np.pi
                    p_pred_deg = euler[:, 0].cpu().numpy()  # Pitch
                    y_pred_deg = euler[:, 1].cpu().numpy()  # Yaw
                    r_pred_deg = euler[:, 2].cpu().numpy()  # Roll

                    # Draw pose axis on the original image (for visualization)
                    original_image = draw_axis(original_image, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0],
                              x + w//2, y + h//2, size=w//2)

                    # Draw head bounding box
                    cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Add confidence score text
                    confidence_text = f"{head['confidence']:.2f}"
                    cv2.putText(original_image, confidence_text, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                except Exception as e:
                    rospy.logerr(f"Error processing head {head}: {e}")
                    continue

            if head_poses:
                head_poses = np.array(head_poses, dtype=np.float32)
            else:
                head_poses = np.empty((0, 10), dtype=np.float32)

        # Create an array for head pose detections
        head_pose_array = Float32MultiArrayStamped()
        head_pose_array.header.stamp = image_msg.header.stamp
        head_pose_array.header.frame_id = image_msg.header.frame_id
        head_pose_array.layout.dim = [MultiArrayDimension(f'dim{i}', head_poses.shape[i],
                                head_poses.shape[i] * head_poses.dtype.itemsize) for i in range(head_poses.ndim)]
        head_pose_array.data = head_poses.flatten()

        # Publish head pose detections as an array
        self.head_pose_pub.publish(head_pose_array)

        # Visualize detections on image
        self.publish_visualization(original_image, image_msg.header)

    def publish_visualization(self, image, image_header):
        """Create and publish visualization of detected heads and their head poses."""
        # Add some information text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, f"YOLO Head Pose Detector", (10, 30), font, 0.7, (0, 255, 255), 2)

        # Resize image for visualization
        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        img_msg = self.bridge.cv2_to_imgmsg(image, encoding='rgb8')

        img_msg.header.stamp = image_header.stamp
        img_msg.header.frame_id = image_header.frame_id
        self.head_pose_vis_pub.publish(img_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('camera_head_detector_yolo', log_level=rospy.INFO)
    node = CameraHeadDetectorYolo()
    node.run()
