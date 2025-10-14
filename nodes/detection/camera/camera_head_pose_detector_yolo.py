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
import torchvision.transforms.v2 as transforms_v2
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

        # New transformation pipeline using transforms v2 and direct tensor conversion
        self.transformations_v2 = transforms_v2.Compose([
            transforms_v2.Resize(size=(224, 224), antialias=True),
            transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

                # Estimate head region (top 25% of person bounding box and 50% of width)
                head_height = int((y2 - y1) * FACE_FRACTION)
                head_width = int((x2 - x1) * 0.5)
                head = {
                    'left': x1 + head_width // 2,  # Center the head in the person bounding box
                    'top': y1,
                    'width': head_width,
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
            # yolo_start_time = time.time()
            bboxes_2d, classes, scores = self.yolo_model.predict(image)
            # yolo_time = time.time() - yolo_start_time
            # rospy.loginfo(f"YOLO detection time: {yolo_time:.3f}s")

            # Extract heads from person detections
            heads = self.extract_heads_from_persons(bboxes_2d, classes, scores)
        except Exception as e:
            rospy.logerr(f"Error in head detection: {e}")
            heads = []

        if len(heads) == 0:
            # No heads detected
            head_poses = np.empty((0, 13), dtype=np.float32)  # [x1, y1, x2, y2, rotation_matrix[0...8]]
        else:
            # Prepare batch of face ROIs
            roi_preprocess_start = time.time()

            # First loop: find the biggest bounding box dimensions
            max_width = 0
            max_height = 0
            for head in heads:
                w = head['width']
                h = head['height']
                margin = int(min(w, h) * 0.2)
                # Calculate final dimensions including margin
                roi_width = min(image.shape[1], head['left'] + w + margin) - max(0, head['left'] - margin)
                roi_height = min(image.shape[0], head['top'] + h + margin) - max(0, head['top'] - margin)

                max_width = max(max_width, roi_width)
                max_height = max(max_height, roi_height)

            # Ensure dimensions are even (sometimes required for certain operations)
            max_width = max_width + (max_width % 2)
            max_height = max_height + (max_height % 2)

            # Initialize arrays for face ROIs and valid heads
            face_rois = np.zeros((len(heads), max_height, max_width, 3), dtype=np.uint8)
            valid_heads = []

            # Second loop: extract and pad all ROIs to the same size
            for i, head in enumerate(heads):
                x = head['left']
                y = head['top']
                w = head['width']
                h = head['height']

                # Extract face ROI with margin
                margin = int(min(w, h) * 0.2)
                x_min = max(0, x - margin)
                y_min = max(0, y - margin)
                x_max = min(image.shape[1], x + w + margin)
                y_max = min(image.shape[0], y + h + margin)

                # Skip if ROI is empty
                if x_max <= x_min or y_max <= y_min:
                    continue

                # Extract the ROI
                face_roi = image[y_min:y_max, x_min:x_max]

                # Center pad to max dimensions
                pad_height_before = (max_height - (y_max - y_min)) // 2
                pad_height_after = max_height - (y_max - y_min) - pad_height_before
                pad_width_before = (max_width - (x_max - x_min)) // 2
                pad_width_after = max_width - (x_max - x_min) - pad_width_before

                # Apply padding
                try:
                    padded_roi = np.pad(face_roi,
                                        ((pad_height_before, pad_height_after),
                                         (pad_width_before, pad_width_after),
                                         (0, 0)),
                                        mode='constant')


                    face_rois[i] = padded_roi

                    # Store valid head with its coordinates for later use
                    valid_heads.append({
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'confidence': head['confidence']
                    })
                except Exception as e:
                    rospy.logerr(f"Error padding ROI: {e}")
                    continue

            # Convert the entire batch of numpy arrays to a tensor at once
            transform_start_time = time.time()

            # Convert the batch from numpy to tensor
            # [N, H, W, C] -> [N, C, H, W] and normalize to 0-1 range
            face_tensors = torch.from_numpy(face_rois).permute(0, 3, 1, 2).float() / 255.0

            # Apply transformations to the entire batch at once
            face_tensors = self.transformations_v2(face_tensors)

            transform_time = time.time() - transform_start_time
            # rospy.loginfo(f"Batch transformation time: {transform_time:.3f}s for {len(valid_heads)} faces, "
            #               f"avg: {transform_time/max(len(valid_heads), 1):.3f}s per face")

            roi_preprocess_time = time.time() - roi_preprocess_start
            # if len(heads) > 0:
            #     rospy.loginfo(f"Face ROI preprocessing time: {roi_preprocess_time:.3f}s for {len(valid_heads)}/{len(heads)} faces, "
            #                   f"avg: {roi_preprocess_time/max(len(valid_heads), 1):.3f}s per face")

            head_poses = []

            if face_tensors is not None:
                try:
                    # Move tensors to device
                    batch_tensor = face_tensors.to(self.device)

                    # Process batch in one forward pass
                    with torch.no_grad():
                        rotation_matrices = self.model(batch_tensor)

                    # Process each result
                    for i, rotation_matrix in enumerate(rotation_matrices):
                        head = valid_heads[i]
                        x, y, w, h = head['x'], head['y'], head['w'], head['h']
                        confidence = head['confidence']

                        # Convert rotation matrix to numpy array
                        rotation_matrix_np = rotation_matrix.cpu().numpy()

                        # Flatten the rotation matrix for transmission
                        rotation_matrix_flat = rotation_matrix_np.flatten()

                        # Store person box and full rotation matrix (x1, y1, x2, y2, rotation_matrix[0...8]) for matching
                        head_pose = np.array([x - 0.5 * w, y, x + 1.5 * w, y + h * (1 / FACE_FRACTION), *rotation_matrix_flat], dtype=np.float32)
                        head_poses.append(head_pose)

                        # For visualization purposes, compute Euler angles
                        euler = compute_euler_angles_from_rotation_matrices(rotation_matrix.unsqueeze(0), full_range=True) * 180/np.pi
                        p_pred_deg = euler[:, 0].cpu().numpy()  # Pitch
                        y_pred_deg = euler[:, 1].cpu().numpy()  # Yaw
                        r_pred_deg = euler[:, 2].cpu().numpy()  # Roll

                        # Draw pose axis on the original image
                        original_image = draw_axis(original_image, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0],
                                    x + w//2, y + h//2, size=w//2)

                        # Draw head bounding box
                        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Draw confidence
                        confidence_text = f"{confidence:.2f}"
                        cv2.putText(original_image, confidence_text, (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                except Exception as e:
                    rospy.logerr(f"Error in batch head pose processing: {e}")

            if head_poses:
                head_poses = np.array(head_poses, dtype=np.float32)
            else:
                head_poses = np.empty((0, 13), dtype=np.float32)

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
