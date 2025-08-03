#!/usr/bin/env python3
# Based on the original head detector model
# Modified to use ONNX runtime instead of TensorFlow

import os
import time
import numpy as np
import cv2
import onnxruntime as ort

class HeadDetector:
    def __init__(self, model_path):
        """Initialize head detector with ONNX model.
        
        Args:
            model_path: Path to the ONNX model file
        """
        self.inference_list = []
        self.count = 0
        
        # Check if the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at {model_path}")
        
        # Create ONNX Runtime session with GPU acceleration
        try:
            # Create inference session with GPU provider
            self.model = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
            
            # Get model input and output details
            self.input_name = self.model.get_inputs()[0].name
            self.input_shape = self.model.get_inputs()[0].shape
            self.output_names = [output.name for output in self.model.get_outputs()]

            print(f"HeadDetector initialized with ONNX model: {model_path}")
            print(f"Using device: {ort.get_device()}")
            print(f"Input name: {self.input_name}, Output names: {self.output_names}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ONNX model: {e}")

    def detect_heads(self, image, confidence_threshold=0.8):
        """Detect heads in an image.
        
        Args:
            image: RGB or BGR image
            confidence_threshold: Minimum confidence score for detection
            
        Returns:
            heads: List of dictionaries containing head detection info
            vis_image: Image with detection boxes drawn
        """
        h, w = image.shape[:2]
        
        # Convert to RGB if it's BGR
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Prepare input tensor - expand dimensions for batch
        input_tensor = np.expand_dims(image_rgb, axis=0).astype(np.uint8)
        
        # Run inference
        outputs = self.model.run(self.output_names, {self.input_name: input_tensor})

        # Process outputs
        boxes = outputs[0][0]  # detection_boxes: [y1, x1, y2, x2] normalized coordinates
        scores = outputs[1][0]  # detection_scores
        classes = outputs[2][0].astype(np.int32)  # detection_classes
        num_detections = int(outputs[3][0])  # num_detections

        # Create list to store detected heads
        heads = []
        
        # Process each detection
        for i in range(num_detections):
            # Only process boxes with class=1 (head) and confidence > threshold
            if classes[i] == 1 and scores[i] > confidence_threshold:
                # Convert normalized coordinates to pixel coordinates
                # Format is [y1, x1, y2, x2] -> convert to [x1, y1, x2, y2]
                y1, x1, y2, x2 = boxes[i]

                # Convert normalized coordinates to absolute pixel values and ensure coordinates are within image boundaries
                left = max(0, int(x1 * w))
                top = max(0, int(y1 * h))
                right = min(w, int(x2 * w))
                bottom = min(h, int(y2 * h))
                
                # Calculate width and height
                width = right - left
                height = bottom - top
                
                # Skip if the box has zero area
                if width <= 0 or height <= 0:
                    continue
                
                # Extract cropped head
                try:
                    cropped_head = image[top:bottom, left:right]
                except Exception as e:
                    print(f"Error cropping head: {e}, box={[top, left, bottom, right]}, image shape={image.shape}")
                    continue
                
                # Create head detection dictionary
                head_dict = {
                    "id": i+1,
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "width": width,
                    "height": height,
                    "confidence": float(scores[i]),
                    "cropped": cropped_head
                }
                heads.append(head_dict)

        return heads

    def get_avg_inference_time(self):
        """Get average inference time in milliseconds."""
        if not self.inference_list:
            return 0
        return (sum(self.inference_list) / len(self.inference_list)) * 1000  # Convert to ms
