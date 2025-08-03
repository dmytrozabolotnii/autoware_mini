#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R
import collections

class HeadPoseFilter:
    """
    Implements temporal filtering for head pose detection to reduce jitter and improve stability.
    """
    def __init__(self, window_size=5, weight_factor=0.6):
        """
        Initialize the head pose filter.

        Args:
            window_size: Number of frames to consider for smoothing
            weight_factor: Higher values give more weight to recent frames
        """
        self.window_size = window_size
        self.weight_factor = weight_factor
        self.rotation_history = {}  # Dictionary to store rotation matrices by object ID
        self.last_filtered_rotation = {}  # Store last filtered rotation for each object

    def update(self, obj_id, rotation_matrix):
        """
        Update the filter with a new rotation matrix.

        Args:
            obj_id: Object ID to track different objects
            rotation_matrix: 3x3 rotation matrix representing the head pose

        Returns:
            Filtered rotation matrix
        """
        # Initialize history for new objects
        if obj_id not in self.rotation_history:
            self.rotation_history[obj_id] = collections.deque(maxlen=self.window_size)
            self.last_filtered_rotation[obj_id] = rotation_matrix.copy()
            self.rotation_history[obj_id].append(rotation_matrix)
            return rotation_matrix

        # Add new rotation matrix to history
        self.rotation_history[obj_id].append(rotation_matrix)

        # Apply weighted average in quaternion space
        quaternions = []
        weights = []

        for i, rot_mat in enumerate(self.rotation_history[obj_id]):
            # Convert rotation matrix to quaternion
            quat = R.from_matrix(rot_mat).as_quat()
            quaternions.append(quat)

            # Exponential weighting - more recent frames have higher weight
            weight = np.exp(self.weight_factor * i / len(self.rotation_history[obj_id]))
            weights.append(weight)

        # Normalize weights
        weights = np.array(weights) / np.sum(weights)

        # Weighted average of quaternions
        avg_quat = np.zeros(4)
        for q, w in zip(quaternions, weights):
            # Ensure quaternions are in the same hemisphere
            if np.dot(q, quaternions[0]) < 0:
                q = -q
            avg_quat += q * w

        # Normalize the averaged quaternion
        avg_quat = avg_quat / np.linalg.norm(avg_quat)

        # Convert back to rotation matrix
        filtered_rotation = R.from_quat(avg_quat).as_matrix()

        # Update last filtered rotation
        self.last_filtered_rotation[obj_id] = filtered_rotation

        return filtered_rotation

    def get_last_filtered_rotation(self, obj_id):
        """Get the last filtered rotation for an object"""
        if obj_id in self.last_filtered_rotation:
            return self.last_filtered_rotation[obj_id]
        return None

    def reset(self, obj_id=None):
        """Reset the filter for a specific object or all objects"""
        if obj_id is not None and obj_id in self.rotation_history:
            del self.rotation_history[obj_id]
            del self.last_filtered_rotation[obj_id]
        elif obj_id is None:
            self.rotation_history.clear()
            self.last_filtered_rotation.clear()
