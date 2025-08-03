import numpy as np
from scipy.optimize import linear_sum_assignment
from autoware_mini.detection import calculate_iou

class BoxMatcher3DTo2D:
    def __init__(self, iou_threshold, projected_3d_boxes_bounds=(-1000, -500, 3000, 2000)):
        self.iou_threshold = iou_threshold
        self.projected_3d_boxes_bounds = projected_3d_boxes_bounds

    def transform_to_camera_frame(self, boxes_3d, T):
        """
        Transforms 3D bounding boxes to camera frame.
        """

        # Convert corners to homogeneous coordinates (N, 8, 4)
        ones = np.ones((boxes_3d.shape[0], 8, 1))  # Shape: (N, 8, 1)
        corners_homogeneous = np.concatenate([boxes_3d, ones], axis=-1)  # (N, 8, 4)
        
        # Apply the transformation matrix
        transformed_corners = np.einsum("ij,nkj->nki", T, corners_homogeneous)  # (N, 8, 4)
        
        # Convert back to 3D by removing the homogeneous coordinate
        return transformed_corners[..., :3]

    def project_to_image(self, boxes_3d_cam, camera_intrinsics):
        """
        Projects 3D bounding box corners onto a 2D image plane using camera intrinsics.
        """

        # Convert to homogeneous coordinates (N, 8, 4)
        ones = np.ones((boxes_3d_cam.shape[0], 8, 1))
        corners_3d_homo = np.concatenate([boxes_3d_cam, ones], axis=-1)  # (N, 8, 4)

        # Apply intrinsic matrix
        projected = np.einsum('ij,nkj->nki', camera_intrinsics, corners_3d_homo[..., :3])  # (N, 8, 3)

        # Normalize by depth (z-axis)
        projected_2d = projected[..., :2] / projected[..., 2:3]  # (N, 8, 2)

        return projected_2d

    def compute_2d_bboxes(self, projected_boxes_3d):
        """
        Computes 2D bounding boxes from projected 3D bounding boxes.
        """
        x_min = np.min(projected_boxes_3d[..., 0], axis=1)
        y_min = np.min(projected_boxes_3d[..., 1], axis=1)
        x_max = np.max(projected_boxes_3d[..., 0], axis=1)
        y_max = np.max(projected_boxes_3d[..., 1], axis=1)

        # Convert to integers
        return np.round(np.stack([x_min, y_min, x_max, y_max], axis=1)).astype(int)  # (N, 4)

    def match_3d_2d_boxes(self, boxes_3d, boxes_2d, transform_matrix, camera_intrinsics):
        """
        Matches 3D bounding boxes to 2D bounding boxes using Hungarian algorithm.
        """
        if len(boxes_3d) == 0 or len(boxes_2d) == 0:
            return [], [], []

        # Transform 3D boxes to camera frame
        boxes_3d_cam = self.transform_to_camera_frame(boxes_3d, transform_matrix)

        # Filters out bounding boxes that are behind camera (all z-coordinates are negative)
        z_coords = boxes_3d_cam[..., 2]
        filtered_3d_boxes_mask = np.any(z_coords >= 0, axis=1)
        kept_3d_indices = np.where(filtered_3d_boxes_mask)[0]
        filtered_boxes_3d_cam = boxes_3d_cam[filtered_3d_boxes_mask]

        if len(filtered_boxes_3d_cam) == 0:
            return [], [], []

        # Project 3D boxes to the image
        projected_3d_boxes = self.project_to_image(filtered_boxes_3d_cam, camera_intrinsics)
        projected_2d_boxes = self.compute_2d_bboxes(projected_3d_boxes)

        # Filter projected 2D bounding boxes to be within predefined bounds
        x_min_bound, y_min_bound, x_max_bound, y_max_bound = self.projected_3d_boxes_bounds

        valid_mask = (
        (projected_2d_boxes[:, 0] >= x_min_bound) & (projected_2d_boxes[:, 1] >= y_min_bound) &  # x_min, y_min in bounds
        (projected_2d_boxes[:, 2] <= x_max_bound) & (projected_2d_boxes[:, 3] <= y_max_bound)    # x_max, y_max in bounds
        )

        valid_projected_2d_boxes = projected_2d_boxes[valid_mask]
        valid_projected_2d_indices = np.where(valid_mask)[0]

        if len(valid_projected_2d_boxes) == 0:
            return [], [], []

        # Perform Hungarian matching between projected 2D boxes and detected 2D boxes using IoU as cost
        # Compute IoU for all pairs
        iou_matrix = calculate_iou(valid_projected_2d_boxes, boxes_2d)

        # Convert IoU to a cost matrix for minimization (negate IoU)
        cost_matrix = -iou_matrix

        # Solve the assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter matches by IoU threshold
        matches = [(i, j) for i, j in zip(row_ind, col_ind) if iou_matrix[i, j] >= self.iou_threshold]

        return matches, valid_projected_2d_boxes, kept_3d_indices[valid_projected_2d_indices]