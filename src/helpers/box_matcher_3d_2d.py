import numpy as np
from scipy.optimize import linear_sum_assignment

class BoxMatcher3DTo2D:
    def __init__(self, iou_threshold):
        self.iou_threshold = iou_threshold
        self.lidar_camera_transform = None
        self.camera_intrinsics = None

    def transform_lidar_to_camera(self, box_3d):
        """
        Transforms 3D bounding box from LiDAR space to Camera space.
        """
        box_3d_hom = np.hstack((box_3d, np.ones((8, 1))))  # Convert to homogeneous (8,4)
        box_cam = (self.lidar_camera_transform @ box_3d_hom.T).T  # Transform to camera frame (8,4)
        return box_cam[:, :3]  # Remove homogeneous coordinate

    def project_to_image(self, box_cam):
        """
        Projects 3D points in the camera frame to the image plane.
        """
        box_img = (self.camera_intrinsics @ box_cam.T).T  # (8,3)
        box_img[:, 0] /= box_img[:, 2]  # Normalize x by depth
        box_img[:, 1] /= box_img[:, 2]  # Normalize y by depth
        return box_img[:, :2]  # Return 2D coordinates

    def compute_2d_bbox(self, box_img):
        """
        Computes 2D bounding box from projected 3D box.
        """
        #print(box_img.shape)
        #print("MIN", np.min(box_img, axis=0).shape)
        #print("MIN", np.min(box_img, axis=0)[0].shape)
        x_min, y_min = np.min(box_img, axis=0)[0]
        x_max, y_max = np.max(box_img, axis=0)[0]
        return (int(x_min), int(y_min), int(x_max), int(y_max))

    def compute_iou(self, boxA, boxB):
        """
        Computes IoU between two 2D bounding boxes.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / float(boxAArea + boxBArea - interArea)

    def match_3d_2d_boxes(self, boxes_3d, boxes_2d):
        """
        Matches 3D bounding boxes to 2D bounding boxes using Hungarian algorithm.
        """
        if self.lidar_camera_transform is None or self.camera_intrinsics is None:
            return [], []
        
        num_3d = len(boxes_3d)
        num_2d = len(boxes_2d)
        cost_matrix = np.zeros((num_3d, num_2d))

        projected_2d_boxes = []
        for i, box_3d in enumerate(boxes_3d):
            box_cam = self.transform_lidar_to_camera(box_3d)
            box_img = self.project_to_image(box_cam)
            projected_2d = self.compute_2d_bbox(box_img)
            projected_2d_boxes.append(projected_2d)

            for j, box_2d in enumerate(boxes_2d):
                iou = self.compute_iou(projected_2d, box_2d)
                #print("BOX", box_3d, box_2d)
                cost_matrix[i, j] = -iou  # We minimize cost (maximize IoU)

        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        for i, j in zip(row_ind, col_ind):
            if -cost_matrix[i, j] >= self.threshold:
                matches.append((i, j))

        return matches, projected_2d_boxes