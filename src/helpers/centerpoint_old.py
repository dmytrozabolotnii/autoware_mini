import numpy as np
import onnxruntime
import time
import shapely
from pypcd4 import PointCloud
import open3d as o3d

import matplotlib.pyplot as plt

VOXEL_SIZE = [0.32, 0.32, 10.0]  # x, y, z size of each voxel
POINT_CLOUD_RANGE = [-76.8, -76.8, -4.0, 76.8, 76.8, 6.0]  # min_x, min_y, min_z, max_x, max_y, max_z
MAX_NUM_POINTS_PER_VOXEL = 10
MAX_VOXELS = 40000
YAW_NORM_THRESHOLDS = [0.3, 0.3, 0.3, 0.3, 0.0]
CLASS_NAMES = ["Car", "Truck", "Bus", "Bicycle", "Pedestrian"]
LIDAR_HEIGHT = 2.11


def visualize_bev_features(spatial_features):
    """
    Visualize all 32 channels of BEV spatial features.
    :param spatial_features: The spatial features from `scatter_voxel_features` of shape (1, 32, H, W).
    """
    # Get the shape of the spatial features
    _, num_channels, H, W = spatial_features.shape

    # Create a figure with subplots to display all channels
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))  # Adjust grid size (4x8) to fit 32 channels
    axes = axes.flatten()

    # Loop over all channels and plot each one
    for i in range(num_channels):
        bev_image = spatial_features[0, i, :, :]  # Extract the i-th channel for visualization
        axes[i].imshow(bev_image, cmap='jet', interpolation='nearest', origin='lower')
        axes[i].set_title(f"Channel {i+1}")
        axes[i].axis('off')  # Hide axis for better visibility

    # Display the grid of images
    plt.tight_layout()
    plt.show()

def visualize_voxel_feature_channel(voxel_features, voxel_coords, channel=11):
    """
    Visualize a single channel of the voxel features in 3D space.

    Args:
        voxel_features: (num_voxels, 1, 32)
        voxel_coords: (num_voxels, 4)  -> [batch_idx, z, y, x]
        channel: which feature channel to visualize
    """
    features = voxel_features.squeeze(1)  # shape: (num_voxels, 32)
    values = features[:, channel]
    
    xs = voxel_coords[:, 3]
    ys = voxel_coords[:, 2]
    zs = voxel_coords[:, 1]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    p = ax.scatter(xs, ys, zs, c=values, cmap='viridis', s=5)
    ax.set_xlabel("X (voxel)")
    ax.set_ylabel("Y (voxel)")
    ax.set_zlabel("Z (voxel)")
    ax.set_title(f"Voxel Features: Channel {channel}")
    fig.colorbar(p, ax=ax, label="Feature Value")
    plt.show()

class CenterpointModel(object):
    """Class for a traffic light detector YOLO model"""

    def __init__(self, encoder_onnx_path, head_onnx_path):
        
        """
        :param encoder_onnx_path: path of the onnx yolo model
        :param head_onnx_path: path of the onnx yolo model
        """
        self.encoder_model = onnxruntime.InferenceSession(encoder_onnx_path, providers=['CUDAExecutionProvider'])
        self.head_model = onnxruntime.InferenceSession(head_onnx_path, providers=['CUDAExecutionProvider'])
        self.voxel_generator = VoxelGenerator(VOXEL_SIZE, POINT_CLOUD_RANGE, MAX_NUM_POINTS_PER_VOXEL, MAX_VOXELS)
    
    def detect(self, pointcloud):
        pointcloud[:, 2] += LIDAR_HEIGHT # use ground as z=0
        pointcloud[:, 3] /= 255 # normalize intensity

        t0 = time.perf_counter()
        voxel_features, voxel_coords = self.voxel_generator.generate(pointcloud)
        print(f"Voxel generator runtime: {time.perf_counter() - t0:.3f}s")
        
        # Run voxel encoder
        t0 = time.perf_counter()
        encoder_outputs = self.encoder_model.run(None, {"input_features": voxel_features})
        print(f"Voxel encoder runtime: {time.perf_counter() - t0:.3f}s")
        print("Encoder outputs:", encoder_outputs[0].shape)

        #visualize_voxel_feature_channel(encoder_outputs[0], voxel_coords)

        # Calculate grid height and width
        H = int((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE[1])
        W = int((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE[0])

        spatial_features = self.scatter_voxel_features(encoder_outputs[0], voxel_coords, (H, W))
        #visualize_bev_features(spatial_features)

        # Run detection head
        head_inputs = {"spatial_features": spatial_features}
        t0 = time.perf_counter()
        head_outputs = self.head_model.run(None, head_inputs)
        print(f"Head inference runtime: {time.perf_counter() - t0:.3f}s")

        # Output parsing
        heatmap = head_outputs[0]         # (1, 5, H, W)
        reg = head_outputs[1]             # (1, 2, H, W)
        height = head_outputs[2]          # (1, 1, H, W)
        dim = head_outputs[3]             # (1, 3, H, W)
        rot = head_outputs[4]             # (1, 2, H, W)
        vel = head_outputs[5]             # (1, 2, H, W)

        print("Output shapes:")
        print("  Heatmap:", heatmap.shape)
        print("  Reg:", reg.shape)
        print("  Height:", height.shape)
        print("  Dim:", dim.shape)
        print("  Rot:", rot.shape)
        print("  Vel:", vel.shape)

        detections = self.post_process_centerpoint(heatmap[0], reg[0], height[0], dim[0], rot[0], vel[0], W, H)
        return detections

    def scatter_voxel_features2(self, voxel_features, voxel_coords, input_shape):
        """
        Scatter the voxel-wise features into a BEV spatial feature map. Assumes batch-size 1.
        Reference: https://github.com/tianweiy/CenterPoint/blob/master/det3d/models/readers/pillar_encoder.py#L182

        Args:
            voxel_features: np.ndarray of shape (num_voxels, 1, 32)
            voxel_coords: np.ndarray of shape (num_voxels, 4), each row = [1, z, y, x]
            input_shape: tuple (1, 32, H, W)

        Returns:
            spatial_features: np.ndarray of shape (1, 32, H, W)
        """

        voxel_features = voxel_features.squeeze(1)  # shape: (num_voxels, 32)
        ny, nx = input_shape

        # Create the canvas for this sample
        canvas = np.zeros((32, nx, ny))

        # Only include non-empty pillars
        #batch_mask = voxel_coords[:, 0] == 0 # We have only 1 batch
        
        #this_coords = voxel_coords[batch_mask, :]
        indices = voxel_coords[:, 3] * nx + voxel_coords[:, 2]

        #voxels = voxel_features[batch_mask, :]
        voxels = voxel_features.T

        # Now scatter the blob back to the canvas
        canvas[:, indices] = voxels

        # Expand to 3-dim tensor 
        batch_canvas = np.expand_dims(canvas, axis=0) # (1, 32, nrows*ncols)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.reshape(1, 32, ny, nx).astype(np.float32) # (1, 32, nrows, ncols)
        #batch_canvas = batch_canvas[:, :, ::-1, :]
        return batch_canvas
    
    def scatter_voxel_features(self, voxel_features, voxel_coords, input_shape):
        """
        Scatter the voxel-wise features into a BEV spatial feature map. Assumes batch-size 1.
        Reference: https://github.com/tianweiy/CenterPoint/blob/master/det3d/models/readers/pillar_encoder.py#L182

        Args:
            voxel_features: np.ndarray of shape (num_voxels, 1, 32)
            voxel_coords: np.ndarray of shape (num_voxels, 4), each row = [1, z, y, x]
            input_shape: tuple (1, 32, H, W)

        Returns:
            spatial_features: np.ndarray of shape (1, 32, H, W)
        """

        voxel_features = voxel_features.squeeze(1)  # shape: (num_voxels, 32)

        C = 32
        H, W = input_shape

        canvas = np.zeros((C, H, W), dtype=np.float32)
        for feature, coord in zip(voxel_features, voxel_coords):
            _, z, y, x = coord
            canvas[:, y, x] = feature  # Place feature into (C, H, W)

        return np.expand_dims(canvas, axis=0)  # Shape: (1, C, H, W)
    
    def post_process_centerpoint(self,
        out_heatmap, out_offset, out_z, out_dim, out_rot, out_vel,
        down_grid_size_x, down_grid_size_y,
        downsample_factor=1, score_threshold=0.1):
        """
        Post-processing pipeline for the centerpoint model outputs. Assumes batch-size 1.
        Reference: https://github.com/autowarefoundation/autoware_universe/blob/main/perception/autoware_lidar_centerpoint/lib/postprocess/postprocess_kernel.cu

        Args:
            voxel_features: np.ndarray of shape (num_voxels, 1, 32)
            voxel_coords: np.ndarray of shape (num_voxels, 4), each row = [1, z, y, x]
            input_shape: tuple (1, 32, H, W)

        Returns:
            spatial_features: np.ndarray of shape (1, 32, H, W)
        """

        detections = []
        full_detections = []

        def sigmoid(x):
            x = x.astype(np.float64)
            return 1 / (1 + np.exp(-x))

        for yi in range(down_grid_size_y):
            for xi in range(down_grid_size_x):

                # Get label with max score
                max_score = -1
                label = -1
                for ci in range(len(CLASS_NAMES)):
                    score = sigmoid(out_heatmap[ci, yi, xi])
                    #print(score, max_score)
                    if score > max_score:
                        max_score = score
                        label = ci

                offset_x = out_offset[0, yi, xi]
                offset_y = out_offset[1, yi, xi]

                x = VOXEL_SIZE[0] * downsample_factor * (xi + offset_x) + POINT_CLOUD_RANGE[0]
                y = VOXEL_SIZE[1] * downsample_factor * (yi + offset_y) + POINT_CLOUD_RANGE[1]
                z = out_z[0, yi, xi]

                w = out_dim[0, yi, xi]
                l = out_dim[1, yi, xi]
                h = out_dim[2, yi, xi]

                yaw_sin = out_rot[0, yi, xi]
                yaw_cos = out_rot[1, yi, xi]
                yaw_norm = np.sqrt(yaw_sin**2 + yaw_cos**2)

                final_score = max_score if yaw_norm >= YAW_NORM_THRESHOLDS[label] else 0.0

                if final_score < score_threshold:
                    continue

                yaw = np.arctan2(yaw_sin, yaw_cos)
                vel_x = out_vel[0, yi, xi]
                vel_y = out_vel[1, yi, xi]

                det = {
                    "label": label,
                    "score": final_score,
                    "x": x,
                    "y": y,
                    "z": z - LIDAR_HEIGHT,
                    "length": np.exp(w),
                    "width": np.exp(l),
                    "height": np.exp(h),
                    "yaw": yaw,
                    "vel_x": vel_x,
                    "vel_y": vel_y
                }

                box = [det["score"], det["x"], det["y"], det["z"], det["length"], det["width"], det["height"], det["yaw"]]
                full_detections.append(box)
                detections.append(det)

        t0 = time.perf_counter()
        detections = self.nms_3d_rotated(np.array(full_detections))
        print(f"NMS runtime: {time.perf_counter() - t0:.3f}s")
        return detections
    

    def get_bev_polygon(self, x, y, l, w, yaw):
        # Get corners of 2d box
        corners = np.array([
            [-l/2, -w/2],
            [-l/2,  w/2],
            [ l/2,  w/2],
            [ l/2, -w/2]
        ])
        poly = shapely.Polygon(corners)
        poly = shapely.affinity.rotate(poly, yaw)
        poly = shapely.affinity.translate(poly, xoff=x, yoff=y)
        return poly

    def compute_iou_3d_rotated(self, box1, box2):
        # Parse boxes
        _, x1, y1, z1, l1, w1, h1, yaw1 = box1
        _, x2, y2, z2, l2, w2, h2, yaw2 = box2

        # BEV (2D) IoU
        poly1 = self.get_bev_polygon(x1, y1, l1, w1, yaw1)
        poly2 = self.get_bev_polygon(x2, y2, l2, w2, yaw2)
        if not poly1.intersects(poly2):
            return 0.0
        
        union_area = poly1.union(poly2).area
        if union_area == 0:
            return 0.0
        
        inter_area = poly1.intersection(poly2).area
        
        # Calculate height (z-axis) overlapping
        z1_min, z1_max = z1 - h1 / 2, z1 + h1 / 2
        z2_min, z2_max = z2 - h2 / 2, z2 + h2 / 2
        inter_height = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))

        inter_vol = inter_area * inter_height
        vol1 = l1 * w1 * h1
        vol2 = l2 * w2 * h2
        union_vol = vol1 + vol2 - inter_vol
        return inter_vol / union_vol if union_vol > 0 else 0.0
    
    def nms_3d_rotated(self, boxes, iou_threshold=0.1):
        boxes = boxes[np.argsort(-boxes[:, 0])]  # Sort by score descending
        keep = []
        while len(boxes) > 0:
            current = boxes[0]
            keep.append(current)
            if len(boxes) == 1:
                break

            rest = boxes[1:]
            ious = np.array([self.compute_iou_3d_rotated(current, box) for box in rest])
            boxes = rest[ious < iou_threshold]

        return np.array(keep)

class VoxelGenerator:
    def __init__(self, voxel_size, point_cloud_range, max_points_per_voxel, max_voxels):
        self.voxel_size = np.array(voxel_size)
        self.point_cloud_range = np.array(point_cloud_range)
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels

        self.grid_size = ((self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size).astype(np.int32)


    def generate(self, points):
        # Filter points within range
        mask = (
            (points[:, 0] >= self.point_cloud_range[0]) & (points[:, 0] < self.point_cloud_range[3]) &
            (points[:, 1] >= self.point_cloud_range[1]) & (points[:, 1] < self.point_cloud_range[4]) &
            (points[:, 2] >= self.point_cloud_range[2]) & (points[:, 2] < self.point_cloud_range[5])
        )
        points = points[mask]

        # Compute voxel indices
        voxel_indices = ((points[:, :3] - self.point_cloud_range[:3]) / self.voxel_size).astype(np.int32)

        # Column-major order
        coords_flat = voxel_indices[:, 0] + voxel_indices[:, 1] * self.grid_size[0] + voxel_indices[:, 2] * self.grid_size[0] * self.grid_size[1]
        #coords_flat = voxel_indices[:, 0]  * self.grid_size[1] * self.grid_size[2] + voxel_indices[:, 1] * self.grid_size[2] + voxel_indices[:, 2]

        # Sort and group points by voxel key
        sort_idx = np.argsort(coords_flat)
        points = points[sort_idx]
        voxel_indices = voxel_indices[sort_idx]
        coords_flat = coords_flat[sort_idx]

        unique_coords, inverse_indices, counts = np.unique(coords_flat, return_inverse=True, return_counts=True)

        # Limit number of voxels
        if len(unique_coords) > self.max_voxels:
            keep_indices = np.argsort(counts)[-self.max_voxels:]  # pick most populated voxels
            mask = np.isin(inverse_indices, keep_indices)
            points = points[mask]
            voxel_indices = voxel_indices[mask]
            inverse_indices = inverse_indices[mask]
            counts = counts[keep_indices]

        num_voxels = len(np.unique(inverse_indices))
        voxel_features = np.zeros((num_voxels, self.max_points_per_voxel, 9), dtype=np.float32)
        voxel_coords = np.zeros((num_voxels, 4), dtype=np.int32)

        voxel_counts = np.zeros(num_voxels, dtype=np.int32)

        for i, (point, voxel_idx, voxel_id) in enumerate(zip(points, voxel_indices, inverse_indices)):
            cnt = voxel_counts[voxel_id]
            if cnt < self.max_points_per_voxel:
                voxel_features[voxel_id, cnt, :4] = point  # x, y, z, intensity
                voxel_coords[voxel_id] = (0, voxel_idx[2], voxel_idx[1], voxel_idx[0])  # batch_idx, z, y, x
                voxel_counts[voxel_id] += 1

        # Compute mean and center features
        for i in range(num_voxels):
            cnt = voxel_counts[i]
            if cnt == 0:
                continue
            pts = voxel_features[i, :cnt, :4]
            mean_xyz = np.mean(pts[:, :3], axis=0)
            center = (voxel_coords[i, 1:][::-1] + 0.5) * self.voxel_size + self.point_cloud_range[:3]  # x, y, z

            # The first four values are just the point features
            # The next three values are the relative coordinates with respect to the voxel average
            # The last two values are the relative coordinates with respect to the voxel center
            voxel_features[i, :cnt, 4:7] = pts[:, :3] - mean_xyz
            voxel_features[i, :cnt, 7:9] = pts[:, :2] - center[:2]

        return voxel_features, voxel_coords

    
    def generate2(self, points):
        """
        Voxelizes the pointcloud.

        Args:
            points: np.ndarray of shape (n, 4), each row = [x, y, z, intensity]

        Returns:
            voxel_features: np.ndarray of shape (num_voxels, num_max_points, 9), each row = [x, y, z, intensity, x_mean, y_mean, z_mean, x_center, y_center]
            voxel_coords: np.ndarray of shape (batch_idx, z, y, x)
        
        """
        # Filter points within range
        mask = (
            (points[:, 0] >= self.point_cloud_range[0]) & (points[:, 0] < self.point_cloud_range[3]) &
            (points[:, 1] >= self.point_cloud_range[1]) & (points[:, 1] < self.point_cloud_range[4]) &
            (points[:, 2] >= self.point_cloud_range[2]) & (points[:, 2] < self.point_cloud_range[5])
        )
        points = points[mask]

        # Calculate voxel indices
        voxel_indices = ((points[:, :3] - self.point_cloud_range[:3]) / self.voxel_size).astype(np.int32)

        voxel_map = {}
        voxel_features = []
        voxel_coords = []

        # Create map between voxel indices and points
        for i in range(points.shape[0]):
            voxel_idx = tuple(voxel_indices[i])
            if voxel_idx not in voxel_map:
                if len(voxel_map) >= self.max_voxels:
                    continue
                voxel_map[voxel_idx] = []
            if len(voxel_map[voxel_idx]) < self.max_points_per_voxel:
                voxel_map[voxel_idx].append(points[i])

        # Create voxel features and 
        for voxel_idx, pts in voxel_map.items():
            pts = np.array(pts)
            num_points = pts.shape[0]

            # Pad or truncate to max points per voxel
            if num_points < self.max_points_per_voxel:
                padding = np.zeros((self.max_points_per_voxel - num_points, pts.shape[1]), dtype=pts.dtype)
                pts = np.vstack((pts, padding))
            elif num_points > self.max_points_per_voxel:
                pts = pts[:self.max_points_per_voxel]

            # Compute voxel mean and voxel center
            mask_valid = np.any(pts != 0, axis=1)
            voxel_mean = np.mean(pts[mask_valid, :3], axis=0)
            voxel_center = (np.array(voxel_idx) + 0.5) * self.voxel_size + self.point_cloud_range[:3]

            # Build 9-dim features
            relative_mean = pts[:, :3] - voxel_mean  # (x - mean_x, ...)
            relative_center = pts[:, :2] - voxel_center[:2]  # (x - center_x, y - center_y)

            # The first four values are just the point features
            # The next three values are the relative coordinates with respect to the voxel average
            # The last two values are the relative coordinates with respect to the voxel center
            feature = np.concatenate([pts, relative_mean, relative_center], axis=1)  # [N, 9]
            voxel_features.append(feature)

            voxel_coords.append((0, voxel_idx[2], voxel_idx[1], voxel_idx[0]))  # batch_idx, z, y, x

        voxel_features = np.array(voxel_features, dtype=np.float32)
        voxel_coords = np.array(voxel_coords, dtype=np.int32)

        return voxel_features, voxel_coords
    
    def generate3(self, points):
        points_copy = points.copy()
        grid_size = np.floor((self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size).astype(np.int32)

        coor_to_voxelidx = np.full((grid_size[2], grid_size[1], grid_size[0]), -1, dtype=np.int32)
        voxels = np.zeros((self.max_voxels, self.max_points_per_voxel, 9), dtype=points_copy.dtype)
        coordinates = np.zeros((self.max_voxels, 3), dtype=np.int32)
        num_points_per_voxel = np.zeros(self.max_voxels, dtype=np.int32)

        points_coords = np.floor((points_copy[:, :3] - self.point_cloud_range[:3]) / self.voxel_size).astype(np.int32)
        mask = ((points_coords >= 0) & (points_coords < grid_size)).all(1)
        points_coords = points_coords[mask, ::-1]
        points_copy = points_copy[mask]
        assert points_copy.shape[0] == points_coords.shape[0]

        voxel_num = 0
        for i, coord in enumerate(points_coords):
            voxel_idx = coor_to_voxelidx[tuple(coord)]
            if voxel_idx == -1:
                voxel_idx = voxel_num
                voxel_num += 1
                if voxel_num > self.max_voxels:
                    break
                coor_to_voxelidx[tuple(coord)] = voxel_idx
                coordinates[voxel_idx] = coord
            point_idx = num_points_per_voxel[voxel_idx]
            if point_idx < self.max_points_per_voxel:
                voxels[voxel_idx, point_idx] = points_copy[i]
                num_points_per_voxel[voxel_idx] += 1

        return voxels[:voxel_num], coordinates[:voxel_num]#, num_points_per_voxel[:voxel_num]
    

    def generate4(self, points):
        """
        Voxelizes the pointcloud.
        Args:
        points: np.ndarray of shape (n, 4), each row = [x, y, z, intensity]
        Returns:
        voxel_features: np.ndarray of shape (num_voxels, num_max_points, 9), each row = [x, y, z, intensity, x_mean, y_mean, z_mean, x_center, y_center]
        voxel_coords: np.ndarray of shape (batch_idx, z, y, x)
        """

        # Filter points within range
        mask = (
            (points[:, 0] >= self.point_cloud_range[0]) & (points[:, 0] < self.point_cloud_range[3]) &
            (points[:, 1] >= self.point_cloud_range[1]) & (points[:, 1] < self.point_cloud_range[4]) &
            (points[:, 2] >= self.point_cloud_range[2]) & (points[:, 2] < self.point_cloud_range[5])
        )
        points = points[mask]
        
        if len(points) == 0:
            return np.zeros((0, self.max_points_per_voxel, 9), dtype=np.float32), np.zeros((0, 4), dtype=np.int32)
        
        # Calculate voxel indices
        voxel_indices = ((points[:, :3] - self.point_cloud_range[:3]) / self.voxel_size).astype(np.int32)
        
        # Create unique voxel identifier (single integer instead of tuple for faster lookup)
        # Using bit shifting for faster hashing - adjust bit sizes based on your expected grid dimensions
        voxel_ids = (voxel_indices[:, 0] << 20) + (voxel_indices[:, 1] << 10) + voxel_indices[:, 2]
        
        # Group points by voxel (using numpy operations where possible)
        unique_voxel_ids, inverse_indices = np.unique(voxel_ids, return_inverse=True)
        
        # Limit to max_voxels if needed
        if len(unique_voxel_ids) > self.max_voxels:
            unique_voxel_ids = unique_voxel_ids[:self.max_voxels]
            keep_mask = np.isin(voxel_ids, unique_voxel_ids)
            points = points[keep_mask]
            voxel_indices = voxel_indices[keep_mask]
            voxel_ids = voxel_ids[keep_mask]
        
        # Initialize output arrays
        voxel_features = []
        voxel_coords = []
        
        # Process each unique voxel
        for i, voxel_id in enumerate(unique_voxel_ids):
            # Get points belonging to this voxel
            voxel_point_mask = (voxel_ids == voxel_id)
            voxel_points = points[voxel_point_mask]
            
            # Limit to max points per voxel if needed
            if voxel_points.shape[0] > self.max_points_per_voxel:
                voxel_points = voxel_points[:self.max_points_per_voxel]
            
            # Get voxel index (recover x, y, z from id)
            x = voxel_id >> 20
            y = (voxel_id >> 10) & 0x3FF  # 10 bits for y
            z = voxel_id & 0x3FF  # 10 bits for z
            voxel_idx = np.array([x, y, z])
            
            # Compute voxel mean
            voxel_mean = np.mean(voxel_points[:, :3], axis=0)
            
            # Compute voxel center
            voxel_center = (voxel_idx + 0.5) * self.voxel_size + self.point_cloud_range[:3]
            
            # Build 9-dim features
            num_points = voxel_points.shape[0]
            feature = np.zeros((self.max_points_per_voxel, 9), dtype=np.float32)
            
            # Fill with actual points
            feature[:num_points, :4] = voxel_points  # Original point features
            feature[:num_points, 4:7] = voxel_points[:, :3] - voxel_mean  # Relative to mean
            feature[:num_points, 7:9] = voxel_points[:, :2] - voxel_center[:2]  # Relative to center
            
            voxel_features.append(feature)
            voxel_coords.append((0, z, y, x))  # batch_idx, z, y, x
        
        voxel_features = np.array(voxel_features, dtype=np.float32)
        voxel_coords = np.array(voxel_coords, dtype=np.int32)
        
        return voxel_features, voxel_coords

def load_velo_scan(file):
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))

def load_pcd_file(file):
    """Load and parse a pcd file."""
    tartu_pcl = PointCloud.from_path(file)#.numpy()
    print(tartu_pcl.fields)
    return tartu_pcl.numpy()[:, :4]

def load_pointcloud(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    return pcd

def create_bounding_box(center, size, yaw, color=[1, 0, 0]):
    """Creates an oriented bounding box from center (x, y, z), size (l, w, h), and yaw."""
    l, w, h = size
    x, y, z = center

    box = o3d.geometry.OrientedBoundingBox()
    box.center = [x, y, z]
    box.extent = [l, w, h]

    # Yaw rotation around Z-axis
    R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_axis_angle([0, 0, yaw])
    box.R = R
    box.color = color
    return box

def visualize(pcd, boxes3d):
    geometries = [pcd]
    for box in boxes3d:
        bbox = create_bounding_box(
            center=[box['x'], box['y'], box['z']],
            size=[box['length'], box['width'], box['height']],
            yaw=box['yaw'],
            color=[1, 0, 0]  # red boxes
        )
        geometries.append(bbox)
    o3d.visualization.draw_geometries(geometries)


def visualize2(pcd, boxes3d):
    geometries = [pcd]
    for box in boxes3d:
        bbox = create_bounding_box(
            center=[box[1], box[2], box[3]],
            size=[box[4], box[5], box[6]],
            yaw=box[7],
            color=[1, 0, 0]  # red boxes
        )
        geometries.append(bbox)
    o3d.visualization.draw_geometries(geometries)


def visualize_voxels_from_generator(pcd, voxel_coords, voxel_size, point_cloud_range):
    voxel_coords = np.array(voxel_coords)
    voxel_size = np.array(voxel_size)
    point_cloud_range = np.array(point_cloud_range)

    geometries = [pcd]

    for coord in voxel_coords:
        _, z, y, x = coord  # assuming (batch_idx, z, y, x)

        # Compute voxel center in metric space
        center = np.array([x, y, z]) * voxel_size + point_cloud_range[:3] + voxel_size / 2

        # Create a cube at voxel center
        box = o3d.geometry.OrientedBoundingBox()
        box.center = center
        box.extent = voxel_size
        box.color = [0.1, 0.9, 0.1]  # green
        geometries.append(box)

    o3d.visualization.draw_geometries(geometries)


def visualize_voxels(voxel_features, voxel_coords, voxel_size, point_cloud_range, max_voxels=100):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Voxel Grid Visualization")

    voxel_size = np.array(voxel_size)
    point_cloud_range = np.array(point_cloud_range)

    num_voxels = min(len(voxel_features), max_voxels)

    for i in range(num_voxels):
        coords = voxel_coords[i]
        # voxel_coords = [batch_idx, z, y, x]
        x, y, z = coords[3], coords[2], coords[1]
        base = point_cloud_range[:3] + voxel_size * np.array([x, y, z])
        
        # Draw voxel cube
        ax.bar3d(base[0], base[1], base[2],
                 voxel_size[0], voxel_size[1], voxel_size[2],
                 shade=True, alpha=0.1, color='cyan')

        # Draw points inside voxel
        points = voxel_features[i][:, :3]
        mask_valid = np.any(points != 0, axis=1)
        points = points[mask_valid]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def create_voxel_wireframe(origin, size, color=[0, 1, 1]):
    """
    Create a wireframe (LineSet) cube at a given origin with given size.
    """
    x, y, z = origin
    dx, dy, dz = size

    # Define the 8 corners of the box
    corners = np.array([
        [x,     y,     z],
        [x+dx,  y,     z],
        [x+dx,  y+dy,  z],
        [x,     y+dy,  z],
        [x,     y,     z+dz],
        [x+dx,  y,     z+dz],
        [x+dx,  y+dy,  z+dz],
        [x,     y+dy,  z+dz]
    ])

    # Define the 12 edges by connecting corner indices
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]

    colors = [color for _ in lines]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def visualize_voxels_open3d_wireframe(voxel_features, voxel_coords, voxel_size, point_cloud_range, max_voxels=MAX_VOXELS):
    voxel_size = np.array(voxel_size)
    point_cloud_range = np.array(point_cloud_range)

    voxel_edges = []
    voxel_points = []

    num_voxels = min(len(voxel_features), max_voxels)

    for i in range(num_voxels):
        coords = voxel_coords[i]
        x, y, z = coords[3], coords[2], coords[1]
        voxel_origin = point_cloud_range[:3] + voxel_size * np.array([x, y, z])

        # Wireframe cube for voxel
        wireframe = create_voxel_wireframe(voxel_origin, voxel_size)
        voxel_edges.append(wireframe)

        # Points inside the voxel
        pts = voxel_features[i][:, :3]
        mask = np.any(pts != 0, axis=1)
        pts = pts[mask]
        voxel_points.append(pts)

    # Combine all voxel points
    geometries = voxel_edges
    if voxel_points:
        all_points = np.vstack(voxel_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.paint_uniform_color([1, 0, 0])  # Red
        geometries.append(pcd)

    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    model_dir = "/home/pilve/autoware_mini_ws/src/autoware_mini/data/models/centerpoint/"
    centerpoint = CenterpointModel(model_dir+"pts_voxel_encoder_centerpoint.onnx", model_dir+"pts_backbone_neck_head_centerpoint.onnx")
    #for i in range(10):
    #point_coud_path = f"/home/pilve/velodyne_points/dataroot/2024-03-25-15-40-16_mapping_tartu/lidar_center/0000{i}0.pcd"
    point_coud_path = "/home/pilve/velodyne_points/dataroot/2024-03-25-15-40-16_mapping_tartu/lidar_center/000240.pcd"
    pcd = load_pointcloud(point_coud_path)

    #voxel_generator = VoxelGenerator(VOXEL_SIZE, POINT_CLOUD_RANGE, MAX_NUM_POINTS_PER_VOXEL, MAX_VOXELS)
    points = load_pcd_file(point_coud_path)
    #voxel_features, voxel_coords = voxel_generator.generate(points)
    #visualize_voxels_from_generator(pcd, voxel_coords, voxel_generator.voxel_size, voxel_generator.point_cloud_range)
    #visualize_voxels_open3d_wireframe(voxel_features, voxel_coords, voxel_generator.voxel_size, voxel_generator.point_cloud_range)
    
    detections = centerpoint.detect(points)
    print(detections[0])
    
    visualize2(pcd, detections)

