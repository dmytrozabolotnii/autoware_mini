#!/usr/bin/env python3
# Based on utils.py from https://github.com/Redhwan-A/6DoFHPE
# Original author: Redhwan Alshammari
# Modified for integration with Autoware Mini

import math
from math import cos, sin
import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
    """
    Draw head pose axis on the image.
    
    Args:
        img: Input image
        yaw: Yaw angle in degrees
        pitch: Pitch angle in degrees
        roll: Roll angle in degrees
        tdx: X-coordinate of the origin (default: image center)
        tdy: Y-coordinate of the origin (default: image center)
        size: Length of the axis lines
        
    Returns:
        img: Image with drawn axis lines
    """
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx is None or tdy is None:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0,0,255), 4)  # X-axis: Red
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0,255,0), 4)  # Y-axis: Green
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255,0,0), 4)  # Z-axis: Blue

    return img

# Normalize vector to unit length
def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    gpu = v.device
    eps = torch.tensor([1e-8], device=gpu)
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v
    
# Cross product of two vectors
def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3
    return out

# Convert 6D rotation representation to rotation matrix
def compute_rotation_matrix_from_ortho6d(poses):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"
    
    Args:
        poses: 6D rotation representation, shape [batch_size, 6]
        
    Returns:
        Rotation matrix of shape [batch_size, 3, 3]
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3
        
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix

# Convert rotation matrix to Euler angles (in radians)
def compute_euler_angles_from_rotation_matrices(rotation_matrices, full_range=False):
    """
    Convert rotation matrices to Euler angles (in radians).
    
    Args:
        rotation_matrices: Rotation matrices of shape [batch_size, 3, 3]
        full_range: Whether to expand yaw angle range to (-180, 180)
        
    Returns:
        Euler angles [x, y, z] in radians, shape [batch_size, 3]
    """
    batch = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
    singular = sy < 1e-6
    singular = singular.float()

    if full_range:
        for i in range(batch):
            if R[i, 0, 0] < 0:
                sy[i] = -sy[i]

    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])  # pitch
    y = torch.atan2(-R[:, 2, 0], sy)  # yaw
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])  # roll

    xs = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    ys = torch.atan2(-R[:, 2, 0], sy)
    zs = torch.zeros_like(R[:, 1, 0])

    # Handle singular cases (gimbal lock)
    device = rotation_matrices.device
    out_euler = torch.zeros(batch, 3, device=device)
    out_euler[:, 0] = x * (1 - singular) + xs * singular  # pitch
    out_euler[:, 1] = y * (1 - singular) + ys * singular  # yaw
    out_euler[:, 2] = z * (1 - singular) + zs * singular  # roll

    return out_euler

# Convert numpy array to rotation matrix
def get_R(x, y, z):
    """
    Get rotation matrix from three rotation angles (radians). right-handed.

    Args:
        x: X-axis rotation angles (in radians)
        y: Y-axis rotation angles (in radians)
        z: Z-axis rotation angles (in radians)

    Returns:
        R: Rotation matrix
    """
    # Create rotation objects from Euler angles
    angles = np.stack([x, y, z], axis=-1)
    if isinstance(angles, np.ndarray):
        r = Rotation.from_euler('xyz', angles, degrees=False)
        R = torch.tensor(r.as_matrix(), dtype=torch.float32)
    else:
        # Handle torch tensors
        angles_np = angles.detach().cpu().numpy()
        r = Rotation.from_euler('xyz', angles_np, degrees=False)
        R = torch.tensor(r.as_matrix(), dtype=torch.float32, device=angles.device)
    
    return R

def stereographic_unproject(a, axis=None):
    """Inverse of stereographic projection: increases dimension by one.

    Args:
        a: Input tensor of shape [batch_size, n]
        axis: Axis to insert the new dimension

    Returns:
        Tensor of shape [batch_size, n+1] after stereographic unprojection
    """
    batch = a.shape[0]
    if axis is None:
        axis = a.shape[1]
    s2 = torch.pow(a, 2).sum(1)  # batch
    ans = torch.zeros(batch, a.shape[1] + 1, device=a.device)  # batch*(n+1)
    unproj = 2 * a / (s2 + 1).view(batch, 1).repeat(1, a.shape[1])  # batch*n
    if axis > 0:
        ans[:, :axis] = unproj[:, :axis]  # batch*(axis-0)
    ans[:, axis] = (s2 - 1) / (s2 + 1)  # batch
    if axis < a.shape[1]:
        ans[:, axis + 1:] = unproj[:, axis:]  # batch*(n-axis)
    return ans
