"""
Adapted from https://github.com/facebookresearch/vggt/issues/82
"""
from typing import List
import os
import os.path as osp
import numpy as np
from scipy.spatial.transform import Rotation as R

import glob
import json
import pdb
import tqdm

def rotation_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Calculate the rotation angle between two rotation matrices.
    
    Args:
        R1 (np.ndarray): First 3x3 rotation matrix
        R2 (np.ndarray): Second 3x3 rotation matrix
        
    Returns:
        float: Rotation angle in degrees between the two matrices
    """
    # R1 and R2 are 3x3 rotation matrices
    R = R1.T @ R2
    # Numerical stability: clamp values into [-1,1]
    val = (np.trace(R) - 1) / 2
    val = np.clip(val, -1.0, 1.0)
    angle_rad = np.arccos(val)
    angle_deg = np.degrees(angle_rad)  # Convert radians to degrees
    return angle_deg

def extrinsic_distance(extrinsic1: np.ndarray, extrinsic2: np.ndarray, lambda_t: float = 1.0) -> float:
    """
    Calculate the extrinsic distance between two camera poses.
    
    Args:
        extrinsic1 (np.ndarray): First camera pose (4x4 matrix)
        extrinsic2 (np.ndarray): Second camera pose (4x4 matrix)
        lambda_t (float): Weight for translation component in distance metric. Default: 1.0
        
    Returns:
        float: Extrinsic distance between the two poses
    """
    R1, t1 = extrinsic1[:3, :3], extrinsic1[:3, 3]
    R2, t2 = extrinsic2[:3, :3], extrinsic2[:3, 3]
    rot_diff = rotation_angle(R1, R2) / 180
    
    center_diff = np.linalg.norm(t1 - t2)
    return rot_diff + lambda_t * center_diff

def rotation_angle_batch(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """
    Calculate the rotation angle between all pairs of rotation matrices.
    
    We want a matrix of rotation angles for all pairs.

    Args:
        R1 (np.ndarray): First set of 3x3 rotation matrices, shape (N, 3, 3)
        R2 (np.ndarray): Second set of 3x3 rotation matrices, shape (N, 3, 3)
        
    Returns:
        np.ndarray: Matrix of rotation angles between all pairs, shape (N, N)
    """
    # 
    # We'll get R1^T R2 for each pair.
    # Expand dimensions to broadcast: 
    # R1^T: (N,3,3) -> (N,1,3,3)
    # R2: (N,3,3) -> (1,N,3,3)
    R1_t = np.transpose(R1, (0, 2, 1))[:, np.newaxis, :, :]  # shape (N,1,3,3)
    R2_b = R2[np.newaxis, :, :, :]                          # shape (1,N,3,3)
    R_mult = np.matmul(R1_t, R2_b)  # shape (N,N,3,3)
    # trace(R) for each pair
    trace_vals = R_mult[..., 0, 0] + R_mult[..., 1, 1] + R_mult[..., 2, 2]  # (N,N)
    val = (trace_vals - 1) / 2
    val = np.clip(val, -1.0, 1.0)
    angle_rad = np.arccos(val)
    angle_deg = np.degrees(angle_rad)
    return angle_deg / 180.0  # normalized rotation difference

def extrinsic_distance_batch(extrinsics: np.ndarray, lambda_t: float = 1.0) -> np.ndarray:
    """
    Calculate the extrinsic distance between all pairs of camera poses.
    
    Args:
        extrinsics (np.ndarray): Camera extrinsic matrices of shape (N, 4, 4)
        lambda_t (float): Weight for translation component in distance metric. Default: 1.0
        
    Returns:
        np.ndarray: Matrix of extrinsic distances between all pairs, shape (N, N)
    """
    # extrinsics: (N,4,4)
    # Extract rotation and translation
    R = extrinsics[:, :3, :3]  # (N,3,3)
    t = extrinsics[:, :3, 3]   # (N,3)
    # Compute all pairwise rotation differences
    rot_diff = rotation_angle_batch(R, R)  # (N,N)
    # Compute all pairwise translation differences
    # For t, shape (N,3). We want all pair differences: t[i] - t[j].
    # t_i: (N,1,3), t_j: (1,N,3)
    t_i = t[:, np.newaxis, :]  # (N,1,3)
    t_j = t[np.newaxis, :, :]  # (1,N,3)
    trans_diff = np.linalg.norm(t_i - t_j, axis=2)  # (N,N)
    dists = rot_diff + lambda_t * trans_diff
    return dists

def rotation_angle_batch_chunked(R: np.ndarray, chunk_size: int) -> np.ndarray:
    """
    Calculate the rotation angle between all pairs of rotation matrices in chunks.
    
    Args:
        R (np.ndarray): Set of 3x3 rotation matrices, shape (N, 3, 3)
        chunk_size (int): Size of chunks to process at a time
        
    Returns:
        np.ndarray: Matrix of rotation angles between all pairs, shape (N, N)
    """
    N = R.shape[0]
    rot_diff = np.empty((N, N), dtype=np.float32)
    # Precompute R transpose once
    R_t = R.transpose(0,2,1)
    
    for i_start in range(0, N, chunk_size):
        i_end = min(N, i_start + chunk_size)
        # Sub-block of R_t
        R_i_t = R_t[i_start:i_end]  # (B,3,3)
        
        for j_start in range(0, N, chunk_size):
            j_end = min(N, j_start + chunk_size)
            R_j = R[j_start:j_end]   # (B,3,3)
            # Compute R_i_t @ R_j for block
            # R_i_t: (B,3,3)
            # R_j:   (B,3,3) but we need pairwise, so we expand dims
            # This still can be large. If even BxB is too big, choose smaller chunks.
            # shape (B,B,3,3)
            R_mult = R_i_t[:, np.newaxis, :, :] @ R_j[np.newaxis, :, :, :]
            # Compute trace
            trace_vals = R_mult[...,0,0] + R_mult[...,1,1] + R_mult[...,2,2]
            val = (trace_vals - 1.0) / 2.0
            np.clip(val, -1.0, 1.0, out=val)
            angle_rad = np.arccos(val)
            angle_deg = np.degrees(angle_rad)
            block_rot_diff = angle_deg / 180.0
            rot_diff[i_start:i_end, j_start:j_end] = block_rot_diff.astype(np.float32)
    return rot_diff

def extrinsic_distance_batch_chunked(extrinsics: np.ndarray, lambda_t: float = 1.0, chunk_size: int = 1000) -> np.ndarray:
    """
    Calculate the extrinsic distance between all pairs of camera poses in chunks.
    
    Args:
        extrinsics (np.ndarray): Camera extrinsic matrices of shape (N, 4, 4)
        lambda_t (float): Weight for translation component in distance metric. Default: 1.0
        chunk_size (int): Size of chunks to process at a time. Default: 1000
        
    Returns:
        np.ndarray: Matrix of extrinsic distances between all pairs, shape (N, N)
    """
    R = extrinsics[:, :3, :3].astype(np.float32)
    t = extrinsics[:, :3, 3].astype(np.float32)
    N = R.shape[0]
    # Compute rotation differences in chunks
    rot_diff = rotation_angle_batch_chunked(R, chunk_size)
    # Compute translation differences in chunks
    dists = np.empty((N, N), dtype=np.float32)
    for i_start in range(0, N, chunk_size):
        i_end = min(N, i_start + chunk_size)
        t_i = t[i_start:i_end]  # (B,3)
        for j_start in range(0, N, chunk_size):
            j_end = min(N, j_start + chunk_size)
            t_j = t[j_start:j_end]  # (B,3)
            
            # broadcasting: (B,1,3) - (1,B,3) => (B,B,3)
            diff = t_i[:, None, :] - t_j[None, :, :]
            trans_diff = np.linalg.norm(diff, axis=2)  # (B,B)
            
            # Add rotation and translation
            dists[i_start:i_end, j_start:j_end] = rot_diff[i_start:i_end, j_start:j_end] + lambda_t * trans_diff
    return dists

def compute_pose_based_ranking(extrinsics: np.ndarray, lambda_t: float = 1.0, normalize: bool = True, batched: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute ranking of camera poses based on extrinsic distance.

    Args:
        extrinsics (np.ndarray): Camera extrinsic matrices of shape (N, 4, 4)
        lambda_t (float): Weight for translation component in distance metric. Default: 1.0
        normalize (bool): Whether to normalize camera centers. Default: True
        batched (bool): Whether to use batched computation. Default: True

    Returns:
        tuple:
            - ranking (np.ndarray): Indices sorted by distance for each pose, shape (N, N)
            - dists (np.ndarray): Pairwise distances between poses, shape (N, N)
    """
    if normalize:
        extrinsics = np.copy(extrinsics)
        camera_center = np.copy(extrinsics[:, :3, 3])
        camera_center_scale = np.linalg.norm(camera_center, axis=1)
        avg_scale = np.mean(camera_center_scale)
        extrinsics[:, :3, 3] = extrinsics[:, :3, 3] / avg_scale
    
    
    if batched:
        if len(extrinsics) > 6000:
            dists = extrinsic_distance_batch_chunked(extrinsics, lambda_t=lambda_t)
        else:
            dists = extrinsic_distance_batch(extrinsics, lambda_t=lambda_t)
    else:
        N = extrinsics.shape[0]
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dists[i,j] = extrinsic_distance(extrinsics[i], extrinsics[j], lambda_t=lambda_t)

    # dists[~check_same_interest_view_batch(extrinsics, extrinsics)] = 999999999
    dists[~check_same_interest_view_batch(extrinsics)] = 999999999
    ranking = np.argsort(dists, axis=1)
    return ranking, dists

# def check_same_interest_view_batch(w2c1, w2c2, angle_threshold=90):
#     """
#     Batch version of checking if two sets of cameras (using extrinsic matrices w2c) have overlapping interests.

#     Parameters:
#         w2c1 : np.ndarray
#             Array of shape (N, 4, 4) representing the extrinsic matrices (world-to-camera) for the first set of cameras.
#         w2c2 : np.ndarray
#             Array of shape (M, 4, 4) representing the extrinsic matrices (world-to-camera) for the second set of cameras.
#         angle_threshold : float
#             Threshold angle (in degrees) to determine if two cameras are pointing towards each other.

#     Returns:
#         np.ndarray
#             A boolean array of shape (N, M) where each element indicates if the corresponding
#             pair of cameras (from `w2c1` and `w2c2`) have overlapping interests.
#     """
#     # Convert angle threshold to cosine threshold
#     cos_threshold = np.cos(np.radians(angle_threshold))

#     # Extract camera positions from w2c matrices (translation part)
#     cam_pos1 = -np.einsum('nij,nj->ni', w2c1[:, :3, :3].transpose(0, 2, 1), w2c1[:, :3, 3])[:, [0, 2]]  # (N, 2) dropping y axis
#     cam_pos2 = -np.einsum('nij,nj->ni', w2c2[:, :3, :3].transpose(0, 2, 1), w2c2[:, :3, 3])[:, [0, 2]]  # (N, 2) dropping y axis

#     # Extract camera directions (assuming the z-axis is the forward direction in camera space)
#     cam_dir1 = w2c1[:, 2, :3][:, [0, 2]]  # (N, 2) - z-axis of the camera in world coordinates, dropping y axis
#     cam_dir2 = w2c2[:, 2, :3][:, [0, 2]]  # (M, 2)

#     # Compute relative distances: (N, 1, 2) - (1, M, 2) -> (N, M, 2)
#     distance1 = cam_pos1[:, np.newaxis, :] - cam_pos2[np.newaxis, :, :]  # (N, M, 2)
#     distance2 = cam_pos2[np.newaxis, :, :] - cam_pos1[:, np.newaxis, :]  # (N, M, 2)

#     # Compute dot products: (N, M)
#     dot1 = np.sum(cam_dir2[np.newaxis, :, :] * distance1, axis=-1)  # (N, M)
#     dot2 = np.sum(cam_dir1[:, np.newaxis, :] * distance2, axis=-1)  # (N, M)

#     # Check flags based on cosine threshold
#     flag1 = dot1 >= cos_threshold  # (N, M)
#     flag2 = dot2 >= cos_threshold  # (N, M)

#     cross1 = cam_dir2[np.newaxis, :, 0] * distance1[:, :, 1] - cam_dir2[np.newaxis, :, 1] * distance1[:, :, 0]
#     cross2 = cam_dir1[np.newaxis, :, 0] * distance2[:, :, 1] - cam_dir1[np.newaxis, :, 1] * distance2[:, :, 0]
#     flag3 = cross1 * cross2 < 0                    # look at the same side relative to baseline

#     valid1 = flag1 & flag2 & flag3

#     rot_angle = np.sum(cam_dir1[:, np.newaxis, :] * cam_dir2[np.newaxis, :, :], axis=-1)  # (N, M)
#     valid2 = rot_angle >= cos_threshold

#     return valid1 | valid2


def check_same_interest_view_batch(w2c, angle_threshold=90):
    """
    Batch version of checking if two sets of cameras (using extrinsic matrices w2c) have overlapping interests.

    Parameters:
        w2c1 : np.ndarray
            Array of shape (N, 4, 4) representing the extrinsic matrices (world-to-camera) for the first set of cameras.
        w2c2 : np.ndarray
            Array of shape (M, 4, 4) representing the extrinsic matrices (world-to-camera) for the second set of cameras.
        angle_threshold : float
            Threshold angle (in degrees) to determine if two cameras are pointing towards each other.

    Returns:
        np.ndarray
            A boolean array of shape (N, M) where each element indicates if the corresponding
            pair of cameras (from `w2c1` and `w2c2`) have overlapping interests.
    """
    # Convert angle threshold to cosine threshold
    cos_threshold = np.cos(np.radians(angle_threshold))

    # Extract camera positions from w2c matrices (translation part)
    cam_pos = -np.einsum('nij,nj->ni', w2c[:, :3, :3].transpose(0, 2, 1), w2c[:, :3, 3])[:, [0, 2]]  # (N, 2) dropping y axis

    # Extract camera directions (assuming the z-axis is the forward direction in camera space)
    cam_dir = w2c[:, 2, :3][:, [0, 2]]  # (N, 2) - z-axis of the camera in world coordinates, dropping y axis

    # Compute relative distances: (N, 1, 2) - (1, M, 2) -> (N, M, 2)
    distance = cam_pos[:, np.newaxis, :] - cam_pos[np.newaxis, :, :]  # (N, M, 2)

    # Compute dot products: (N, M)
    dot1 = np.sum(cam_dir[np.newaxis, :, :] * distance, axis=-1)  # (N, M)
    dot2 = np.sum(cam_dir[:, np.newaxis, :] * -distance, axis=-1)  # (N, M)

    # Check flags based on cosine threshold
    flag1 = dot1 >= cos_threshold  # (N, M)
    flag2 = dot2 >= cos_threshold  # (N, M)

    cross1 = cam_dir[np.newaxis, :, 0] * distance[:, :, 1] - cam_dir[np.newaxis, :, 1] * distance[:, :, 0]
    cross2 = cam_dir[:, np.newaxis, 0] * -distance[:, :, 1] - cam_dir[:, np.newaxis, 1] * -distance[:, :, 0]
    flag3 = cross1 * cross2 < 0                    # look at the same side relative to baseline

    valid1 = flag1 & flag2 & flag3

    rot_angle = np.sum(cam_dir[:, np.newaxis, :] * cam_dir[np.newaxis, :, :], axis=-1)  # (N, M)
    valid2 = rot_angle >= cos_threshold

    return valid1 | valid2