import numpy as np
import cv2

from .constants import DEPTH_IMG_SCALE, PC_MIN, PC_MAX

def depth2pcd(depth: np.ndarray, intrinsics:np.ndarray) -> np.ndarray:
    '''
    depth: (H, W)
    intrinsics: (3, 3)
    '''
    height, width = depth.shape
    v, u = np.meshgrid(
        range(height), range(width),
        indexing='ij'
    ) # (H, W), (H, W)
    u, v = u.flatten(), v.flatten() # (HW,), (HW,)
    depth_flat = depth.flatten() # (HW,)
    valid = depth_flat > 0 # (N,)

    u = u[valid] # (N,)
    v = v[valid] # (N,)
    depth_flat = depth_flat[valid] # (N,)

    pixels = np.stack([u, v, np.ones_like(u)], axis=0) # (3, N)
    rays = np.linalg.inv(intrinsics) @ pixels # (3, N)
    points = rays * depth_flat[np.newaxis, :] # (3, N)
    return points.T # (N, 3)

def load_depth(depth_image_path:str):
    depth_array = (
        np.array(
            cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
        )
        / DEPTH_IMG_SCALE
    )
    return depth_array

def pcd_camera2world(pcd_camera: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    return np.einsum("ab,nb->na", camera_pose[:3, :3], pcd_camera) + camera_pose[:3, 3]

def pcd_world2object(pcd_world: np.ndarray, object_pose: np.ndarray) -> np.ndarray:
    return np.einsum(
        "ba,nb->na", object_pose[:3, :3], pcd_world - object_pose[:3, 3]
    )

def get_workspace_mask(pc:np.ndarray) -> np.ndarray:
    pc_mask = (
        (pc[:, 0] > PC_MIN[0])
        & (pc[:, 0] < PC_MAX[0])
        & (pc[:, 1] > PC_MIN[1])
        & (pc[:, 1] < PC_MAX[1])
        & (pc[:, 2] > PC_MIN[2])
        & (pc[:, 2] < PC_MAX[2])
    )
    return pc_mask