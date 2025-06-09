
import json
import os
import struct
from pathlib import Path


import numpy as np
import cv2
import open3d as o3d

from tqdm import tqdm
import torch
from internal.utils.colmap import *
from internal.dataparsers.colmap_dataparser import ColmapDataParser

def load_depth_maps(depth_maps_path, step=1):
    """Loads depth maps from a directory, optionally skipping frames."""
    depth_maps = []
    filenames = []
    image_height = None
    image_width = None

    # Get list of files and sort them to ensure consistent ordering
    all_filenames = sorted([f for f in os.listdir(depth_maps_path) if f.endswith(".jpg.tiff")])

    for i in tqdm(range(0, len(all_filenames), step)):
        filename = all_filenames[i]
        filepath = os.path.join(depth_maps_path, filename)
        # Read the TIFF file
        depth_map = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if depth_map is not None:
            depth_maps.append(depth_map)
            filenames.append(filename)
            if image_height is None or image_width is None:
                image_height, image_width = depth_map.shape[:2]
        else:
            print(f"Warning: Could not read depth map from {filepath}")

    if not depth_maps:
        raise FileNotFoundError(f"No depth maps found in {depth_maps_path} with step {step}")

    return depth_maps, image_height, image_width, filenames

# Remove custom Colmap binary reading functions
# def read_colmap_images_bin(images_bin_path):
#     """Reads the images.bin file from Colmap and extracts extrinsics."""
#     images = {}
#     with open(images_bin_path, "rb") as f:
#         num_images = struct.unpack("<I", f.read(4))[0]
#         for _ in range(num_images):
#             image_id = struct.unpack("<I", f.read(4))[0]
#             qvec = np.frombuffer(f.read(32), dtype=np.float64) # Read 32 bytes for quaternion (4 floats)
#             tvec = np.frombuffer(f.read(24), dtype=np.float64)
#             camera_id = struct.unpack("<I", f.read(4))[0]
#             name = b""
#             char = struct.unpack("<c", f.read(1))[0]
#             while char != b"\x00":
#                 name += char
#                 char = struct.unpack("<c", f.read(1))[0]
#             name = name.decode("utf-8", errors='ignore')

#             # Convert quaternion and translation vector to c2w matrix
#             # Colmap stores w, x, y, z for quaternion
#             q = np.array([qvec[1], qvec[2], qvec[3], qvec[0]]) # Convert to x, y, z, w
#             R = quaternion_to_rotation_matrix(q)
#             c2w = np.eye(4)
#             c2w[:3, :3] = R
#             c2w[:3, 3] = tvec

#             images[name] = c2w

#             # Read number of points2D
#             num_points2D = struct.unpack("<I", f.read(4))[0]
#             # Skip point2D coordinates (x, y) and point3D_ids for each point
#             for _ in range(num_points2D):
#                 f.read(24) # Each point is 2 doubles (x, y) + 1 unsigned long long (point3D_id) = 8 + 8 + 8 = 24 bytes

#     return images

# def quaternion_to_rotation_matrix(q):
#     """Converts a quaternion (x, y, z, w) to a rotation matrix."""
#     # Normalize the quaternion to prevent overflow in calculations
#     q = q / np.linalg.norm(q)
#     x, y, z, w = q
#     R = np.array([
#         [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
#         [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
#         [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
#     ])
#     return R

# def load_colmap_extrinsics(colmap_sparse_path):
#     """Loads camera extrinsics (c2w matrices) from Colmap images.bin."""
#     images_bin_path = os.path.join(colmap_sparse_path, "images.bin")
#     if not os.path.exists(images_bin_path):
#         raise FileNotFoundError(f"images.bin not found at {images_bin_path}")

#     images_data = read_colmap_images_bin(images_bin_path)

#     # Extract c2w matrices in the order of image names from depth maps
#     # Assuming image names in depth maps match those in Colmap images.bin
#     # This might need adjustment based on actual naming conventions
#     c2w_list = []
#     # Need the list of image names from depth maps to match the order
#     # For now, return a dictionary and handle ordering in main
#     return images_data

from internal.utils import colmap as colmap_utils

def visualize_simple(voxel_grid, voxel_size, grid_origin):

    occupied_indices = np.argwhere(voxel_grid > 0)
    occupied_points = occupied_indices * voxel_size + grid_origin

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(occupied_points)

    voxel_grid_vis = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

    # o3d.visualization.draw_geometries([voxel_grid_vis])
    # visualiz points
    # o3d.visualization.draw_geometries([pcd])
    return pcd


def visualize_points(depth_maps, c2w_list, fx, fy, cx,cy):

    points = []

    for depth, c2w in zip(depth_maps, c2w_list):
        points_world, origin = depth_to_point_cloud(depth, c2w, fx, fy, cx, cy)
        points.append(points_world)

    points = np.concatenate(points, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([pcd])
    return pcd


def raycast_and_update(voxel_grid, origin, point, voxel_size, grid_origin):
    direction = point - origin
    length = np.linalg.norm(direction)
    direction /= length
    steps = int(length / (voxel_size * 0.5))
    for step in range(steps):
        pos = origin + direction * step * voxel_size * 0.5
        voxel_idx = np.floor((pos - grid_origin) / voxel_size).astype(int)
        voxel_idx_tuple = tuple(voxel_idx)
        voxel_grid[voxel_idx_tuple] = 0  # 清空该voxel

def create_occupancy_grid_raycast(depth_maps, c2w_list, fx, fy, cx, cy, 
                                  voxel_size=0.1, grid_size=(64, 64, 64), grid_origin=(-5, -5, -5)):
    voxel_grid = np.ones(grid_size, dtype=np.uint8)

    for depth, c2w in zip(depth_maps, c2w_list):
        points, camera_origin = depth_to_point_cloud(depth, c2w, fx, fy, cx, cy)
        for pt in points:
            raycast_and_update(voxel_grid, camera_origin, pt, voxel_size, grid_origin)

    return voxel_grid   

def depth_to_point_cloud(depth, c2w, fx, fy, cx, cy):
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth.flatten()
    x = (u.flatten() - cx)  / fx
    y = (v.flatten() - cy) / fy
    points_camera = np.vstack((x, y, z, np.ones_like(z)))

    coordinates = np.vstack((x, y, -np.ones_like(x))).T

    coordinates[..., 1] *= -1
    
    rotation = c2w[:3, :3]

    directions = np.sum( coordinates[...,None, :] * rotation, axis=-1)

    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    points_world = directions*z[:, None] + c2w[:3, 3]

    return points_world, c2w[:3, 3]



def depth_to_world_points_torch(depth, c2w, fx, fy, cx, cy):
    h, w = depth.shape
    device = c2w.device  # 让所有计算在相同设备上进行
    u, v = torch.meshgrid(torch.arange(w, device=device), torch.arange(h, device=device), indexing='xy')
  

    z = depth.flatten()

    x = (u.flatten() - cx) / fx
    y = (v.flatten() - cy) / fy
    # points_camera = torch.stack([x, y, z, torch.ones_like(z)], dim=0)

    coordinates = torch.stack([x, y, -torch.ones_like(x)], dim=-1)

    coordinates[..., 1] *= -1
    rotation = c2w[:3, :3]
    directions = torch.sum(coordinates[..., None, :] * rotation, dim=-1)
    directions /= torch.norm(directions, dim=-1, keepdim=True)

    points_world = directions * z[:, None] + c2w[:3, 3]

    # points_world = (c2w @ points_camera)[:3].T  # 变换到世界坐标
    # origin_world = c2w[:3, 3]  # 相机中心
    return c2w[:3, 3], points_world

def dda_3d_batch_torch(origin, points, voxel_size, grid_origin, grid_size):
    """
    批量化3D DDA算法，计算所有射线穿过的voxel索引
    """
    device = points.device
    origin = origin.to(device)
    grid_origin = torch.tensor(grid_origin, device=device)

    directions = points - origin[None, :]  # 计算射线方向

    # make the length 1 voxel shorter to avoid going out of bounds
    lengths = torch.norm(directions, dim=1, keepdim=True) - 5*voxel_size
    # make sure length is larger than 0
    lengths = torch.clamp(lengths, min=0)
    directions = directions / (lengths + 1e-8)  # 归一化

    max_steps = torch.floor(lengths / (voxel_size * 0.5)).long().squeeze()  # 计算每条射线的步数
    max_steps = torch.clamp(max_steps, min=0)  # 确保至少有一步

    steps = torch.arange(max_steps.max(), device=device)[None, :].expand(len(points), -1)
    mask = steps < max_steps[:, None]  # 只计算需要的步数

    pos = origin[None, :] + directions[:, None, :] * steps[..., None] * voxel_size * 0.5
    voxel_indices = torch.floor((pos - grid_origin) / voxel_size).long()  # 转换到网格索引

    voxel_indices = voxel_indices[mask]  # 仅保留有效的索引
    voxel_indices = torch.unique(voxel_indices,  dim=0)  # 去重，提高效率

    valid_mask = ((voxel_indices >= 0) & (voxel_indices < torch.tensor(grid_size, device=device))).all(dim=1)
    return voxel_indices[valid_mask]  # 过滤掉超出边界的点

def create_occupancy_grid_batch_torch(depth_maps, c2w_list, fx, fy, cx, cy,
                                      voxel_size=0.5, grid_size=(32,32,32), grid_origin=(-5,-5,-4), device="cuda"):
    voxel_grid = torch.ones(grid_size, dtype=torch.uint8, device=device)  # 初始化网格

    grid_origin = torch.tensor(grid_origin, dtype=torch.float32, device=device)
    depth_maps = [torch.tensor(d, dtype=torch.float32, device=device) for d in depth_maps]
    c2w_list = [torch.tensor(c, dtype=torch.float32, device=device) for c in c2w_list]

    for depth, c2w in zip(depth_maps, c2w_list):
        origin, points = depth_to_world_points_torch(depth, c2w, fx, fy, cx, cy)
        voxel_indices = dda_3d_batch_torch(origin, points, voxel_size, grid_origin, grid_size)

        voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 0  # 标记清空区域

    return voxel_grid


def precompute_voxel_indices(grid_size, voxel_size, grid_origin, device="cuda"):
    """ 预计算3D网格索引，减少重复计算 """
    device = torch.device(device)
    grid_indices = torch.meshgrid(
        torch.arange(grid_size[0], device=device),
        torch.arange(grid_size[1], device=device),
        torch.arange(grid_size[2], device=device),
        indexing='ij'
    )

    grid_origin = torch.tensor(grid_origin, device=device)
    grid_indices = torch.stack(grid_indices, dim=-1).reshape(-1, 3).float()
    voxel_positions = grid_indices * voxel_size + grid_origin  # 转换到世界坐标
    voxel_positions_homo = torch.cat([voxel_positions, torch.ones((voxel_positions.shape[0], 1), device=device)], dim=-1)
    return grid_indices.long(), voxel_positions_homo

def project_voxels_to_camera(voxel_grid, voxel_positions_homo, depth_map, w2c, image_height, image_width, voxel_size=0.05,  device="cuda"):
    """ 投影Voxel到当前相机，基于Depth Map决定占用情况 (ERP Camera Model) """
    device = torch.device(device)
    voxel_grid = voxel_grid.to(device)
    w2c = torch.tensor(w2c, dtype=torch.float32, device=device)
    depth_map = torch.tensor(depth_map, dtype=torch.float32, device=device)


    R = w2c[:3, :3]
    T = w2c[:3, 3]

    # 将 Voxel 转换到相机坐标系
    voxel_camera = voxel_positions_homo[..., :3] @ R.T + T

    # voxel_camera[..., 1] *= -1
    # 获取相机坐标系中的 (x, y, z)
    x_c = voxel_camera[..., 0]
    y_c = voxel_camera[..., 1]
    z_c = voxel_camera[..., 2]

    # 计算球面坐标
    r = torch.sqrt(x_c**2 + y_c**2 + z_c**2)
    theta = torch.atan2(x_c, z_c)  # [-π, π]
    phi = torch.asin(y_c / r)       # [-π/2, π/2]
    
    normalized_theta = theta / np.pi
    normalized_phi = phi / np.pi * 2
    # 映射到像素坐标
    x_p = ((normalized_theta + 1) / 2) * image_width
    y_p = ((normalized_phi + 1) / 2) * image_height
    # to i , j
    i = torch.clamp(y_p.long(), min=0, max=image_height - 1)
    j = torch.clamp(x_p.long(), min=0, max=image_width - 1)

    depth_map = depth_map[i, j] 
    # 计算射线方向上的距离（Range Distance）
    range_distance = r

    # 计算占用状态 (free/occupied)
    free_mask = (range_distance < (depth_map - 2*voxel_size))
    voxel_grid -= free_mask.long()

    return voxel_grid

def create_occupancy_grid_from_depth(depth_maps, w2c_list, image_height, image_width,
                                     voxel_grid, grid_indices, voxel_positions_homo, voxel_size,
                                     depth_scale=1.0, device="cuda"):
    """ Through multiple Depth Maps determine the occupancy grid, optimized calculation (ERP Camera Model) """
    device = torch.device(device)

    for depth, w2c in tqdm(zip(depth_maps, w2c_list), total=len(depth_maps), desc="Creating Occupancy Grid"):
        voxel_grid = project_voxels_to_camera(voxel_grid, grid_indices, voxel_positions_homo,
                                              depth, w2c, image_height, image_width, voxel_size,
                                              depth_scale=depth_scale, device=device)

    return voxel_grid

import random

def construct_c2w_from_colmap(image_data):
    """ Construct camera-to-world matrix from COLMAP image data """
    # Extract rotation (as quaternion) and translation
    R = qvec2rotmat(image_data.qvec)
    T = image_data.tvec

    # Construct world-to-camera matrix
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = T

    # Invert to get c2w
    c2w = np.linalg.inv(w2c)
    R_flip = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    return c2w
    # Apply the flip to adjust the coordinate system
    c2w_nerf = np.eye(4)
    c2w_nerf[:3, :3] = R_flip @ c2w[:3, :3]
    c2w_nerf[:3, 3] = R_flip @ c2w[:3, 3]

    return c2w_nerf

def get_w2c_from_colmap(image_data):
    """ Get world-to-camera matrix from COLMAP image data """
    # Extract rotation (as quaternion) and translation
    R = qvec2rotmat(image_data.qvec)
    T = image_data.tvec

    # Construct world-to-camera matrix
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = T

    return w2c


def build_grid(origin_w, R, scene_size, spacing, centre_origin=True):
    """
    centre_origin=True  → 方案 A（origin = 网格中心）
    centre_origin=False → 方案 B（origin = 最小角）
    """
    dx, dy, dz = np.broadcast_to(spacing, 3)
    nx, ny, nz = np.ceil(scene_size / (dx, dy, dz)).astype(int)

    # 索引到本地坐标
    ix = (np.arange(nx) + 0.5) * dx
    iy = (np.arange(ny) + 0.5) * dy
    iz = (np.arange(nz) + 0.5) * dz
    x_loc, y_loc, z_loc = np.meshgrid(ix, iy, iz, indexing='ij')
    p_local = np.stack([x_loc, y_loc, z_loc], -1)          # (nx,ny,nz,3)

    if centre_origin:    # 方案 A：把网格搬到几何中心
        p_local -= np.array([nx*dx, ny*dy, nz*dz]) / 2.0

    # 本地 → 世界
    coords_w = (p_local @ R.T) + origin_w
    coords_w = coords_w.astype(np.float32).transpose(2,1,0,3)  # →(nz,ny,nx,3)

    grid = np.ones((nz, ny, nx), dtype=np.int32)

    return grid, coords_w


def get_occupancy_grid(source_path, output_path, all_grid_info ):

    depth_maps_path = os.path.join(source_path, "semantic/depths")
    
    depth_maps, image_height, image_width, filenames = load_depth_maps(depth_maps_path, step = 20)

    # Load Colmap image data using colmap_utils
    images_bin_path = os.path.join(source_path, "sparse", "images.bin")
    if not os.path.exists(images_bin_path):
        raise FileNotFoundError(f"images.bin not found at {images_bin_path}")
    colmap_images_data = read_images_binary(images_bin_path)

    # Create a dictionary mapping image names to Image objects for easier lookup
    colmap_images_by_name = {img_data.name: img_data for img_id, img_data in colmap_images_data.items()}

    # Create ordered list of c2w matrices based on depth map filenames
    w2c_list = []
    
    for filename in filenames:
        colmap_image_name = filename.replace('.jpg.tiff', '.jpg')
        colmap_image_data = colmap_images_by_name[colmap_image_name]
        w2c = get_w2c_from_colmap(colmap_image_data)
        w2c_list.append(w2c)

    # Create occupancy grid
    global_transformation = all_grid_info['global_transformation'] # 4,4
    global_scene_size = all_grid_info['global_scene_size'] # 3, in x, y, z full length
    spacing_coarse = all_grid_info['spacing_coarse'] 
    spacing_fine = all_grid_info['spacing_fine']

    global_origin = global_transformation[:3, 3] # center
    global_rotation = global_transformation[:3, :3]
    
    occupancy_global, coords_global = build_grid(
        global_origin, global_rotation, global_scene_size, spacing_coarse, centre_origin=True
    )

    # to torch
    occupancy_global = torch.from_numpy(occupancy_global).cuda()
    coords_global = torch.from_numpy(coords_global).cuda()

    additional_grids = []
    # Save occupancy grid
    for addtional_grid in all_grid_info['additional_boxes']:
        local_origin = addtional_grid['origin']
        local_size = addtional_grid['scene_size']
        occupancy_local, coords_local = build_grid(
            local_origin, global_rotation, local_size, spacing_fine, centre_origin=True
        )

        occupancy_local = torch.from_numpy(occupancy_local).cuda()
        coords_local = torch.from_numpy(coords_local).cuda()

        additional_grids.append((occupancy_local, coords_local))

    
    for (depth, w2c) in tqdm(zip(depth_maps, w2c_list), total=len(depth_maps), desc="Creating Occupancy Grid"):

        occupancy_global = project_voxels_to_camera(occupancy_global, coords_global, depth, w2c, image_height, image_width, spacing_coarse)

        for i, (occupancy, coords) in enumerate(additional_grids):
            occupancy = project_voxels_to_camera(occupancy, coords, depth, w2c, image_height, image_width, spacing_fine)

            # overwrite
            additional_grids[i] = (occupancy, coords)

    # Save occupancy grid
    try:
        save_to_npz(output_path, {
            'global_grid': (occupancy_global, coords_global),
            'additional_grids': additional_grids,
            'global_rotation': global_rotation
        })
    except Exception as e:  
        print(f"Error saving occupancy grid: {e}")

    return {
        'global_grid': (occupancy_global, coords_global),
        'additional_grids': additional_grids,
        'global_rotation': global_rotation
    }
        



def save_to_npz(out_path: str, occ_dict: dict) -> None:
    """
    occ_dict 结构同 get_occupancy_grid 的返回值:
        {
          'global_grid'    : (occ_g, coords_g),
          'additional_grids': [(occ_l0, coords_l0), ...],
          'global_rotation': R
        }
    """
    arrays = {}
    # --- 全局 ---
    occ_g, coords_g = occ_dict['global_grid']

    # to numpy if they are torch tensors
    if isinstance(occ_g, torch.Tensor):
        occ_g = occ_g.cpu().numpy()
    if isinstance(coords_g, torch.Tensor):
        coords_g = coords_g.cpu().numpy()

    arrays['global_occ']    = occ_g > 0
    arrays['global_coords'] = coords_g
    arrays['global_rot']    = occ_dict['global_rotation']

    # --- 局部 ---
    for i, (occ, coords) in enumerate(occ_dict['additional_grids']):
        # to numpy if they are torch tensors
        if isinstance(occ, torch.Tensor):
            occ = occ.cpu().numpy()
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()
        arrays[f'add{i}_occ']    = occ > 0
        arrays[f'add{i}_coords'] = coords

    # --- 打包 ---
    np.savez_compressed(out_path, **arrays)

    return


def main():
    # Load depth maps and Colmap extrinsics
    depth_maps_path = '/mnt/e/nanjing_project/scene_april/panorama/semantic/depths'
    colmap_sparse_path = '/mnt/e/nanjing_project/scene_april/panorama/sparse'

    depth_maps, image_height, image_width, filenames = load_depth_maps(depth_maps_path, step = 10)

    # Load Colmap image data using colmap_utils
    images_bin_path = os.path.join(colmap_sparse_path, "images.bin")
    if not os.path.exists(images_bin_path):
        raise FileNotFoundError(f"images.bin not found at {images_bin_path}")
    colmap_images_data = read_images_binary(images_bin_path)

    # Create a dictionary mapping image names to Image objects for easier lookup
    colmap_images_by_name = {img_data.name: img_data for img_id, img_data in colmap_images_data.items()}

    # Create ordered list of c2w matrices based on depth map filenames
    w2c_list = []
    
    for filename in filenames:
        colmap_image_name = filename.replace('.jpg.tiff', '.jpg')

        if colmap_image_name in colmap_images_by_name:
            image_data = colmap_images_by_name[colmap_image_name]
            w2c = get_w2c_from_colmap(image_data)
            w2c_list.append(w2c)

            
        else:
             print(f"Warning: Image {colmap_image_name} not found in Colmap images.bin data.")


    if not w2c_list:
        raise RuntimeError("No matching camera extrinsics found for the loaded depth maps.")

    sub_division = 128

    scene_size = 12
    scene_offset = (0,0,5)
    grid_origin = torch.tensor([-scene_size/2, -scene_size/2, -scene_size/2]) + torch.tensor(scene_offset)

    # grid_origin = (0,0,0)
    voxel_size = scene_size/sub_division
    grid_size = (sub_division,sub_division,int(sub_division*2/4))

    # Note: fx, fy, cx, cy are not used for ERP projection in create_occupancy_grid_from_depth
    # The function will be modified to use image_height and image_width instead.
    fx, fy, cx, cy = None, None, None, None # These are not used for ERP projection


    c2w = torch.linalg.inv(torch.tensor(w2c_list))  

    camera_pos = c2w[:, :3, 3]
    
    up = -torch.mean(c2w[:, :3, 1], dim=0)
    up = up / torch.linalg.norm(up)

    rotation = ColmapDataParser.rotation_matrix( up, torch.tensor([0, 0, 1], dtype=up.dtype))
    transform = torch.eye(4)
    transform[:3, :3] = rotation
    transform = transform.to(device='cuda')

    voxel_grid = torch.ones(grid_size, dtype=torch.long, device='cuda')  # Initialize as occupied (1)

    # Precompute 3D grid indices
    grid_indices, voxel_positions_homo = precompute_voxel_indices(grid_size, voxel_size, grid_origin, device='cuda')

    # transform to the upright direction
    voxel_positions_homo = voxel_positions_homo @ transform


    occupancy_grid = create_occupancy_grid_from_depth(depth_maps, w2c_list,
                                                      image_height=image_height, image_width=image_width,
                                                      voxel_grid=voxel_grid,
                                                      grid_indices=grid_indices,
                                                      voxel_positions_homo=voxel_positions_homo,
                                                      voxel_size=voxel_size,
                                                      depth_scale=1.0,
                                                      )


    voxel_pos = voxel_positions_homo[occupancy_grid.flatten() > 0, :3]

    print('nodes',len(voxel_pos),'ratio',len(voxel_pos)/len(occupancy_grid.flatten()))
    # occupancy_grid = occupancy_grid.numpy(force=True)



    # occ = visualize_simple(occupancy_grid, voxel_size, grid_origin)

    occ_pcd = o3d.geometry.PointCloud()
    occ_pcd.points = o3d.utility.Vector3dVector(voxel_pos.numpy(force=True))

    occ = o3d.geometry.VoxelGrid.create_from_point_cloud(occ_pcd, voxel_size=voxel_size)

    # create a point cloud for cameras (optional, for visualization)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(camera_pos)
    # use black
    pcd.paint_uniform_color([0, 0, 0])

    # depth = visualize_points(depth_maps, c2w_list, fx, fy, cx, cy) # This function still uses fx, fy, cx, cy and pinhole projection

    o3d.visualization.draw_geometries([occ_pcd])

    # save the the point positions
    np.save('voxel_pos.npy', voxel_pos.numpy(force=True))


