import numpy as np
import trimesh
from plyfile import PlyData
import open3d as o3d
from probreg import l2dist_regs, cpd
use_cuda = True
if use_cuda:
    import cupy as cp
    to_cpu = cp.asnumpy
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
else:
    cp = np
    to_cpu = lambda x: x
import copy
import time

def load_gaussian_points(ply_path):
    ply = PlyData.read(ply_path)
    xyz = np.stack([ply['vertex'][axis] for axis in ['x', 'y', 'z']], axis=-1)
    return xyz.astype(np.float32)


def sample_mesh_surface(obj_path, num_points=5000):
    mesh = trimesh.load(obj_path, process=False)
    # mesh.apply_scale(0.0005)
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points.astype(np.float32), mesh

def sample_mesh_volume(obj_path, num_points):
    mesh = trimesh.load(obj_path, process=False)
    bounds = mesh.bounds
    min_bound, max_bound = bounds
    collected_points = []

    batch_size = int(num_points * 3)  # 每批尝试多采一些
    total_collected = 0

    while total_collected < num_points:
        # 在 bounding box 中采样点
        sample = np.random.uniform(low=min_bound, high=max_bound, size=(batch_size, 3))

        # 判断哪些点在 mesh 内部
        inside_mask = mesh.contains(sample)
        inside_points = sample[inside_mask]

        if inside_points.shape[0] > 0:
            collected_points.append(inside_points)
            total_collected += inside_points.shape[0]

    all_points = np.vstack(collected_points)
    return all_points[:num_points], mesh
def estimate_normals(pcd, radius=0.05):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    pcd.normalize_normals()

def icp_registration(source_points, target_points):
    src_center = np.mean(source_points, axis=0)
    tgt_center = np.mean(target_points, axis=0)

    src_normalized = source_points - src_center
    tgt_normalized = target_points - tgt_center

    src_extent = np.max(np.linalg.norm(src_normalized, axis=1))
    tgt_extent = np.max(np.linalg.norm(tgt_normalized, axis=1))
    scale = tgt_extent / src_extent
    src_normalized *= scale

    src_pc = o3d.geometry.PointCloud()
    tgt_pc = o3d.geometry.PointCloud()
    src_pc.points = o3d.utility.Vector3dVector(src_normalized)
    tgt_pc.points = o3d.utility.Vector3dVector(tgt_normalized)

    estimate_normals(src_pc)
    estimate_normals(tgt_pc)

    # src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    #     src_pc, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))
    # tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    #     tgt_pc, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))

    # ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    #     src_pc, tgt_pc, src_fpfh, tgt_fpfh, True, 0.1,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    #     4, [
    #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
    #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.1)
    #     ],
    #     o3d.pipelines.registration.RANSACConvergenceCriteria(40000, 500)
    # )
    loss = o3d.pipelines.registration.TukeyLoss(k=0.02)
    # print(ransac_result.transformation)
    # refine with ICP
    icp_result = o3d.pipelines.registration.registration_icp(
        src_pc, tgt_pc, 0.2, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True,)

    )

    icp_transform = icp_result.transformation

    # 补偿变换： T_final = T_tgt_center @ T_icp @ T_scale @ T_src_center_inv
    T_src_center = np.eye(4)
    T_src_center[:3, 3] = -src_center

    T_scale = np.eye(4)
    T_scale[:3, :3] *= scale

    T_tgt_center = np.eye(4)
    T_tgt_center[:3, 3] = tgt_center

    full_transform = T_tgt_center @ icp_transform @ T_scale @ T_src_center

    return full_transform, scale

def align_mesh_to_gaussian(obj_path, ply_path, output_path="aligned_mesh.obj", num_sample=3000):
    gaussian_pts = load_gaussian_points(ply_path)
    gaussian_pts = remove_outliers(gaussian_pts, nb_neighbors=40, std_ratio=2.0)
    sampled_pts, mesh = sample_mesh_volume(obj_path, num_sample)
    transform, scale = icp_registration(sampled_pts, gaussian_pts)

    # 输出旋转、平移、缩放
    R = transform[:3, :3] / scale  # 去掉 scale，便于显示真实旋转
    T = transform[:3, 3]

    print("Rotation matrix R:\n", R)
    print("Translation vector T:", T)
    print("Uniform scale factor:", scale)

    aligned_mesh = apply_transform_to_mesh(mesh, R, T, scale)
    aligned_mesh.export(output_path)
    print(f"Saved aligned mesh to {output_path}")

    # 可视化结果（使用采样点而非 mesh）
    aligned_sampled = (sampled_pts @ (R.T * scale)) + T
    sampled_pc = o3d.geometry.PointCloud()
    sampled_pc.points = o3d.utility.Vector3dVector(aligned_sampled)
    sampled_pc.paint_uniform_color([1.0, 0.5, 0.2])

    gauss_pc = o3d.geometry.PointCloud()
    gauss_pc.points = o3d.utility.Vector3dVector(gaussian_pts)
    gauss_pc.paint_uniform_color([0.1, 0.6, 0.8])

    o3d.visualization.draw_geometries([sampled_pc, gauss_pc])

    return R, T, scale, transform, aligned_mesh


def remove_outliers(points, nb_neighbors=20, std_ratio=2.0):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    cl, ind = pc.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return np.asarray(cl.points)

def apply_transform_to_mesh(mesh, R, T, scale):
    transform = np.eye(4)
    transform[:3, :3] = R * scale
    transform[:3, 3] = T
    mesh = mesh.copy()
    mesh.apply_transform(transform)
    return mesh


def test_alignment_on_mesh(obj_path, num_points=2000):
    # 采样原始点
    pts, mesh = sample_mesh_surface(obj_path, num_points)

    # 构造随机变换
    np.random.seed(42)
    rand_rot = o3d.geometry.get_rotation_matrix_from_xyz(np.random.uniform(0, np.pi, 3))
    rand_trans = np.random.uniform(-0.5, 0.5, 3)
    rand_scale = np.random.uniform(0.8, 1.2)

    pts_transformed = (pts @ (rand_rot.T * rand_scale)) + rand_trans

    # 对齐
    estimated_T, estimated_scale = icp_registration(pts_transformed, pts)
    R_est = estimated_T[:3, :3] / estimated_scale
    T_est = estimated_T[:3, 3]

    # 变换回去
    aligned_pts = (pts_transformed @ (R_est.T * estimated_scale)) + T_est

    # 可视化
    pc_src = o3d.geometry.PointCloud()
    pc_src.points = o3d.utility.Vector3dVector(pts)
    pc_src.paint_uniform_color([0, 1, 0])

    pc_dst = o3d.geometry.PointCloud()
    pc_dst.points = o3d.utility.Vector3dVector(pts_transformed)
    pc_dst.paint_uniform_color([1, 0, 0])

    pc_aligned = o3d.geometry.PointCloud()
    pc_aligned.points = o3d.utility.Vector3dVector(aligned_pts)
    pc_aligned.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([pc_src, pc_dst, pc_aligned])

def run_cpd_rigid_scaling(source, target):
    
    source = cp.asarray(source)
    target = cp.asarray(target)

    rcpd = cpd.RigidCPD(source, use_cuda=use_cuda)

    tf_param, _, _ = rcpd.registration(target, maxiter=100)

    # 获取变换参数
    R = tf_param.rot
    t = tf_param.t
    s = tf_param.scale

    # 应用变换
    source_transformed = (s * source) @ R.T + t
    
    source_transformed = cp.asnumpy(source_transformed)
    R = cp.asnumpy(R)
    t = cp.asnumpy(t)
    s = cp.asnumpy(s)
    return source_transformed, R, t, s

def run_gmm_rigid_scaling(source, target):
    """使用 GMMReg 执行刚性+缩放配准。"""
    # 将点云转换为 Open3D 格式
    source_pc = o3d.geometry.PointCloud()
    source_pc.points = o3d.utility.Vector3dVector(source)
    target_pc = o3d.geometry.PointCloud()
    target_pc.points = o3d.utility.Vector3dVector(target)

    # 执行 GMMReg 配准
    tf_param = l2dist_regs.registration_gmmreg(source, target, tf_type_name='rigid')

    # 获取变换参数
    R = tf_param.rot
    t = tf_param.t
    s = tf_param.scale

    # 应用变换
    source_transformed = (s * source) @ R.T + t

    return source_transformed, R, t, s

def align_with_gmm(
    obj_path,
    ply_path,
    use_volume=False,
    num_points=3000,
    visualize=True,
    save_aligned=False,
    output_path="aligned.obj"
):
    # 加载目标点云（PLY）
    target = load_gaussian_points(ply_path)
    # remove outliers
    target = remove_outliers(target, nb_neighbors=50, std_ratio=2.0)
    # 从源网格（OBJ）采样点云
    if use_volume:
        source, mesh = sample_mesh_volume(obj_path, num_points)
    else:
        source, mesh = sample_mesh_surface(obj_path, num_points)

    # 执行 GMM 配准（仅考虑刚性和缩放）
    aligned_points, R, t, s = run_cpd_rigid_scaling(source, target)

    # 可视化对齐结果
    if visualize:
        source_pc = o3d.geometry.PointCloud()
        source_pc.points = o3d.utility.Vector3dVector(aligned_points)
        source_pc.paint_uniform_color([1.0, 0.3, 0.3])  # 红色

        target_pc = o3d.geometry.PointCloud()
        target_pc.points = o3d.utility.Vector3dVector(target)
        target_pc.paint_uniform_color([0.2, 0.6, 1.0])  # 蓝色

        o3d.visualization.draw_geometries([target_pc, source_pc])

    # 保存对齐后的网格
    if save_aligned:
        transform = np.eye(4)
        transform[:3, :3] = R * s
        transform[:3, 3] = t
        aligned_mesh = copy.deepcopy(mesh)
        aligned_mesh.apply_transform(transform)
        aligned_mesh.export(output_path)
        print(f"[✓] 对齐后的网格已保存至: {output_path}")

    return R, t, s



# def align_with_cpd_simple(
#     target_points,
#     source_points,
#     visualize=False,
# ):
#     # 加载目标点云
#     target_pts = target_points
#     target_pts = remove_outliers(target_pts, nb_neighbors=50, std_ratio=2.0)


#     # 采样源点云
#     source_pts = source_points

#     source = o3d.geometry.PointCloud()
#     source.points = o3d.utility.Vector3dVector(source_pts)

#     target = o3d.geometry.PointCloud()
#     target.points = o3d.utility.Vector3dVector(target_pts)

#     # 配准
#     tf_param, _, _ = cpd.registration_cpd(source, target, w=0.1, tf_type_name='rigid', update_scale=True)

#     # 应用变换
#     transform = np.eye(4)
#     transform[:3, :3] = tf_param.rot * tf_param.scale
#     transform[:3, 3] = tf_param.t
#     print("Rotation:\n", tf_param.rot)
#     print("Translation:\n", tf_param.t)
#     print("Scale:\n", tf_param.scale)

#     return transform

def align_with_cpd_simple(
    target_points,
    source_points,
    visualize=False,
):
    # 预处理
    # target_pts = remove_outliers(target_points, nb_neighbors=50, std_ratio=2.0)
    target_pts = target_points
    source_pts = source_points

    def normalize(points):
        center = np.mean(points, axis=0)
        scale = np.max(np.linalg.norm(points - center, axis=1))
        normalized = (points - center) / scale
        return normalized, center, scale

    source_norm, source_center, source_scale = normalize(source_pts)
    target_norm, target_center, target_scale = normalize(target_pts)

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_norm)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_norm)

    tf_param, _, _ = cpd.registration_cpd(
        source,
        target,
        w=0.1,
        tf_type_name='rigid',
        update_scale=True,
    )

    # 正确恢复变换
    R = tf_param.rot
    s = tf_param.scale
    t = tf_param.t

    M = np.eye(4)
    M[:3, :3] = (target_scale / source_scale) * s * R
    M[:3, 3] = target_scale * (t - (R @ (source_center / source_scale))) + target_center

    print("Rotation:\n", R)
    print("Translation:\n", t)
    print("Scale (from CPD):\n", s)
    print("Recovered transform:\n", M)

    return M

if __name__ == "__main__":
    obj_path = "meshes/antenna.obj"
    ply_path = "sensor1_full.ply"
    # align_mesh_to_gaussian(obj_path, ply_path, output_path="meshes/aligned_sensor2.obj")
    # R, t, s = align_with_gmm(obj_path, ply_path, use_volume=False, visualize=True, save_aligned=True)
    # test_alignment_on_mesh(obj_path)
    R, t, s = align_with_cpd_simple(obj_path, ply_path, use_volume=True, visualize=True)