import numpy as np

def estimate_similarity_transform(source_points: np.ndarray,
                                  target_points: np.ndarray):
    """
    用 Umeyama 方法估计源点到目标点的相似变换（旋转+均匀缩放+平移）。

    Args:
        source_points: (N,3) numpy 数组，源点坐标
        target_points: (N,3) numpy 数组，目标点坐标

    Returns:
        4×4 齐次变换矩阵，或在失败时返回 None。
    """
    if source_points.shape != target_points.shape or source_points.shape[0] < 3:
        print("Error: 至少需要 3 对同形状点。")
        return None

    # 1) 坐标去中心
    mu_src = source_points.mean(axis=0)
    mu_tgt = target_points.mean(axis=0)
    X = source_points - mu_src  # (N,3)
    Y = target_points - mu_tgt  # (N,3)

    # 2) 计算协方差矩阵
    S = (Y.T @ X) / source_points.shape[0]  # (3,3)

    # 3) SVD 分解
    U, Sigma, Vt = np.linalg.svd(S)

    # 4) 构造修正矩阵 D 以保证无反射
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1

    # 5) 计算纯旋转矩阵
    R = U @ D @ Vt

    # 6) 计算源点集方差 σ²
    var_src = np.sum((X ** 2)) / source_points.shape[0]

    # 7) 计算均匀缩放因子
    scale = np.sum(Sigma * np.diag(D)) / var_src

    # 8) 构造 4×4 变换
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = mu_tgt - scale * (R @ mu_src)

    return T
