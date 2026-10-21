"""光滑微分求导核心算法（脚本3重构）。

KDE-adaptive Gaussian Derivative Estimation
计算因变量 Y 对各自变量 X 的偏导数。

方法:
1. 使用高斯核密度估计(KDE)计算每个数据点的局部密度
2. 自适应权重与密度成反比（稀疏区域每个点代表更大空间，权重更高）
3. 对每个数据点，使用高斯核加权其邻域点
4. 组合权重 = KDE自适应权重 x 高斯核距离权重
5. 对每个点拟合局部线性模型，提取偏导数
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.linalg import lstsq

from config import get_config
from utils.data_io import (
    resolve_columns,
    read_data,
    validate_data,
    default_output_path,
    materialize_inline_data,
)


def gaussian_kde(points, bandwidth):
    """高斯核密度估计。

    对每个点计算局部密度: f(x_i) = (1/N) * sum_j K_h(x_i - x_j)
    其中 K_h(u) = (2*pi*h^2)^(-d/2) * exp(-||u||^2 / (2h^2))

    使用 KDTree 加速，只考虑距离 < 3*h 的邻域点（超出范围贡献极小）。
    """
    n = points.shape[0]
    dim = points.shape[1]
    tree = KDTree(points)
    density = np.zeros(n)

    # 截断半径：3倍带宽外的高斯核贡献可忽略
    cutoff = 3.0 * bandwidth
    norm_const = (2 * np.pi * bandwidth ** 2) ** (-dim / 2.0)

    # 批量查询邻域
    indices_list = tree.query_ball_point(points, cutoff)

    for i in range(n):
        neighbors = indices_list[i]
        if len(neighbors) == 0:
            density[i] = norm_const  # 孤立点的最小密度估计
            continue
        diffs = points[neighbors] - points[i]
        sq_dists = np.sum(diffs ** 2, axis=1)
        kernel_vals = norm_const * np.exp(-sq_dists / (2 * bandwidth ** 2))
        density[i] = np.sum(kernel_vals) / n

    return density


def compute_derivatives(
    input_file=None,
    output_file=None,
    data=None,
    n_indep=None,
    x_cols=None,
    y_col=None,
    k_neighbors=None,
    k_bandwidth=None,
    space="log10",
):
    """对输入数据计算因变量对各自变量的偏导数。

    Args:
        input_file: 数据文件路径（.xlsx 或 .csv），支持服务器本地路径
            数据集；与 data 参数二选一
        data: 行内数据 {"columns": [...], "rows": [[...], ...]}，
            适用于无法向服务器传文件的调用方（如平台智能体）
        space: 输入数据的空间，"log10"（默认，已是log10对数空间）或 "physical"
            （物理空间原始值，工具内部自动取log10——LIGGGHTS 仿真结果等物理
            空间数据直传本参数即可，无需预处理）
        output_file: 偏导数结果保存路径（默认自动生成）
        n_indep: 自变量列数（默认 总列数-1）
        x_cols: 自变量列名列表（覆盖 n_indep）
        y_col: 因变量列名（默认末列）
        k_neighbors: 局部拟合邻居数（默认随维度缩放）
        k_bandwidth: 带宽估计近邻数（默认 30）

    Returns:
        dict: 结果摘要，包含 output_file / n_indep / x_cols / y_col /
              derivative_cols / stats / data_points
    """
    cfg = get_config()

    # ========== 0. 行内数据落盘 / 输入校验 ==========
    if input_file is None and data is not None:
        input_file = materialize_inline_data(data)
    if input_file is None:
        raise ValueError("input_file 与 data 参数必须提供其一")

    # ========== 1. 读取数据 ==========
    df = read_data(input_file)
    x_cols, y_col = resolve_columns(df, n_indep=n_indep, x_cols=x_cols, y_col=y_col)
    data, _ = validate_data(df, x_cols, y_col)

    # 物理空间输入自动转 log10（输出文件统一为 log10 空间，供下游工具使用）
    if space == "physical":
        for c in x_cols + [y_col]:
            vals = data[c].astype(float)
            if (vals <= 0).any():
                is_y = (c == y_col)
                hint = ("响应量含负值（如有符号的偏析量）时幂律拟合不适用。"
                        "可改用分步分析绕开 y 取对数：自行把自变量列变换为 log10 后"
                        "以 space='log10' 调 compute_derivatives（半对数灵敏度："
                        "∂y/∂log10·x），再 cluster_analysis 做梯度空间聚类。" if is_y else
                        "自变量须恒为正才能取对数，请检查数据或剔除该列。")
                raise ValueError(
                    f"space='physical' 但列 '{c}' 含非正数（值域 "
                    f"[{vals.min():g}, {vals.max():g}]），无法取对数。{hint}")
            data[c] = np.log10(vals)
    elif space != "log10":
        raise ValueError("space 只支持 'log10' 或 'physical'")

    X = data[x_cols].values  # (N, d)
    Y = data[y_col].values   # (N,)
    N = X.shape[0]
    d = X.shape[1]

    # ========== 2. 数据标准化 ==========
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_scaled = (X - X_mean) / X_std

    # ========== 3. 高斯核密度估计(KDE) ==========
    # 使用 Silverman 法则确定 KDE 带宽
    silverman_factor = (4.0 / (d + 2)) ** (1.0 / (d + 4))
    h_kde = silverman_factor * N ** (-1.0 / (d + 4)) * np.mean(X_std)

    density = gaussian_kde(X_scaled, h_kde)

    # ========== 4. 计算KDE自适应权重 ==========
    # 稀疏区域（低密度）的点代表更大空间，应赋予更高权重
    kde_weights = 1.0 / density
    kde_weights /= kde_weights.sum()

    # ========== 5. 确定高斯核距离权重带宽 ==========
    tree = KDTree(X_scaled)
    if k_bandwidth is None:
        k_bandwidth = cfg.k_bandwidth_k
    k_bandwidth = min(k_bandwidth, N - 1)
    dists, _ = tree.query(X_scaled, k=k_bandwidth + 1)
    knn_dists = dists[:, -1]
    h = np.median(knn_dists)

    # ========== 6. KDE自适应加权高斯求导 ==========
    if k_neighbors is None:
        k_neighbors = min(max(cfg.k_neighbors_base, d * cfg.k_neighbors_per_dim), N - 1)
    else:
        k_neighbors = min(k_neighbors, N - 1)

    partial_derivatives = np.zeros((N, d))

    for i in range(N):
        # 找到邻居（包含自身，之后排除）
        dists_i, indices_i = tree.query(X_scaled[i], k=k_neighbors + 1)
        mask = indices_i != i
        neighbor_indices = indices_i[mask]
        neighbor_dists = dists_i[mask]

        if len(neighbor_indices) < d + 1:
            dists_i, indices_i = tree.query(X_scaled[i], k=min(d + 10, N))
            mask = indices_i != i
            neighbor_indices = indices_i[mask]
            neighbor_dists = dists_i[mask]

        # 高斯核距离权重
        gauss_weights = np.exp(-0.5 * (neighbor_dists / h) ** 2)

        # KDE自适应权重
        kde_w = kde_weights[neighbor_indices]

        # 组合权重 = KDE自适应权重 x 高斯核距离权重
        combined_weights = kde_w * gauss_weights
        combined_weights /= combined_weights.sum()

        # 局部线性拟合
        dX = X_scaled[neighbor_indices] - X_scaled[i]
        dY = Y[neighbor_indices] - Y[i]

        # 加权最小二乘
        W_sqrt = np.sqrt(combined_weights)
        A = dX * W_sqrt[:, np.newaxis]
        b_vec = dY * W_sqrt

        try:
            result, _, _, _ = lstsq(A, b_vec)
            partial_derivatives[i] = result
        except Exception:
            partial_derivatives[i] = np.nan

    # ========== 7. 将标准化空间导数转换回原始空间 ==========
    for j in range(d):
        partial_derivatives[:, j] /= X_std[j]

    # ========== 8. 保存结果 ==========
    if output_file is None:
        output_file = default_output_path(input_file, "_偏导数")
    result_df = data.copy()
    derivative_cols = [f"dY_d{col}" for col in x_cols]
    for j, col in enumerate(derivative_cols):
        result_df[col] = partial_derivatives[:, j]

    result_df.to_excel(output_file, index=False)

    # ========== 9. 统计信息 ==========
    stats = {}
    for j, col in enumerate(derivative_cols):
        vals = partial_derivatives[:, j]
        stats[col] = {
            "mean": float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else None,
            "std": float(np.nanstd(vals)) if np.any(~np.isnan(vals)) else None,
            "min": float(np.nanmin(vals)) if np.any(~np.isnan(vals)) else None,
            "max": float(np.nanmax(vals)) if np.any(~np.isnan(vals)) else None,
        }

    nan_count = int(np.isnan(partial_derivatives).any(axis=1).sum())

    return {
        "output_file": output_file,
        "n_indep": d,
        "x_cols": x_cols,
        "y_col": y_col,
        "derivative_cols": derivative_cols,
        "stats": stats,
        "nan_count": nan_count,
        "data_points": N,
    }
