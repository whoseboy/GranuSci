"""
KDE-adaptive Gaussian Derivative Estimation
计算 Y 相对于 X1~X6 的偏导数

方法:
1. 使用高斯核密度估计(KDE)计算每个数据点的局部密度
2. 自适应权重与密度成反比（稀疏区域每个点代表更大空间，权重更高）
3. 对每个数据点，使用高斯核加权其邻域点
4. 组合权重 = KDE自适应权重 × 高斯核距离权重
5. 对每个点拟合局部线性模型，提取偏导数
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.linalg import lstsq
import os

# 当前脚本所在文件夹
base_dir = os.path.dirname(os.path.abspath(__file__))

# ========== 1. 读取数据 ==========
print("正在读取数据...")
df = pd.read_excel(os.path.join(base_dir, "对数空间数据-720h.xlsx"))
X_cols = ["X1", "X2", "X3", "X4", "X5", "X6"]
X = df[X_cols].values  # (N, 6)
Y = df["Y"].values      # (N,)
N = X.shape[0]
d = X.shape[1]  # 维度
print(f"数据规模: {N} 个点, {d} 个自变量")

# ========== 2. 数据标准化 ==========
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1.0
X_scaled = (X - X_mean) / X_std

# ========== 3. 高斯核密度估计(KDE) ==========
print("正在计算高斯核密度估计...")

# 使用Silverman法则确定KDE带宽
# h_silverman = (4 / (d+2))^(1/(d+4)) * N^(-1/(d+4)) * std
silverman_factor = (4.0 / (d + 2)) ** (1.0 / (d + 4))
h_kde = silverman_factor * N ** (-1.0 / (d + 4)) * np.mean(X_std)
print(f"KDE带宽(Silverman法则) h_kde = {h_kde:.6f}")


def gaussian_kde(points, bandwidth):
    """
    高斯核密度估计
    对每个点计算局部密度: f(x_i) = (1/N) * Σ_j K_h(x_i - x_j)
    其中 K_h(u) = (2πh^2)^(-d/2) * exp(-||u||^2 / (2h^2))

    使用KDTree加速，只考虑距离 < 3*h 的邻域点（超出范围贡献极小）
    """
    N = points.shape[0]
    dim = points.shape[1]
    tree = KDTree(points)
    density = np.zeros(N)

    # 截断半径：3倍带宽外的高斯核贡献可忽略
    cutoff = 3.0 * bandwidth
    norm_const = (2 * np.pi * bandwidth ** 2) ** (-dim / 2.0)

    # 批量查询邻域
    indices_list = tree.query_ball_point(points, cutoff)

    for i in range(N):
        neighbors = indices_list[i]
        if len(neighbors) == 0:
            density[i] = norm_const  # 孤立点的最小密度估计
            continue
        diffs = points[neighbors] - points[i]
        sq_dists = np.sum(diffs ** 2, axis=1)
        kernel_vals = norm_const * np.exp(-sq_dists / (2 * bandwidth ** 2))
        density[i] = np.sum(kernel_vals) / N

    return density


density = gaussian_kde(X_scaled, h_kde)
print(f"KDE密度统计: min={density.min():.6e}, max={density.max():.6e}, "
      f"mean={density.mean():.6e}")

# ========== 4. 计算KDE自适应权重 ==========
# 稀疏区域（低密度）的点代表更大空间，应赋予更高权重
# 自适应权重 ∝ 1 / f(x_i)，然后归一化
print("正在计算KDE自适应权重...")
kde_weights = 1.0 / density
kde_weights /= kde_weights.sum()
print(f"KDE自适应权重统计: min={kde_weights.min():.6e}, max={kde_weights.max():.6e}, "
      f"mean={kde_weights.mean():.6e}")

# ========== 5. 确定高斯核距离权重带宽 ==========
print("正在确定高斯核距离权重带宽...")
tree = KDTree(X_scaled)
k_bandwidth = min(30, N - 1)
dists, _ = tree.query(X_scaled, k=k_bandwidth + 1)
knn_dists = dists[:, -1]
h = np.median(knn_dists)
print(f"高斯核距离权重带宽 h = {h:.6f}")

# ========== 6. KDE自适应加权高斯求导 ==========
print("正在计算偏导数...")
k_neighbors = min(80, N - 1)

partial_derivatives = np.zeros((N, d))

for i in range(N):
    if (i + 1) % 100 == 0:
        print(f"  处理进度: {i + 1}/{N}")

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

    # 组合权重 = KDE自适应权重 × 高斯核距离权重
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

print(f"  处理进度: {N}/{N}")

# ========== 7. 将标准化空间导数转换回原始空间 ==========
# dY/dX_j = (dY/dX_scaled_j) / std_j
for j in range(d):
    partial_derivatives[:, j] /= X_std[j]

# ========== 8. 保存结果 ==========
print("正在保存结果...")
result_df = df.copy()
for j, col in enumerate(X_cols):
    result_df[f"dY_d{col}"] = partial_derivatives[:, j]

output_path = os.path.join(base_dir, "对数空间数据-720h_偏导数.xlsx")
result_df.to_excel(output_path, index=False)
print(f"\n结果已保存至: {output_path}")

# 打印偏导数统计信息
print("\n===== 偏导数统计 =====")
for j, col in enumerate(X_cols):
    vals = partial_derivatives[:, j]
    print(f"dY/d{col}: mean={np.nanmean(vals):.6f}, std={np.nanstd(vals):.6f}, "
          f"min={np.nanmin(vals):.6f}, max={np.nanmax(vals):.6f}")

nan_count = np.isnan(partial_derivatives).any(axis=1).sum()
if nan_count > 0:
    print(f"\n警告: {nan_count} 个点的偏导数计算失败(NaN)")
