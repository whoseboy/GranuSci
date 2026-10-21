"""FCM 聚类核心算法（脚本4重构）。

使用 subclust+FCM 复合聚类算法:
1. subclust（减法聚类）确定初始聚类中心和聚类数量
2. FCM 进行精细聚类
3. 特征分解，输出聚类标签、中心、特征值/特征向量
"""

import gc
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import eig
from scipy.spatial.distance import cdist
from tqdm import tqdm

warnings.filterwarnings("ignore")

from config import get_config  # noqa: E402
from utils.data_io import (  # noqa: E402
    infer_columns_from_derivative_file,
    read_excel,
    ensure_dir,
)


class AdvancedClusteringAnalysis:
    """高级聚类分析类（维度无关）。"""

    def __init__(
        self,
        data_file,
        n_clusters=None,
        batch_size=None,
        n_indep=None,
        x_cols=None,
        y_col=None,
        fcm_m=None,
        fcm_error=None,
        fcm_maxiter=None,
        max_samples=None,
        subclust_radius_factor=None,
        subclust_suppression_ratio=None,
        membership_threshold=None,
        subclust_accept_ratio=None,
        subclust_reject_ratio=None,
    ):
        """初始化聚类分析。

        Args:
            data_file: 偏导数数据文件路径（Excel，含 X 列和 dY_dX 偏导数列）
            n_clusters: 聚类数；0 或 "auto" 表示由 subclust 自动确定（论文核心特性）
            batch_size: 批次大小
            n_indep: 自变量列数（用于自动识别列）
            x_cols: 自变量列名列表
            y_col: 因变量列名
            fcm_m / fcm_error / fcm_maxiter: FCM 参数
            max_samples: 抽样上限
            subclust_radius_factor / subclust_suppression_ratio: subclust 参数
            membership_threshold: 最大隶属度阈值（默认0.6），低于则标记为模糊点(-1)；
                0 关闭模糊点剔除
        """
        cfg = get_config()
        self.data_file = data_file
        if n_clusters is None:
            n_clusters = cfg.n_clusters
        self.auto_clusters = n_clusters in (0, "auto")
        self.n_clusters = None if self.auto_clusters else int(n_clusters)
        self.membership_threshold = (
            cfg.membership_threshold if membership_threshold is None else membership_threshold
        )
        self.subclust_accept_ratio = (
            cfg.subclust_accept_ratio if subclust_accept_ratio is None else subclust_accept_ratio
        )
        self.subclust_reject_ratio = (
            cfg.subclust_reject_ratio if subclust_reject_ratio is None else subclust_reject_ratio
        )
        self.batch_size = cfg.batch_size if batch_size is None else batch_size
        self.fcm_m = cfg.fcm_m if fcm_m is None else fcm_m
        self.fcm_error = cfg.fcm_error if fcm_error is None else fcm_error
        self.fcm_maxiter = cfg.fcm_maxiter if fcm_maxiter is None else fcm_maxiter
        self.max_samples = cfg.max_samples if max_samples is None else max_samples
        self.subclust_radius_factor = (
            cfg.subclust_radius_factor if subclust_radius_factor is None else subclust_radius_factor
        )
        self.subclust_suppression_ratio = (
            cfg.subclust_suppression_ratio
            if subclust_suppression_ratio is None
            else subclust_suppression_ratio
        )
        self.n_indep = n_indep
        self.x_cols_override = x_cols
        self.y_col_override = y_col

        # 由数据推断的列
        self.X_cols = None
        self.G_cols = None
        self.Y_col = None
        self.n_dims = None

        # 存储结果
        self.x = None
        self.pFpX = None
        self.D = None
        self.F = None

        # 保存聚类组件（用于分类）
        self.clustering_scaler = None
        self.features = None
        self.fcm_centers = None
        self.fcm_parameters = None
        self.labels = None
        # 设计仪器 schema 暴露（v1.3）：FCM 隶属度矩阵与局部特征值持久化
        self.u_matrix = None
        self.local_eigenvalues = None

    def load_data_chunked(self):
        """从 Excel 文件加载数据（包含坐标和偏导数）。

        通过 "dY_d" 前缀自动识别偏导数列，自变量/因变量列自动推断。
        """
        df = read_excel(self.data_file)
        x_cols, y_col, g_cols = infer_columns_from_derivative_file(
            df, n_indep=self.n_indep
        )
        if self.x_cols_override:
            x_cols = [c for c in self.x_cols_override if c in df.columns]
        if self.y_col_override:
            y_col = self.y_col_override
        if not g_cols:
            raise ValueError(
                f"数据文件未找到偏导数列（列名以 'dY_d' 开头）: {list(df.columns)}"
            )

        self.X_cols = x_cols
        self.G_cols = g_cols
        self.Y_col = y_col
        self.n_dims = len(g_cols)

        self.x = df[x_cols].values.astype(np.float32)
        self.pFpX = df[g_cols].values.astype(np.float32)

        if len(self.x) != len(self.pFpX):
            raise ValueError(
                f"数据点数不匹配: 坐标 {len(self.x)}, 偏微分 {len(self.pFpX)}"
            )

        return self.x, self.pFpX

    def compute_local_features_batch(self):
        """分块计算局部特征矩阵和主导特征向量（使用偏微分信息）。

        返回原始（未缩放）的符号对齐主导特征向量：
        W = pFp·pFᵀ 对梯度符号不敏感（±g 给出同一 W），特征向量符号由数值
        过程任意决定，须按首分量符号对齐（对应 MATLAB cluster_myfcm_o.m 中
        `if s(i,1)<0: s(i,:)=-s(i,:)`），否则同一物理方向的点会落到特征
        空间两侧。缩放交给下游 subclust_fcm_clustering 统一处理。

        Returns:
            tuple: (local_features, eigenvalues)
        """
        N = len(self.pFpX)
        local_features = np.zeros((N, self.n_dims), dtype=np.float32)
        eigenvalues = np.zeros(N, dtype=np.float32)

        n_batches = (N + self.batch_size - 1) // self.batch_size

        for batch_idx in tqdm(range(n_batches), desc="计算局部特征"):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, N)
            batch_size = end_idx - start_idx

            batch_pFpX = self.pFpX[start_idx:end_idx].astype(np.float32)

            for i in range(batch_size):
                pFp_i = batch_pFpX[i].reshape(self.n_dims, 1)
                W = pFp_i @ pFp_i.T

                eig_vals, eig_vecs = np.linalg.eigh(W)

                sorted_indices = np.argsort(eig_vals)[::-1]

                dominant_eigenvector = eig_vecs[:, sorted_indices[0]].copy()
                dominant_eigenvalue = eig_vals[sorted_indices[0]]

                # 符号对齐：首分量为正（W 不区分 ±g，符号须归一）
                if dominant_eigenvector[0] < 0:
                    dominant_eigenvector = -dominant_eigenvector

                local_features[start_idx + i] = dominant_eigenvector
                eigenvalues[start_idx + i] = dominant_eigenvalue

            del batch_pFpX

            if batch_idx % 5 == 0:
                gc.collect()

        return local_features, eigenvalues

    def sample_features_for_clustering(self, features, max_samples=None):
        """对特征进行抽样以减少内存占用和计算时间。"""
        if max_samples is None:
            max_samples = self.max_samples
        n_samples = features.shape[0]
        if n_samples <= max_samples:
            return features, np.arange(n_samples), features.shape

        sample_indices = np.random.choice(n_samples, size=max_samples, replace=False)
        sampled_features = features[sample_indices]

        return sampled_features, sample_indices, features.shape

    def subclust_fcm_clustering(self, features, eigenvalues):
        """使用 subclust+FCM 复合聚类算法进行聚类。

        Returns:
            labels: 聚类标签
            centers: 聚类中心
            u: 模糊隶属度矩阵
        """
        from skfuzzy.cluster import cmeans
        from sklearn.preprocessing import RobustScaler

        features = features.astype(np.float64)
        eigenvalues = eigenvalues.astype(np.float64)

        n_samples = features.shape[0]

        # 大数据集抽样
        max_samples_for_clustering = self.max_samples
        if n_samples > max_samples_for_clustering:
            sampled_features, sample_indices, original_shape = (
                self.sample_features_for_clustering(
                    features, max_samples=max_samples_for_clustering
                )
            )
            is_sampled = True
            features_to_use = sampled_features
        else:
            features_to_use = features
            is_sampled = False
            sample_indices = np.arange(n_samples)

        # 特征标准化
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_to_use)
        features_reduced = features_scaled

        # 第一步：subclust 确定初始中心
        n_samples_reduced = features_reduced.shape[0]
        if n_samples_reduced > 5000:
            from sklearn.cluster import KMeans

            n_sample_clusters = min(50000, max(100, n_samples_reduced // 30))
            kmeans_sampler = KMeans(n_clusters=n_sample_clusters, random_state=42, n_init=3)
            cluster_labels = kmeans_sampler.fit_predict(features_reduced)
            unique_labels = np.unique(cluster_labels)
            selected_indices = []
            samples_per_cluster = max(1, 50000 // len(unique_labels))

            for label in unique_labels:
                cluster_mask = cluster_labels == label
                cluster_indices = np.where(cluster_mask)[0]
                if len(cluster_indices) <= samples_per_cluster:
                    selected_indices.extend(cluster_indices)
                else:
                    randomly_selected = np.random.choice(
                        cluster_indices, size=samples_per_cluster, replace=False
                    )
                    selected_indices.extend(randomly_selected)

            subclust_sample_indices = np.array(selected_indices)
            sample_data = features_reduced[subclust_sample_indices]
        else:
            sample_data = features_reduced
            subclust_sample_indices = np.arange(n_samples_reduced)

        # Chiu 减法聚类（MATLAB subclust 语义）：数据逐维归一化到 [0,1]，
        # 潜力 D_i = Σ_j exp(-4/ra² · ||x_i-x_j||²)，压制 exp(-4/rb² · d²)，rb=sq·ra。
        # subclust 在原始（符号对齐）特征上进行，中心映射回缩放空间用于 FCM 初始化。
        raw_for_subclust = features_to_use[subclust_sample_indices]
        d_min = np.min(raw_for_subclust, axis=0)
        d_max = np.max(raw_for_subclust, axis=0)
        d_range = np.maximum(d_max - d_min, 1e-12)
        norm_data = (raw_for_subclust - d_min) / d_range
        radius = float(np.clip(self.subclust_radius_factor, 1e-3, 1.0))

        # 减法聚类
        def subtractive_clustering_optimized(data, radius, target_clusters=None):
            """Chiu 减法聚类。target_clusters=None 时按接受/拒绝准则自动确定聚类数
            （论文附录A.1.4：p/p1* >= ζ̄ 接受；<= ς̄ 拒绝停止；否则 d_min/ra + p/p1* >= 1 接受）。"""
            ra_sq = (radius / 2.0) ** 2
            rb_sq = ((radius * self.subclust_suppression_ratio) / 2.0) ** 2

            n_points = data.shape[0]
            potentials = np.zeros(n_points)

            for i in range(n_points):
                distances_sq = np.sum((data - data[i]) ** 2, axis=1)
                potentials[i] = np.sum(np.exp(-distances_sq / ra_sq))

            centers = []
            selected_indices = []
            first_potential = None
            while True:
                max_idx = np.argmax(potentials)
                max_potential = potentials[max_idx]

                if target_clusters is not None and len(centers) >= target_clusters:
                    break

                if target_clusters is None and len(centers) > 0:
                    # Chiu 接受/拒绝准则（自动确定聚类数）
                    ratio = max_potential / first_potential
                    if ratio <= self.subclust_reject_ratio:
                        break
                    if ratio < self.subclust_accept_ratio:
                        # 附加判断：与已有中心的最小距离
                        d_min = np.min(
                            np.linalg.norm(np.array(centers) - data[max_idx], axis=1)
                        )
                        if d_min / radius + ratio < 1.0:
                            # 拒绝该点，置零其潜力继续找
                            potentials[max_idx] = -np.inf
                            continue
                    if max_potential < 0.005 * first_potential:
                        break

                centers.append(data[max_idx].copy())
                selected_indices.append(max_idx)
                if first_potential is None:
                    first_potential = max_potential

                center_point = data[max_idx]
                distances_sq = np.sum((data - center_point) ** 2, axis=1)
                reduction_factor = np.exp(-distances_sq / rb_sq)
                potentials -= max_potential * reduction_factor

                potentials[max_idx] = -np.inf

            return np.array(centers), selected_indices

        # 执行减法聚类获得初始中心（归一化空间），映射回原始特征空间，
        # 再经 RobustScaler 变换到 FCM 使用的缩放空间
        centers_norm, subclust_indices = subtractive_clustering_optimized(
            norm_data, radius, target_clusters=self.n_clusters
        )
        centers_raw = d_min + centers_norm * d_range
        n_clusters_subclust = len(centers_raw)
        subclust_centers = scaler.transform(centers_raw)

        X = features_reduced

        if subclust_centers.shape[0] >= n_clusters_subclust:
            cluster_centers = subclust_centers[:n_clusters_subclust, :]
        else:
            cluster_centers = (
                np.random.rand(n_clusters_subclust, X.shape[1])
                * (
                    np.max(X, axis=0, keepdims=True)
                    - np.min(X, axis=0, keepdims=True)
                )
                + np.min(X, axis=0, keepdims=True)
            )

        # 执行 FCM 聚类（用 subclust 中心构造初始隶属度，对应论文附录A.1.4
        # "将中心点矩阵和中心点个数导入FCM聚类算法中"）
        X_transposed = X.T
        u_init = None
        if cluster_centers is not None and len(cluster_centers) == n_clusters_subclust:
            dist_init = cdist(X, cluster_centers)  # (n, c)
            dist_init = np.maximum(dist_init, 1e-12)
            expo = 2.0 / (self.fcm_m - 1.0)
            u0_raw = dist_init ** (-expo)           # (n, c)
            u_init = (u0_raw / u0_raw.sum(axis=1, keepdims=True)).T  # (c, n)
        cntr, u, u0, d, jm, p, fpc = cmeans(
            X_transposed,
            c=n_clusters_subclust,
            m=self.fcm_m,
            error=self.fcm_error,
            maxiter=self.fcm_maxiter,
            init=u_init,
        )

        coarse_labels = np.argmax(u, axis=0)

        # 模糊点剔除（论文附录A.1.4：最大隶属度低于阈值的点不归入任何子域）
        if self.membership_threshold and self.membership_threshold > 0:
            max_u = np.max(u, axis=0)
            coarse_labels = np.where(max_u >= self.membership_threshold, coarse_labels, -1)

        final_labels = coarse_labels
        fine_centers = cntr

        # 抽样标签映射回原始数据
        if is_sampled:
            full_labels = np.full(original_shape[0], -1, dtype=int)
            full_labels[sample_indices] = final_labels

            unsampled_indices = np.setdiff1d(
                np.arange(original_shape[0]), sample_indices
            )
            if len(unsampled_indices) > 0:
                unsampled_features = features[unsampled_indices]
                unsampled_features_scaled = scaler.transform(unsampled_features)
                unsampled_features_reduced = unsampled_features_scaled

                if len(fine_centers) > 0:
                    centers_scaled_back = fine_centers

                    if unsampled_features_reduced.shape[1] != centers_scaled_back.shape[1]:
                        centers_scaled_back = centers_scaled_back.T

                    distances_to_centers = cdist(
                        unsampled_features_reduced, centers_scaled_back
                    )

                    m = 2.0
                    distances_safe = distances_to_centers.copy()
                    zero_mask = distances_safe == 0
                    if np.any(zero_mask):
                        distances_safe[zero_mask] = 1e-10

                    n_unsampled = distances_safe.shape[0]
                    n_clusters = distances_safe.shape[1]

                    distances_safe = distances_safe.astype(np.float64)

                    exponent = 2.0 / (m - 1.0)
                    d_pow = distances_safe ** exponent
                    d_inv_pow = distances_safe ** (-exponent)

                    d_inv_pow[np.isinf(d_inv_pow)] = 0.0

                    sum_d_inv_pow = np.sum(d_inv_pow, axis=1, keepdims=True)
                    denominator = d_pow * sum_d_inv_pow

                    u_unsampled = 1.0 / denominator

                    if np.any(zero_mask):
                        for j in range(n_unsampled):
                            for i in range(n_clusters):
                                if zero_mask[j, i]:
                                    u_unsampled[j, :] = 0.0
                                    u_unsampled[j, i] = 1.0

                    row_sums = u_unsampled.sum(axis=1, keepdims=True)
                    u_unsampled = u_unsampled / row_sums

                    closest_clusters = np.argmax(u_unsampled, axis=1)
                    if self.membership_threshold and self.membership_threshold > 0:
                        max_u_uns = np.max(u_unsampled, axis=1)
                        closest_clusters = np.where(
                            max_u_uns >= self.membership_threshold, closest_clusters, -1
                        )
                    full_labels[unsampled_indices] = closest_clusters
                else:
                    full_labels[unsampled_indices] = 0

            final_labels = full_labels

        # 构建完整的隶属度矩阵
        if is_sampled:
            n_clusters_total = fine_centers.shape[0]
            n_total = original_shape[0]

            u_all = np.zeros((n_clusters_total, n_total))

            u_all[:, sample_indices] = u

            if len(unsampled_indices) > 0 and "u_unsampled" in locals():
                u_unsampled_T = u_unsampled.T
                u_all[:, unsampled_indices] = u_unsampled_T
        else:
            u_all = u

        # 保存聚类组件用于后续分类
        self.clustering_scaler = scaler
        self.features = features
        self.fcm_centers = fine_centers
        self.fcm_parameters = {
            "m": self.fcm_m,
            "error": self.fcm_error,
            "maxiter": self.fcm_maxiter,
            "n_clusters": fine_centers.shape[0],
            "feature_dimension": features.shape[1],
        }
        self.labels = final_labels

        return final_labels, fine_centers, u_all

    def organize_clustering_results(self, labels, centers, u, features=None):
        """组织聚类结果。"""
        self.D = {"sub": []}
        self.F = {"sub": []}

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]

        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            n_points = len(cluster_indices)

            if features is not None and len(cluster_indices) > 0:
                cluster_data = features[cluster_indices]
                if len(cluster_data) > 0:
                    actual_center = np.mean(cluster_data, axis=0)
                else:
                    actual_center = np.zeros(self.n_dims)
            elif len(centers) > 0:
                label_pos = np.where(unique_labels == label)[0]
                if len(label_pos) > 0 and label_pos[0] < len(centers):
                    actual_center = centers[label_pos[0]]
                else:
                    actual_center = np.zeros(self.n_dims)
            else:
                actual_center = np.zeros(self.n_dims)

            membership_values = None
            if u is not None and u.shape[0] > label and u.shape[1] > max(cluster_indices):
                try:
                    membership_values = u[label, cluster_indices]
                    avg_membership = np.mean(membership_values) if len(membership_values) else 0.0
                    max_membership = np.max(membership_values) if len(membership_values) else 0.0
                    min_membership = np.min(membership_values) if len(membership_values) else 0.0
                except Exception:
                    membership_values = None

            sub_info = {
                "index": cluster_indices,
                "n_points": n_points,
                "percentage": n_points / len(labels) * 100,
                "center": actual_center,
            }

            if membership_values is not None:
                sub_info["membership"] = membership_values
                sub_info["avg_membership"] = avg_membership
                sub_info["max_membership"] = max_membership
                sub_info["min_membership"] = min_membership

            self.D["sub"].append(sub_info)

            self.F["sub"].append({"center": sub_info["center"], "n_points": n_points})

    def compute_global_features(self):
        """计算每个子区域的全局特征矩阵。"""
        for i in range(len(self.D["sub"])):
            cluster_indices = self.D["sub"][i]["index"]
            n_points = len(cluster_indices)

            if n_points == 0:
                continue

            C_total = np.zeros((self.n_dims, self.n_dims))

            n_batches = (n_points + self.batch_size - 1) // self.batch_size

            for batch_idx in tqdm(range(n_batches), desc=f"子区域 {i+1} 特征累加"):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, n_points)

                batch_indices = cluster_indices[start_idx:end_idx]
                batch_pFpX = self.pFpX[batch_indices]

                batch_vectors = batch_pFpX.reshape(-1, self.n_dims, 1)
                batch_matrices = batch_vectors @ batch_pFpX.reshape(-1, 1, self.n_dims)
                C_total += np.sum(batch_matrices, axis=0)

            self.D["sub"][i]["C1"] = C_total * n_points

            # C1 为对称半正定矩阵，特征值/特征向量取实部（eig 对称输入返回
            # complex dtype，直接落盘 Excel 会变成字符串导致下游解析错误）
            eigenvalues, eigenvectors = eig(self.D["sub"][i]["C1"])
            sorted_indices = np.argsort(eigenvalues.real)[::-1]
            self.D["sub"][i]["eigenvalues"] = np.real(eigenvalues[sorted_indices])
            self.D["sub"][i]["eigenvectors"] = np.real(eigenvectors[:, sorted_indices])

            C1_matrix = self.D["sub"][i]["C1"]
            self.D["sub"][i]["matrix_stats"] = {
                "frobenius_norm": np.linalg.norm(C1_matrix, "fro"),
                "condition_number": np.linalg.cond(C1_matrix),
                "determinant": np.linalg.det(C1_matrix),
                "trace": np.trace(C1_matrix),
            }

        return self.D

    def save_cluster_coordinates(self, output_dir=None, max_points_per_cluster=200000):
        """保存聚类坐标信息到单独的 CSV 文件。"""
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.abspath(self.data_file)), "cluster_coordinates"
            )
        ensure_dir(output_dir)

        for i, sub_info in enumerate(self.D["sub"]):
            cluster_indices = sub_info["index"]

            if len(cluster_indices) > 0:
                cluster_coords = self.x[cluster_indices]

                if len(cluster_coords) > max_points_per_cluster:
                    selected_indices = np.random.choice(
                        len(cluster_coords), size=max_points_per_cluster, replace=False
                    )
                    cluster_coords = cluster_coords[selected_indices]

                coord_df = pd.DataFrame(cluster_coords, columns=self.X_cols)
                coord_df.to_csv(
                    os.path.join(output_dir, f"cluster_{i}_coordinates.csv"), index=False
                )

    def save_cluster_coordinates_with_gradients(
        self, output_dir=None, max_points_per_cluster=200000
    ):
        """保存聚类坐标及对应偏导数到单独的 CSV 文件。"""
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.abspath(self.data_file)),
                "cluster_coordinates_with_gradients",
            )
        ensure_dir(output_dir)

        for i, sub_info in enumerate(self.D["sub"]):
            cluster_indices = sub_info["index"]

            if len(cluster_indices) > 0:
                cluster_coords = self.x[cluster_indices]
                cluster_gradients = self.pFpX[cluster_indices]

                if len(cluster_coords) > max_points_per_cluster:
                    selected_indices = np.random.choice(
                        len(cluster_coords), size=max_points_per_cluster, replace=False
                    )
                    cluster_coords = cluster_coords[selected_indices]
                    cluster_gradients = cluster_gradients[selected_indices]

                data_dict = {}
                for j, col in enumerate(self.X_cols):
                    data_dict[col] = cluster_coords[:, j]
                for j, col in enumerate(self.G_cols):
                    data_dict[col] = cluster_gradients[:, j]

                combined_df = pd.DataFrame(data_dict)
                combined_df.to_csv(
                    os.path.join(
                        output_dir, f"cluster_{i}_coordinates_with_gradients.csv"
                    ),
                    index=False,
                )

    def save_results(self, output_dir=None):
        """保存聚类结果到 Excel 文件。"""
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.abspath(self.data_file)),
                "advanced_clustering_results",
            )
        ensure_dir(output_dir)

        excel_file = os.path.join(output_dir, "clustering_eigen_analysis.xlsx")

        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            # 1. 特征值矩阵
            eigenvalues_data = []
            for i in range(len(self.D["sub"])):
                if "eigenvalues" in self.D["sub"][i]:
                    cluster_eigenvalues = self.D["sub"][i]["eigenvalues"]
                    for j, eigenvalue in enumerate(cluster_eigenvalues):
                        eigenvalues_data.append(
                            {
                                "cluster_id": i,
                                "eigenvalue_index": j + 1,
                                "eigenvalue": eigenvalue,
                            }
                        )

            pd.DataFrame(eigenvalues_data).to_excel(
                writer, sheet_name="特征值矩阵", index=False
            )

            # 2. 特征向量矩阵
            for i in range(len(self.D["sub"])):
                if "eigenvectors" in self.D["sub"][i]:
                    cluster_eigenvectors = self.D["sub"][i]["eigenvectors"]

                    eigenvector_data = {}
                    for j in range(cluster_eigenvectors.shape[1]):
                        eigenvector_data[f"eigenvector_{j+1}"] = cluster_eigenvectors[:, j]

                    eigenvector_data["component_index"] = list(
                        range(1, cluster_eigenvectors.shape[0] + 1)
                    )

                    columns = ["component_index"] + [
                        f"eigenvector_{j+1}"
                        for j in range(cluster_eigenvectors.shape[1])
                    ]
                    pd.DataFrame(eigenvector_data, columns=columns).to_excel(
                        writer, sheet_name=f"聚类{i}_特征向量", index=False
                    )

            # 3. 聚类摘要
            cluster_info = []
            for i in range(len(self.D["sub"])):
                cluster_info.append(
                    {
                        "cluster_id": i,
                        "n_points": self.D["sub"][i]["n_points"],
                        "percentage": self.D["sub"][i]["n_points"] / len(self.x) * 100,
                    }
                )

            pd.DataFrame(cluster_info).to_excel(writer, sheet_name="聚类摘要", index=False)

            # 4. 全局特征矩阵信息
            matrix_info_data = []
            for i in range(len(self.D["sub"])):
                if "C1" in self.D["sub"][i]:
                    matrix = self.D["sub"][i]["C1"]
                    eigenvalues = self.D["sub"][i]["eigenvalues"]
                    matrix_info_data.append(
                        {
                            "cluster_id": i,
                            "matrix_size": f"{self.n_dims}x{self.n_dims}",
                            "trace": np.trace(matrix),
                            "determinant": np.linalg.det(matrix),
                            "frobenius_norm": np.linalg.norm(matrix, "fro"),
                            "max_eigenvalue": np.max(eigenvalues),
                            "min_eigenvalue": np.min(eigenvalues),
                        }
                    )

            pd.DataFrame(matrix_info_data).to_excel(
                writer, sheet_name="全局特征矩阵信息", index=False
            )

            # 5. 算法参数
            algorithm_params = {
                "n_clusters": len(self.D["sub"]),
                "total_points": len(self.x),
                "batch_size": self.batch_size,
                "algorithm": "subclust+FCM",
                "feature_dimension": self.n_dims,
                "data_file": os.path.basename(self.data_file),
            }

            pd.DataFrame([algorithm_params]).to_excel(
                writer, sheet_name="算法参数", index=False
            )

        return excel_file

    def save_clustering_components(self, output_dir=None):
        """保存聚类组件到文件，用于后续分类。

        组件固定保存到 {output_dir}/clustering_components/ 子目录，
        与 fit_power_function 的读取路径（clustering_dir/clustering_components/）一致。
        """
        import joblib

        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.abspath(self.data_file)), "advanced_clustering_results"
            )
        output_dir = os.path.join(output_dir, "clustering_components")
        ensure_dir(output_dir)

        if self.clustering_scaler is None or self.fcm_centers is None:
            raise ValueError("clustering_scaler / fcm_centers 未设置，无法保存组件")

        scaler_path = os.path.join(output_dir, "clustering_scaler.joblib")
        joblib.dump(self.clustering_scaler, scaler_path)

        centers_path = os.path.join(output_dir, "fcm_centers.npy")
        np.save(centers_path, self.fcm_centers)

        if self.fcm_parameters is not None:
            params_path = os.path.join(output_dir, "fcm_parameters.json")
            with open(params_path, "w") as f:
                json.dump(self.fcm_parameters, f, indent=2)

        if self.labels is not None:
            labels_path = os.path.join(output_dir, "cluster_labels.npy")
            np.save(labels_path, self.labels)

        # 设计仪器 schema 暴露（v1.3，加法）：隶属度矩阵与局部特征值落盘，
        # 供 core/design.py 与下游诊断使用；缺失时下游自动降级
        if self.u_matrix is not None:
            np.save(os.path.join(output_dir, "u_matrix.npy"), self.u_matrix)
        if self.local_eigenvalues is not None:
            np.save(os.path.join(output_dir, "local_eigenvalues.npy"), self.local_eigenvalues)

        feature_info = {
            "original_dimensions": self.n_dims,
            "feature_dimensions": self.fcm_centers.shape[1],
            "n_clusters": self.fcm_centers.shape[0],
            "X_cols": self.X_cols,
            "G_cols": self.G_cols,
            "Y_col": self.Y_col,
            "gradient_norm_included": False,
        }

        feature_info_path = os.path.join(output_dir, "feature_info.json")
        with open(feature_info_path, "w") as f:
            json.dump(feature_info, f, indent=2)

        if self.x is not None and len(self.x) > 0:
            param_ranges = {
                "min_values": self.x.min(axis=0).tolist(),
                "max_values": self.x.max(axis=0).tolist(),
                "mean_values": self.x.mean(axis=0).tolist(),
                "std_values": self.x.std(axis=0).tolist(),
                "n_samples": len(self.x),
                "n_dimensions": self.x.shape[1],
            }

            param_ranges_path = os.path.join(output_dir, "parameter_ranges.json")
            with open(param_ranges_path, "w") as f:
                json.dump(param_ranges, f, indent=2)

        return output_dir

    def run_analysis(self, output_dir=None):
        """运行完整的聚类分析流程。

        Args:
            output_dir: 结果输出目录（None 则默认在输入文件同目录）

        Returns:
            dict: 结构化结果摘要
        """
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(self.data_file))

        # 1. 加载数据
        self.load_data_chunked()

        # 2. 计算局部特征
        local_features, eigenvalues = self.compute_local_features_batch()

        # 3. subclust+FCM 复合聚类
        final_labels, fcm_centers, u = self.subclust_fcm_clustering(
            local_features, eigenvalues
        )
        self.u_matrix = u                          # schema 暴露：隶属度矩阵 (c, n)
        self.local_eigenvalues = eigenvalues       # schema 暴露：逐点 |g|²

        # 4. 组织聚类结果
        self.organize_clustering_results(final_labels, fcm_centers, u, local_features)

        # 5. 计算全局特征
        self.compute_global_features()

        # 6. 保存结果
        results_excel = self.save_results(output_dir)

        # 7. 保存聚类组件
        components_dir = self.save_clustering_components(output_dir)

        # 8. 保存聚类坐标
        self.save_cluster_coordinates(output_dir)

        # 9. 保存聚类坐标及偏导数
        self.save_cluster_coordinates_with_gradients(output_dir)

        # 10. 结构化摘要
        cluster_summary = [
            {
                "cluster_id": i,
                "n_points": sub["n_points"],
                "percentage": round(sub["percentage"], 2),
            }
            for i, sub in enumerate(self.D["sub"])
        ]
        n_fuzzy = int(np.sum(self.labels < 0)) if self.labels is not None else 0

        return {
            "output_dir": output_dir,
            "n_clusters": len(self.D["sub"]),
            "n_clusters_auto": self.auto_clusters,
            "n_fuzzy_points": n_fuzzy,
            "membership_threshold": self.membership_threshold,
            "n_dims": self.n_dims,
            "x_cols": self.X_cols,
            "y_col": self.Y_col,
            "g_cols": self.G_cols,
            "cluster_summary": cluster_summary,
            "files": {
                "eigen_analysis": results_excel,
                "components_dir": components_dir,
                "cluster_coordinates": os.path.join(output_dir, "cluster_coordinates"),
                "cluster_coordinates_with_gradients": os.path.join(
                    output_dir, "cluster_coordinates_with_gradients"
                ),
            },
        }
