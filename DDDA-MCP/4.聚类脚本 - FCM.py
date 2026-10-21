import os
# 在脚本最开头（import numpy 之前）添加
os.environ['OPENBLAS_NUM_THREADS'] = '4'      # 进一步降低线程数
os.environ['OMP_NUM_THREADS'] = '4'           # 控制 OpenMP 线程
os.environ['MKL_NUM_THREADS'] = '4'           # 如果使用 MKL
os.environ['NUMEXPR_NUM_THREADS'] = '4'       # NumExpr 线程控制

import numpy as np
import pandas as pd
from scipy.linalg import eig
from scipy.spatial.distance import cdist
import gc
from tqdm import tqdm
import glob
import warnings
warnings.filterwarnings('ignore')

class AdvancedClusteringAnalysis:
    """高级聚类分析类"""

    def __init__(self, data_file, n_clusters=3, batch_size=100000):
        """
        初始化聚类分析

        Args:
            data_file: 偏导数数据文件路径（Excel格式，包含坐标X1~X6和偏导数dY_dX1~dY_dX6）
            n_clusters: 聚类数
            batch_size: 批次大小
        """
        self.data_file = data_file
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.n_dims = 6  # 数据维度（X1~X6）
        self.X_cols = ["X1", "X2", "X3", "X4", "X5", "X6"]
        self.G_cols = ["dY_dX1", "dY_dX2", "dY_dX3", "dY_dX4", "dY_dX5", "dY_dX6"]

        # 存储结果
        self.x = None  # 坐标数据
        self.pFpX = None  # 偏微分数据
        self.D = None  # 聚类结果
        self.F = None  # 聚类中心

        # 保存聚类组件（用于分类）
        self.clustering_scaler = None  # 用于标准化梯度特征的RobustScaler
        self.features = None  # 梯度特征（6维）
        self.fcm_centers = None  # FCM聚类中心（标准化梯度特征空间）
        self.fcm_parameters = None  # FCM参数（m=2.0, error=0.0003, maxiter=2000）

    def load_data_chunked(self):
        """从Excel文件加载数据（包含坐标和偏导数）"""
        print("加载数据...")

        # 读取Excel文件
        df = pd.read_excel(self.data_file)

        # 检查必要的列是否存在
        missing_cols = [c for c in self.X_cols + self.G_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"数据文件缺少以下列: {missing_cols}")

        self.x = df[self.X_cols].values.astype(np.float32)
        self.pFpX = df[self.G_cols].values.astype(np.float32)

        if len(self.x) != len(self.pFpX):
            raise ValueError(f"数据点数不匹配: 坐标数据 {len(self.x)}, 偏微分数据 {len(self.pFpX)}")

        print(f"数据加载完成: {len(self.x)} 个数据点, {self.n_dims} 维")
        print(f"内存使用: 坐标数据 {self.x.nbytes/1024/1024:.2f} MB, 偏微分数据 {self.pFpX.nbytes/1024/1024:.2f} MB")
        return self.x, self.pFpX

    def compute_local_features_batch(self):
        """分块计算局部特征矩阵和主导特征向量（使用偏微分信息）

        Returns:
            tuple: (local_features, eigenvalues) - 特征向量和对应的特征值
        """
        N = len(self.pFpX)
        print(f"开始计算局部特征，共 {N} 个数据点")

        # 存储6维主导特征向量
        local_features = np.zeros((N, self.n_dims), dtype=np.float32)
        # 存储主导特征值
        eigenvalues = np.zeros(N, dtype=np.float32)

        # 分块处理
        n_batches = (N + self.batch_size - 1) // self.batch_size

        for batch_idx in tqdm(range(n_batches), desc="计算局部特征"):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, N)
            batch_size = end_idx - start_idx

            # 批量处理当前批次
            batch_pFpX = self.pFpX[start_idx:end_idx].astype(np.float32)

            # 向量化计算批次内的所有特征
            for i in range(batch_size):
                # 局部偏微分向量
                pFp_i = batch_pFpX[i].reshape(self.n_dims, 1)

                # 局部特征矩阵: W = pFp_i^T * pFp_i
                W = pFp_i @ pFp_i.T

                # 特征分解（使用eigh确保实对称矩阵的特征值都是实数）
                eig_vals, eig_vecs = np.linalg.eigh(W)

                # 对特征值排序（降序）
                sorted_indices = np.argsort(eig_vals)[::-1]

                # 提取主导特征向量（对应最大特征值）
                dominant_eigenvector = eig_vecs[:, sorted_indices[0]]
                # 提取主导特征值
                dominant_eigenvalue = eig_vals[sorted_indices[0]]

                # 存储主导特征向量和特征值
                local_features[start_idx + i] = dominant_eigenvector
                eigenvalues[start_idx + i] = dominant_eigenvalue

            # 释放批次数据内存
            del batch_pFpX

            # 定期释放内存
            if batch_idx % 5 == 0:
                gc.collect()

        # 对特征进行标准化处理，提高聚类效果
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        local_features = scaler.fit_transform(local_features)
        print("特征鲁棒标准化处理完成")

        return local_features, eigenvalues

    def sample_features_for_clustering(self, features, max_samples=200000):
        """
        对特征进行抽样以减少内存占用和计算时间

        Args:
            features: 特征向量矩阵
            max_samples: 最大采样数量

        Returns:
            sampled_features: 抽样后的特征
            sample_indices: 抽样索引
            original_shape: 原始形状
        """
        n_samples = features.shape[0]
        if n_samples <= max_samples:
            return features, np.arange(n_samples), features.shape

        # 随机抽样
        sample_indices = np.random.choice(n_samples, size=max_samples, replace=False)
        sampled_features = features[sample_indices]
        print(f"数据抽样完成: 从 {n_samples} 个样本抽样至 {max_samples} 个样本")

        return sampled_features, sample_indices, features.shape

    def subclust_fcm_clustering(self, features, eigenvalues):
        """
        使用subclust+FCM复合聚类算法进行聚类
        先用subclust确定初始聚类中心，再用FCM进行精细聚类

        Args:
            features: 特征向量矩阵（6维梯度向量）
            eigenvalues: 对应的特征值数组

        Returns:
            labels: 聚类标签
            centers: 聚类中心
            u: 模糊隶属度矩阵
        """
        print("开始subclust+FCM复合聚类...")

        # 确保特征矩阵是float类型
        features = features.astype(np.float64)
        eigenvalues = eigenvalues.astype(np.float64)

        print(f"原始特征形状: {features.shape}")

        n_samples = features.shape[0]
        print(f"数据集大小: {n_samples} 个样本")

        # 对大数据集进行抽样处理
        max_samples_for_clustering = 200000
        if n_samples > max_samples_for_clustering:
            print(f"数据集过大，使用抽样进行subclust+FCM聚类...")
            sampled_features, sample_indices, original_shape = self.sample_features_for_clustering(
                features, max_samples=max_samples_for_clustering)
            is_sampled = True
            features_to_use = sampled_features
        else:
            features_to_use = features
            is_sampled = False
            sample_indices = np.arange(n_samples)

        # 对特征进行标准化处理
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_to_use)
        features_reduced = features_scaled
        print("特征鲁棒标准化处理完成")

        # 第一步：使用subclust+FCM将数据分为粗粒度子空间
        print("使用subclust+FCM进行粗粒度聚类...")
        from skfuzzy.cluster import cmeans
        from scipy.spatial.distance import cdist

        # 使用减法聚类(subclust)确定聚类数量和初始中心
        n_samples = features_reduced.shape[0]
        if n_samples > 5000:
            from sklearn.cluster import KMeans
            n_sample_clusters = min(50000, max(100, n_samples // 30))
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
                    randomly_selected = np.random.choice(cluster_indices, size=samples_per_cluster, replace=False)
                    selected_indices.extend(randomly_selected)

            subclust_sample_indices = np.array(selected_indices)
            sample_data = features_reduced[subclust_sample_indices]
            print(f"对大数据集进行分层采样: 从 {n_samples} 个样本减少到 {len(sample_data)} 个样本用于subclust")
        else:
            sample_data = features_reduced
            subclust_sample_indices = np.arange(n_samples)

        # 计算数据范围以确定减法聚类参数
        ranges = np.max(sample_data, axis=0) - np.min(sample_data, axis=0)
        radius = 0.30 * np.mean(ranges)
        print(f"减法聚类半径: {radius}")

        # 优化的减法聚类(subclust)算法
        def subtractive_clustering_optimized(data, radius, target_clusters=None):
            ra_sq = (radius ** 2)
            rb_sq = (radius * 1.5) ** 2

            n_points = data.shape[0]
            potentials = np.zeros(n_points)

            for i in range(n_points):
                distances_sq = np.sum((data - data[i])**2, axis=1)
                potentials[i] = np.sum(np.exp(-distances_sq / ra_sq))

            centers = []
            selected_indices = []
            original_potentials = potentials.copy()
            while True:
                max_idx = np.argmax(potentials)
                max_potential = potentials[max_idx]

                if target_clusters is not None and len(centers) >= target_clusters:
                    break

                # 仅在未指定目标聚类数时，使用势能阈值提前终止
                if target_clusters is None and len(centers) > 0 and max_potential < 0.005 * original_potentials[selected_indices[0]]:
                    break

                centers.append(data[max_idx].copy())
                selected_indices.append(max_idx)

                center_point = data[max_idx]
                distances_sq = np.sum((data - center_point)**2, axis=1)
                reduction_factor = np.exp(-distances_sq / rb_sq)
                potentials -= max_potential * reduction_factor

                potentials[max_idx] = -np.inf

            return np.array(centers), selected_indices

        # 执行减法聚类获得初始中心
        subclust_centers, subclust_indices = subtractive_clustering_optimized(sample_data, radius, target_clusters=self.n_clusters)
        n_clusters_subclust = len(subclust_centers)

        print(f"原始特征数据形状: {features_reduced.shape}")
        X = features_reduced
        print(f"FCM输入数据形状: {X.shape}")

        print(f"subclust_centers形状: {subclust_centers.shape}")

        expected_shape = (n_clusters_subclust, X.shape[1])

        if subclust_centers.shape[0] >= n_clusters_subclust:
            cluster_centers = subclust_centers[:n_clusters_subclust, :]
            print(f"使用减法聚类中心，形状: {cluster_centers.shape}")
        else:
            print(f"警告: 减法聚类中心数量({subclust_centers.shape[0]})少于期望聚类数({n_clusters_subclust})，使用随机初始化")
            cluster_centers = np.random.rand(n_clusters_subclust, X.shape[1]) * \
                             (np.max(X, axis=0, keepdims=True) - np.min(X, axis=0, keepdims=True)) + \
                             np.min(X, axis=0, keepdims=True)
            print(f"随机初始化中心，形状: {cluster_centers.shape}")

        print(f"FCM初始化中心形状: {cluster_centers.shape}")
        print(f"期望聚类数: {n_clusters_subclust}")

        # 执行FCM聚类
        print("开始执行FCM聚类...")

        print(f"FCM输入X形状: {X.shape}")
        X_transposed = X.T
        print(f"转置后X_transposed形状: {X_transposed.shape}")

        cntr, u, u0, d, jm, p, fpc = cmeans(
            X_transposed,
            c=n_clusters_subclust,
            m=2.0,
            error=0.0003,
            maxiter=2000
        )

        print(f"cmeans返回的cntr形状: {cntr.shape}")

        coarse_labels = np.argmax(u, axis=0)

        print(f"FCM聚类完成，聚类中心形状: {cntr.shape}")
        print(f"隶属度矩阵形状: {u.shape}")
        print(f"粗粒度聚类标签长度: {len(coarse_labels)}")
        coarse_centers = cntr

        print(f"subclust+FCM粗粒度聚类完成，得到{n_clusters_subclust}个子空间")

        final_labels = coarse_labels
        fine_centers = cntr
        print(f"FCM聚类中心形状: {fine_centers.shape}")

        print(f"FCM聚类完成: {len(np.unique(final_labels))} 个聚类")

        # 如果进行了抽样，需要将标签映射回原始数据
        if is_sampled:
            print("将聚类结果映射回原始数据...")
            full_labels = np.full(original_shape[0], -1, dtype=int)
            full_labels[sample_indices] = final_labels

            unsampled_indices = np.setdiff1d(np.arange(original_shape[0]), sample_indices)
            if len(unsampled_indices) > 0:
                unsampled_features = features[unsampled_indices]
                unsampled_features_scaled = scaler.transform(unsampled_features)
                unsampled_features_reduced = unsampled_features_scaled

                if len(fine_centers) > 0:
                    centers_scaled_back = fine_centers
                    print(f"unsampled_features_reduced形状: {unsampled_features_reduced.shape}")
                    print(f"centers_scaled_back形状: {centers_scaled_back.shape}")

                    if unsampled_features_reduced.shape[1] != centers_scaled_back.shape[1]:
                        centers_scaled_back = centers_scaled_back.T
                        print(f"转置后的centers_scaled_back形状: {centers_scaled_back.shape}")

                    distances_to_centers = cdist(unsampled_features_reduced, centers_scaled_back)
                    print(f"未抽样点距离矩阵形状: {distances_to_centers.shape}")

                    m = 2.0
                    exponent = 2.0 / (m - 1.0)

                    distances_safe = distances_to_centers.copy()
                    zero_mask = distances_safe == 0
                    if np.any(zero_mask):
                        distances_safe[zero_mask] = 1e-10

                    n_unsampled = distances_safe.shape[0]
                    n_clusters = distances_safe.shape[1]

                    distances_safe = distances_safe.astype(np.float64)

                    exponent = 4.0
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
                    full_labels[unsampled_indices] = closest_clusters

                    print(f"未抽样点隶属度计算完成，形状: {u_unsampled.shape}")
                else:
                    full_labels[unsampled_indices] = 0

            final_labels = full_labels
        else:
            final_labels = final_labels

        print(f"FCM聚类完成: {len(np.unique(final_labels))} 个聚类")

        # 构建完整的隶属度矩阵
        if is_sampled:
            n_clusters = fine_centers.shape[0]
            n_total = original_shape[0]

            u_all = np.zeros((n_clusters, n_total))

            u_all[:, sample_indices] = u

            if len(unsampled_indices) > 0 and 'u_unsampled' in locals():
                u_unsampled_T = u_unsampled.T
                u_all[:, unsampled_indices] = u_unsampled_T
                print(f"完整隶属度矩阵构建完成，形状: {u_all.shape}")
            else:
                print(f"完整隶属度矩阵构建完成（仅抽样点），形状: {u_all.shape}")
        else:
            u_all = u
            print(f"完整隶属度矩阵（未抽样），形状: {u_all.shape}")

        # 保存聚类组件用于后续分类
        self.clustering_scaler = scaler
        self.features = features
        self.fcm_centers = fine_centers
        self.fcm_parameters = {
            'm': 2.0,
            'error': 0.0003,
            'maxiter': 2000,
            'n_clusters': fine_centers.shape[0],
            'feature_dimension': features.shape[1]
        }
        # 保存标签（供拟合脚本直接使用，无需重新计算特征和分配聚类）
        self.labels = final_labels
        print(f"聚类组件已保存: RobustScaler, {features.shape[1]}维特征, {fine_centers.shape[0]}个聚类中心, {len(final_labels)}个标签")

        return final_labels, fine_centers, u_all

    def organize_clustering_results(self, labels, centers, u, features=None):
        """组织聚类结果"""
        print("组织聚类结果...")

        self.D = {'sub': []}
        self.F = {'sub': []}

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
                    center_from_fcm = centers[label_pos[0]]
                    actual_center = center_from_fcm
                else:
                    actual_center = np.zeros(self.n_dims)
            else:
                actual_center = np.zeros(self.n_dims)

            membership_values = None
            if u is not None and u.shape[0] > label and u.shape[1] > max(cluster_indices) if len(cluster_indices) > 0 else False:
                try:
                    membership_values = u[label, cluster_indices]
                    avg_membership = np.mean(membership_values) if len(membership_values) > 0 else 0.0
                    max_membership = np.max(membership_values) if len(membership_values) > 0 else 0.0
                    min_membership = np.min(membership_values) if len(membership_values) > 0 else 0.0
                except Exception as e:
                    print(f"获取聚类 {label} 的隶属度值时出错: {e}")
                    membership_values = None

            sub_info = {
                'index': cluster_indices,
                'n_points': n_points,
                'percentage': n_points / len(labels) * 100,
                'center': actual_center
            }

            if membership_values is not None:
                sub_info['membership'] = membership_values
                sub_info['avg_membership'] = avg_membership
                sub_info['max_membership'] = max_membership
                sub_info['min_membership'] = min_membership

            self.D['sub'].append(sub_info)

            f_sub_info = {
                'center': sub_info['center'],
                'n_points': n_points
            }

            self.F['sub'].append(f_sub_info)

        print(f"聚类结果组织完成: {len(self.D['sub'])} 个子空间")

    def compute_global_features(self):
        """计算每个子区域的全局特征矩阵"""
        print("计算各子区域的全局特征矩阵...")

        for i in range(len(self.D['sub'])):
            cluster_indices = self.D['sub'][i]['index']
            n_points = len(cluster_indices)

            if n_points == 0:
                continue

            print(f"处理子区域 {i+1}: {n_points} 个数据点")

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

            self.D['sub'][i]['C1'] = C_total * n_points

            eigenvalues, eigenvectors = eig(self.D['sub'][i]['C1'])

            sorted_indices = np.argsort(eigenvalues.real)[::-1]
            self.D['sub'][i]['eigenvalues'] = eigenvalues.real[sorted_indices]
            self.D['sub'][i]['eigenvectors'] = eigenvectors[:, sorted_indices]

            C1_matrix = self.D['sub'][i]['C1']
            matrix_norm = np.linalg.norm(C1_matrix, 'fro')
            condition_number = np.linalg.cond(C1_matrix)
            determinant = np.linalg.det(C1_matrix)
            trace = np.trace(C1_matrix)

            self.D['sub'][i]['matrix_stats'] = {
                'frobenius_norm': matrix_norm,
                'condition_number': condition_number,
                'determinant': determinant,
                'trace': trace
            }

            print(f"子区域 {i+1} 特征分析完成:")
            print(f"  特征值范围: {self.D['sub'][i]['eigenvalues'][0]:.6e} ~ {self.D['sub'][i]['eigenvalues'][-1]:.6e}")

        return self.D

    def save_cluster_coordinates(self, output_dir=None, max_points_per_cluster=200000):
        """保存聚类坐标信息到单独的CSV文件"""
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(self.data_file)), "cluster_coordinates")
        print("保存聚类坐标信息...")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, sub_info in enumerate(self.D['sub']):
            cluster_indices = sub_info['index']

            if len(cluster_indices) > 0:
                cluster_coords = self.x[cluster_indices]

                if len(cluster_coords) > max_points_per_cluster:
                    print(f"聚类 {i} 包含 {len(cluster_coords)} 个点，超过限制 {max_points_per_cluster}，进行随机采样")
                    selected_indices = np.random.choice(len(cluster_coords), size=max_points_per_cluster, replace=False)
                    cluster_coords = cluster_coords[selected_indices]

                coord_df = pd.DataFrame(cluster_coords, columns=self.X_cols)
                coord_df.to_csv(f'{output_dir}/cluster_{i}_coordinates.csv', index=False)

                print(f"聚类 {i} 坐标数据已保存: {len(cluster_coords)} 个点")

    def save_cluster_coordinates_with_gradients(self, output_dir=None, max_points_per_cluster=200000):
        """保存聚类坐标信息及其对应的偏导数到单独的CSV文件"""
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(self.data_file)), "cluster_coordinates_with_gradients")
        print("保存聚类坐标及其对应的偏导数信息...")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, sub_info in enumerate(self.D['sub']):
            cluster_indices = sub_info['index']

            if len(cluster_indices) > 0:
                cluster_coords = self.x[cluster_indices]
                cluster_gradients = self.pFpX[cluster_indices]

                if len(cluster_coords) > max_points_per_cluster:
                    print(f"聚类 {i} 包含 {len(cluster_coords)} 个点，超过限制 {max_points_per_cluster}，进行随机采样")
                    selected_indices = np.random.choice(len(cluster_coords), size=max_points_per_cluster, replace=False)
                    cluster_coords = cluster_coords[selected_indices]
                    cluster_gradients = cluster_gradients[selected_indices]

                data_dict = {}
                for j, col in enumerate(self.X_cols):
                    data_dict[col] = cluster_coords[:, j]
                for j, col in enumerate(self.G_cols):
                    data_dict[col] = cluster_gradients[:, j]

                combined_df = pd.DataFrame(data_dict)
                combined_df.to_csv(f'{output_dir}/cluster_{i}_coordinates_with_gradients.csv', index=False)

                print(f"聚类 {i} 坐标和偏导数数据已保存: {len(cluster_coords)} 个点")

    def save_results(self, output_dir=None):
        """保存聚类结果到Excel文件"""
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(self.data_file)), "advanced_clustering_results")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"保存结果到 {output_dir} 目录...")

        excel_file = f"{output_dir}/clustering_eigen_analysis.xlsx"

        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # 1. 保存特征值矩阵
            eigenvalues_data = []
            for i in range(len(self.D['sub'])):
                if 'eigenvalues' in self.D['sub'][i]:
                    cluster_eigenvalues = self.D['sub'][i]['eigenvalues']
                    for j, eigenvalue in enumerate(cluster_eigenvalues):
                        eigenvalues_data.append({
                            'cluster_id': i,
                            'eigenvalue_index': j + 1,
                            'eigenvalue': eigenvalue
                        })

            eigenvalues_df = pd.DataFrame(eigenvalues_data)
            eigenvalues_df.to_excel(writer, sheet_name='特征值矩阵', index=False)

            # 2. 保存特征向量矩阵
            for i in range(len(self.D['sub'])):
                if 'eigenvectors' in self.D['sub'][i]:
                    cluster_eigenvectors = self.D['sub'][i]['eigenvectors']

                    eigenvector_data = {}
                    for j in range(cluster_eigenvectors.shape[1]):
                        eigenvector_data[f'eigenvector_{j+1}'] = cluster_eigenvectors[:, j]

                    eigenvector_data['component_index'] = list(range(1, cluster_eigenvectors.shape[0] + 1))

                    columns = ['component_index'] + [f'eigenvector_{j+1}' for j in range(cluster_eigenvectors.shape[1])]
                    eigenvectors_df = pd.DataFrame(eigenvector_data, columns=columns)
                    eigenvectors_df.to_excel(writer, sheet_name=f'聚类{i}_特征向量', index=False)

            # 3. 保存聚类摘要
            cluster_info = []
            for i in range(len(self.D['sub'])):
                cluster_info.append({
                    'cluster_id': i,
                    'n_points': self.D['sub'][i]['n_points'],
                    'percentage': self.D['sub'][i]['n_points'] / len(self.x) * 100
                })

            cluster_summary_df = pd.DataFrame(cluster_info)
            cluster_summary_df.to_excel(writer, sheet_name='聚类摘要', index=False)

            # 4. 保存全局特征矩阵信息
            matrix_info_data = []
            for i in range(len(self.D['sub'])):
                if 'C1' in self.D['sub'][i]:
                    matrix = self.D['sub'][i]['C1']
                    eigenvalues = self.D['sub'][i]['eigenvalues'] if 'eigenvalues' in self.D['sub'][i] else []
                    matrix_info_data.append({
                        'cluster_id': i,
                        'matrix_size': f"{self.n_dims}x{self.n_dims}",
                        'trace': np.trace(matrix),
                        'determinant': np.linalg.det(matrix),
                        'frobenius_norm': np.linalg.norm(matrix, 'fro'),
                        'max_eigenvalue': np.max(eigenvalues) if len(eigenvalues) > 0 else 0,
                        'min_eigenvalue': np.min(eigenvalues) if len(eigenvalues) > 0 else 0
                    })

            matrix_info_df = pd.DataFrame(matrix_info_data)
            matrix_info_df.to_excel(writer, sheet_name='全局特征矩阵信息', index=False)

            # 5. 保存算法参数
            algorithm_params = {
                'n_clusters': len(self.D['sub']),
                'total_points': len(self.x),
                'batch_size': self.batch_size,
                'algorithm': 'subclust+FCM',
                'feature_dimension': self.n_dims,
                'data_file': os.path.basename(self.data_file)
            }

            algorithm_df = pd.DataFrame([algorithm_params])
            algorithm_df.to_excel(writer, sheet_name='算法参数', index=False)

        print(f"所有结果已保存到 {excel_file}")

    def save_clustering_components(self, output_dir=None):
        """保存聚类组件到文件，用于后续分类"""
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(self.data_file)), "clustering_components")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"保存聚类组件到 {output_dir} 目录...")

        if self.clustering_scaler is None:
            print("警告: clustering_scaler 未设置，无法保存聚类组件")
            return

        if self.fcm_centers is None:
            print("警告: fcm_centers 未设置，无法保存聚类组件")
            return

        try:
            import joblib
            import json

            scaler_path = os.path.join(output_dir, "clustering_scaler.joblib")
            joblib.dump(self.clustering_scaler, scaler_path)
            print(f"RobustScaler已保存到: {scaler_path}")

            centers_path = os.path.join(output_dir, "fcm_centers.npy")
            np.save(centers_path, self.fcm_centers)
            print(f"FCM聚类中心已保存到: {centers_path}")

            if self.fcm_parameters is not None:
                params_path = os.path.join(output_dir, "fcm_parameters.json")
                with open(params_path, 'w') as f:
                    json.dump(self.fcm_parameters, f, indent=2)
                print(f"FCM参数已保存到: {params_path}")

            # 保存聚类标签（供拟合脚本直接使用）
            if hasattr(self, 'labels') and self.labels is not None:
                labels_path = os.path.join(output_dir, "cluster_labels.npy")
                np.save(labels_path, self.labels)
                print(f"聚类标签已保存到: {labels_path} ({len(self.labels)} 个标签)")

            feature_info = {
                'original_dimensions': self.n_dims,
                'feature_dimensions': self.fcm_centers.shape[1] if self.fcm_centers is not None else None,
                'n_clusters': self.fcm_centers.shape[0] if self.fcm_centers is not None else None,
                'X_cols': self.X_cols,
                'G_cols': self.G_cols,
                'gradient_norm_included': False
            }

            feature_info_path = os.path.join(output_dir, "feature_info.json")
            with open(feature_info_path, 'w') as f:
                json.dump(feature_info, f, indent=2)
            print(f"特征信息已保存到: {feature_info_path}")

            if self.x is not None and len(self.x) > 0:
                param_ranges = {
                    'min_values': self.x.min(axis=0).tolist(),
                    'max_values': self.x.max(axis=0).tolist(),
                    'mean_values': self.x.mean(axis=0).tolist(),
                    'std_values': self.x.std(axis=0).tolist(),
                    'n_samples': len(self.x),
                    'n_dimensions': self.x.shape[1]
                }

                param_ranges_path = os.path.join(output_dir, "parameter_ranges.json")
                with open(param_ranges_path, 'w') as f:
                    json.dump(param_ranges, f, indent=2)
                print(f"参数范围信息已保存到: {param_ranges_path}")

                print(f"训练数据参数范围摘要:")
                for i in range(min(self.n_dims, len(param_ranges['min_values']))):
                    print(f"  {self.X_cols[i]}: [{param_ranges['min_values'][i]:.6f}, {param_ranges['max_values'][i]:.6f}]")
            else:
                print("警告: 训练坐标数据未加载，无法保存参数范围")

            print("所有聚类组件已保存完成")

        except Exception as e:
            print(f"保存聚类组件时出错: {e}")

    def run_analysis(self):
        """运行完整的聚类分析流程"""
        print("=" * 60)
        print("开始高级聚类分析")
        print("=" * 60)

        # 1. 加载数据
        self.load_data_chunked()

        # 2. 计算局部特征（使用偏微分信息进行聚类）
        local_features, eigenvalues = self.compute_local_features_batch()

        # 3. subclust+FCM复合聚类
        final_labels, fcm_centers, u = self.subclust_fcm_clustering(local_features, eigenvalues)

        # 4. 组织聚类结果
        self.organize_clustering_results(final_labels, fcm_centers, u, local_features)

        # 5. 计算全局特征
        self.compute_global_features()

        # 6. 保存结果
        self.save_results()

        # 7. 保存聚类组件（用于后续分类）
        self.save_clustering_components()

        # 8. 保存聚类坐标信息
        self.save_cluster_coordinates()

        # 9. 保存聚类坐标及其对应的偏导数信息
        self.save_cluster_coordinates_with_gradients()

        # 输出最终统计信息
        self.print_summary()

        return self.D, self.F

    def print_summary(self):
        """打印聚类分析总结"""
        print("=" * 60)
        print("聚类分析总结")
        print("=" * 60)

        total_points = len(self.x)
        n_clusters = len(self.D['sub'])

        print(f"总数据点数: {total_points}")
        print(f"聚类数量: {n_clusters}")

        for i, sub_info in enumerate(self.D['sub']):
            n_points = sub_info['n_points']
            percentage = sub_info['percentage']
            print(f"聚类 {i}: {n_points} 点 ({percentage:.2f}%)")

        print("=" * 60)


def main():
    """主函数"""
    # 使用脚本所在目录的偏导数文件
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base_dir, "对数空间数据-720h_偏导数.xlsx")

    if not os.path.exists(data_file):
        print(f"错误: 数据文件不存在: {data_file}")
        print("请先运行 3.光滑微分求导.py 生成偏导数文件")
        return

    print(f"使用数据文件: {data_file}")

    # 创建聚类实例
    analyzer = AdvancedClusteringAnalysis(data_file, n_clusters=3)

    # 运行完整的聚类分析
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
