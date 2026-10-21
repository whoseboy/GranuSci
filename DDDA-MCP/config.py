"""DDDA-MCP 配置管理。

集中管理默认参数、线程数、路径等配置项。
"""

import os


def get_default_n_threads():
    """获取默认线程数。若配置了环境变量则优先使用，否则使用 CPU 核数。"""
    env_threads = os.environ.get("DDDA_NUM_THREADS")
    if env_threads and env_threads.isdigit():
        return max(1, int(env_threads))
    cpu_count = os.cpu_count()
    return max(1, cpu_count or 4)


class Config:
    """DDDA-MCP 全局配置。"""

    def __init__(self):
        # 线程数（影响 OpenBLAS / OMP / MKL 等数值库）
        self.n_threads = get_default_n_threads()

        # Tool 1: compute_derivatives 默认参数
        self.k_bandwidth_k = 30        # 带宽估计近邻数
        self.k_neighbors_base = 80     # 局部拟合邻居数基数
        self.k_neighbors_per_dim = 15  # 每维额外邻居数

        # Tool 2: cluster_analysis 默认参数
        self.n_clusters = 1
        self.batch_size = 100000
        self.max_samples = 200000
        self.fcm_m = 2.0
        self.fcm_error = 0.0003
        self.fcm_maxiter = 2000
        # subclust 半径因子（MATLAB ra 语义，[0,1] 归一化特征空间；论文/MATLAB 用 0.1）
        self.subclust_radius_factor = 0.10
        # 压制半径倍数 sq（rb = sq·ra，MATLAB subclust 默认 1.25）
        self.subclust_suppression_ratio = 1.25
        # FCM 隶属度阈值：最大隶属度低于该值的点标记为模糊点(标签-1)，
        # 不参与拟合（对应论文/MATLAB 中 maxU<0.6 剔除模糊点，附录A.1.4）。设 0 关闭。
        self.membership_threshold = 0.6
        # subclust 自动确定聚类数时的 Chiu 接受/拒绝常数（论文附录A.1.4）
        self.subclust_accept_ratio = 0.5
        self.subclust_reject_ratio = 0.15

        # Tool 3: fit_power_function 默认参数
        self.max_clean_ratio = 0.1
        self.c_mode = "fixed"
        self.fixed_c = 1.0
        self.di_threshold = 0.05
        self.min_fit_points = 10
        self.curve_fit_maxfev = 20000
        # 幂指数 b 的拟合边界（论文扫描范围 [-2,2]，略放宽）；退化判定阈值
        self.fit_b_bounds = (-3.0, 3.0)
        self.fit_b_degenerate = 0.02

        # 可视化默认字体
        self.font_family = "Microsoft YaHei"
        self.font_fallback = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]


_config = None


def get_config() -> Config:
    """获取全局配置单例。"""
    global _config
    if _config is None:
        _config = Config()
    return _config
