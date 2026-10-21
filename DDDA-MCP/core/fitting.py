"""幂函数拟合核心算法（脚本5重构）。

在物理空间中对每个聚类子空间拟合幂函数 A = a * C^b + c，
其中 C = B1^D1 * B2^D2 * ... * Bn^Dn（由主导特征向量 Di 加权）。

输入数据固定在 log10 对数空间，物理空间通过 B=10^X、A=10^Y 转换。
支持固定 c（默认 1.0）与自由拟合 c 两种模式，含残差离群点清洗。
"""

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=RuntimeWarning)

from config import get_config  # noqa: E402
from utils.data_io import (  # noqa: E402
    infer_columns_from_derivative_file,
    read_excel,
    ensure_dir,
)
from utils.visualization import plt  # noqa: E402


# ========== 数学函数 ==========

def power_func(C, a, b, c=1.0):
    """幂函数: A = a * C^b + c"""
    return a * np.power(C, b) + c


def calculate_C_vectorized(B, Di):
    """向量化计算 C = prod(B_i ^ D_i)。

    Args:
        B: (N, n_dims) 物理空间坐标
        Di: (n_dims,) 主导特征向量分量

    Returns:
        C: (N,) 组合变量
    """
    mask = (B > 0) & (Di != 0)
    log_B = np.where(mask, np.log(np.where(B > 0, B, 1.0)), 0.0)
    log_C = log_B @ Di
    return np.exp(log_C)


# ========== 数据加载 ==========

def _load_derivative_df(derivative_file, n_indep=None, x_cols=None, y_col=None):
    """读取偏导数数据并推断列名。"""
    df = read_excel(derivative_file)
    inferred_x, inferred_y, g_cols = infer_columns_from_derivative_file(
        df, n_indep=n_indep
    )
    if x_cols:
        inferred_x = [c for c in x_cols if c in df.columns]
    if y_col:
        inferred_y = y_col
    if not g_cols:
        raise ValueError(
            f"数据文件未找到偏导数列（列名以 'dY_d' 开头）: {list(df.columns)}"
        )
    return df, inferred_x, inferred_y, g_cols


def _load_cluster_data_from_files(clustering_dir, n_indep=None):
    """从聚类输出目录加载标签和特征向量。

    目录结构:
      clustering_dir/clustering_eigen_analysis.xlsx
      clustering_dir/clustering_components/cluster_labels.npy
    """
    labels_path = os.path.join(clustering_dir, "clustering_components", "cluster_labels.npy")
    # 兼容旧布局（组件直接存于 output_dir 根下）
    if not os.path.exists(labels_path):
        labels_path = os.path.join(clustering_dir, "cluster_labels.npy")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"聚类标签文件不存在: {labels_path}\n"
            f"请先运行 cluster_analysis 完成聚类并保存标签"
        )
    labels = np.load(labels_path)

    eigen_path = os.path.join(clustering_dir, "clustering_eigen_analysis.xlsx")
    if not os.path.exists(eigen_path):
        raise FileNotFoundError(
            f"聚类特征向量文件不存在: {eigen_path}\n"
            f"请先运行 cluster_analysis 完成聚类"
        )

    n_clusters = len(np.unique(labels[labels >= 0]))
    cluster_eigenvectors = {}

    for c in range(n_clusters):
        sheet_name = f"聚类{c}_特征向量"
        try:
            df_eig = pd.read_excel(eigen_path, sheet_name=sheet_name)
            if "component_index" in df_eig.columns:
                eig_mat = df_eig.drop("component_index", axis=1).values
            else:
                eig_mat = df_eig.values
            # 保持 (n_dims x n_dims) 布局：列 j = 第 j 个特征向量（特征值降序），
            # 列 0 为主导特征向量 —— 与内存格式一致
            cluster_eigenvectors[c] = eig_mat
        except Exception as e:
            warnings.warn(f"无法加载子空间 {c} 的特征向量: {e}")

    return labels, n_clusters, cluster_eigenvectors


def _load_cluster_data_from_analysis(analysis):
    """从 AdvancedClusteringAnalysis 实例（内存）加载标签和特征向量。"""
    labels = analysis.labels
    if labels is None:
        raise ValueError("聚类分析尚未运行，labels 为空")

    n_clusters = len(analysis.D["sub"])
    cluster_eigenvectors = {}

    for i in range(n_clusters):
        sub = analysis.D["sub"][i]
        if "eigenvectors" in sub:
            # eigenvectors shape: (n_dims, n_dims)，列按特征值降序
            cluster_eigenvectors[i] = sub["eigenvectors"]
        else:
            warnings.warn(f"子空间 {i} 缺少特征向量")

    return labels, n_clusters, cluster_eigenvectors


def _get_Di(eigenvectors, di_threshold):
    """从特征向量矩阵提取主导特征向量 Di，并对小分量置零。

    Args:
        eigenvectors:
            - 文件加载: shape (n_eigvecs, n_dims)，第0行为主导
            - 内存加载: shape (n_dims, n_dims)，第0列为主导
        di_threshold: |Di| < threshold 时置零

    Returns:
        Di: (n_dims,) numpy array
    """
    if eigenvectors.shape[0] == eigenvectors.shape[1]:
        # 方阵：内存格式 (n_dims, n_dims)，列0为主导
        Di = eigenvectors[:, 0].copy()
    else:
        # 文件加载格式 (n_eigvecs, n_dims)，行0为主导
        Di = eigenvectors[0, :].copy()
    Di[np.abs(Di) < di_threshold] = 0
    return Di


# ========== 拟合 ==========

def _robust_fit(
    C_values, A_values,
    c_mode="fixed", fixed_c=1.0,
    max_clean_ratio=0.1,
    min_fit_points=10,
    maxfev=20000,
    b_bounds=(-3.0, 3.0),
    b_degenerate=0.02,
):
    """幂函数拟合，含残差离群点清洗、幂指数边界约束与退化回退。

    相比原版的改进：
    - b 限制在 b_bounds 内（论文扫描范围 [-2,2] 的放宽版），避免无物理意义的解
    - 拟合结果 |b| < b_degenerate（退化为常数）时，回退到论文式指数扫描：
      n 在 [-2, 2] 内以 0.1 步长扫描，每个 n 线性求解 a（及 c），以平均相对
      误差最小者作为最优（论文式2.17-2.19 协议）
    - 输出论文指标 delta（平均相对误差）

    Returns:
        (result_dict, C_cleaned, A_cleaned, C_removed, A_removed)
    """
    result = {}

    positive_mask = C_values > 0
    C_pos = C_values[positive_mask]
    A_pos = A_values[positive_mask]

    if len(C_pos) < min_fit_points:
        result["error"] = f"正数 C 值不足: {len(C_pos)} < {min_fit_points}"
        return result, C_values, A_values, np.array([]), np.array([])

    if c_mode == "free":
        fit_func = lambda C, a, b, c: a * np.power(C, b) + c
        bounds = ([-np.inf, b_bounds[0], -np.inf], [np.inf, b_bounds[1], np.inf])
        p0 = [np.mean(A_pos), 1.0, 0.0]
    else:
        fit_func = lambda C, a, b: a * np.power(C, b) + fixed_c
        bounds = ([-np.inf, b_bounds[0]], [np.inf, b_bounds[1]])
        p0 = [np.mean(A_pos), 1.0]

    def _fit_once(C_arr, A_arr):
        popt, _ = curve_fit(fit_func, C_arr, A_arr, p0=p0, bounds=bounds, maxfev=maxfev)
        if c_mode == "fixed":
            return [popt[0], popt[1], fixed_c]
        return list(popt)

    def _fit_scan(C_arr, A_arr):
        """论文协议回退：指数 n 扫描 + 线性最小二乘，最小化平均相对误差。"""
        best = None
        for n in np.arange(-2.0, 2.001, 0.1):
            Cn = np.power(C_arr, n)
            if c_mode == "fixed":
                # A - fixed_c = a * C^n
                y = A_arr - fixed_c
                denom = np.sum(Cn ** 2)
                if denom <= 0:
                    continue
                a = np.sum(Cn * y) / denom
                c = fixed_c
            else:
                # A = a * C^n + c -> 线性设计矩阵 [C^n, 1]
                M = np.column_stack([Cn, np.ones_like(Cn)])
                coef, *_ = np.linalg.lstsq(M, A_arr, rcond=None)
                a, c = coef
            pred = a * Cn + c
            rel = np.mean(np.abs((A_arr - pred) / np.maximum(np.abs(A_arr), 1e-12)))
            if best is None or rel < best[0]:
                best = (rel, [a, float(n), float(c)], "scan")
        if best is None:
            raise RuntimeError("指数扫描回退失败")
        return best[1]

    try:
        try:
            params_initial = _fit_once(C_pos, A_pos)
        except Exception:
            params_initial = _fit_scan(C_pos, A_pos)

        # 退化检测：|b| 过小 => 常数拟合，回退论文扫描协议
        if abs(params_initial[1]) < b_degenerate:
            params_initial = _fit_scan(C_pos, A_pos)

        A_pred_initial = power_func(C_pos, *params_initial)
        residuals = np.abs(A_pos - A_pred_initial)

        n_points = len(C_pos)
        n_to_remove = int(np.floor(n_points * max_clean_ratio))

        if n_to_remove > 0 and n_points - n_to_remove >= min_fit_points:
            sorted_idx = np.argsort(residuals)
            keep_idx = sorted_idx[:-n_to_remove]
            remove_idx = sorted_idx[-n_to_remove:]

            C_clean = C_pos[keep_idx]
            A_clean = A_pos[keep_idx]
            C_rem = C_pos[remove_idx]
            A_rem = A_pos[remove_idx]

            try:
                params = _fit_once(C_clean, A_clean)
                if abs(params[1]) < b_degenerate:
                    params = _fit_scan(C_clean, A_clean)
            except Exception:
                params = _fit_scan(C_clean, A_clean)

            A_pred = power_func(C_clean, *params)
            ss_res = np.sum((A_clean - A_pred) ** 2)
            ss_tot = np.sum((A_clean - np.mean(A_clean)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            delta = float(np.mean(np.abs((A_clean - A_pred) / np.maximum(np.abs(A_clean), 1e-12))))

            result.update({
                "params": {"a": params[0], "b": params[1], "c": params[2]},
                "r2": float(r2),
                "delta": delta,
                "func": f"A = {params[0]:.6f} * C^{params[1]:.6f} + {params[2]:.6f}",
                "cleaned": True,
                "n_initial": n_points,
                "n_cleaned": n_to_remove,
            })
            return result, C_clean, A_clean, C_rem, A_rem
        else:
            params = params_initial
            A_pred = power_func(C_pos, *params)
            ss_res = np.sum((A_pos - A_pred) ** 2)
            ss_tot = np.sum((A_pos - np.mean(A_pos)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            delta = float(np.mean(np.abs((A_pos - A_pred) / np.maximum(np.abs(A_pos), 1e-12))))

            result.update({
                "params": {"a": params[0], "b": params[1], "c": params[2]},
                "r2": float(r2),
                "delta": delta,
                "func": f"A = {params[0]:.6f} * C^{params[1]:.6f} + {params[2]:.6f}",
                "cleaned": False,
                "n_initial": n_points,
                "n_cleaned": 0,
            })
            return result, C_pos, A_pos, np.array([]), np.array([])

    except Exception as e:
        result["error"] = str(e)
        return result, C_values, A_values, np.array([]), np.array([])


# ========== 可视化 ==========

def _create_visualizations(all_results, n_dims, vis_dir):
    """生成总览图和各子空间详细拟合图。"""
    ensure_dir(vis_dir)
    cluster_ids = sorted(all_results.keys())
    n_clusters = len(cluster_ids)

    # --- 总览图 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 样本数量
    ax = axes[0, 0]
    sizes = [all_results[c]["sample_size"] for c in cluster_ids]
    ax.bar(cluster_ids, sizes, color=plt.cm.Set3(np.linspace(0, 1, max(n_clusters, 2))))
    ax.set_title("各子空间样本数量")
    ax.set_xlabel("子空间编号")
    ax.set_ylabel("样本数量")
    for i, v in enumerate(sizes):
        ax.text(cluster_ids[i], v, str(v), ha="center", va="bottom")

    # R2
    ax = axes[0, 1]
    r2_values = []
    for c in cluster_ids:
        fr = all_results[c]["fitting_results"]
        r2 = fr.get("r2", 0) if "error" not in fr else 0
        r2_values.append(r2)
    colors = [
        "green" if r2 > 0.4 else "orange" if r2 > 0.2 else "red"
        for r2 in r2_values
    ]
    bars = ax.bar(cluster_ids, r2_values, color=colors, alpha=0.7)
    ax.set_title("各子空间幂函数拟合 R²")
    ax.set_xlabel("子空间编号")
    ax.set_ylabel("R² 值")
    ax.axhline(y=0.4, color="red", linestyle="--", alpha=0.5, label="良好拟合")
    ax.axhline(y=0.2, color="orange", linestyle="--", alpha=0.5, label="一般拟合")
    ax.legend()
    for bar, r2 in zip(bars, r2_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{r2:.3f}", ha="center", va="bottom", fontsize=9,
        )

    # Di 热力图
    ax = axes[1, 0]
    Di_matrix = np.array([all_results[c]["Di"] for c in cluster_ids])
    im = ax.imshow(Di_matrix, cmap="coolwarm", aspect="auto")
    plt.colorbar(im, ax=ax, label="Di 值")
    ax.set_title("各子空间 Di 值热力图")
    ax.set_xlabel("特征维度")
    ax.set_ylabel("子空间编号")
    ax.set_xticks(range(n_dims))
    ax.set_xticklabels([f"D{i+1}" for i in range(n_dims)])
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels(cluster_ids)

    # C-A 散点
    ax = axes[1, 1]
    scatter_colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 2)))
    for c, color in zip(cluster_ids, scatter_colors):
        result = all_results[c]
        if len(result["C_values"]) > 0:
            ax.scatter(
                result["C_values"], result["A_values"],
                color=color, alpha=0.6, s=10, label=f"子空间{c}",
            )
    ax.set_title("各子空间 C-A 散点图")
    ax.set_xlabel("C 值")
    ax.set_ylabel("A 值")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xscale("log")

    plt.tight_layout()
    overview_path = os.path.join(vis_dir, "fitting_overview.png")
    plt.savefig(overview_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- 各子空间详细图 ---
    detail_paths = []
    for c in cluster_ids:
        result = all_results[c]
        C_vals = result["C_values"]
        A_vals = result["A_values"]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(C_vals, A_vals, alpha=0.6, s=30, color="blue", label="数据点")

        if len(result["C_not_cleaned"]) > 0:
            ax.scatter(
                result["C_not_cleaned"], result["A_not_cleaned"],
                alpha=0.3, s=20, color="gray", marker="x", label="已清洗点",
            )

        fr = result["fitting_results"]
        if "error" not in fr and "params" in fr:
            params = [fr["params"]["a"], fr["params"]["b"], fr["params"]["c"]]
            r2 = fr["r2"]
            C_range = np.logspace(
                np.log10(np.min(C_vals)), np.log10(np.max(C_vals)), 500
            )
            A_pred = power_func(C_range, *params)
            ax.plot(
                C_range, A_pred, color="red", linewidth=2,
                label=f"幂函数 (R²={r2:.4f})",
            )

        ax.set_title(f"子空间 {c} 拟合曲线\n样本数: {result['sample_size']}")
        ax.set_xlabel("C 值（物理空间）")
        ax.set_ylabel("A 值（物理空间）")
        ax.legend()
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        detail_path = os.path.join(vis_dir, f"subspace_{c}_fitting.png")
        plt.savefig(detail_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        detail_paths.append(detail_path)

    return {"overview": overview_path, "details": detail_paths}


# ========== 数据导出 ==========

def _export_fitting_data(all_results, output_dir):
    """导出拟合数据和曲线到 Excel。"""
    ensure_dir(output_dir)
    excel_path = os.path.join(output_dir, "fitting_data.xlsx")
    cluster_ids = sorted(all_results.keys())

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for c in cluster_ids:
            result = all_results[c]
            C_vals = result["C_values"]
            A_vals = result["A_values"]

            df_data = pd.DataFrame({"C值": C_vals, "A值": A_vals})
            df_data.to_excel(writer, sheet_name=f"子空间{c}_数据", index=False)

            fr = result["fitting_results"]
            if "error" not in fr and "params" in fr and len(C_vals) > 0:
                params = [fr["params"]["a"], fr["params"]["b"], fr["params"]["c"]]
                C_range = np.logspace(
                    np.log10(np.min(C_vals)), np.log10(np.max(C_vals)), 500
                )
                A_pred = power_func(C_range, *params)
                df_curve = pd.DataFrame({"C值": C_range, "拟合A值": A_pred})
                df_curve.to_excel(
                    writer, sheet_name=f"子空间{c}_拟合曲线", index=False
                )

    return excel_path


def _generate_report(all_results, report_path):
    """生成文本分析报告。"""
    cluster_ids = sorted(all_results.keys())

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("聚类幂函数拟合分析报告\n")
        f.write("=" * 70 + "\n\n")
        f.write("方法:\n")
        f.write("- 在真实物理空间中拟合: C = B1^D1 * ... * Bn^Dn\n")
        f.write("- 幂函数模型: A = a * C^b + c\n")
        f.write("- 数据清洗: 去除残差最大的数据点\n\n")

        f.write("拟合结果:\n")
        f.write("-" * 70 + "\n")

        best_r2 = -1
        best_cluster = None

        for c in cluster_ids:
            result = all_results[c]
            f.write(f"\n子空间 {c}:\n")
            f.write(f"  样本数量: {result['sample_size']}\n")
            if result.get("not_cleaned_size", 0) > 0:
                f.write(f"  清洗数据点: {result['not_cleaned_size']}\n")
            f.write(f"  Di 值: {result['Di']}\n")

            fr = result["fitting_results"]
            if "error" not in fr and "params" in fr:
                f.write(f"  拟合函数: {fr['func']}\n")
                f.write(f"  R² = {fr['r2']:.6f}\n")
                if fr["r2"] > best_r2:
                    best_r2 = fr["r2"]
                    best_cluster = c
            else:
                f.write(f"  拟合失败: {fr.get('error', '未知')}\n")

        f.write(f"\n{'=' * 70}\n")
        f.write("拟合效果评估:\n")
        for c in cluster_ids:
            fr = all_results[c]["fitting_results"]
            r2 = fr.get("r2", -1) if "error" not in fr else -1
            if r2 > 0.4:
                eval_str = "良好拟合"
            elif r2 > 0.2:
                eval_str = "一般拟合"
            else:
                eval_str = "较差拟合"
            f.write(f"  子空间 {c}: {eval_str} (R² = {r2:.6f})\n")

        if best_cluster is not None:
            f.write(f"\n最佳拟合: 子空间 {best_cluster} (R² = {best_r2:.6f})\n")

    return report_path


# ========== 主入口 ==========

def fit_power_function(
    derivative_file,
    output_dir=None,
    n_indep=None,
    x_cols=None,
    y_col=None,
    clustering_dir=None,
    clustering_analysis=None,
    c_mode=None,
    fixed_c=None,
    max_clean_ratio=None,
    di_threshold=None,
    min_fit_points=None,
    curve_fit_maxfev=None,
):
    """对每个聚类子空间在物理空间中拟合幂函数 A = a * C^b + c。

    Args:
        derivative_file: 偏导数数据文件路径（Excel，含 X、Y、dY_dX 列）。
            数据固定在 log10 对数空间。
        output_dir: 结果输出目录（None 则使用 derivative_file 同目录）。
        n_indep: 自变量列数（用于自动推断列名）。
        x_cols: 显式自变量列名列表。
        y_col: 显式因变量列名。
        clustering_dir: 聚类结果目录（含 clustering_eigen_analysis.xlsx 和
            clustering_components/cluster_labels.npy）。None 时使用
            derivative_file 同目录。与 clustering_analysis 二选一。
        clustering_analysis: AdvancedClusteringAnalysis 实例（pipeline 模式，
            直接从内存读取聚类结果，无需落盘文件）。
        c_mode: "fixed"（固定 c=fixed_c）或 "free"（自由拟合 c）。
        fixed_c: 固定 c 值（默认 1.0），仅 c_mode="fixed" 时生效。
        max_clean_ratio: 离群点清洗比例（默认 0.1）。
        di_threshold: |Di| 小于此阈值时置零（默认 0.05）。
        min_fit_points: 拟合所需最少数据点（默认 10）。
        curve_fit_maxfev: curve_fit 最大迭代次数（默认 20000）。

    Returns:
        dict: 结构化结果，包含 fitting_results、cluster_info、files。
    """
    cfg = get_config()
    c_mode = cfg.c_mode if c_mode is None else c_mode
    fixed_c = cfg.fixed_c if fixed_c is None else fixed_c
    max_clean_ratio = (
        cfg.max_clean_ratio if max_clean_ratio is None else max_clean_ratio
    )
    di_threshold = cfg.di_threshold if di_threshold is None else di_threshold
    min_fit_points = (
        cfg.min_fit_points if min_fit_points is None else min_fit_points
    )
    curve_fit_maxfev = (
        cfg.curve_fit_maxfev if curve_fit_maxfev is None else curve_fit_maxfev
    )

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(derivative_file))
    ensure_dir(output_dir)

    # 1. 读取偏导数数据
    df, X_cols, Y_col, G_cols = _load_derivative_df(
        derivative_file, n_indep=n_indep, x_cols=x_cols, y_col=y_col
    )
    n_dims = len(G_cols)
    X_log = df[X_cols].values.astype(np.float64)
    Y_log = df[Y_col].values.astype(np.float64)

    # 2. 加载聚类结果
    if clustering_analysis is not None:
        labels, n_clusters, cluster_eigenvectors = (
            _load_cluster_data_from_analysis(clustering_analysis)
        )
    else:
        if clustering_dir is None:
            clustering_dir = os.path.dirname(os.path.abspath(derivative_file))
        labels, n_clusters, cluster_eigenvectors = (
            _load_cluster_data_from_files(clustering_dir, n_indep=n_indep)
        )

    if len(labels) != len(df):
        raise ValueError(
            f"聚类标签数量 ({len(labels)}) 与数据行数 ({len(df)}) 不一致"
        )

    # 3. 转换到物理空间
    B_data = 10.0 ** X_log
    A_data = 10.0 ** Y_log

    # 4. 提取各子空间 Di
    cluster_Di = {}
    for c in range(n_clusters):
        if c in cluster_eigenvectors:
            cluster_Di[c] = _get_Di(cluster_eigenvectors[c], di_threshold)

    # 5. 计算 C 值并按子空间分组
    cluster_C = {c: [] for c in range(n_clusters)}
    cluster_A = {c: [] for c in range(n_clusters)}

    for c in range(n_clusters):
        if c not in cluster_Di:
            continue
        mask = labels == c
        if not np.any(mask):
            continue
        B_sub = B_data[mask]
        A_sub = A_data[mask]
        C_sub = calculate_C_vectorized(B_sub, cluster_Di[c])

        valid = (C_sub > 0) & np.isfinite(C_sub) & (A_sub > 0) & np.isfinite(A_sub)
        cluster_C[c] = C_sub[valid]
        cluster_A[c] = A_sub[valid]

    # 6. 逐子空间拟合
    all_results = {}
    for c in range(n_clusters):
        C_arr = cluster_C[c]
        A_arr = cluster_A[c]

        if len(C_arr) < min_fit_points:
            continue

        fit_result, C_clean, A_clean, C_rem, A_rem = _robust_fit(
            C_arr, A_arr,
            c_mode=c_mode,
            fixed_c=fixed_c,
            max_clean_ratio=max_clean_ratio,
            min_fit_points=min_fit_points,
            maxfev=curve_fit_maxfev,
            b_bounds=cfg.fit_b_bounds,
            b_degenerate=cfg.fit_b_degenerate,
        )

        all_results[c] = {
            "cluster_id": c,
            "Di": cluster_Di.get(c, np.zeros(n_dims)).tolist(),
            "C_values": C_clean,
            "A_values": A_clean,
            "C_not_cleaned": C_rem,
            "A_not_cleaned": A_rem,
            "fitting_results": fit_result,
            "sample_size": len(C_clean),
            "not_cleaned_size": len(C_rem),
        }

    if not all_results:
        return {
            "error": "没有获得有效的拟合结果（所有子空间数据点不足）",
            "fitting_results": [],
            "files": {},
        }

    # 7. 可视化
    vis_dir = os.path.join(output_dir, "拟合可视化")
    vis_files = _create_visualizations(all_results, n_dims, vis_dir)

    # 8. 导出数据
    export_dir = os.path.join(output_dir, "拟合导出数据")
    excel_path = _export_fitting_data(all_results, export_dir)

    # 9. 报告
    report_path = os.path.join(output_dir, "拟合分析报告.txt")
    _generate_report(all_results, report_path)

    # 10. 结构化摘要
    fitting_summary = []
    total_pts, total_rel_err = 0, 0.0
    for c in sorted(all_results.keys()):
        result = all_results[c]
        fr = result["fitting_results"]
        entry = {
            "cluster_id": c,
            "sample_size": result["sample_size"],
            "n_cleaned": result["not_cleaned_size"],
            "Di": result["Di"],
        }
        if "error" not in fr and "params" in fr:
            entry.update({
                "a": fr["params"]["a"],
                "b": fr["params"]["b"],
                "c": fr["params"]["c"],
                "r2": fr["r2"],
                "delta": fr.get("delta"),
                "func": fr["func"],
                "cleaned": fr["cleaned"],
            })
            # 有效指数: A = a * Π B_i^(Di_i * b) + c —— 各变量的等效幂次
            eff = {X_cols[j]: round(fr["params"]["b"] * result["Di"][j], 4)
                   for j in range(n_dims)}
            entry["effective_exponents"] = eff
            entry["effective_formula"] = "A ≈ %.4g * (%s) + %.4g" % (
                fr["params"]["a"],
                " * ".join("%s^%.3f" % (X_cols[j], fr["params"]["b"] * result["Di"][j])
                           for j in range(n_dims)),
                fr["params"]["c"],
            )
            # 加权累计综合平均相对误差（论文式2.19 δ*）
            if fr.get("delta") is not None:
                total_pts += result["sample_size"]
                total_rel_err += fr["delta"] * result["sample_size"]
        else:
            entry["error"] = fr.get("error", "未知")
        fitting_summary.append(entry)

    delta_star = round(total_rel_err / total_pts, 6) if total_pts else None

    # 11. 持久化拟合结果 JSON（供 predict 等下游工具复用）
    summary_path = os.path.join(output_dir, "fitting_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_clusters": n_clusters, "n_dims": n_dims,
            "x_cols": X_cols, "y_col": Y_col, "c_mode": c_mode,
            "delta_star": delta_star,
            "fitting_results": fitting_summary,
        }, f, ensure_ascii=False, indent=1)

    return {
        "n_clusters": n_clusters,
        "n_dims": n_dims,
        "x_cols": X_cols,
        "y_col": Y_col,
        "c_mode": c_mode,
        "delta_star": delta_star,
        "fitting_results": fitting_summary,
        "files": {
            "visualization_overview": vis_files["overview"],
            "visualization_details": vis_files["details"],
            "fitting_data_excel": excel_path,
            "fitting_results_json": summary_path,
            "report": report_path,
            "visualization_dir": vis_dir,
        },
    }


# ========== 预测（规律 -> 应用闭环） ==========

def predict(clustering_dir, params, fitting_json=None, k=7):
    """给定新的参数点（物理空间），用已完成的 DDDA 分区幂律模型预测响应量。

    Args:
        clustering_dir: cluster_analysis 的 output_dir（含 cluster_coordinates/ 子目录）。
        params: dict，{自变量列名: 物理空间值}，例如 {"ad": 1.0, "gamma": 10.0, "F": 8}。
        fitting_json: fitting_results.json 路径。默认依次在 clustering_dir 及其上级目录查找
            （run_full_pipeline 模式下两者同目录）。
        k: KNN 投票近邻数（log10 空间），用于判断参数点所属子域。

    Returns:
        dict: {ok, subdomain_id, predicted_value, formula, effective_exponents,
               knn_votes, input_params, hint}
    """
    import glob as _glob

    # 1) 定位 fitting_results.json
    candidates = []
    if fitting_json:
        candidates.append(fitting_json)
    else:
        candidates.append(os.path.join(clustering_dir, "fitting_results.json"))
        candidates.append(os.path.join(os.path.dirname(clustering_dir), "fitting_results.json"))
    fj = next((p for p in candidates if os.path.exists(p)), None)
    if fj is None:
        return {"ok": False, "error": "fitting_json_not_found",
                "message": "未找到 fitting_results.json，尝试路径: %s；请先运行 fit_power_function" % candidates}

    with open(fj, encoding="utf-8") as f:
        fit_data = json.load(f)
    x_cols = fit_data["x_cols"]
    y_col = fit_data["y_col"]

    # 2) 校验输入参数（params 允许是 x_cols 的超集——固定常数列等额外参数直接忽略）
    missing = [c for c in x_cols if c not in params]
    if missing:
        return {"ok": False, "error": "missing_params",
                "message": "缺少自变量参数: %s（需要: %s）" % (missing, x_cols)}
    ignored = [c for c in params if c not in x_cols]
    B = np.array([float(params[c]) for c in x_cols])
    if np.any(B <= 0):
        return {"ok": False, "error": "non_positive_params",
                "message": "自变量须为正数（物理空间）: %s" % dict(zip(x_cols, B))}
    X_log = np.log10(B)

    # 3) 加载各子域坐标（log10 空间）并 KNN 投票
    #    兼容两种布局: 直接在 clustering_dir 下, 或 clustering_dir/cluster_coordinates/ 子目录
    all_pts, all_labels = [], []
    for c in range(len(fit_data["fitting_results"])):
        fname = "cluster_%d_coordinates.csv" % c
        p = os.path.join(clustering_dir, fname)
        if not os.path.exists(p):
            p = os.path.join(clustering_dir, "cluster_coordinates", fname)
        if not os.path.exists(p):
            continue
        df_c = pd.read_csv(p)
        lack = [col for col in x_cols if col not in df_c.columns]
        if lack:
            return {"ok": False, "error": "fitting_cluster_mismatch",
                    "message": ("fitting_json 的自变量 %s 不在聚类坐标列 %s 中——"
                                "fitting_json 与 clustering_dir 属于不同的分析"
                                "（fitting_json: %s）。请用同一次分析的输出目录，"
                                "或显式传 fitting_json。" % (lack, list(df_c.columns), fj))}
        arr = df_c[x_cols].values
        all_pts.append(arr)
        all_labels.append(np.full(len(arr), c))
    if not all_pts:
        return {"ok": False, "error": "no_coordinates",
                "message": "cluster_coordinates 目录为空"}
    pts = np.vstack(all_pts)
    labels = np.concatenate(all_labels)

    d = np.linalg.norm(pts - X_log, axis=1)
    order = np.argsort(d)[:k]
    votes = {}
    for i in order:
        votes[int(labels[i])] = votes.get(int(labels[i]), 0) + 1
    sub_id = max(votes, key=votes.get)

    # 4) 用该子域的拟合公式预测
    entry = next((e for e in fit_data["fitting_results"] if e.get("cluster_id") == sub_id), None)
    if entry is None or "a" not in entry:
        return {"ok": False, "error": "subdomain_not_fitted",
                "message": "参数点归属于子域 %s，但该子域拟合失败，无法预测" % sub_id,
                "knn_votes": votes}
    Di = np.array(entry["Di"])
    C = float(np.prod(np.power(B, Di)))
    a, b, c0 = entry["a"], entry["b"], entry["c"]
    A = a * np.power(C, b) + c0
    if not np.isfinite(A) or A <= 0:
        return {"ok": True, "subdomain_id": sub_id, "predicted_value": None,
                "message": "预测值超出物理有效范围（A=%.4g），该点可能超出训练参数空间" % A,
                "knn_votes": votes, "input_params": params}

    return {
        "ok": True,
        "subdomain_id": sub_id,
        "predicted_value": round(float(A), 6),
        "predicted_col": y_col,
        "formula": entry.get("effective_formula") or entry.get("func"),
        "effective_exponents": entry.get("effective_exponents"),
        "delta_of_subdomain": entry.get("delta"),
        "knn_votes": {str(kk): vv for kk, vv in votes.items()},
        "nearest_distance_log10": round(float(d[order[0]]), 4),
        "input_params": params,
        "ignored_params": ignored,
        "hint": "subdomain_id 对应聚类分析中的子域编号；nearest_distance_log10 偏大说明该点远离训练数据",
    }
