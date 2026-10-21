"""
基于偏导数的聚类拟合函数（带数据清洗）
利用脚本4预计算的聚类标签直接分类，然后在真实物理空间中拟合幂函数 A = a * C^b + c

流程:
1. 读取偏导数数据文件
2. 加载脚本4预计算的聚类标签（无需重新计算特征和FCM分配）
3. 从聚类结果中加载每个子空间的全局特征向量，提取Di值
4. 在真实物理空间中计算 C = B1^D1 * B2^D2 * ... * B6^D6
5. 对每个子空间拟合 A = a * C^b + c（含数据清洗）
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
import os
import warnings
import time

warnings.filterwarnings('ignore', category=RuntimeWarning)

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stixsans'
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['font.size'] = 12

base_dir = os.path.dirname(os.path.abspath(__file__))

X_cols = ["X1", "X2", "X3", "X4", "X5", "X6"]
G_cols = ["dY_dX1", "dY_dX2", "dY_dX3", "dY_dX4", "dY_dX5", "dY_dX6"]
n_dims = 6


def load_derivative_data():
    """读取偏导数数据文件"""
    data_file = os.path.join(base_dir, "对数空间数据-720h_偏导数.xlsx")
    print(f"读取偏导数数据: {data_file}")
    df = pd.read_excel(data_file)

    missing = [c for c in X_cols + G_cols + ["Y"] if c not in df.columns]
    if missing:
        raise ValueError(f"数据文件缺少列: {missing}")

    X = df[X_cols].values  # 对数空间 (N, 6)
    Y = df["Y"].values      # 对数空间 (N,)
    pFpX = df[G_cols].values  # 偏导数 (N, 6)

    print(f"数据规模: {len(X)} 个点, {n_dims} 维")
    return df, X, Y, pFpX


def load_cluster_labels():
    """
    直接加载脚本4保存的聚类标签，无需重新计算特征和FCM分配
    返回标签数组和聚类数量
    """
    components_dir = os.path.join(base_dir, "clustering_components")
    labels_path = os.path.join(components_dir, "cluster_labels.npy")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"聚类标签文件不存在: {labels_path}\n"
            f"请先运行 4.聚类脚本 - FCM.py 完成聚类并保存标签"
        )

    labels = np.load(labels_path)
    n_clusters = len(np.unique(labels[labels >= 0]))

    print(f"已加载聚类标签: {labels_path}")
    print(f"标签数量: {len(labels)}, 聚类数: {n_clusters}")
    for c in range(n_clusters):
        count = np.sum(labels == c)
        print(f"  子空间 {c}: {count} 个点 ({count/len(labels)*100:.1f}%)")

    return labels, n_clusters


def load_eigenvectors(cluster_id):
    """从聚类结果Excel中加载指定子空间的特征向量"""
    results_dir = os.path.join(base_dir, "advanced_clustering_results")
    file_path = os.path.join(results_dir, "clustering_eigen_analysis.xlsx")
    sheet_name = f"聚类{cluster_id}_特征向量"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"聚类结果文件不存在: {file_path}\n请先运行 4.1网格聚类脚本 - FCM.py")

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    if 'component_index' in df.columns:
        eigenvectors = df.drop('component_index', axis=1).values
    else:
        eigenvectors = df.values

    # 转置后形状为 (n_eigenvectors, n_components)，行=特征向量，列=分量
    eigenvectors = eigenvectors.T
    return eigenvectors


def get_Di_values(eigenvectors):
    """
    从特征向量矩阵中提取Di值
    eigenvectors 形状: (n_eigenvectors, n_components)
    Di = 主导特征向量（最大特征值对应，即第0行）
    如果|Di| < 0.05，则令Di=0
    """
    Di = eigenvectors[0, :].copy()  # 第0行 = 主导特征向量
    Di[np.abs(Di) < 0.05] = 0
    return Di


def calculate_C(B, Di):
    """计算 C = B1^D1 * B2^D2 * ... * B6^D6，使用对数避免溢出"""
    log_C = 0.0
    for i in range(len(B)):
        if B[i] > 0 and Di[i] != 0:
            log_C += Di[i] * np.log(B[i])
    return np.exp(log_C)


def power_func(C, a, b, c=1):
    """幂函数: A = a * C^b + c"""
    return a * np.power(C, b) + c


def robust_fit_models(C_values, A_values, max_clean_ratio=0.1):
    """
    幂函数拟合，含数据清洗
    返回: (results, C_cleaned, A_cleaned, C_not_cleaned, A_not_cleaned)
    """
    results = {}

    print(f"  幂函数拟合 - C值数量: {len(C_values)}, A值数量: {len(A_values)}")
    print(f"  C值范围: [{np.min(C_values):.6e}, {np.max(C_values):.6e}]")
    print(f"  A值范围: [{np.min(A_values):.6e}, {np.max(A_values):.6e}]")

    try:
        C_positive = C_values[C_values > 0]
        A_positive = A_values[C_values > 0]

        if len(C_positive) < 10:
            results['power'] = {'error': f'正数C值不足: {len(C_positive)}'}
            return results, C_values, A_values, np.array([]), np.array([])

        # 初步拟合
        fixed_c_func = lambda C, a, b: power_func(C, a, b, 1)
        popt_initial, _ = curve_fit(fixed_c_func, C_positive, A_positive, maxfev=20000)
        popt_initial_with_c = [popt_initial[0], popt_initial[1], 1.0]

        # 计算残差
        A_pred_initial = power_func(C_positive, *popt_initial_with_c)
        residuals = np.abs(A_positive - A_pred_initial)

        # 确定清洗数量
        n_points = len(C_positive)
        n_to_remove = int(np.floor(n_points * max_clean_ratio))

        if n_to_remove > 0:
            print(f"  清洗 {n_to_remove} 个残差最大的数据点 ({max_clean_ratio*100}%)")
            sorted_indices = np.argsort(residuals)
            keep_indices = sorted_indices[:-n_to_remove]
            remove_indices = sorted_indices[-n_to_remove:]

            C_cleaned = C_positive[keep_indices]
            A_cleaned = A_positive[keep_indices]
            C_not_cleaned = C_positive[remove_indices]
            A_not_cleaned = A_positive[remove_indices]

            if len(C_cleaned) > 10:
                popt, _ = curve_fit(fixed_c_func, C_cleaned, A_cleaned, maxfev=20000)
                popt_with_c = [popt[0], popt[1], 1.0]
                A_pred = power_func(C_cleaned, *popt_with_c)
                r2 = 1 - np.sum((A_cleaned - A_pred)**2) / np.sum((A_cleaned - np.mean(A_cleaned))**2)

                results['power'] = {
                    'params': popt_with_c,
                    'r2': r2,
                    'func': f'A = {popt_with_c[0]:.6f} * C^{popt_with_c[1]:.6f} + {popt_with_c[2]:.6f}',
                    'cleaned': True,
                    'n_initial': n_points,
                    'n_cleaned': n_to_remove
                }
                print(f"  拟合成功（已清洗）: {results['power']['func']}, R2={r2:.6f}")
                return results, C_cleaned, A_cleaned, C_not_cleaned, A_not_cleaned
            else:
                results['power'] = {'error': f'清洗后数据不足: {len(C_cleaned)}'}
                return results, C_positive, A_positive, np.array([]), np.array([])
        else:
            # n_to_remove == 0，直接复用初步拟合结果
            popt_with_c = popt_initial_with_c
            A_pred = power_func(C_positive, *popt_with_c)
            r2 = 1 - np.sum((A_positive - A_pred)**2) / np.sum((A_positive - np.mean(A_positive))**2)

            results['power'] = {
                'params': popt_with_c,
                'r2': r2,
                'func': f'A = {popt_with_c[0]:.6f} * C^{popt_with_c[1]:.6f} + {popt_with_c[2]:.6f}',
                'cleaned': False
            }
            print(f"  拟合成功（未清洗）: {results['power']['func']}, R2={r2:.6f}")
            return results, C_positive, A_positive, np.array([]), np.array([])

    except Exception as e:
        results['power'] = {'error': str(e)}
        return results, C_values, A_values, np.array([]), np.array([])


def process_and_fit(sample_size=None):
    """
    主处理流程：
    1. 读取偏导数数据
    2. 加载脚本4预计算的聚类标签
    3. 在真实物理空间中拟合幂函数
    """
    print("=" * 60)
    print("开始聚类拟合分析")
    print("=" * 60)

    # 1. 读取数据
    df, X_log, Y_log, pFpX = load_derivative_data()

    # 2. 加载预计算的聚类标签
    labels, n_clusters = load_cluster_labels()

    # 可选抽样
    if sample_size and len(df) > sample_size:
        print(f"数据抽样: {len(df)} → {sample_size}")
        idx = np.random.choice(len(df), sample_size, replace=False)
        idx.sort()
        X_log = X_log[idx]
        Y_log = Y_log[idx]
        labels = labels[idx]  # 同步抽样标签

    # 3. 转换到真实物理空间
    B_data = 10.0 ** X_log  # B_i = 10^X_i（物理空间）
    A_data = 10.0 ** Y_log  # A = 10^Y（物理空间）

    # 4. 加载各聚类的Di值
    cluster_Di = {}
    for c in range(n_clusters):
        try:
            eigvecs = load_eigenvectors(c)
            cluster_Di[c] = get_Di_values(eigvecs)
            print(f"子空间 {c} 的 Di 值: {cluster_Di[c]}")
        except Exception as e:
            print(f"警告: 无法加载子空间 {c} 的特征向量: {e}")

    # 5. 计算每个点的C值并按子空间分组
    cluster_data = {c: {'C': [], 'A': [], 'B': [], 'X': []} for c in range(n_clusters)}

    for i in range(len(X_log)):
        c = labels[i]
        if c not in cluster_Di:
            continue

        C = calculate_C(B_data[i], cluster_Di[c])
        if C > 0 and np.isfinite(C) and A_data[i] > 0 and np.isfinite(A_data[i]):
            cluster_data[c]['C'].append(C)
            cluster_data[c]['A'].append(A_data[i])
            cluster_data[c]['B'].append(B_data[i])
            cluster_data[c]['X'].append(X_log[i])

    # 6. 对每个子空间拟合
    all_results = {}
    for c in range(n_clusters):
        C_arr = np.array(cluster_data[c]['C'])
        A_arr = np.array(cluster_data[c]['A'])
        B_arr = np.array(cluster_data[c]['B'])
        X_arr = np.array(cluster_data[c]['X'])

        if len(C_arr) < 10:
            print(f"子空间 {c}: 有效数据不足 ({len(C_arr)})，跳过")
            continue

        print(f"\n子空间 {c}: {len(C_arr)} 个有效数据点")
        print(f"  Di = {cluster_Di[c]}")

        # 清洗规则：统一10%
        max_clean_ratio = 0.1

        fitting_results, C_cleaned, A_cleaned, C_removed, A_removed = robust_fit_models(
            C_arr, A_arr, max_clean_ratio=max_clean_ratio)

        all_results[c] = {
            'cluster_id': c,
            'Di': cluster_Di[c],
            'C_values': C_cleaned,
            'A_values': A_cleaned,
            'C_not_cleaned': C_removed,
            'A_not_cleaned': A_removed,
            'B_values': B_arr,
            'X_values': X_arr,
            'fitting_results': fitting_results,
            'sample_size': len(C_cleaned),
            'not_cleaned_size': len(C_removed)
        }

    return all_results


def create_visualizations(all_results):
    """创建可视化图表"""
    print("\n生成可视化图表...")
    vis_dir = os.path.join(base_dir, "拟合可视化")
    os.makedirs(vis_dir, exist_ok=True)

    n_clusters = len(all_results)
    cluster_ids = sorted(all_results.keys())

    # 1. 总览图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 样本分布
    ax = axes[0, 0]
    sizes = [all_results[c]['sample_size'] for c in cluster_ids]
    ax.bar(cluster_ids, sizes, color=plt.cm.Set3(np.linspace(0, 1, n_clusters)))
    ax.set_title('各子空间样本数量')
    ax.set_xlabel('子空间编号')
    ax.set_ylabel('样本数量')
    for i, v in enumerate(sizes):
        ax.text(cluster_ids[i], v, str(v), ha='center', va='bottom')

    # R2值
    ax = axes[0, 1]
    r2_values = []
    for c in cluster_ids:
        fr = all_results[c]['fitting_results']
        r2 = fr['power']['r2'] if 'power' in fr and 'error' not in fr['power'] else 0
        r2_values.append(r2)
    colors = ['green' if r2 > 0.4 else 'orange' if r2 > 0.2 else 'red' for r2 in r2_values]
    bars = ax.bar(cluster_ids, r2_values, color=colors, alpha=0.7)
    ax.set_title('各子空间幂函数拟合R²')
    ax.set_xlabel('子空间编号')
    ax.set_ylabel('R²值')
    ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='良好拟合')
    ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='一般拟合')
    ax.legend()
    for bar, r2 in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{r2:.3f}', ha='center', va='bottom', fontsize=9)

    # Di值热力图
    ax = axes[1, 0]
    Di_matrix = np.array([all_results[c]['Di'] for c in cluster_ids])
    im = ax.imshow(Di_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(im, ax=ax, label='Di值')
    ax.set_title('各子空间Di值热力图')
    ax.set_xlabel('特征维度')
    ax.set_ylabel('子空间编号')
    ax.set_xticks(range(n_dims))
    ax.set_xticklabels([f'D{i+1}' for i in range(n_dims)])
    ax.set_yticks(range(len(cluster_ids)))
    ax.set_yticklabels(cluster_ids)

    # C-A散点图
    ax = axes[1, 1]
    colors_scatter = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    for (c, result), color in zip(all_results.items(), colors_scatter):
        if len(result['C_values']) > 0:
            ax.scatter(result['C_values'], result['A_values'], color=color,
                      alpha=0.6, s=10, label=f'子空间{c}')
    ax.set_title('各子空间C-A散点图')
    ax.set_xlabel('C值')
    ax.set_ylabel('A值')
    ax.legend()
    ax.set_yscale('log')
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'fitting_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 每个子空间的详细拟合图
    for c in cluster_ids:
        result = all_results[c]
        C_vals = result['C_values']
        A_vals = result['A_values']

        fig, ax = plt.subplots(figsize=(10, 8))

        # 散点
        ax.scatter(C_vals, A_vals, alpha=0.6, s=30, color='blue', label='数据点')

        # 被清洗的点
        if len(result['C_not_cleaned']) > 0:
            ax.scatter(result['C_not_cleaned'], result['A_not_cleaned'],
                      alpha=0.3, s=20, color='gray', marker='x', label='已清洗点')

        # 拟合曲线
        fr = result['fitting_results']
        if 'power' in fr and 'error' not in fr['power']:
            params = fr['power']['params']
            r2 = fr['power']['r2']
            C_range = np.logspace(np.log10(np.min(C_vals)), np.log10(np.max(C_vals)), 500)
            A_pred = power_func(C_range, *params)
            ax.plot(C_range, A_pred, color='red', linewidth=2,
                   label=f'幂函数 (R²={r2:.4f})')

        ax.set_title(f'子空间 {c} 拟合曲线\n样本数: {result["sample_size"]}')
        ax.set_xlabel('C值（物理空间）')
        ax.set_ylabel('A值（物理空间）')
        ax.legend()
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid(True, which='both', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'subspace_{c}_fitting.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 3. 导出数据到Excel
    export_dir = os.path.join(base_dir, "拟合导出数据")
    os.makedirs(export_dir, exist_ok=True)

    with pd.ExcelWriter(os.path.join(export_dir, 'fitting_data.xlsx'), engine='openpyxl') as writer:
        for c in cluster_ids:
            result = all_results[c]
            C_vals = result['C_values']
            A_vals = result['A_values']

            # 实际数据
            df_data = pd.DataFrame({'C值': C_vals, 'A值': A_vals})
            df_data.to_excel(writer, sheet_name=f'子空间{c}_数据', index=False)

            # 拟合曲线
            fr = result['fitting_results']
            if 'power' in fr and 'error' not in fr['power']:
                params = fr['power']['params']
                C_range = np.logspace(np.log10(np.min(C_vals)), np.log10(np.max(C_vals)), 500)
                A_pred = power_func(C_range, *params)
                df_curve = pd.DataFrame({'C值': C_range, '拟合A值': A_pred})
                df_curve.to_excel(writer, sheet_name=f'子空间{c}_拟合曲线', index=False)

    print(f"可视化图表已保存至: {vis_dir}")
    print(f"拟合数据已导出至: {export_dir}")
    return vis_dir


def generate_report(all_results):
    """生成分析报告"""
    report_path = os.path.join(base_dir, "拟合分析报告.txt")
    cluster_ids = sorted(all_results.keys())

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("聚类幂函数拟合分析报告\n")
        f.write("=" * 70 + "\n\n")
        f.write("方法:\n")
        f.write("- 从偏导数计算特征矩阵空间，分配到已有3个子空间\n")
        f.write("- 在真实物理空间中拟合: C = B1^D1 * ... * B6^D6\n")
        f.write("- 幂函数模型: A = a * C^b + c\n")
        f.write("- 数据清洗: 去除残差最大的10%数据点\n\n")

        f.write("拟合结果:\n")
        f.write("-" * 70 + "\n")

        best_r2 = -1
        best_cluster = None

        for c in cluster_ids:
            result = all_results[c]
            f.write(f"\n子空间 {c}:\n")
            f.write(f"  样本数量: {result['sample_size']}\n")
            if result['not_cleaned_size'] > 0:
                f.write(f"  清洗数据点: {result['not_cleaned_size']}\n")
            f.write(f"  Di值: {result['Di']}\n")

            fr = result['fitting_results']
            if 'power' in fr and 'error' not in fr['power']:
                f.write(f"  拟合函数: {fr['power']['func']}\n")
                f.write(f"  R² = {fr['power']['r2']:.6f}\n")
                if fr['power']['r2'] > best_r2:
                    best_r2 = fr['power']['r2']
                    best_cluster = c
            else:
                f.write(f"  拟合失败: {fr['power'].get('error', '未知')}\n")

        f.write(f"\n{'='*70}\n")
        f.write("拟合效果评估:\n")
        for c in cluster_ids:
            fr = all_results[c]['fitting_results']
            r2 = fr['power']['r2'] if 'power' in fr and 'error' not in fr['power'] else -1
            if r2 > 0.4:
                eval_str = "良好拟合"
            elif r2 > 0.2:
                eval_str = "一般拟合"
            else:
                eval_str = "较差拟合"
            f.write(f"  子空间 {c}: {eval_str} (R² = {r2:.6f})\n")

        if best_cluster is not None:
            f.write(f"\n最佳拟合: 子空间 {best_cluster} (R² = {best_r2:.6f})\n")

        # 各子空间数据点示例
        f.write(f"\n{'='*70}\n")
        f.write("各子空间数据点示例（物理空间）:\n")
        f.write("-" * 70 + "\n")

        for c in cluster_ids:
            result = all_results[c]
            f.write(f"\n子空间 {c} (样本数: {result['sample_size']}):\n")
            if result['sample_size'] >= 3:
                indices = np.random.choice(result['sample_size'], min(3, result['sample_size']), replace=False)
                for k, idx in enumerate(indices):
                    B_vals = result['B_values'][idx]
                    A_val = result['A_values'][idx]
                    B_str = "  ".join([f"{b:.4f}" for b in B_vals])
                    f.write(f"  #{k+1}: B=[{B_str}]  A={A_val:.6f}\n")

    print(f"分析报告已保存至: {report_path}")


def main():
    """主函数"""
    total_start = time.time()

    # 处理数据并拟合
    all_results = process_and_fit(sample_size=None)

    if not all_results:
        print("没有获得有效的拟合结果")
        return

    # 输出结果摘要
    print("\n" + "=" * 70)
    print("拟合结果摘要")
    print("=" * 70)

    for c in sorted(all_results.keys()):
        result = all_results[c]
        print(f"\n子空间 {c} (样本数: {result['sample_size']})")
        print(f"  Di: {result['Di']}")
        fr = result['fitting_results']
        if 'power' in fr and 'error' not in fr['power']:
            print(f"  {fr['power']['func']}")
            print(f"  R² = {fr['power']['r2']:.6f}")
        else:
            print(f"  拟合失败: {fr['power'].get('error', '')}")

    # 可视化
    create_visualizations(all_results)

    # 报告
    generate_report(all_results)

    total_time = time.time() - total_start
    print(f"\n总耗时: {total_time:.1f}秒")

    return all_results


if __name__ == "__main__":
    all_results = main()
