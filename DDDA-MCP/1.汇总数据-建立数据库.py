import pandas as pd
import os

# ============================================================
# 汇总 DDDA2nd-1 ~ DDDA2nd-12 的 720h 结果
# 输出: 汇总数据-720h.xlsx
# ============================================================

data_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(data_dir, '汇总数据-720h.xlsx')

all_rows = []

for i in range(1, 5):
    file_path = os.path.join(data_dir, f'DDDA2nd-{i}.xlsx')
    print(f'正在处理: DDDA2nd-{i}.xlsx')

    # 读取原始数据，跳过前5行（metadata），第5行是表头
    df = pd.read_excel(file_path, header=None, skiprows=4, dtype=str)

    # 第5行（即skiprows=4后的第1行）是列名
    columns = df.iloc[0].tolist()
    df = df.iloc[1:].reset_index(drop=True)
    df.columns = columns

    # 重命名列方便处理
    col_names = list(df.columns)

    # --- 单位换算（基于列名中的单位信息）---
    # 列名示例: 'k (m/s)', 'E0 (MPa)', 'p (kPa)', '时间 (h)'
    # 统一换算到: k -> m/s, E0 -> MPa, p -> kPa, 时间 -> h
    # 如果列名中的单位不同，进行换算

    for ci, cn in enumerate(col_names):
        cn_str = str(cn)

        # k: 渗透系数，统一换算到 m/s
        if cn_str.startswith('k'):
            if 'cm/s' in cn_str:
                # cm/s -> m/s: 除以100
                df.iloc[:, ci] = df.iloc[:, ci].astype(float) / 100.0
                col_names[ci] = 'k (m/s)'
                print(f'  -> k 单位 cm/s 已换算为 m/s')
            elif 'm/d' in cn_str.lower() or 'm/day' in cn_str.lower():
                # m/day -> m/s: 除以86400
                df.iloc[:, ci] = df.iloc[:, ci].astype(float) / 86400.0
                col_names[ci] = 'k (m/s)'
                print(f'  -> k 单位 m/day 已换算为 m/s')
            else:
                col_names[ci] = 'k (m/s)'  # 统一列名

        # E0: 弹性模量，统一换算到 MPa
        elif cn_str.startswith('E0'):
            if 'Pa' in cn_str and 'MPa' not in cn_str and 'kPa' not in cn_str and 'GPa' not in cn_str:
                # Pa -> MPa: 除以1e6
                df.iloc[:, ci] = df.iloc[:, ci].astype(float) / 1e6
                col_names[ci] = 'E0 (MPa)'
                print(f'  -> E0 单位 Pa 已换算为 MPa')
            elif 'kPa' in cn_str:
                # kPa -> MPa: 除以1000
                df.iloc[:, ci] = df.iloc[:, ci].astype(float) / 1000.0
                col_names[ci] = 'E0 (MPa)'
                print(f'  -> E0 单位 kPa 已换算为 MPa')
            elif 'GPa' in cn_str:
                # GPa -> MPa: 乘以1000
                df.iloc[:, ci] = df.iloc[:, ci].astype(float) * 1000.0
                col_names[ci] = 'E0 (MPa)'
                print(f'  -> E0 单位 GPa 已换算为 MPa')
            else:
                col_names[ci] = 'E0 (MPa)'

        # p: 围压，统一换算到 kPa
        elif cn_str.startswith('p'):
            if 'Pa' in cn_str and 'kPa' not in cn_str and 'MPa' not in cn_str:
                # Pa -> kPa: 除以1000
                df.iloc[:, ci] = df.iloc[:, ci].astype(float) / 1000.0
                col_names[ci] = 'p (kPa)'
                print(f'  -> p 单位 Pa 已换算为 kPa')
            elif 'MPa' in cn_str:
                # MPa -> kPa: 乘以1000
                df.iloc[:, ci] = df.iloc[:, ci].astype(float) * 1000.0
                col_names[ci] = 'p (kPa)'
                print(f'  -> p 单位 MPa 已换算为 kPa')
            else:
                col_names[ci] = 'p (kPa)'

        # 时间: 统一换算到 h
        elif '时间' in cn_str or 'Time' in cn_str.lower():
            if 'min' in cn_str.lower():
                # min -> h: 除以60
                df.iloc[:, ci] = df.iloc[:, ci].astype(float) / 60.0
                col_names[ci] = '时间 (h)'
                print(f'  -> 时间单位 min 已换算为 h')
            elif 's' in cn_str and 'min' not in cn_str.lower() and 'h' not in cn_str:
                # s -> h: 除以3600
                df.iloc[:, ci] = df.iloc[:, ci].astype(float) / 3600.0
                col_names[ci] = '时间 (h)'
                print(f'  -> 时间单位 s 已换算为 h')
            elif 'day' in cn_str.lower() or 'd' == cn_str.strip().lower():
                # day -> h: 乘以24
                df.iloc[:, ci] = df.iloc[:, ci].astype(float) * 24.0
                col_names[ci] = '时间 (h)'
                print(f'  -> 时间单位 day 已换算为 h')
            else:
                col_names[ci] = '时间 (h)'

        else:
            # 保留原列名
            pass

    df.columns = col_names

    # 转换为数值类型
    df = df.apply(pd.to_numeric, errors='coerce')

    # --- 筛选时间为 720h 的数据 ---
    # 找到时间列
    time_col = None
    for cn in col_names:
        if '时间' in str(cn) or 'Time' in str(cn):
            time_col = cn
            break

    if time_col is None:
        print(f'  ⚠ 未找到时间列，跳过')
        continue

    # 筛选 720h 的数据（考虑浮点误差），用 .copy() 避免 SettingWithCopyWarning
    df_720 = df[(df[time_col] >= 719.999) & (df[time_col] <= 720.001)].copy()

    if len(df_720) == 0:
        print(f'  ⚠ 未找到时间=720h的数据')
        continue

    print(f'  找到 {len(df_720)} 条时间=720h的数据')

    # 添加来源文件信息
    df_720['来源文件'] = f'DDDA2nd-{i}'

    # 提取需要的列：6个自变量 + 因变量（去掉时间列）
    # 6个自变量是前6列: ns, k, E0, p, fc, ue
    # 因变量是第8列（第7列是时间）
    # 但为稳健计，通过列名定位

    # 识别6个自变量
    var_names_map = {
        'ns': 'ns',
        'k': 'k (m/s)',
        'E0': 'E0 (MPa)',
        'p': 'p (kPa)',
        'fc': 'fc',
        'ue': 'ue'
    }

    selected_cols = []
    for short_name, full_name in var_names_map.items():
        for cn in col_names:
            if str(cn).strip() == full_name or str(cn).strip().startswith(short_name):
                if cn not in selected_cols:
                    selected_cols.append(cn)
                    break

    # 因变量（第8列）- 不是时间且不是6个自变量的那列
    indep_names = set(selected_cols)
    all_cols = set(col_names)
    dep_col = None
    for cn in all_cols:
        if cn not in indep_names and cn != time_col and '来源' not in str(cn):
            dep_col = cn
            break

    if dep_col is None:
        print(f'  ⚠ 未找到因变量列')
        continue

    # 统一因变量列名为 "void"
    df_720.rename(columns={dep_col: 'void'}, inplace=True)

    # 只选取需要的列
    result_cols = selected_cols + ['void', '来源文件']
    df_result = df_720[result_cols].copy()

    all_rows.append(df_result)

    print(f'  → 提取 {len(df_result)} 条记录')

# --- 合并所有结果 ---
if all_rows:
    final_df = pd.concat(all_rows, ignore_index=True)
    final_df.to_excel(output_file, index=False)
    print(f'\n✅ 汇总完成！共 {len(final_df)} 条记录')
    print(f'   输出文件: {output_file}')
    print(f'   列名: {list(final_df.columns)}')
else:
    print('\n❌ 未找到任何有效数据')
