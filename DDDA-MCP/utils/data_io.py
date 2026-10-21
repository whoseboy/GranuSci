"""数据读写与校验工具。

提供 Excel 读写、列名推断、数据质量校验等通用函数。
"""

import os

import numpy as np
import pandas as pd

from config import get_config


# ========== 列推断 ==========

def resolve_columns(df, n_indep=None, x_cols=None, y_col=None):
    """根据 n_indep 或显式列名，推断自变量列和因变量列。

    Args:
        df: pandas DataFrame
        n_indep: 自变量列数（默认 总列数-1）
        x_cols: 显式自变量列名列表（提供后覆盖 n_indep）
        y_col: 显式因变量列名（默认末列）

    Returns:
        (x_cols, y_col): 自变量列名列表和因变量列名
    """
    all_cols = list(df.columns)
    if x_cols and y_col:
        return list(x_cols), y_col
    if n_indep is None:
        n_indep = len(all_cols) - 1
    if n_indep < 1:
        raise ValueError(f"自变量列数必须 >= 1，当前为 {n_indep}")
    if y_col is None:
        y_col = all_cols[-1]
    if x_cols is None:
        x_cols = all_cols[:n_indep]
    x_cols = [c for c in x_cols if c != y_col]
    if len(x_cols) != n_indep:
        raise ValueError(
            f"自变量列数 {len(x_cols)} 与指定 n_indep={n_indep} 不一致")
    return list(x_cols), y_col


def infer_columns_from_derivative_file(df, n_indep=None):
    """从含偏导数的 DataFrame 中推断 X/Y/G 列。

    偏导数列通过 "dY_d" 前缀识别。

    Returns:
        (x_cols, y_col, g_cols)
    """
    all_cols = list(df.columns)
    g_cols = [c for c in all_cols if c.startswith("dY_d")]
    non_g_cols = [c for c in all_cols if c not in g_cols]
    if n_indep is None:
        n_indep = len(non_g_cols) - 1
    if n_indep < 1:
        raise ValueError(f"自变量列数必须 >= 1，当前为 {n_indep}")
    x_cols = non_g_cols[:n_indep]
    y_col = non_g_cols[-1]
    return x_cols, y_col, g_cols


# ========== 输入解析 ==========

def resolve_input(file_path):
    """输入文件路径原样返回（内置数据集机制已移除——分析服务只做通用方法，
    不携带任何特定研究的数据；数据一律经 input_file 路径或 data 行内参数传入）。"""
    return file_path


# ========== 数据读取 ==========

def read_excel(file_path):
    """读取 Excel 文件并校验存在性。

    Returns:
        pandas DataFrame
    """
    file_path = resolve_input(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    return pd.read_excel(file_path)


def read_data(file_path):
    """按扩展名读取数据文件（.xlsx/.xls -> Excel, .csv -> CSV），
    供上游工具（如仿真服务的 CSV 输出）直接对接。"""
    file_path = resolve_input(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    if str(file_path).lower().endswith(".csv"):
        return pd.read_csv(file_path)
    return pd.read_excel(file_path)


def materialize_inline_data(data, workdir_name="uploads"):
    """把行内数据（{"columns": [...], "rows": [[...], ...]}）落盘为 Excel 并返回路径。

    平台智能体无法向 MCP 服务器上传文件，但可以把数据行直接放进工具参数。
    """
    import pandas as pd  # noqa: F811
    if isinstance(data, str):
        # 平台工作流/部分客户端把工具参数全序列化为字符串（2026-10-16 托管实测）
        import json as _json
        try:
            data = _json.loads(data)
        except Exception:
            raise ValueError('data 参数为字符串时须是 {columns, rows} 的 JSON 串')
    if not isinstance(data, dict) or "columns" not in data or "rows" not in data:
        raise ValueError('data 参数须为 dict: {columns: [...], rows: [[...], ...]} 结构')
    cols = data["columns"]
    rows = data["rows"]
    if not cols or not rows:
        raise ValueError("columns/rows 不能为空")
    if any(len(r) != len(cols) for r in rows):
        raise ValueError(f"存在行长度与列数({len(cols)})不一致的数据行")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 平台托管容器根文件系统只读（仅 /tmp 可写，2026-10-16 实测 PermissionError）——env 覆盖
    updir = os.environ.get("DDDA_UPLOAD_DIR") or os.path.join(base, workdir_name)
    os.makedirs(updir, exist_ok=True)
    import time as _t
    out = os.path.join(updir, "inline_%s.xlsx" % _t.strftime("%Y%m%d_%H%M%S"))
    pd.DataFrame(rows, columns=cols).to_excel(out, index=False)
    return out


# ========== 数据校验 ==========

def validate_data(df, x_cols, y_col):
    """校验数据质量。

    检查项:
      - 必需列是否存在
      - 是否存在 NaN / Inf
      - 是否存在常数列（std=0）
      - 数值类型强制转换

    Returns:
        (valid_df, warnings): 清洗后的 DataFrame 与警告信息列表
    """
    warnings = []

    # 1. 必需列检查
    missing = [c for c in x_cols + [y_col] if c not in df.columns]
    if missing:
        raise ValueError(f"数据文件缺少列: {missing}")

    # 2. 数值类型强制转换
    data = df[x_cols + [y_col]].apply(pd.to_numeric, errors="coerce")

    # 3. NaN / Inf 检查
    n_before = len(data)
    n_nan = data.isna().sum().sum()
    if n_nan > 0:
        data = data.dropna()
        warnings.append(f"已删除 {n_nan} 个包含 NaN/文本 的单元格所在行")

    # 4. 常数列检查
    for col in x_cols:
        std = data[col].std()
        if np.isnan(std) or std == 0:
            warnings.append(f"自变量列 '{col}' 为常数列 (std=0)，可能导致导数不稳定")

    # 5. 行数不足检查
    if len(data) < 3:
        raise ValueError(f"数据行数不足: {len(data)} < 3")

    if n_before - len(data) > 0:
        warnings.append(f"有效数据 {len(data)}/{n_before} 行")

    return data, warnings


# ========== 输出路径工具 ==========

def default_output_path(input_file, suffix, ext=".xlsx"):
    """在输入文件同目录生成默认输出路径。

    Args:
        input_file: 输入文件路径
        suffix: 追加到文件名的后缀，如 "_偏导数"
        ext: 输出扩展名

    Returns:
        str: 输出文件路径
    """
    base_dir = os.path.dirname(os.path.abspath(input_file))
    stem = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(base_dir, f"{stem}{suffix}{ext}")


def ensure_dir(path):
    """确保目录存在。"""
    os.makedirs(path, exist_ok=True)
    return path


def download_to_server(url, filename=None, max_bytes=52428800, workdir_name="uploads"):
    """从 URL 下载数据文件到服务器本地，返回本地路径（可直接传给其他工具）。

    MCP 协议没有文件传输原语，跨系统数据引入采用"传 URL、服务器自取"模式。
    仅支持 http/https，默认大小上限 50MB。
    """
    import urllib.request
    import urllib.parse
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise ValueError("url 须以 http:// 或 https:// 开头")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    updir = os.environ.get("DDDA_UPLOAD_DIR") or os.path.join(base, workdir_name)
    os.makedirs(updir, exist_ok=True)
    if not filename:
        name = os.path.basename(urllib.parse.urlparse(url).path) or "download.bin"
        import time as _t
        filename = "%s_%s" % (_t.strftime("%Y%m%d_%H%M%S"), name)
    out = os.path.join(updir, os.path.basename(filename))

    def _progress(blocks, block_size, total):
        if blocks * block_size > max_bytes:
            raise ValueError("文件超过大小上限 %d MB" % (max_bytes // 1048576))

    urllib.request.urlretrieve(url, out, reporthook=_progress)
    if os.path.getsize(out) == 0:
        raise ValueError("下载内容为空，请检查 URL 是否可公开访问")
    return out
