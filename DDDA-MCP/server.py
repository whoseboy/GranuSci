"""DDDA-MCP: 数据驱动维度分析 MCP 服务。

提供四个工具:
  1. compute_derivatives  - KDE 自适应光滑偏导数计算
  2. cluster_analysis     - subclust+FCM 复合聚类
  3. fit_power_function   - 物理空间幂函数拟合（含离群清洗）
  4. run_full_pipeline    - 顺序执行以上三步

输入数据要求:
  - Excel 格式 (.xlsx)
  - 前 N-1 列为自变量，最后一列为因变量（列数与列名可任意）
  - 所有数据固定在 log10 对数空间（即已对原始物理量取以10为底的对数）
  - 列名通过第一行表头识别；偏导数工具输出列以 "dY_d" 为前缀
  - 后续聚类与拟合工具自动识别 "dY_d" 前缀列作为偏导数
"""

import os
import sys
import traceback

# 确保项目根目录在 sys.path 中（无论从哪个工作目录启动）
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mcp.server.fastmcp import FastMCP  # noqa: E402

from core.derivatives import compute_derivatives  # noqa: E402
from utils.data_io import materialize_inline_data  # noqa: E402
from core.clustering import AdvancedClusteringAnalysis  # noqa: E402
from core.fitting import fit_power_function, predict  # noqa: E402
from auth_middleware import TokenAuth  # noqa: E402


SERVER_NAME = "ddda-mcp"
SERVER_VERSION = "1.0.0"
SERVER_DESCRIPTION = (
    "数据驱动维度分析（DDDA）工具集：光滑偏导数计算、FCM 聚类、幂函数拟合。"
    "所有输入数据须为 log10 对数空间下的 Excel 表格，前 N-1 列为自变量，末列为因变量。"
)
SERVER_INSTRUCTIONS = __doc__

mcp = FastMCP(
    name=SERVER_NAME,
    instructions=f"{SERVER_DESCRIPTION}\n\n{SERVER_INSTRUCTIONS}",
    host=os.getenv("MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("MCP_PORT", "8000")),
    sse_path=os.getenv("MCP_SSE_PATH", "/sse"),
    message_path=os.getenv("MCP_MESSAGE_PATH", "/messages/"),
    streamable_http_path=os.getenv("MCP_STREAMABLE_HTTP_PATH", "/mcp"),
    stateless_http=os.getenv("MCP_STATELESS_HTTP", "1").strip().lower() not in {"0", "false", "no"},
    # json_response: 以普通 JSON 响应 SSE 流——兼容不带 Accept: text/event-stream 头的客户端（磐石平台chat调用器）
    json_response=os.getenv("MCP_JSON_RESPONSE", "1").strip().lower() not in {"0", "false", "no"},
)


EMPTY_STRINGS = ("", chr(39)*2, chr(34)*2, chr(39)+chr(34)*2+chr(39))


def _clean(v):
    """平台工作流引擎会把未填参数发成空串或带引号空串——清洗为 None；数字/JSON串转原生类型。"""
    if isinstance(v, str):
        s = v.strip()
        if s in EMPTY_STRINGS:
            return None
        import json as _j
        try:
            return _j.loads(s)
        except Exception:
            return v
    return v


def _clean_args(kwargs, str_list_keys=()):
    """清洗工具参数：空->None；声明为列表的字段，字符串尝试JSON解析否则按逗号拆分。"""
    import json as _j
    out = {}
    for k, v in kwargs.items():
        v = _clean(v)
        if v is None:
            out[k] = None
        elif k in str_list_keys and isinstance(v, str):
            s = v.strip()
            if s.startswith("["):
                try:
                    out[k] = _j.loads(s)
                except Exception:
                    out[k] = [x.strip() for x in s.split(",") if x.strip()]
            else:
                out[k] = [x.strip() for x in s.split(",") if x.strip()]
        else:
            out[k] = v
    return out


class AcceptHeaderCompat:
    """ASGI 中间件：请求未携带 MCP 协议要求的 Accept 头时自动补全。

    部分简化客户端（如磐石平台的 chat 工具调用器）不发送
    Accept: application/json, text/event-stream，标准 FastMCP 以 406 拒绝。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = scope.get("headers", [])
            accept = None
            for k, v in headers:
                if k == b"accept":
                    accept = v.decode("latin-1")
                    break
            if ("application/json" not in (accept or "")
                    or "text/event-stream" not in (accept or "")):
                new_headers = [(k, v) for k, v in headers if k != b"accept"]
                new_headers.append((b"accept", b"application/json, text/event-stream"))
                scope = dict(scope)
                scope["headers"] = new_headers
        await self.app(scope, receive, send)


def _error(message: str, exc: Exception | None = None) -> dict:
    """统一错误返回结构。"""
    detail = message
    if exc is not None:
        detail = f"{message}: {type(exc).__name__}: {exc}"
    return {"ok": False, "error": detail, "traceback": traceback.format_exc() if exc else None}


# ========== Tool 1: 偏导数 ==========

@mcp.tool(
    name="compute_derivatives",
    description=(
        "计算因变量 Y 对各自变量 X 的光滑偏导数（KDE 自适应高斯加权局部线性拟合）。"
        "输入 Excel 须为 log10 对数空间数据，前 N-1 列为自变量、末列为因变量。"
        "输出为在原数据后追加 dY_d{列名} 偏导数列的新 Excel 文件，返回文件路径与统计摘要。"
    ),
)
def tool_compute_derivatives(
    input_file: str | list | dict | int | float | None = None,
    output_file: str | list | dict | int | float | None = None,
    data: str | list | dict | int | float | None = None,
    n_indep: str | list | dict | int | float | None = None,
    x_cols: str | list | dict | int | float | None = None,
    y_col: str | list | dict | int | float | None = None,
    k_neighbors: str | list | dict | int | float | None = None,
    k_bandwidth: str | list | dict | int | float | None = None,
    space: str | list | dict | int | float | None = None) -> dict:
    """计算光滑偏导数。

    Args:
        input_file: 输入数据文件路径（.xlsx，log10 对数空间）；与 data 二选一。
        data: 行内数据 {"columns": [...], "rows": [[...], ...]}（log10 空间），
            供无法向服务器上传文件的调用方（平台智能体）直接传数据。
        output_file: 输出 Excel 路径，None 时在输入文件同目录自动生成。
        n_indep: 自变量列数；None 表示总列数-1。
        x_cols: 显式指定自变量列名（提供后覆盖 n_indep）。
        y_col: 显式指定因变量列名；None 时取末列。
        k_neighbors: 局部线性拟合的邻居数，默认随维度自适应。
        k_bandwidth: KDE 带宽估计所用近邻数，默认 30。
    """
    try:
        kw = _clean_args(locals(), str_list_keys=("x_cols",))
        result = compute_derivatives(
            input_file=kw.get("input_file"),
            output_file=kw.get("output_file"),
            data=kw.get("data"),
            n_indep=kw.get("n_indep"),
            x_cols=kw.get("x_cols"),
            y_col=kw.get("y_col"),
            k_neighbors=kw.get("k_neighbors"),
            k_bandwidth=kw.get("k_bandwidth"),
            space=kw.get("space") or "log10",
        )
        return {"ok": True, **result}
    except Exception as e:
        return _error("偏导数计算失败", e)


# ========== Tool 2: 聚类 ==========

@mcp.tool(
    name="cluster_analysis",
    description=(
        "对偏导数数据执行 subclust（减法聚类）+ FCM（模糊C均值）复合聚类，"
        "识别数据所在的低维子空间并输出每个子空间的特征值、特征向量（含主导 Di 值）。"
        "输入应为 compute_derivatives 输出的含 dY_d 前缀列的 Excel 文件。"
        "结果保存为聚类特征分析 Excel、聚类组件（标签/中心）及聚类坐标 CSV。"
    ),
)
def tool_cluster_analysis(
    input_file: str | list | dict | int | float | None = None,
    output_dir: str | list | dict | int | float | None = None,
    n_clusters: str | list | dict | int | float | None = None,
    n_indep: str | list | dict | int | float | None = None,
    x_cols: str | list | dict | int | float | None = None,
    y_col: str | list | dict | int | float | None = None,
    fcm_m: str | list | dict | int | float | None = None,
    fcm_error: str | list | dict | int | float | None = None,
    fcm_maxiter: str | list | dict | int | float | None = None,
    max_samples: str | list | dict | int | float | None = None,
    subclust_radius_factor: str | list | dict | int | float | None = None,
    subclust_suppression_ratio: str | list | dict | int | float | None = None,
    membership_threshold: str | list | dict | int | float | None = None) -> dict:
    """执行 FCM 复合聚类分析。

    Args:
        input_file: 含偏导数（dY_d 前缀列）的 Excel 文件路径。
        output_dir: 结果输出目录，None 时使用输入文件同目录。
        n_clusters: 目标聚类数，默认 1；传 0 表示由 subclust 按 Chiu 接受/拒绝
            准则自动确定聚类数（"自动分区"特性）。
        n_indep: 自变量列数，用于自动识别列。
        x_cols: 显式自变量列名列表。
        y_col: 显式因变量列名。
        fcm_m: FCM 模糊指数，默认 2.0。
        fcm_error: FCM 收敛阈值，默认 0.0003。
        fcm_maxiter: FCM 最大迭代次数，默认 2000。
        max_samples: 聚类抽样上限，默认 200000。
        subclust_radius_factor: 减法聚类半径因子，默认 0.30。
        subclust_suppression_ratio: 减法聚类抑制比，默认 1.5。
        membership_threshold: FCM 最大隶属度阈值，默认 0.6；低于该值的点标记为
            模糊点（标签-1，不参与后续拟合）。传 0 关闭模糊点剔除。
    """
    try:
        kw = _clean_args(locals())
        ncl = kw.get("n_clusters")
        analysis = AdvancedClusteringAnalysis(
            data_file=kw.get("input_file"),
            n_clusters=1 if ncl is None else ncl,
            n_indep=kw.get("n_indep"),
            x_cols=kw.get("x_cols"),
            y_col=kw.get("y_col"),
            fcm_m=kw.get("fcm_m"),
            fcm_error=kw.get("fcm_error"),
            fcm_maxiter=kw.get("fcm_maxiter"),
            max_samples=kw.get("max_samples"),
            subclust_radius_factor=kw.get("subclust_radius_factor"),
            subclust_suppression_ratio=kw.get("subclust_suppression_ratio"),
            membership_threshold=kw.get("membership_threshold"),
        )
        result = analysis.run_analysis(output_dir=output_dir)
        return {"ok": True, **result}
    except Exception as e:
        return _error("聚类分析失败", e)


# ========== Tool 3: 拟合 ==========

@mcp.tool(
    name="fit_power_function",
    description=(
        "在物理空间中对每个聚类子空间拟合幂函数 A = a·C^b + c，其中组合变量 "
        "C = B1^D1·B2^D2·...·Bn^Dn（Di 为聚类主导特征向量，|Di|<0.05 置零）。"
        "工具内部将 log10 数据转换为物理空间（B=10^X, A=10^Y）。含残差离群点清洗。"
        "需先运行 cluster_analysis 生成聚类结果；输出拟合曲线图、总览图、数据 Excel 和文本报告。"
    ),
)
def tool_fit_power_function(
    derivative_file: str | list | dict | int | float | None = None,
    output_dir: str | list | dict | int | float | None = None,
    clustering_dir: str | list | dict | int | float | None = None,
    n_indep: str | list | dict | int | float | None = None,
    x_cols: str | list | dict | int | float | None = None,
    y_col: str | list | dict | int | float | None = None,
    c_mode: str | list | dict | int | float | None = None,
    fixed_c: str | list | dict | int | float | None = None,
    max_clean_ratio: str | list | dict | int | float | None = None,
    di_threshold: str | list | dict | int | float | None = None,
    min_fit_points: str | list | dict | int | float | None = None,
    curve_fit_maxfev: str | list | dict | int | float | None = None) -> dict:
    """拟合幂函数 A = a·C^b + c。

    Args:
        derivative_file: 含偏导数的 Excel 文件路径（log10 空间）。
        output_dir: 结果输出目录，None 时使用输入文件同目录。
        clustering_dir: 聚类结果目录（含 clustering_eigen_analysis.xlsx 与
            clustering_components/cluster_labels.npy）；None 时使用输入文件同目录。
        n_indep: 自变量列数。
        x_cols: 显式自变量列名列表。
        y_col: 显式因变量列名。
        c_mode: 常数项模式，"fixed"（固定 c=fixed_c）或 "free"（自由拟合），默认 "fixed"。
        fixed_c: 固定 c 值，默认 1.0，仅 c_mode="fixed" 生效。
        max_clean_ratio: 离群点清洗比例（按残差最大的比例剔除），默认 0.1。
        di_threshold: |Di| 小于该阈值置零，默认 0.05。
        min_fit_points: 拟合所需最少正 C 值数据点，默认 10。
        curve_fit_maxfev: curve_fit 最大迭代次数，默认 20000。
    """
    try:
        kw = _clean_args(locals())
        result = fit_power_function(
            derivative_file=kw.get("derivative_file"),
            output_dir=kw.get("output_dir"),
            clustering_dir=kw.get("clustering_dir"),
            n_indep=kw.get("n_indep"),
            x_cols=kw.get("x_cols"),
            y_col=kw.get("y_col"),
            c_mode=kw.get("c_mode"),
            fixed_c=kw.get("fixed_c"),
            max_clean_ratio=kw.get("max_clean_ratio"),
            di_threshold=kw.get("di_threshold"),
            min_fit_points=kw.get("min_fit_points"),
            curve_fit_maxfev=kw.get("curve_fit_maxfev"),
        )
        return {"ok": "error" not in result, **result}
    except Exception as e:
        return _error("幂函数拟合失败", e)


# ========== Tool 4: 完整流水线 ==========

@mcp.tool(
    name="run_full_pipeline",
    description=(
        "依次执行完整 DDDA 流水线：(1) 光滑偏导数计算 -> (2) subclust+FCM 聚类 -> "
        "(3) 物理空间幂函数拟合。输入为 log10 空间原始数据 Excel。"
        "聚类与拟合在内存中直接传递中间结果，避免中间文件依赖；所有最终结果保存到 output_dir。"
    ),
)
def tool_run_full_pipeline(
    input_file: str | list | dict | int | float | None = None,
    output_dir: str | list | dict | int | float | None = None,
    data: str | list | dict | int | float | None = None,
    n_indep: str | list | dict | int | float | None = None,
    x_cols: str | list | dict | int | float | None = None,
    y_col: str | list | dict | int | float | None = None,
    k_neighbors: str | list | dict | int | float | None = None,
    k_bandwidth: str | list | dict | int | float | None = None,
    space: str | list | dict | int | float | None = None,
    n_clusters: str | list | dict | int | float | None = None,
    fcm_m: str | list | dict | int | float | None = None,
    fcm_error: str | list | dict | int | float | None = None,
    fcm_maxiter: str | list | dict | int | float | None = None,
    max_samples: str | list | dict | int | float | None = None,
    subclust_radius_factor: str | list | dict | int | float | None = None,
    subclust_suppression_ratio: str | list | dict | int | float | None = None,
    membership_threshold: str | list | dict | int | float | None = None,
    c_mode: str | list | dict | int | float | None = None,
    fixed_c: str | list | dict | int | float | None = None,
    max_clean_ratio: str | list | dict | int | float | None = None,
    di_threshold: str | list | dict | int | float | None = None,
    min_fit_points: str | list | dict | int | float | None = None,
    curve_fit_maxfev: str | list | dict | int | float | None = None) -> dict:
    """执行完整 DDDA 流水线。

    Args:
        input_file: 原始数据文件路径（.xlsx/.csv，服务器本地路径）；与 data 二选一。
        data: 行内数据 {"columns": [...], "rows": [[...], ...]}。
        output_dir: 最终结果输出目录，None 时使用输入文件同目录。
        n_indep: 自变量列数。
        x_cols: 显式自变量列名列表。
        y_col: 显式因变量列名。
        k_neighbors: 偏导数局部拟合邻居数。
        k_bandwidth: KDE 带宽估计近邻数。
        space: 输入数据空间，"log10"（默认）或 "physical"（物理空间原始值，
            自动转 log10；LIGGGHTS 仿真结果 CSV 可直传）。
        n_clusters: 目标聚类数，默认 1；传 0 表示由 subclust 自动确定（自动分区）。
        fcm_m: FCM 模糊指数。
        fcm_error: FCM 收敛阈值。
        fcm_maxiter: FCM 最大迭代次数。
        max_samples: 聚类抽样上限。
        subclust_radius_factor: 减法聚类半径因子。
        subclust_suppression_ratio: 减法聚类抑制比。
        membership_threshold: FCM 最大隶属度阈值，默认 0.6；低于则标记为模糊点不参与拟合。
        c_mode: 常数项模式 "fixed"/"free"。
        fixed_c: 固定 c 值。
        max_clean_ratio: 离群点清洗比例。
        di_threshold: Di 置零阈值。
        min_fit_points: 拟合最少数据点。
        curve_fit_maxfev: 拟合最大迭代次数。
    """
    try:
        kw = _clean_args(locals(), str_list_keys=("x_cols",))
        input_file = kw.get("input_file")
        output_dir = kw.get("output_dir")
        data = kw.get("data")
        n_indep = kw.get("n_indep"); x_cols = kw.get("x_cols"); y_col = kw.get("y_col")
        k_neighbors = kw.get("k_neighbors"); k_bandwidth = kw.get("k_bandwidth")
        space = kw.get("space") or "log10"
        ncl = kw.get("n_clusters"); n_clusters = 1 if ncl is None else ncl
        fcm_m = kw.get("fcm_m"); fcm_error = kw.get("fcm_error"); fcm_maxiter = kw.get("fcm_maxiter")
        max_samples = kw.get("max_samples")
        srf = kw.get("subclust_radius_factor"); ssr = kw.get("subclust_suppression_ratio")
        mt = kw.get("membership_threshold")
        c_mode = kw.get("c_mode"); fixed_c = kw.get("fixed_c")
        max_clean_ratio = kw.get("max_clean_ratio"); di_threshold = kw.get("di_threshold")
        min_fit_points = kw.get("min_fit_points"); curve_fit_maxfev = kw.get("curve_fit_maxfev")

        if output_dir is None:
            if input_file and isinstance(input_file, str):
                output_dir = os.path.dirname(os.path.abspath(input_file))
            else:
                # 托管容器可能根只读——env 覆盖（Dockerfile.platform 设 /tmp/ddda_out）
                output_dir = os.environ.get("DDDA_DEFAULT_OUT", "/tmp/ddda_out")
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: 偏导数
        if input_file is None and data is not None:
            input_file = materialize_inline_data(data)
        if input_file is None:
            return _error("input_file 与 data 参数必须提供其一")

        base_name = ("inline_data" if not input_file
                     else os.path.splitext(os.path.basename(str(input_file)))[0])
        derivative_output = os.path.join(output_dir, base_name + "_偏导数.xlsx")

        deriv_result = compute_derivatives(
            input_file=input_file,
            output_file=derivative_output,
            n_indep=n_indep,
            x_cols=x_cols,
            y_col=y_col,
            k_neighbors=k_neighbors,
            k_bandwidth=k_bandwidth,
            space=space,
        )

        # Step 2: 聚类（结果落盘，供后续排查；同时保留内存实例）
        analysis = AdvancedClusteringAnalysis(
            data_file=derivative_output,
            n_clusters=n_clusters,
            n_indep=n_indep,
            x_cols=x_cols,
            y_col=y_col,
            fcm_m=fcm_m,
            fcm_error=fcm_error,
            fcm_maxiter=fcm_maxiter,
            max_samples=max_samples,
            subclust_radius_factor=srf,
            subclust_suppression_ratio=ssr,
            membership_threshold=mt,
        )
        cluster_result = analysis.run_analysis(output_dir=output_dir)

        # Step 3: 拟合（直接传入内存中的 analysis，避免文件依赖）
        fitting_result = fit_power_function(
            derivative_file=derivative_output,
            output_dir=output_dir,
            clustering_analysis=analysis,
            n_indep=n_indep,
            x_cols=x_cols,
            y_col=y_col,
            c_mode=c_mode,
            fixed_c=fixed_c,
            max_clean_ratio=max_clean_ratio,
            di_threshold=di_threshold,
            min_fit_points=min_fit_points,
            curve_fit_maxfev=curve_fit_maxfev,
        )

        return {
            "ok": "error" not in fitting_result,
            "derivatives": deriv_result,
            "clustering": cluster_result,
            "fitting": fitting_result,
            "output_dir": output_dir,
        }
    except Exception as e:
        return _error("完整流水线执行失败", e)


# ========== Tool 5: 预测 ==========

@mcp.tool(
    name="predict",
    description=(
        "给定新的参数点（物理空间），用已完成的 DDDA 分区幂律模型预测响应量，"
        "并判断该参数点属于哪个子域（KNN投票）。实现规律->应用的闭环："
        "先运行 run_full_pipeline（或 cluster_analysis + fit_power_function），"
        "再用本工具对新参数做预测。"
    ),
)
def tool_predict(
    clustering_dir: str | list | dict | int | float | None,
    params: str | dict | list | int | float | None,
    fitting_json: str | list | dict | int | float | None = None,
    k: str | int | float | None = 7,
) -> dict:
    """用已拟合的分区幂律模型预测新参数点的响应量。

    Args:
        clustering_dir: cluster_analysis 的输出目录（含 cluster_*_coordinates.csv）。
            run_full_pipeline 模式下即其 output_dir。
        params: dict，各自变量的物理空间值，如 {"ad": 1.0, "gamma": 10.0, "F": 8.0}，
            键名须与建模时的自变量列名一致。
        fitting_json: fitting_results.json 路径；默认在 clustering_dir 及其上级目录查找。
        k: KNN 投票近邻数（log10 空间），默认 7。
    """
    try:
        kw = _clean_args(locals())
        result = predict(
            clustering_dir=kw.get("clustering_dir"),
            params=kw.get("params"),
            fitting_json=kw.get("fitting_json"),
            k=kw.get("k") if kw.get("k") is not None else 7,
        )
        return result
    except Exception as e:
        return _error("预测失败", e)


# ========== Tool 6: URL 数据引入 ==========

@mcp.tool(
    name="load_data_from_url",
    description=(
        "从 URL 下载数据文件到服务器本地，返回可直接用于其他工具（如 compute_derivatives / "
        "run_full_pipeline 的 input_file）的本地路径。适用于平台数据集、共享存储等以 URL "
        "形式提供的数据源——MCP 工具参数无法传文件，传 URL 由服务器自取是标准做法。"
    ),
)
def tool_load_data_from_url(
    url: str | list | dict | int | float | None,
    filename: str | list | dict | int | float | None = None,
) -> dict:
    # filename 为空串时视为未提供
    """下载数据文件到服务器并返回本地路径。

    Args:
        url: 数据文件的 http/https 地址（须可公开访问或同网段可达）。
        filename: 保存文件名（可选，默认按 URL 和时间自动命名）。
    """
    try:
        from utils.data_io import download_to_server
        path = download_to_server(url, filename=filename)
        return {"ok": True, "local_path": path, "size_bytes": os.path.getsize(path),
                "hint": "将 local_path 作为 input_file 传给后续工具"}
    except Exception as e:
        return _error("下载失败", e)


@mcp.tool(
    name="derive_pi_groups",
    description=(
        "自动量纲分析（Buckingham π 定理）：给定候选物理参数及其量纲，自动推导全部独立"
        "无量纲组合（π 群）。这是 DDDA 流程的第 0 步——为新的科研问题自动确定无量纲"
        "自变量。量纲表达式支持两种写法：基本量纲组合（'L'、'L*T^-2'、'M*L/T^3'）或"
        "单位形式（'m'、'm/s^2'、'kg'、'Hz'）；无量纲参数写 'dimensionless'。"
        "验证案例：{a:L, f:T^-1, d:L, g:L*T^-2} 自动导出 a/d 与 g/(a·f²)，"
        "经典教科书结果一致（如单摆周期、管流雷诺数）。"
    ),
)
def tool_derive_pi_groups(
    params: str | list | dict | None,
) -> dict:
    """Buckingham π 定理自动推导无量纲数。

    Args:
        params: {参数名: 量纲表达式} 的字典；也接受 JSON 字符串形式。
                示例：{"a": "L", "f": "T^-1", "d": "L", "g": "L*T^-2"}
                或单位形式 {"a": "m", "f": "Hz", "d": "m", "g": "m/s^2"}。
    """
    try:
        from core.pi_groups import derive_pi_groups
        p = _clean(params)
        if isinstance(p, str):
            import json as _json
            p = _json.loads(p)
        if not isinstance(p, dict):
            return {"ok": False, "error": "params 须为 {参数名: 量纲表达式} 字典或其 JSON 串"}
        return derive_pi_groups({str(k): v for k, v in p.items()})
    except Exception as e:
        return _error("量纲分析失败", e)


# ========== Tool 8: 批量归属查询 ==========
@mcp.tool(
    name="batch_query",
    description=(
        "批量归属查询：对一组候选工况点（物理空间参数，与 predict 同口径）一次性返回 "
        "KNN 归属投票、校准归属概率 π（票比→实测正确率的 isotonic 校准）、最近已仿真点"
        "距离（log10）。predict 的批量+概率化版本，供设计分析与建议解释使用。只读。"
    ),
)
def tool_batch_query(
    clustering_dir: str | list | dict | int | float | None,
    params_list: str | list | dict | None = None,
    fitting_json: str | list | dict | int | float | None = None,
) -> dict:
    """批量归属查询。

    Args:
        clustering_dir: 聚类分析输出目录。
        params_list: [{参数名: 物理空间值}, ...] 候选点列表（JSON）。
        fitting_json: fitting_results.json 路径（可选，默认自动定位）。
    """
    try:
        kw = _clean_args(locals())
        import numpy as np
        from core.design import DesignInstrument
        inst = DesignInstrument(kw.get("clustering_dir"), fitting_json=kw.get("fitting_json"))
        pl = kw.get("params_list")
        if not isinstance(pl, list) or not pl:
            return {"ok": False, "error": "params_list 须为非空 [{参数名:值}, ...] 列表"}
        X = np.log10(np.array([[float(p[c]) for c in inst.x_cols] for p in pl]))
        votes, dnn = inst.votes_and_dnn(X)
        pi = inst._apply_kappa(votes)
        return {"ok": True, "n": len(pl), "K": inst.K,
                "results": [{"params": p,
                             "votes": votes[i].astype(int).tolist(),
                             "pi": np.round(pi[i], 3).tolist(),
                             "nearest_distance_log10": round(float(dnn[i]), 4)}
                            for i, p in enumerate(pl)]}
    except Exception as e:
        return _error("批量归属查询失败", e)


# ========== Tool 9: 设计建议 ==========
@mcp.tool(
    name="design_advice",
    description=(
        "区制可辨识性驱动的实验设计建议：在当前 DDDA 分析状态上，对候选工况计算复合"
        "期望信息增益 score=(A_bnd+A_law+w_e·A_exp)/c 并输出批次建议（含逐点成分分解、"
        "归属概率 π、两区预测分布与人话解释）。A_bnd=归属信息（校准票→π，两区预测混合"
        "信道容量，nat）；A_law=定律信息（贝叶斯杠杆 ½log(1+h̃)，nat）；A_exp=探索"
        "（最近点距离，声明的模型外启发项）；c 为成本（v0 恒 1）。建议为咨询性质："
        "agent 可采纳/修改/否决，决策应入账。candidates/exclude 为物理空间坐标；"
        "缺省时在 parameter_ranges 网格上自动铺候选（grid_n 每维）。只读。"
    ),
)
def tool_design_advice(
    clustering_dir: str | list | dict | int | float | None,
    fitting_json: str | list | dict | int | float | None = None,
    budget: str | list | dict | int | float | None = 30,
    w_e: str | list | dict | int | float | None = 0.3,
    candidates: str | list | dict | None = None,
    grid_n: str | list | dict | int | float | None = 20,
    exclude: str | list | dict | None = None,
    repel_radius: str | list | dict | int | float | None = None,
) -> dict:
    """生成下一批工况的设计建议。

    Args:
        clustering_dir: 聚类分析输出目录（含 clustering_components 与拟合产物）。
        fitting_json: fitting_results.json 路径（可选）。
        budget: 批次大小 m（默认 30）。
        w_e: 探索项权重（默认 0.3，预注册值；0=纯 EIG）。
        candidates: 候选点列表（物理空间 [[x1,x2,x3],...]，可选；缺省自动铺网格）。
        grid_n: 自动网格每维点数（默认 20）。
        exclude: 排除点列表（物理空间，如已仿真/已列入批次的点）。
        repel_radius: 批内排斥半径（log10 距离；默认=已仿真点中位近邻距×1.5）。
    """
    try:
        kw = _clean_args(locals(), str_list_keys=("candidates", "exclude"))
        from core.design import DesignInstrument
        inst = DesignInstrument(kw.get("clustering_dir"), fitting_json=kw.get("fitting_json"))
        out = inst.advise(
            budget=int(kw.get("budget") or 30),
            w_e=float(kw.get("w_e") if kw.get("w_e") is not None else 0.3),
            candidates=kw.get("candidates"),
            grid_n=int(kw.get("grid_n") or 20),
            exclude=kw.get("exclude"),
            repel_radius=(float(kw["repel_radius"])
                          if kw.get("repel_radius") is not None else None),
        )
        out["ok"] = True
        return out
    except Exception as e:
        return _error("设计建议生成失败", e)


def main():
    """启动 MCP 服务。

    默认使用 stdio，兼容本地 MCP 客户端；需要 HTTP 接入时设置：
      MCP_TRANSPORT=streamable-http python server.py
    可选值：stdio、sse、streamable-http。
    """
    # 平台托管运行时语义免疫（2026-10-14 H2 假设防线）：start_command 可能以
    # execve 或 shell 方式执行、env 可能不注入——缺省值兜底成 HTTP 模式
    os.environ.setdefault("MCP_TRANSPORT", "streamable-http")
    os.environ.setdefault("MCP_HOST", "0.0.0.0")
    os.environ.setdefault("MCP_PORT", "8000")
    # FastMCP 在模块导入时就按 MCP_HOST 构造完毕（host 为回环时自动启用仅放行 localhost 的
    # DNS rebinding 保护，且 uvicorn 同样用 settings.host 绑定）——上面的 setdefault 来得太晚，
    # 必须把 settings 一并改过来，否则无 env 时只监听容器内 127.0.0.1、外部（平台 probe）连不上
    mcp.settings.host = os.environ["MCP_HOST"]
    mcp.settings.port = int(os.environ["MCP_PORT"])
    if mcp.settings.host not in ("127.0.0.1", "localhost", "::1"):
        from mcp.server.transport_security import TransportSecuritySettings
        mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    transport = os.getenv("MCP_TRANSPORT", "stdio").strip().lower().replace("_", "-")
    if transport not in {"stdio", "sse", "streamable-http"}:
        raise ValueError("MCP_TRANSPORT must be one of: stdio, sse, streamable-http")
    if transport == "streamable-http":
        import uvicorn

        class PathNorm:
            """平台托管探测经网关带前缀路径（.../mcpserver/<id>/mcp）——
            凡以 /mcp 结尾的请求一律归一到 /mcp（2026-10-14 托管 mcp_probe 404 假设修复）。"""
            def __init__(self, app):
                self.app = app

            async def __call__(self, scope, receive, send):
                if scope.get("type") == "http":
                    p = scope.get("path", "")
                    _acc = dict(scope.get("headers") or []).get(b"accept", b"")
                    # healthz，以及不要 SSE 流的普通 GET（健康检查/探活）一律直接答 200
                    if p.rstrip("/").endswith("/healthz") or (
                            scope.get("method") == "GET" and b"text/event-stream" not in _acc):
                        await send({"type": "http.response.start", "status": 200,
                                    "headers": [(b"content-type", b"application/json")]})
                        await send({"type": "http.response.body", "body": b'{"ok":true}'})
                        return
                    # 容器只提供 MCP：一切其他路径归一到 /mcp
                    # （平台网关可能带前缀、可能打根路径，2026-10-14 probe 失败二轮修复）
                    scope["path"] = "/mcp"
                await self.app(scope, receive, send)

        uvicorn.run(PathNorm(TokenAuth(AcceptHeaderCompat(mcp.streamable_http_app()))),
                    host=mcp.settings.host, port=mcp.settings.port)
    else:
        mcp.run(transport=transport)


if __name__ == "__main__":
    main()
