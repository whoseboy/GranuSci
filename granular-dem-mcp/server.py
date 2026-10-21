# -*- coding: utf-8 -*-
"""granular-dem MCP 服务入口。

4 个工具（与 DDDA-MCP 同风格，供平台智能体/科研助手调用）：
  1. generate_design  - 生成批量振动工况实验设计（无量纲参数网格）
  2. run_simulation   - 提交批量 LIGGGHTS 仿真（异步，返回 job_id）
  3. job_status       - 查询作业进度
  4. postprocess      - 解析 dump，输出 responses.csv（物理空间，DDDA-MCP 可直传）

运行环境：LIGGGHTS + Python 一体化 Docker 容器。
文件布局：容器内 /work ↔ 宿主机 {HOST_WORK_DIR}（docker run -v 挂载），
工具返回的路径一律为宿主机路径（方便同机的 DDDA-MCP 直接读取）。
"""
import json
import os
import sys
import traceback

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mcp.server.fastmcp import FastMCP  # noqa: E402

from legacy.design import generate_design  # noqa: E402  # 战役定制（energy/segregation），待 case 生成器替代
from legacy.runner import JobManager  # noqa: E402  # 振动槽模板 runner，待新执行器替代
from legacy.postprocess import (postprocess_job, merge_responses,  # noqa: E402  # 响应量定义待可插拔化
                              get_snapshot, get_profiles)
from auth_middleware import TokenAuth  # noqa: E402

WORK_DIR = os.getenv("WORK_DIR", "/work")
HOST_WORK_DIR = os.getenv("HOST_WORK_DIR", WORK_DIR)
LIGGGHTS_CMD = os.getenv("LIGGGHTS_CMD", "/opt/liggghts/bin/liggghts")

mcp = FastMCP(
    name="granular-dem-mcp",
    instructions=(
        "LIGGGHTS 批量离散元仿真服务：生成振动颗粒体系实验设计、批量运行仿真、"
        "后处理提取响应量。输出 responses.csv 为物理空间原始值，可直接传给 "
        "DDDA-MCP 的 run_full_pipeline（space=\"physical\"）。"
    ),
    host=os.getenv("MCP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_PORT", "8001")),
    stateless_http=True,
    # json_response: 兼容不带 Accept: text/event-stream 头的客户端（磐石平台chat调用器）
    json_response=True,
)

_jobs = JobManager(WORK_DIR, LIGGGHTS_CMD, HOST_WORK_DIR)

# casegen v1：结构化 case 描述 → 渲染 → 运行（C1 判决架构：LLM 只产 JSON，
# 语法由渲染器唯一出口封死；见 core/casegen/SCHEMA.md）
from core.casegen.jobs import CaseJobManager  # noqa: E402
from core.casegen.SCHEMA_TEXT import SCHEMA_DOC  # noqa: E402
_casegen = CaseJobManager(WORK_DIR, LIGGGHTS_CMD)


class AcceptHeaderCompat:
    """ASGI 中间件：补全 MCP 协议要求的 Accept 头（兼容磐石平台chat调用器）。"""

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


EMPTY_STRINGS = ("", chr(39)*2, chr(34)*2)


def _clean(v):
    """平台工作流/LLM 会把参数发成字符串——空串转None，数字/JSON串转原生类型。"""
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


def _error(message, exc=None):
    detail = message if exc is None else "%s: %s: %s" % (message, type(exc).__name__, exc)
    return {"ok": False, "error": detail,
            "traceback": traceback.format_exc()[-400:] if exc else None}


@mcp.tool(
    name="generate_design",
    description=(
        "生成振动颗粒体系批量仿真的实验设计（全排列网格）。study='energy'（默认）：单分散体系，"
        "核心无量纲参数为无量纲振幅 a/d、振动强度 Gamma=a(2*pi*f)^2/g、颗粒层数 F；"
        "study='segregation'：二元混合偏析体系（小颗粒 type1 + 大/重颗粒 type2），"
        "额外无量纲参数为粒径比 r_s=d_l/d_s、密度比 r_rho=rho_l/rho_s、大颗粒体积分数 x_large，"
        "a/d 与 F 以小颗粒直径 d_s 计（默认 d_s=0.002m，受槽宽限制 r_s<=2）。"
        "自动换算振幅/振动周期/时间步数，返回 design_id 与工况列表。"
    ),
)
def tool_generate_design(
    ad_list: str | list | None = None,
    gamma_list: str | list | None = None,
    F_list: str | list | None = None,
    d: str | float | None = None,
    dev_periods: str | int | None = None,
    stat_periods: str | int | None = None,
    study: str | None = None,
    r_s_list: str | list | None = None,
    r_rho_list: str | list | None = None,
    x_large_list: str | list | None = None,
) -> dict:
    """生成实验设计。

    Args:
        ad_list: 无量纲振幅取值列表，如 [0.25, 1.0, 4.0]；缺省用演示网格 [0.25,1,4]×[1.25,10,80]×[4,8,16]。
        gamma_list: 振动强度取值列表。
        F_list: 颗粒层数取值列表。
        d: 参考颗粒直径（米），必填。energy 为单分散颗粒直径；segregation 为小颗粒直径。
        dev_periods: 发展段振动周期数（统计前充分发展），默认 20。
        stat_periods: 统计段振动周期数，默认 8。
        study: 'energy'（默认，单分散能量规律）或 'segregation'（二元振动偏析）。
        r_s_list: （segregation）粒径比 d_l/d_s 列表，范围 [1,2]。
        r_rho_list: （segregation）密度比 rho_l/rho_s 列表，范围 [0.25,4]。
        x_large_list: （segregation）大颗粒体积分数列表，范围 [0.05,0.6]。
    """
    try:
        kw = _clean_args(locals(), str_list_keys=(
            "ad_list", "gamma_list", "F_list",
            "r_s_list", "r_rho_list", "x_large_list"))
        return generate_design(
            WORK_DIR, kw.get("ad_list"), kw.get("gamma_list"), kw.get("F_list"),
            float(kw.get("d") or 0),           # 0 -> design 内部按 study 取默认
            int(kw.get("dev_periods") or 20), int(kw.get("stat_periods") or 8),
            study=kw.get("study") or "energy",
            r_s_list=kw.get("r_s_list"),
            r_rho_list=kw.get("r_rho_list"),
            x_large_list=kw.get("x_large_list"))
    except Exception as e:
        return _error("生成实验设计失败", e)


@mcp.tool(
    name="run_simulation",
    description=(
        "提交批量 LIGGGHTS 仿真。默认异步（立即返回 job_id，用 job_status 查询）；"
        "wait=true 时同步等待作业完成（每5秒轮询，最长 wait_timeout 秒）后返回最终状态，"
        "适合工作流一次性调用。传 wait_chunk=N 则改为分段等待：单次调用最多阻塞 N 秒，"
        "未完成即返回进度（waited=false），带 job_id 再次调用可续接等待（平台工作流轮询链模式，"
        "规避长调用通道超时）。同一 design_id 已有作业时自动续接、不重复提交。"
        "每个颗粒层数 F 首次运行会先生成自然堆积初始床层并缓存。"
    ),
)
def tool_run_simulation(
    design_id: str | None = None,
    job_id: str | None = None,
    max_parallel: str | int | None = None,
    wait: str | bool | None = None,
    wait_timeout: str | int | None = None,
    wait_chunk: str | int | None = None,
) -> dict:
    """提交或续接批量仿真作业。

    Args:
        design_id: generate_design 返回的设计ID。
        job_id: 续接已有作业（分段等待模式第 2+ 次调用时传）。
        max_parallel: 同时运行的工况数上限（每个工况占约1核），默认 4。
        wait: true 时同步等待作业完成再返回（工作流模式）。
        wait_timeout: wait 模式最长等待秒数，默认 240（未传 wait_chunk 时的传统整段语义）。
        wait_chunk: 分段等待秒数（传入即启用分段模式，单次调用最多阻塞这么久）。
    """
    try:
        kw = _clean_args(locals())
        jid = str(kw.get("job_id") or "").strip() or None
        if jid is None and kw.get("design_id"):
            # 同设计已有作业（轮询链重复提交/引擎重试）→ 续接，防双花
            jid = _jobs.find_job_by_design(kw.get("design_id"))
        if jid is None:
            r = _jobs.submit(kw.get("design_id"),
                             int(kw.get("max_parallel") or 4))
            if not r.get("ok"):
                return r
            jid = r["job_id"]
        if not str(kw.get("wait") or "").lower() in ("true", "1", "yes"):
            s = _jobs.status(jid)
            s["mode"] = "async"
            return s
        import time as _t
        wc = kw.get("wait_chunk")
        if wc is not None:
            cap = max(1, int(float(wc)))          # 分段模式
        else:
            cap = int(kw.get("wait_timeout") or 240)  # 传统整段模式
        deadline = _t.time() + cap
        while True:
            s = _jobs.status(jid)
            if s.get("status") in ("done", "failed"):
                s["waited"] = True
                s["mode"] = "chunked" if wc is not None else "blocking"
                return s
            if _t.time() >= deadline:
                break
            _t.sleep(3 if wc is not None else 5)
        s = _jobs.status(jid)
        s["waited"] = False
        s["mode"] = "chunked" if wc is not None else "blocking"
        s["message"] = ("分段等待 %ss 已到，作业仍在运行；带 job_id=%s 再次调用以续接"
                        % (cap, jid)) if wc is not None else \
                       ("等待超时(>%ss)，作业仍在运行，可继续用 job_status 查询" % cap)
        return s
    except Exception as e:
        return _error("提交仿真失败", e)


@mcp.tool(
    name="job_status",
    description="查询仿真作业进度（status: running/done/failed，progress，失败详情，日志尾部）。"
)
def tool_job_status(job_id: str) -> dict:
    """查询作业状态。

    Args:
        job_id: run_simulation 返回的作业ID。
    """
    try:
        return _jobs.status(job_id)
    except Exception as e:
        return _error("查询失败", e)


@mcp.tool(
    name="postprocess",
    description=(
        "对已完成的仿真作业做后处理：解析颗粒信息 dump，计算无量纲响应量——"
        "体系平均动能 E_k*=mean(0.5|U|^2)/(g*d)、纵向平均温度 T_y*（2d 网格胞元"
        "长时间统计）、质心高度 h_m*、平均离底高度 ep*。二元偏析设计（study=segregation）"
        "额外输出偏析响应量：物种高度差 seg_dh_star=（大颗粒平均高度-小颗粒平均高度)/d_s"
        "（正=大颗粒上浮）、Lacey 混合指数 mix_index（0=完全偏析，1=完全混合），"
        "设计参数列 r_s/r_rho/x_large 一并写入。输出物理空间 CSV，可直传 DDDA-MCP。"
        "rows_limit：summary 返回的数据行数（默认 5；跨容器聚合分析时传 500 可取回全部行，"
        "将返回的行内联传给 DDDA run_full_pipeline 的 data 参数）。"
    ),
)
def tool_postprocess(job_id: str, rows_limit: str | int | None = None) -> dict:
    """后处理仿真结果。

    Args:
        job_id: 已完成的作业ID。
        rows_limit: summary 返回行数（默认5，上限500）。
    """
    try:
        kw = _clean_args(locals())
        return postprocess_job(_jobs, job_id, WORK_DIR, HOST_WORK_DIR,
                               rows_limit=kw.get("rows_limit") or 5)
    except Exception as e:
        return _error("后处理失败", e)



@mcp.tool(
    name="merge_responses",
    description=(
        "合并多个仿真作业的响应数据（responses.csv）为一个文件，同一参数组合自动去重"
        "（最新优先）。用于自适应多轮采样：把第一轮广扫描与第二轮边界加密的数据合并后"
        "交给 DDDA 分析，可显著提升边界区域的采样密度。"
        "rows_limit：返回的 rows 内联行数（默认 5；传 500 取回全部合并行，"
        "可直接内联传给 DDDA run_full_pipeline 的 data 参数，跨容器无需共享文件系统）。"
    ),
)
def tool_merge_responses(
    job_ids: str | list | None,
    rows_limit: str | int | None = None,
) -> dict:
    """合并多作业响应数据。

    Args:
        job_ids: 作业ID列表（JSON数组字符串或逗号分隔），如 ["job_20260827_052016","job_20260827_052046"]。
    """
    try:
        kw = _clean_args(locals())
        raw = kw.get("job_ids")
        if isinstance(raw, str):
            raw = raw.strip().strip(chr(34)).strip(chr(39))
            import json as _json
            try:
                raw = _json.loads(raw)
            except Exception:
                raw = [x.strip() for x in raw.split(",") if x.strip()]
        if not isinstance(raw, list) or not raw:
            return {"ok": False, "error": "需要 job_ids 列表"}
        return merge_responses(WORK_DIR, HOST_WORK_DIR, [str(j) for j in raw],
                               rows_limit=kw.get("rows_limit") or 5)
    except Exception as e:
        return _error("合并失败", e)


@mcp.tool(
    name="get_snapshot",
    description=(
        "提取单个工况的某帧颗粒快照（并行数组：x/y/vx/vy/speed/type/radius），"
        "供渲染成快照图与视觉机制分析（分层/对流/气隙等结构）。frame=-1 取最后一帧，"
        "也可取任意帧号（负数从末尾数）。数据量较大：配合渲染工具的引用机制使用。"
    ),
)
def tool_get_snapshot(
    job_id: str,
    case_id: str | int,
    frame: str | int | None = None,
) -> dict:
    """提取颗粒快照。

    Args:
        job_id: 已完成作业ID。
        case_id: 工况编号（responses.csv 的 case_id 列）。
        frame: 帧号（0 起；-1 最后一帧，默认）。
    """
    try:
        kw = _clean_args(locals())
        return get_snapshot(_jobs, kw.get("job_id"), kw.get("case_id"),
                            int(kw.get("frame") if kw.get("frame") is not None else -1))
    except Exception as e:
        return _error("快照提取失败", e)


@mcp.tool(
    name="get_profiles",
    description=(
        "计算单个工况的时间平均观测剖面与场（机制诊断）：大颗粒浓度沿高度剖面 "
        "large_fraction(y)（偏析结构）、时间平均速度场 vx/vy_field（对流环诊断）、"
        "局部体积分数场 solid_fraction_field（气隙/流化诊断）、逐帧物种高度差时间序列"
        "（收敛性检查）。dump 需保留在作业目录。"
    ),
)
def tool_get_profiles(
    job_id: str,
    case_id: str | int,
    max_frames: str | int | None = None,
) -> dict:
    """计算观测剖面与场。

    Args:
        job_id: 已完成作业ID。
        case_id: 工况编号。
        max_frames: 参与平均的最大帧数（均匀抽样），默认 300。
    """
    try:
        kw = _clean_args(locals())
        return get_profiles(_jobs, kw.get("job_id"), kw.get("case_id"),
                            int(kw.get("max_frames") or 300))
    except Exception as e:
        return _error("剖面计算失败", e)


# ================================================================ casegen v1 工具

@mcp.tool(
    name="build_case",
    description=(
        "构建离散元仿真 case：提交结构化 case 描述 JSON（契约见下），校验+渲染为"
        " LIGGGHTS 脚本并落盘（不运行）。返回 job_id 或逐条校验错误（按错误消息"
        "修复 JSON 后重新提交）。物理护栏：杨氏模量≥5e6(SI)、时间步≤0.3倍Rayleigh"
        "上限、插入区堆积分数≤0.55且每维厚度≥2倍最大粒径、粒子总数≤5000、总步数"
        "≤2e6。== case 描述契约 == " + SCHEMA_DOC
    ),
)
def tool_build_case(case: str | dict | None = None) -> dict:
    """构建 case（校验+渲染+落盘）。

    Args:
        case: case 描述 JSON（对象或 JSON 字符串，契约见 description）。
    """
    try:
        if case is None:
            return {"ok": False, "error": "missing_arg",
                    "message": "须提供 case JSON"}
        if isinstance(case, str):
            case = json.loads(case)
        return _casegen.build(case)
    except json.JSONDecodeError as e:
        return {"ok": False, "error": "bad_json",
                "message": f"case 不是合法 JSON: {e}"}
    except Exception as e:
        return _error("构建 case 失败", e)


@mcp.tool(
    name="run_case",
    description=(
        "运行 build_case 构建的仿真作业（单核后台，默认超时 600s）。wait=true 时"
        "同步等待完成。运行结束后用 case_result 取结果。"
    ),
)
def tool_run_case(job_id: str, wait: str | bool | None = None,
                  timeout: str | int | None = None) -> dict:
    """运行 case 作业。

    Args:
        job_id: build_case 返回的作业ID。
        wait: true 时同步等待完成。
        timeout: 墙钟超时秒数，默认 600。
    """
    try:
        w = str(wait or "").lower() in ("true", "1", "yes")
        return _casegen.run(job_id, wait=w,
                            timeout=int(timeout or 600))
    except Exception as e:
        return _error("运行失败", e)


@mcp.tool(
    name="case_result",
    description=(
        "获取 case 作业的状态与结果摘要：运行状态、dump 文件列表、末帧统计"
        "（粒子数/type 分布/质心/包围盒——可用作物理合理性快速检查：颗粒未飞出域、"
        "两类颗粒都在、沉降到位等）。失败时附日志尾部。"
    ),
)
def tool_case_result(job_id: str) -> dict:
    """读取 case 作业结果摘要。

    Args:
        job_id: build_case 返回的作业ID。
    """
    try:
        return _casegen.result(job_id)
    except Exception as e:
        return _error("读取结果失败", e)


@mcp.tool(
    name="analyze_frame",
    description=(
        "对 case 作业的末帧（或指定 dump 文件）做标准几何/物理量分析，返回"
        "审计友好的定量结果（含计算口径说明）。算子：bed_geometry——床层"
        "高度/床层包围体积/堆积分数 packing_fraction/孔隙率（按重力轴，含"
        "粒径修正）；density_profile——沿重力轴分层的局部堆积分数剖面"
        "（分层/流化/表面结构诊断）。不要手工从 bbox 推导这些量——直接用"
        "本工具，数字自带口径可过数值审计。"
    ),
)
def tool_analyze_frame(job_id: str, op: str = "bed_geometry",
                       n_layers: str | int = 10,
                       dump_file: str | None = None) -> dict:
    """标准帧分析算子。

    Args:
        job_id: build_case 返回的作业ID。
        op: bed_geometry（床层几何+堆积分数）或 density_profile（分层密度剖面）。
        n_layers: density_profile 的层数（默认 10）。
        dump_file: 指定 dump 文件名（缺省用最新一帧）。
    """
    try:
        import json as _json
        jdir = os.path.join(_casegen.root, job_id)
        case_path = os.path.join(jdir, "case.json")
        if not os.path.exists(case_path):
            return {"ok": False, "message": f"未知 job_id: {job_id}"}
        with open(case_path, encoding="utf-8") as f:
            case = _json.load(f)
        gravity = case.get("fields", {}).get("gravity", [0, 0, -9.81])
        dumps = sorted(f for f in os.listdir(jdir)
                       if f.startswith("out_") and f.endswith(".txt"))
        if not dumps:
            return {"ok": False, "message": "该作业无 dump 文件"}
        target = dump_file if dump_file else dumps[-1]
        # dump 按步数排序取最大
        if dump_file is None:
            def _step(fn):
                try:
                    return int(fn.rsplit("_", 1)[1].split(".")[0])
                except (ValueError, IndexError):
                    return 0
            target = max(dumps, key=_step)
        path = os.path.join(jdir, target)
        from core.casegen.analysis_ops import bed_geometry, density_profile
        if op == "bed_geometry":
            return bed_geometry(path, gravity)
        if op == "density_profile":
            return density_profile(path, gravity, int(n_layers))
        return {"ok": False, "message": f"未知算子: {op}（bed_geometry|density_profile）"}
    except Exception as e:
        return _error("帧分析失败", e)


@mcp.tool(
    name="modify_case",
    description=(
        "仿真门处置工具：对已构建的 case 打补丁并重建为新作业（原作业保留作对照）。"
        "patch 为段级深合并的 JSON，例：{'phases':[{'name':'settle','steps':600000,"
        "'thermo_every':10000}]} 延长沉降；{'particles':{'insertion':{'region':"
        "[...]}}} 缩小插入区。典型处置映射：D2=RUNNING_IN 或 D6=FAIL → 加"
        "phases 步数（沉降时间加倍起）；D1=FAIL/done_leak → 修 walls（补底壁"
        "/封域）；插入区堆积分数过低导致颗粒雨 → 缩小 insertion.region。两轮"
        "处置仍 FAIL → 停止重试，向用户报告诊断详情（超预案上报）。"
    ),
)
def tool_modify_case(job_id: str, patch: str | dict | None = None) -> dict:
    """补丁重建 case。

    Args:
        job_id: 源作业ID。
        patch: 段级深合并补丁 JSON。
    """
    try:
        if patch is None:
            return {"ok": False, "message": "须提供 patch JSON"}
        if isinstance(patch, str):
            patch = json.loads(patch)
        return _casegen.modify_and_rebuild(job_id, patch)
    except json.JSONDecodeError as e:
        return {"ok": False, "message": f"patch 不是合法 JSON: {e}"}
    except Exception as e:
        return _error("补丁重建失败", e)



@mcp.tool(
    name="analyze_data",
    description=(
        "对工况-响应数据做统计分析：op=fit_power_law（幂律/多元 log-log "
        "回归，返回系数±SE、95%CI、R² 双报 log/物理空间）、op=model_compare"
        "（幂律 vs 线性 vs 二次，AIC/BIC 裁决函数形式）。数据格式：每行 "
        "[x1,...,xk,y]（CSV 文本或行数组）。输出自带统计口径说明——报告引用"
        "这些数字时口径随行（审稿可溯源）。"
    ),
)
def tool_analyze_data(data: str, op: str = "fit_power_law",
                     var_index: str | int = 0) -> dict:
    """统计分析算子（A2-d）。

    Args:
        data: 数据（CSV 文本，每行 x1...xk,y；首行表头自动跳过）。
        op: fit_power_law 或 model_compare。
    """
    try:
        from core.casegen.analysis_ops import fit_power_law, model_compare
        if op == "fit_power_law":
            return fit_power_law(data)
        if op == "model_compare":
            return model_compare(data)
        if op == "residual_view":
            from core.casegen.analysis_ops import residual_view
            return residual_view(data)
        if op == "diff_view":
            from core.casegen.analysis_ops import diff_view
            return diff_view(data, int(var_index or 0))
        return {"ok": False, "message": f"未知算子: {op}"}
    except Exception as e:
        return _error("数据分析失败", e)


def main():
    transport = os.getenv("MCP_TRANSPORT", "streamable-http").strip().lower().replace("_", "-")
    if transport not in {"stdio", "sse", "streamable-http"}:
        raise ValueError("MCP_TRANSPORT must be one of: stdio, sse, streamable-http")
    if transport == "streamable-http":
        import uvicorn
        mcp.settings.json_response = True
        # 平台托管免疫（与 DDDA-MCP 同款，2026-10-16）：非回环绑定时关闭 DNS rebinding
        # 保护——平台网关以内部主机名转发 Host 头，会被 mcp 1.x 默认防护拒为 421
        if mcp.settings.host not in ("127.0.0.1", "localhost", "::1"):
            from mcp.server.transport_security import TransportSecuritySettings
            mcp.settings.transport_security = TransportSecuritySettings(
                enable_dns_rebinding_protection=False)

        class PathNorm:
            """平台托管探测经网关带前缀路径（.../mcpserver/<id>/mcp）——
            一切 HTTP 请求路径归一到 /mcp；healthz 与不带 SSE 头的 GET 直接答 200。"""
            def __init__(self, app):
                self.app = app

            async def __call__(self, scope, receive, send):
                if scope.get("type") == "http":
                    p = scope.get("path", "")
                    _acc = dict(scope.get("headers") or []).get(b"accept", b"")
                    if p.rstrip("/").endswith("/healthz") or (
                            scope.get("method") == "GET" and b"text/event-stream" not in _acc):
                        await send({"type": "http.response.start", "status": 200,
                                    "headers": [(b"content-type", b"application/json")]})
                        await send({"type": "http.response.body", "body": b'{"ok":true}'})
                        return
                    scope["path"] = "/mcp"
                await self.app(scope, receive, send)

        uvicorn.run(PathNorm(TokenAuth(AcceptHeaderCompat(mcp.streamable_http_app()))),
                    host=mcp.settings.host, port=mcp.settings.port)
    else:
        mcp.run(transport=transport)


if __name__ == "__main__":
    main()
