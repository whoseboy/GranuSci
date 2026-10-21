# -*- coding: utf-8 -*-
"""分区假设运行时（结构中介制四机制的纯逻辑层）。

论文阶段二核心实现（骨架文档第四节）：假设循环的四环**触发**是结构性的，
**环内决策**保留给 agent——本模块只实现触发与记录，不做科学判断。

四机制与函数对应：
- a. 实验设计强制引用注册假设 → gate_design()
    dem_build_case / dem_modify_case 须带 hypothesis_id（在档非 retired）
    或显式 exploratory=true；存在未对账报数时一律拒绝新工况（机制 b 的
    硬约束面——agent 不能绕过对账继续铺工况）。
- b. 每批数据返回强制对账 → record_measurement()
    agent 报数（测量提取是环内决策）→ 系统记账 + 对账建议 +
    pending_reconciliation 挂账；hypothesis_update 对相应 hid 的成功
    裁决清账（tools_node 接线层调用 clear_pending）。
- c. 假设死亡强制结构化后继 → on_refuted()
    refuted 迁移自动生成 open_question 账本条目（refuted 位点=新问题
    坐标），agent 无需任何动作后继已结构化存在。
- d. DDDA 分区分析自动触发 → ddda_request() / absorb_ddda()
    报数矩阵满足触发条件时系统直接构造 DDDA 全管线请求（内联数据），
    结果吸收为 last_partition（分区结构+边界+加密建议供机制 a 附加）。

hr 状态结构（dict，走 langgraph state checkpoint 持久化；tools_node
每轮与进程内 hr_store 双向同步——工具闭包不得引用 state，NameError 教训）：
{
  "jobs": {job_id: {"hid": str, "exploratory": bool, "ts": float}},
  "measurements": [{mid, job_id, hid, params: {...}, values: {...},
                    evidence, ts}],
  "pending": [mid, ...],          # 已报数未对账的测量 id
  "last_partition": {...},        # 最近一次 DDDA 分区结果摘要
  "ddda_runs": [{ts, n_rows, y_col, ok, note}],   # 自动触发留痕（验收可指认）
  "enforced": {"builds_gated": 0, "reconciliations": 0, "successors": 0},
}
"""
from __future__ import annotations

import time

# 机制 a 拦截的实验设计工具（granusci 侧拦截，服务器 schema 不感知——
# hypothesis_id/exploratory 两个字段在转发前剥离）
DESIGN_TOOLS = ("dem_build_case", "dem_modify_case")


def augment_design_tool_schemas(bus) -> None:
    """设计门字段进工具 schema（v3hr 教训：拦截消息教 agent 加
    hypothesis_id，但字段不在服务器 schema 里——强模型不塞未声明参数，
    门与模型陷入"拒绝-重试"死循环。schema 层声明，agent 才有加的通道）。

    在 bus.start 后调用：向两个设计工具的 inputSchema 注入
    hypothesis_id/exploratory 字段 + 描述尾注。转发前 gate_design 剥离。
    """
    for name in DESIGN_TOOLS:
        t = getattr(bus, "tools", {}).get(name)
        if t is None:
            continue
        props = t.schema.setdefault("properties", {})
        props["hypothesis_id"] = {
            "type": "string",
            "description": "本工况检验的假设 ID（hypothesis_register 返回的 hid）。与 exploratory 二选一必填——省略会被设计门拒绝"}
        props["exploratory"] = {
            "type": "boolean",
            "description": "显式探索工况标记（无假设依据的扫描/试点）。与 hypothesis_id 二选一必填"}
        t.description = (t.description.rstrip()
                         + "\n【假设运行时】本工具须带 hypothesis_id（在档假设）"
                           "或 exploratory=true，否则被设计门拒绝；工况成功后"
                           "系统自动附最近 DDDA 分区分析的边界加密建议。")

# 报数矩阵触发自动 DDDA 的条件：总行数下限 + 自上次运行后的新增行数
# （run2 教训：>6 时小预算战役全程不触发；incline 教训：4 行触发了但
# 聚类“子空间数据点不足”——8-10 行是分区拟合的实际下限，6 行折中：
# 触发后 n_clusters=1 单区兜底，分区发现等矩阵长大）
DDDA_MIN_ROWS = 6

# job 完成态的观测来源（系统从工具结果里看 job 状态，不依赖 agent 报告）
_DONE_STATES = ("done", "done_warn")


def mark_done_jobs(hr: dict, bus_call_log: list) -> None:
    """从工具调用日志观测 job 完成并把 done 标记写进 hr.jobs（持久——
    resume 后 bus 日志清空但标记随 checkpoint 存活）。

    v3hr run1 教训：报数靠 LLM 自觉 100% 缺席——机制 b/d 全链死亡。
    系统必须自己看见 job 完成，报数门才有依据。
    """
    for e in bus_call_log or []:
        if str(e.get("tool", "")) not in ("dem_case_result", "dem_run_case",
                                          "dem_job_status"):
            continue
        r = e.get("result") or {}
        jid = str(r.get("job_id") or (e.get("args") or {}).get("job_id") or "")
        info = hr.get("jobs", {}).get(jid)
        if info is not None and str(r.get("status")) in _DONE_STATES:
            info["done"] = True


def unreported_done_jobs(hr: dict) -> list[str]:
    """已完成但未报数的工况（报数门依据）：hr.jobs 里 done 且不在
    measurements（job_id 已报数）也不在 reported_none（显式声明无测量
    产出）里的 job。"""
    reported = {m["job_id"] for m in hr.get("measurements", [])}
    reported |= set(hr.get("reported_none", {}).keys())
    return [j for j, info in hr.get("jobs", {}).items()
            if info.get("done") and j not in reported]
DDDA_MIN_NEW = 2


def new_hr() -> dict:
    return {"jobs": {}, "measurements": [], "pending": [],
            "last_partition": {}, "ddda_runs": [],
            "enforced": {"builds_gated": 0, "reconciliations": 0,
                         "successors": 0}}


# ---------------- 机制 a+b：实验设计门 ----------------

def gate_design(name: str, args: dict, hr: dict,
                hyp_exists) -> tuple[dict, dict | None]:
    """实验设计工具的准入裁决。

    返回 (verdict, cleaned_args)：
    - verdict["ok"]=True  → 放行；cleaned_args 为剥离 hypothesis_id/
      exploratory 后的转发参数（服务器 schema 不感知这两个字段）
    - verdict["ok"]=False → 拒绝；verdict["message"] 为教学性错误说明
    hyp_exists(hid)->bool 校验假设在档（由调用方注入 HypothesisStore 查询）。
    """
    if name not in DESIGN_TOOLS:
        return {"ok": True}, args
    hr.setdefault("enforced", {}).setdefault("builds_gated", 0)
    args = dict(args)
    hid = str(args.pop("hypothesis_id", "") or "").strip()
    exploratory = bool(args.pop("exploratory", False))

    # 机制 b 硬约束：未对账报数阻断一切新工况（探索也不行——对账是前提）
    if hr.get("pending"):
        n = len(hr["pending"])
        hr["enforced"]["builds_gated"] += 1
        return {"ok": False, "gated": "unreconciled",
                "message": (
                    f"【假设运行时-对账门】有 {n} 批已报数数据未对账"
                    f"（测量 id: {hr['pending'][:6]}）——新工况构建被暂停。"
                    "先完成对账：对每批数据涉及的假设调用 hypothesis_update"
                    "（testing/supported/refuted，evidence 引用测量 id 与出处），"
                    "或用 hypothesis_query 查看待对账假设的 prediction 与本报数值"
                    "逐条比对。对账完成前不得开新工况、不得提交论文/报告。")}, None

    # 报数门（v3hr run1 教训：报数靠 LLM 自觉 100% 缺席，机制 b/d 全链死亡
    # ——已完成工况未报数时阻断新设计，报数由系统观测的 done 状态强制）
    unreported = unreported_done_jobs(hr)
    if unreported:
        hr["enforced"]["builds_gated"] += 1
        return {"ok": False, "gated": "unreported",
                "message": (
                    f"【假设运行时-报数门】有 {len(unreported)} 个已完成工况未报数"
                    f"（{unreported[:6]}）——新工况构建被暂停。每个已完成工况"
                    "须先 hyp_report_measurement 报数（job_id + params 调控参数 + "
                    "values 响应量测值 + evidence 工具记录出处）；确认无数值测量"
                    "产出的工况（如纯结构观察/失败诊断），报数时给 "
                    "no_measurement_reason 参数说明。报数是分区分析与对账的数据"
                    "源——不报数的工况等于白跑。")}, None

    if not hid and not exploratory:
        hr["enforced"]["builds_gated"] += 1
        return {"ok": False, "gated": "no_reference",
                "message": (
                    "【假设运行时-设计门】工况构建必须引用一条注册假设"
                    "（args 加 hypothesis_id=<hid>，hypothesis_register 返回的 ID）"
                    "或显式标记探索（args 加 exploratory=true）。"
                    "纯探索工况也合法，但必须显式声明——隐式无依据铺网格"
                    "正是假设运行时要拦的行为。若尚无假设，先从文献/推理"
                    "hypothesis_register 登记再设计工况。")}, None

    if hid and not hyp_exists(hid):
        hr["enforced"]["builds_gated"] += 1
        return {"ok": False, "gated": "bad_hid",
                "message": f"hypothesis_id={hid} 不在假设档案中"
                           "（hypothesis_query 可查在档 ID）——不得引用虚构 ID。"}, None

    verdict = {"ok": True, "hid": hid, "exploratory": exploratory}
    return verdict, args


def note_job(hr: dict, job_id: str, verdict: dict) -> None:
    """设计工具成功返回后登记 job → 假设关联（供报数校验与验收统计）。"""
    if not job_id:
        return
    hr["jobs"][job_id] = {"hid": verdict.get("hid", ""),
                          "exploratory": verdict.get("exploratory", False),
                          "ts": round(time.time(), 2)}


# ---------------- 机制 b：报数与强制对账 ----------------

def record_measurement(hr: dict, args: dict, hyp_active) -> tuple[dict, list[dict]]:
    """报数记账：测量入矩阵 + open_question 之外的对账建议。

    hyp_active() -> [{hid, statement, prediction, state}] 当前
    registered/testing 假设（由调用方注入查询）。
    返回 (record, suggestions)；record 含 mid（测量 id）。
    """
    job_id = str(args.get("job_id", "") or "").strip()
    params = args.get("params") or {}
    values = args.get("values") or {}
    evidence = str(args.get("evidence", "") or "").strip()
    hid = str(args.get("hypothesis_id", "") or "").strip()
    no_meas = str(args.get("no_measurement_reason", "") or "").strip()
    if no_meas:
        # 逃生通道：确认无测量产出的工况（纯结构观察/失败诊断）显式销账
        if job_id not in hr.get("jobs", {}):
            return ({"ok": False, "message": f"job {job_id} 不在设计门登记的"
                     "工况里——no_measurement_reason 只用于已构建工况"}, [], None)
        hr.setdefault("reported_none", {})[job_id] = no_meas[:300]
        return ({"job_id": job_id, "no_measurement": True,
                 "message": f"工况 {job_id} 已显式声明无测量产出（{no_meas[:100]}）"
                            "——报数门销账。"}, [], None)
    if not job_id:
        return {"ok": False, "message": "报数须带 job_id（对应 build/run 的工况）"}, [], None
    if not isinstance(params, dict) or not params:
        return {"ok": False, "message": "报数须带 params：本工况的调控参数"
                "（如 {\"gamma\": 2.0, \"D_over_d\": 6}）——供分区矩阵使用"}, [], None
    if not isinstance(values, dict) or not values:
        return {"ok": False, "message": "报数须带 values：本工况的响应量测值"
                "（如 {\"discharge_rate\": 1.45e-3}）"}, [], None
    if not evidence:
        return {"ok": False, "message": "报数须带 evidence：测量值的工具记录"
                "出处（plot_script 输出/dem_analyze_frame 结果等）——"
                "无出处的测量不被接受"}, [], None

    mid = f"M{int(time.time() * 1000) % 10**10:010d}"

    def _numeric(d: dict) -> dict:
        """数值清洗：DDDA 要数值列——"2.0"/2 都归一为 float（不可解析的保留原值，
        会在 ddda_request 的非常数列判定中被字符串集合去重处理）。"""
        out = {}
        for k, v in d.items():
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
        return out

    params, values = _numeric(params), _numeric(values)
    rec = {"mid": mid, "job_id": job_id, "hid": hid,
           "params": params, "values": values,
           "evidence": evidence[:300], "ts": round(time.time(), 2)}
    # 同 job 重复报数=更新而非追加（v3hr_gamma 教训：过程量验证先报 0.0 占位、
    # 正式拟合值后报——追加使矩阵同 job 双行，DDDA 矩阵重复计数）
    hr["measurements"] = [m for m in hr.get("measurements", [])
                          if m.get("job_id") != job_id]
    hr["measurements"].append(rec)

    # 对账建议：注册/检验中假设 × 本批数据（有 hid 的报数优先对账其引用假设）
    suggestions = []
    actives = hyp_active()
    targets = [h for h in actives if (not hid or h["hid"] == hid)]
    for h in targets:
        suggestions.append({
            "hid": h["hid"], "state": h["state"],
            "prediction": h.get("prediction", "") or h.get("statement", ""),
            "this_batch": {**params, **values},
        })
    if suggestions:
        # 涉及的 hid 持久化进测量记录（clear_pending 按此关联销账）
        rec["hypotheses"] = [s["hid"] for s in suggestions]
        hr.setdefault("pending", []).append(mid)
    # 键名一致性预警（v3hr_incline 教训：values 键名跨工况漂移——
    # transport_velocity_ms/v_char_ms/centroid_x_m 混用使 DDDA 矩阵
    # 列对齐失败 9 连败。报数时即时提醒，别等 DDDA 炸了才知道）
    _warn = None
    if len(hr["measurements"]) >= 2:
        prev_keys = {k for m in hr["measurements"][:-1]
                     for k in m.get("values", {})}
        new_keys = set(values.keys())
        if prev_keys and new_keys and not (new_keys & prev_keys):
            _warn = (f"注意：本次报数 values 键名 {sorted(new_keys)} 与此前测量"
                     f"（{sorted(prev_keys)}）完全不同——同一战役的同一响应量"
                     "请用一致键名，否则系统分区分析的矩阵无法对齐。")
    return rec, suggestions, _warn


def clear_pending(hr: dict, hid: str) -> list[str]:
    """hypothesis_update 成功裁决后清账：该 hid 涉及的 pending 测量出列。

    pending 的 mid 只有在所有关联假设都已裁决后才该移除；简化实现：
    对 hid 的裁决视为完成该批数据的对账动作——报数建议列表里出现的
    hid 已裁决即销账（保守：只清与该 hid 直接关联的测量）。
    """
    cleared = []
    for m in list(hr.get("measurements", [])):
        if m["mid"] in hr.get("pending", []) and hid in (m.get("hypotheses") or []):
            hr["pending"].remove(m["mid"])
            cleared.append(m["mid"])
    if cleared:
        hr.setdefault("enforced", {}).setdefault("reconciliations", 0)
        hr["enforced"]["reconciliations"] += len(cleared)
    return cleared


def pending_blocker(hr: dict) -> str:
    """submit_paper/submit_report 预检：未对账报数 → 拒绝理由文本（空=放行）。"""
    if not hr or not hr.get("pending"):
        return ""
    n = len(hr["pending"])
    return (f"【假设运行时-对账门】有 {n} 批已报数数据未对账"
            f"（{hr['pending'][:6]}），不得提交——先 hypothesis_update 完成"
            "每批数据与注册假设的对账（测量提取已完成，裁决不能拖欠）。")


# ---------------- 机制 c：死亡 → 结构化后继 ----------------

def on_refuted(hid: str, rec: dict, evidence: str) -> dict | None:
    """refuted 迁移 → open_question 账本条目（自动，agent 零动作）。

    返回 record_finding 兼容的账本条目；rec 为假设档案记录。
    """
    if not rec:
        return None
    locus = (rec.get("statement", "") or "")[:200]
    return {
        "kind": "open_question",
        "title": f"假设 {hid} 死亡位点（结构化后继）",
        "content": (f"被推翻的主张：{locus}。死亡证据：{evidence[:200]}。"
                    "该位点与新数据一起构成新科学问题候选——后继假设"
                    "（hypothesis_register）应针对失效机理重新表述，"
                    "或在论文 limitation/next-step 引用本条。"),
        "ref_hid": hid,
        "provenance": f"hypothesis_update {hid} -> refuted",
    }


# ---------------- 机制 d：DDDA 自动触发 ----------------

def ddda_request(hr: dict) -> dict | None:
    """满足触发条件时构造 DDDA 全管线请求（内联数据），否则 None。

    触发：矩阵行数 ≥ DDDA_MIN_ROWS 且自上次自动运行后新增 ≥ DDDA_MIN_NEW。
    x_cols 取跨测量非常数参数列；每个响应列各跑一次（上限 3 列）。
    """
    rows = hr.get("measurements", [])
    if len(rows) < DDDA_MIN_ROWS:
        return None
    last_n = hr.get("ddda_runs", [{}])[-1].get("n_rows", 0) if hr.get("ddda_runs") else 0
    if len(rows) - last_n < DDDA_MIN_NEW:
        return None

    # 拼矩阵列：params 列 + values 列
    # 拟合质量列（R²/rmse/误差类）不是响应量——进矩阵只会给 DDDA 增加噪声列
    _QUALITY_RE = ("r2", "rmse", "stderr", "error_", "_err")
    param_cols, value_cols = [], []
    for m in rows:
        for k in m["params"]:
            if k not in param_cols:
                param_cols.append(k)
        for k in m["values"]:
            if k not in value_cols and not any(q in k.lower() for q in _QUALITY_RE):
                value_cols.append(k)
    if not param_cols or not value_cols:
        return None
    # None/缺失列剔除（v3hr_incline 教训：f_hz 整列缺失 → DDDA validate 后
    # 0 行；values 键名跨测量不统一 → y 列大量 None——列构造按实际覆盖筛选：
    # 参数列须 ≥半数行且 ≥3 行有值；覆盖完整的常数参数列保留（分区维度））
    def _col_cover(getter, col):
        present = [v for m in rows if (v := getter(m).get(col)) is not None]
        return len(present) >= max(3, len(rows) // 2)

    x_cols = [c for c in param_cols if _col_cover(lambda m: m["params"], c)]
    # 响应列须数值型（v3hr_incline 教训：regime="quasi-static" 字符串列
    # 混进 y 列——分区标签是 regime 判读结果，不是标度分析的响应量）
    def _numeric_col(getter, col):
        vals = [v for m in rows if isinstance((v := getter(m).get(col)), (int, float))]
        return len(vals) >= max(3, len(rows) // 2)

    value_covered = [c for c in value_cols
                     if _numeric_col(lambda m: m["values"], c)]
    if not x_cols or not value_covered:
        return None

    # 非正数行剔除（v3hr_incline 教训：Γ=0 基线行使 gamma 列含 0，
    # physical 空间对数变换拒绝整次分析。基线行不进自动矩阵——
    # 基线锚点的价值在对账，不在标度拟合）
    def _row_positive(m):
        return all((m["params"].get(c) or 0) > 0 for c in x_cols) and \
            all(isinstance(m["values"].get(c), (int, float))
                and m["values"].get(c) > 0 for c in value_covered[:3])

    fit_rows = [m for m in rows if _row_positive(m)]
    if len(fit_rows) < DDDA_MIN_ROWS:
        return None

    columns = x_cols + value_covered
    table = []
    for m in fit_rows:
        table.append([m["params"].get(c) for c in x_cols]
                     + [m["values"].get(c) for c in value_covered])
    y_cols = value_covered[:3]   # 一次最多分析 3 个响应量
    return {"columns": columns, "rows": table, "x_cols": x_cols,
            "y_cols": y_cols, "n_rows": len(rows)}


def absorb_ddda(hr: dict, y_col: str, result: dict) -> dict:
    """吸收一次 DDDA 结果 → last_partition 摘要 + 留痕。返回摘要文本。"""
    ok = bool(result.get("ok"))
    # 错误信息提取（v3hr_incline 教训：DDDA 失败走 error/traceback 字段，
    # message 为空——note 空白导致 9 连败原因不可诊断）
    err_msg = str(result.get("message") or result.get("error") or "")[:200]
    hr.setdefault("ddda_runs", []).append({
        "ts": round(time.time(), 2), "n_rows": len(hr.get("measurements", [])),
        "y_col": y_col, "ok": ok, "note": "" if ok else err_msg})
    if not ok:
        return (f"【假设运行时-DDDA】响应量 {y_col} 的自动分区分析失败："
                f"{err_msg}——常见原因：报数 params/values 的列名跨工况不一致"
                "（同一战役同一响应量须用同一键名）或某参数列大量缺失。"
                "请检查最近几次报数的键名一致性；修正后后续报数会自动重试。")
    fitting = result.get("fitting") or {}
    parts = []
    for fr in (fitting.get("fitting_results") or [])[:6]:
        exps = fr.get("effective_exponents") or {}
        exp_str = ", ".join(f"{k}^{v:.3f}" for k, v in exps.items())
        parts.append(f"R²={fr.get('r2', 0):.3f}, y∝{exp_str}"
                     + (f" [区 {fr.get('cluster', '?')}]" if fr.get("cluster") is not None else ""))
    summary = (f"δ*={fitting.get('delta_star', '?')}；{len(fitting.get('fitting_results') or [])} 个拟合区："
               + "；".join(parts))
    hr["last_partition"] = {"y_col": y_col, "summary": summary,
                            "n_rows": len(hr.get("measurements", [])),
                            "ts": round(time.time(), 2)}
    return (f"【假设运行时-DDDA 自动分区分析】响应量 {y_col}（{len(hr.get('measurements', []))} 个测量点）："
            f"{summary}。分区结构与边界是下一批工况设计的依据"
            "（边界附近加密采样优于均匀铺网格）；此分析为系统自动触发，"
            "你的裁决权保留：可与你的机制直觉对照，分歧点值得记录。")


def boundary_advice(hr: dict) -> str:
    """机制 a 附加：最近分区结果 → 边界加密建议文本（空=无建议）。"""
    lp = hr.get("last_partition") or {}
    if not lp:
        return ""
    return (f"\n【假设运行时-设计建议】最近分区分析（{lp.get('y_col')}，"
            f"{lp.get('n_rows')} 点）：{lp.get('summary')}。"
            "新工况优先落在区边界附近（加密定界）与带外留出点（检验外推），"
            "而非均匀铺网格——采纳与否是你的裁决权。")
