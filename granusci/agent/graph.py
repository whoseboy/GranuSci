# -*- coding: utf-8 -*-
"""LangGraph 图：主循环（agent ⇄ tools）+ 审稿人门（reviewer）。

拓扑：
  START → agent ⇄ tools（标准工具循环，全量工具自主调用）
  tools →（submit_report 后）reviewer
  reviewer → END（通过或两轮未过）/ agent（不通过，反馈回改）
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .. import config, cross_memory, figures, kb, latex_tools, llm, lit_survey, openalex_lit, scievo, slides_tools
from ..mcp_bus import McpBus, Tool
from ..skills import load_skills
from . import prompts
from .state import ResearchState

SUBMIT_MARKER = "__SUBMIT_REPORT__"
PLAN_MARKER = "__SUBMIT_PLAN__"
LEDGER_MARKER = "__LEDGER_ENTRY__"
PAPER_MARKER = "__SUBMIT_PAPER__"   # submit_paper 的论文 tex（审稿对象，勿进 report.md）
# loop-guard 豁免名单：同参重复调用=合法轮询而非死循环。
# 注意区分两个概念（审计 A-3 教训）：豁免 loop-guard ≠ 允许缓存结果。
# dem_job_status 是时变状态（pending→running→done），曾被错误地放进
# idem_cache 永久缓存名单——异步轮询永远拿到首个状态，收割流程被堵死。
_IDEMPOTENT_POLL_TOOLS = {"dem_postprocess", "kb_search", "litf_read",
                          "dem_job_status", "litx_cite_lookup"}
# 结果可安全缓存的工具（真正幂等：同参数同结果）。dem_job_status 不在此列。
_IDEM_CACHE_TOOLS = {"dem_postprocess", "kb_search", "litf_read"}
# 启动协议结构强制（2026-10-12 webplan 实测教训：意图里一句"快速协议"就能让
# agent 跳过 submit_research_plan 直奔仿真——prompt instruct 不保证计划门触发）。
# 计划批准前仿真类工具不放行；--yolo 在 plan_gate 自动批准，故批量实验不受阻。
_PLAN_GATE_TOOLS = ("dem_generate_design", "dem_build_case",
                    "dem_modify_case", "dem_run_simulation")


def _sanitize_assistant(message: dict) -> dict:
    """只保留协议字段，剔除 reasoning_content 等扩展字段（回传会被 API 拒）。"""
    out = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("tool_calls"):
        out["tool_calls"] = message["tool_calls"]
    return out


# ================================================================ 研究状态账本
_LEDGER_KINDS = {"law", "observation", "number", "open_question", "decision", "artifact"}
_NUM_RE_LEDGER = re.compile(r"\d+\.\d+")


def _render_ledger(ledger: list | None, max_entries: int = 40,
                   max_chars: int = 3500) -> str:
    """账本的提示词视图：compact 单行条目，最新优先（压缩锚点）。"""
    if not ledger:
        return ""
    lines = ["# 研究状态账本（权威记录——消息历史可能被压缩，关键数值以此为准；勿重复记录）"]
    # 勘误加权：errata_for 条目全保留（C-F17：错误条目与更正并存时
    # 后来者只读旧条目就继承错误——勘误链在视图里必须可见）
    keep = []
    for e in ledger[-max_entries:]:
        if e.get("errata_for"):
            keep.append(e)
    for e in ledger[-max_entries:]:
        if not e.get("errata_for") and e not in keep:
            keep.append(e)
    keep = keep[-max_entries:]
    for e in keep:
        content = str(e.get("content", "")).strip()
        if len(content) > 300:
            content = content[:300] + "…"
        prov = str(e.get("provenance", "")).strip()
        era = f" [勘误 L{e['errata_for']}]" if e.get("errata_for") else ""
        lines.append(f"- L{e.get('seq', '?')} [{e.get('kind', '?')}|{e.get('confidence', '?')}]"
                     f"{era} {e.get('title', '')}: {content}（来源: {prov}）")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…（账本视图截断，完整条目见 ledger.json）"
    return text


def _dump_ledger_json(run_dir: Path | None, ledger: list):
    if run_dir is None:
        return
    try:
        (run_dir / "ledger.json").write_text(
            json.dumps(ledger, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass  # 落盘失败不拦研究主流程


def _validate_ledger_entry(args: dict) -> dict | str:
    """record_finding 参数校验。返回 entry dict 或错误信息 str。"""
    kind = str(args.get("kind", "")).strip()
    title = str(args.get("title", "")).strip()
    content = str(args.get("content", "")).strip()
    provenance = str(args.get("provenance", "")).strip()
    confidence = str(args.get("confidence", "medium")).strip() or "medium"
    if kind not in _LEDGER_KINDS:
        return f"kind 须为 {sorted(_LEDGER_KINDS)} 之一"
    if not title:
        return "title 必填（短标题）"
    if len(title) > 60:
        return f"标题过长（{len(title)} 字），压缩到 60 字内"
    if not content:
        return "content 必填"
    if not provenance:
        return "provenance 必填（工具名+关键参数，或 call_ref 编号）"
    if kind in ("law", "number") and not _NUM_RE_LEDGER.search(content):
        return f"{kind} 类条目必须含具体小数数值（照抄工具返回，不许凭记忆写整数/无量纲）"
    errata_for = str(args.get("errata_for", "")).strip()
    return {"kind": kind, "title": title[:60], "content": content,
            "provenance": provenance, "confidence": confidence,
            **({"errata_for": errata_for} if errata_for else {})}


def _merge_tool_log(prev: list, bus_log: list, already_merged: int) -> list:
    """tool_log 合并：state 前段 + 本进程 bus 的新增尾部。

    历史缺陷一：直接 list(bus.call_log) 覆盖——续跑后 bus 只含新分段，
    checkpoint 里的前半程记录被抹掉（数值审计语料与审稿证据链全丢）。
    历史缺陷二：前缀延续判定——续跑进程 bus 永远不含前段，判定每轮失败、
    每轮重复拼接，tool_log 随轮数膨胀出重复条目（加固测试抓出）。
    正确方案：already_merged 记录本进程已并入 state 的 bus 条数（闭包计数器）。
    """
    return list(prev or []) + list(bus_log or [])[already_merged:]


# ================================================================ from_call 字段解析与结构提示（模块级——
# _make_local_tools 与 build_graph 的 tools_node 都要用）
def _dig(src, path: str):
    """按点路径取嵌套字段：'time_series.dh' -> src['time_series']['dh']。
    列表通配：'summary.*.gamma' -> [每元素的 gamma]（跳过缺失项）。
    列表索引：'summary.0.gamma' -> 第 0 个元素的 gamma。"""
    cur = src
    for part in str(path).split("."):
        if isinstance(cur, list):
            if part == "*":
                continue  # 在下一级对每个元素取值（由调用处的展开逻辑处理）
            if part.isdigit() and int(part) < len(cur):
                cur = cur[int(part)]
                continue
            return None
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _dig_expand(src, path: str):
    """支持列表通配的完整版：返回扁平数组。"""
    parts = str(path).split(".")
    cur = src
    # 逐段走，遇 '*' 展开为多值
    values = [cur]
    for part in parts:
        nxt = []
        for v in values:
            if v is None:
                continue
            if isinstance(v, list):
                if part == "*":
                    nxt.extend(v)
                elif part.isdigit() and int(part) < len(v):
                    nxt.append(v[int(part)])
            elif isinstance(v, dict) and part in v:
                nxt.append(v[part])
        values = nxt
        if not values:
            return None
    return values


def _field_tree(src, depth=2, max_items=12):
    """诊断用字段树：顶层字段 + 嵌套结构 + **可直接引用的点路径**（不导出数据，只给形状）。"""
    lines = []
    for k, v in list(src.items())[:max_items]:
        if isinstance(v, dict):
            sub = ", ".join(list(v.keys())[:6])
            paths = ", ".join(f"{k}.{sk}" for sk in list(v.keys())[:6])
            lines.append(f"{k} (dict: {sub}) → 可用路径: {paths}")
            # 二层 dict 再展一层
            for sk, sv in list(v.items())[:4]:
                if isinstance(sv, dict) and depth > 1:
                    lines.append(f"  {k}.{sk} (dict: {', '.join(list(sv.keys())[:6])})")
        elif isinstance(v, list):
            if v and isinstance(v[0], dict):
                cols = ', '.join(list(v[0].keys())[:8])
                lines.append(f"{k} (list[{len(v)}] of dict) → 通配路径: {k}.*.{cols.split(', ')[0]} 等；列: {cols}")
            elif v and isinstance(v[0], list):
                lines.append(f"{k} (矩阵 {len(v)}x{len(v[0])})")
            else:
                lines.append(f"{k} (list[{len(v)}])")
        else:
            lines.append(f"{k} = {str(v)[:30]}")
    return "\n".join(lines)


# ================================================================ 本地工具
def _make_local_tools(skills: dict, run_dir: Path | None = None,
                      bus=None, log_fn: Callable[[str], None] | None = None,
                      agent_model: str | None = None,
                      blind_kb: bool = False,
                      hr_store: dict | None = None,
                      hr_enabled: bool = False,
                      hyptest_root: Path | None = None) -> list[Tool]:
    # 子代理（deep_read/lit_survey）用主智能体同款模型——模型消融实验的有效性前提：
    # 主模型换弱时精读质量也须同步变弱，否则消融被子代理的 SOTA 污染。
    sub_model = agent_model or config.MODEL_PRESETS["agent"]
    fig_dir = (run_dir / "figs") if run_dir else Path("figs")
    _existing = sorted(int(p.stem[4:]) for p in fig_dir.glob("fig_*.pdf") if p.stem[4:].isdigit()) if fig_dir.exists() else []
    fig_state = {"n": _existing[-1] if _existing else 0}  # 续号防覆盖（断点续跑场景）

    def _resolve_refs(obj):
        """把 {"from_call": N, "x": "字段名", ...} 解析为第 N 次工具调用返回中的数组。

        大数据（快照/场/剖面）不经过 LLM 上下文——工具结果超长被截断时，
        完整数据留在 bus.call_log，LLM 用 from_call 引用即可传递给 make_figure。
        字段名支持点路径（如 "time_series.dh"；列表元素通配 "summary.*.gamma"）。
        只给 {"from_call": N} 时，拉取该结果的全部数组字段（快照直传场景）。
        """
        if not isinstance(obj, dict) or "from_call" not in obj:
            return obj
        idx = int(obj["from_call"])
        if bus is None or idx >= len(bus.call_log):
            raise ValueError(f"from_call={idx} 超出工具调用记录范围")
        src = bus.call_log[idx].get("result") or {}
        mapping = {k: v for k, v in obj.items() if k != "from_call"}
        if not mapping:
            return {k: v for k, v in src.items() if isinstance(v, list)}
        out = {}
        for k, f in mapping.items():
            out[k] = _dig_expand(src, f) if isinstance(f, str) else f
        # 任一字段解析为 None 都明确报错——不允许静默画空图。
        # 错误消息带结构化字段树+正确语法示例（paper11 教训：盲试 5+ 次）
        missing = [k for k, v in out.items() if v is None]
        if missing:
            raise ValueError(
                f"from_call={idx} 的字段解析失败: {missing}。该结果的结构：\n"
                f"{_field_tree(src)}\n"
                f"语法：嵌套字段用点路径（如 time_series.dh）；对列表内每个 dict 取字段用 * 通配"
                f"（如 summary.*.gamma 取 summary 数组每个元素的 gamma 列）。不要用 [] 或 [*]。"
                f"小数据（<20 点）也可直接把数值抄进 series 的 x/y。")
        return out

    def kb_search(args: dict) -> dict:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"error": True, "message": "query 不能为空"}
        hits = kb.search(query, k=int(args.get("k", 5)))
        if not hits:
            return {"message": "知识库中没有检索到相关内容", "hits": []}
        blocks = []
        for i, h in enumerate(hits, 1):
            blocks.append(f"[{i}] 来源: {h['source']} | 章节: {h['section'] or '(无)'}\n{h['text']}")
        return {"hits": len(hits), "content": "\n\n".join(blocks)}

    def load_skill(args: dict) -> dict:
        name = str(args.get("name", "")).strip()
        s = skills.get(name)
        if not s:
            return {"error": True, "message": f"无此技能: {name}。可用: {sorted(skills)}"}
        deep = str(args.get("deep", "")).strip()
        if deep:
            # 参考层（三层渐进加载的第三层）：载入 references/ 深层文档
            if deep not in s.refs:
                return {"error": True,
                        "message": f"技能 {name} 无参考文档 {deep}。可用参考: {s.ref_list}"}
            return {"skill": name, "reference": deep, "content": s.refs[deep]}
        result = {"skill": name, "content": s.body}
        if s.refs:
            result["available_references"] = s.ref_list
            result["hint"] = ("本技能有深层参考文档，需要完整框架时用 "
                              "load_skill(name=..., deep=\"<参考名>\") 展开")
        return result

    def record_finding(args: dict) -> dict:
        """研究状态账本：每确认一条关键发现当下就记（数值照抄工具返回）。"""
        rv = _validate_ledger_entry(args)
        if isinstance(rv, str):
            return {"error": True, "message": rv}
        return {LEDGER_MARKER: rv,
                "message": f"已入账 [{rv['kind']}] {rv['title']}。账本会持续出现在你的上下文中，写论文时以账本为数据底稿。"}

    # ---- 假设层与战役编排（A4+B1+B6）----
    from granusci.hypothesis import HypothesisStore  # noqa: E402  研究记忆存储
    _hyp_store = HypothesisStore(root=hyptest_root)
    # run_tag 从 run_dir 名派生（<时间戳>_<tag>）——v1_dryrun 教训：闭包里引用
    # 不存在的 state 变量导致 hypothesis_register 运行时 NameError（单测只测了
    # HypothesisStore 类、没测闭包接线，漏网）
    _run_tag = (run_dir.name.split("_", 2)[2]
                if run_dir is not None and run_dir.name.count("_") >= 2 else "")

    def _mk_hyp_register():
        def handler(args: dict) -> dict:
            r = _hyp_store.register(args, run_tag=_run_tag)
            if not r.get("ok"):
                return {"error": True, "message": r["message"]}
            h = r["hypothesis"]
            return {"message": (f"假设已登记 {r['hid']}（{h['kind']}/"
                                f"{h['confidence']}）。进入生命周期管理：检验时 "
                                f"hypothesis_update(hid, action=state, state=testing)。"
                                f"假设的生死记录是论文核心素材。"),
                    "hid": r["hid"]}
        return handler

    def _mk_hyp_update():
        def handler(args: dict) -> dict:
            hid = str(args.get("hid", ""))
            action = args.get("action", "state")
            if action == "revise":
                if not args.get("new_statement"):
                    return {"error": True, "message": "revise 须给 new_statement"}
                r = _hyp_store.revise(hid, args["new_statement"],
                                      args.get("why", ""),
                                      args.get("new_prediction", ""))
            else:
                r = _hyp_store.update_state(
                    hid, args.get("state", ""), evidence=args.get("evidence", ""),
                    note=args.get("note", ""))
            if not r.get("ok"):
                return {"error": True, "message": r["message"]}
            msg = f"假设 {hid} → {r.get('state', action)}"
            out = {"message": msg}
            if hr_enabled and hr_store is not None:
                import granusci.agent.hyp_runtime as _hrt
                # 机制 b 收口：对该 hid 的成功裁决清待对账账目
                cleared = _hrt.clear_pending(hr_store, hid)
                if cleared:
                    out["reconciled_measurements"] = cleared
                    out["message"] += (f"（已销账 {len(cleared)} 批测量: "
                                       f"{cleared[:6]}）")
                # 机制 c：假设死亡 → 结构化后继自动落账本（open_question）
                if r.get("state") == "refuted":
                    rec = _hyp_store.lifecycle(hid).get("hypothesis")
                    succ = _hrt.on_refuted(hid, rec, args.get("evidence", ""))
                    if succ:
                        hr_store.setdefault("enforced", {})
                        hr_store["enforced"]["successors"] = \
                            hr_store["enforced"].get("successors", 0) + 1
                        out[LEDGER_MARKER] = succ
                        out["message"] += ("；死亡位点已自动记为 open_question"
                                           f"（账本 L?：{succ['title']}）")
            if r.get("note"):
                out["message"] += r["note"]
            return out
        return handler

    def _mk_report_measurement():
        """机制 b+d 的 agent 入口：测量报数（数据落地的结构化声明）。

        报数 → 入账本（kind=measurement）→ 对账建议（registered/testing
        假设 × 本批数据）→ 矩阵达阈值时系统自动运行 DDDA 分区分析并把
        结果附在本返回里（生成环触发——不依赖 agent 主动调 DDDA）。
        """
        async def handler(args: dict) -> dict:
            import granusci.agent.hyp_runtime as _hrt
            def _hyp_active():
                actives = []
                for st in ("registered", "testing"):
                    for h in _hyp_store.query(state=st)["hypotheses"]:
                        actives.append({"hid": h["hid"], "state": h["state"],
                                        "statement": h.get("statement", ""),
                                        "prediction": h.get("prediction", "")})
                return actives
            rec, suggestions, keyname_warn = _hrt.record_measurement(
                hr_store, args, _hyp_active)
            if rec.get("no_measurement"):
                # 逃生通道返回：显式销账（无 error 字段——这是合法路径）
                return {"message": rec.get("message", "已销账")}
            if not rec.get("mid"):
                return {"error": True, "message": rec.get(
                    "message", "报数参数不完整")}
            out = {"message": (f"测量已入账 {rec['mid']}（job {rec['job_id']}，"
                               f"响应 {list(rec['values'].keys())}）——"
                               "已记入研究状态账本（kind=measurement）。"),
                   "mid": rec["mid"]}
            # 测量自动入账本（审计可溯源 + 断点续跑不丢）
            out[LEDGER_MARKER] = {
                "kind": "measurement",
                "title": f"测量 {rec['mid']}（job {rec['job_id']}）",
                "content": (f"params={json.dumps(rec['params'], ensure_ascii=False)}；"
                            f"values={json.dumps(rec['values'], ensure_ascii=False)}；"
                            f"出处：{rec['evidence']}"),
                "provenance": f"hyp_report_measurement {rec['mid']}"}
            if suggestions:
                sug_lines = [
                    f"- {s['hid']}（{s['state']}）预测：{s['prediction'][:160]}"
                    f" ↔ 本批数据：{json.dumps(s['this_batch'], ensure_ascii=False)}"
                    for s in suggestions]
                out["message"] += ("\n\n【假设运行时-对账建议】本批数据涉及在档假设，"
                                   "逐条比对后用 hypothesis_update 裁决"
                                   "（evidence 引用测量 id " + rec["mid"] + "）：\n"
                                   + "\n".join(sug_lines)
                                   + "\n（未完成对账前新工况构建与提交均被暂停）")
                out["reconciliation_suggestions"] = suggestions
            if keyname_warn:
                out["message"] += "\n\n【假设运行时-键名预警】" + keyname_warn
            # 机制 d：矩阵达阈值 → 系统直接跑 DDDA（生成环结构性触发）
            req = _hrt.ddda_request(hr_store)
            if req is not None and bus is not None:
                ddda_parts = []
                for y_col in req["y_cols"]:
                    payload = {"data": {"columns": req["columns"], "rows": req["rows"]},
                               "x_cols": req["x_cols"], "y_col": y_col,
                               "space": "physical", "n_clusters": 1}
                    try:
                        result = await bus.call("ddda_run_full_pipeline", payload)
                        ddda_parts.append(_hrt.absorb_ddda(hr_store, y_col, result))
                    except Exception as e:  # noqa: BLE001  DDDA 故障不阻塞报数
                        ddda_parts.append(
                            _hrt.absorb_ddda(hr_store, y_col,
                                             {"ok": False, "message": f"{type(e).__name__}: {e}"}))
                out["message"] += "\n\n" + "\n\n".join(ddda_parts)
            return out
        return handler

    def _mk_hyp_query():
        def handler(args: dict) -> dict:
            r = _hyp_store.query(state=args.get("state", "") or "",
                                 kind=args.get("kind", "") or "")
            return {"message": f"假设档案 {r['n']} 条",
                    "hypotheses": r["hypotheses"]}
        return handler

    from granusci.hypothesis import KnowledgeLibrary as _KLib
    _klib = _KLib()

    def _mk_knowledge():
        def handler(args: dict) -> dict:
            r = _klib.deposit(args, run_tag=_run_tag)
            if not r.get("ok"):
                return {"error": True, "message": r["message"]}
            return {"message": f"知识已入库 {r['kid']}——后续战役检索本库即为先验"}
        return handler

    def _mk_knowledge_query():
        def handler(args: dict) -> dict:
            r = _klib.query(keyword=str(args.get("keyword", "")),
                            kind=str(args.get("kind", "") or ""))
            return {"message": f"知识库 {r['n']} 条", "entries": r["entries"]}
        return handler

    def _mk_campaign():
        def handler(args: dict) -> dict:
            phase = str(args.get("phase", "") or "").strip()
            if not phase:
                return _hyp_store.campaign_log()
            return _hyp_store.campaign_phase(
                phase, note=str(args.get("note", "")),
                budget_used=args.get("budget_used") or {})
        return handler


    def submit_research_plan(args: dict) -> dict:
        plan = str(args.get("plan", "")).strip()
        if len(plan) < 600:
            return {"error": True,
                    "message": f"研究计划过短（{len(plan)} 字）。计划须含：候选方向与依据、"
                               "科学问题与假设、可行性边界、参数空间与预算、风险与开放问题。"}
        required_kw = ["科学问题"]
        lack = [k for k in required_kw if k not in plan]
        if lack:
            return {"error": True,
                    "message": f"计划缺少必要要素: {lack}（须明确科学问题表述）。补全后重交。"}
        return {PLAN_MARKER: plan, "message": "计划已提交，等待用户批复。"}

    def lit_search_openalex(args: dict) -> dict:
        q = str(args.get("query", "")).strip()
        if not q:
            return {"error": True, "message": "query 必填"}
        return openalex_lit.search(q, limit=int(args.get("limit", 10)))

    def litf_acquire(args: dict) -> dict:
        r = scievo.acquire(doi=args.get("doi"), title=args.get("title"))
        # paper12 教训：agent 反复重试排队中的解析（9 连击）。附等待策略提示
        if isinstance(r, dict) and r.get("status") == "metadata_only":
            r["hint"] = ("MinerU 解析已触发但尚未完成（DOI 路径约 1-2 分钟，题名路径可能更久）。"
                         "不要立即重试本工具——先推进其他研究步骤，稍后（如一个仿真批次后）"
                         "用 litf_read 或 deep_read_paper 读取；排队中的重试不加速解析。")
        return r

    def litf_read(args: dict) -> dict:
        pid = str(args.get("paper_id", "")).strip()
        if not pid:
            return {"error": True, "message": "paper_id 必填（litf_acquire 返回）"}
        return scievo.read(pid, max_chars=int(args.get("max_chars", 24000)),
                           offset=int(args.get("offset", 0)))

    def lit_survey_tool(args: dict) -> dict:
        """批量文献调研子代理：检索→选篇→精读→综述全在独立上下文完成。"""
        topic = str(args.get("topic", "")).strip()
        if not topic:
            return {"error": True, "message": "topic 必填（调研主题）"}
        questions = str(args.get("questions", "")).strip() or topic
        try:
            return lit_survey.run_survey(
                topic, questions,
                max_papers=int(args.get("max_papers", 8)),
                deep_read=int(args.get("deep_read", 2)),
                model=sub_model, mode=str(args.get("mode", "survey")),
                log_fn=log_fn)
        except Exception as e:
            return {"error": True, "message": f"调研子代理失败: {type(e).__name__}: {e}"}

    def deep_read_paper(args: dict) -> dict:
        """精读子代理：独立上下文里通读全文，主上下文只收结构化摘要。

        paper10 事故根因：38+ 次 litf_read 把 450KB 全文灌进主上下文导致死循环。
        本工具把阅读隔离在一次性子会话中。
        """
        pid = str(args.get("paper_id", "")).strip()
        questions = str(args.get("questions", "")).strip() or (
            "1) 论文核心方法与设置 2) 关键定量结果（含数值）"
            " 3) 与偏析机制相关的结论 4) 对本研究可引用的要点")
        if not pid:
            return {"error": True, "message": "paper_id 必填"}
        first = scievo.read(pid, max_chars=24000, offset=0)
        if not first.get("ok"):
            # 排队态给出可执行的等待策略（paper12：9 次盲重试全部失败）
            state = str(first.get("state") or first.get("message") or "")
            if "no_assets" in state or "pending" in state.lower() or "未" in state:
                return {"error": True, "state": first.get("state", "no_assets"),
                        "message": "全文解析排队中（MinerU 处理需分钟级）。"
                                   "连续重试不加速解析——先推进其他研究步骤"
                                   "（仿真/分析/写作均可），完成一个实质步骤后再回来重试本工具。"}
            return first
        total = first.get("total_chars", 0)
        n_seg = max(1, (total + 23999) // 24000)
        conv = [
            {"role": "system", "content":
                "你是文献精读子代理。你将分段接收一篇论文的全文，"
                "最后按用户问题输出结构化摘要。摘要须含关键数值与其所在章节，"
                "标注不确定之处。不要在接收分段时输出。"},
            {"role": "user", "content":
                f"精读问题：{questions}\n\n[分段 1/{n_seg}]\n{first.get('content', '')}"}]
        for k in range(1, n_seg):
            seg = scievo.read(pid, max_chars=24000, offset=k * 24000)
            if not seg.get("ok"):
                break
            conv.append({"role": "user", "content": f"[分段 {k + 1}/{n_seg}]\n{seg.get('content', '')}"})
        conv.append({"role": "user", "content":
            "全文发送完毕。现在输出结构化摘要回答精读问题（中文，含关键数值）。"})
        summary = llm.chat_text(sub_model, conv,
                                temperature=0.1, purpose="deep_read")
        return {"ok": True, "paper_id": pid, "total_chars": total,
                "segments": n_seg, "summary": summary,
                "hint": "全文已在子上下文消化完毕；主对话只需本摘要，不要再用 litf_read 重读本文"}

    def make_figure(args: dict) -> dict:
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_state["n"] += 1
        try:
            spec = dict(args)
            # series / particles 里的 from_call 引用解析为真实数组
            if isinstance(spec.get("series"), list):
                spec["series"] = [_resolve_refs(s) if isinstance(s, dict) else s
                                  for s in spec["series"]]
            if isinstance(spec.get("particles"), dict):
                spec["particles"] = _resolve_refs(spec["particles"])
            # panels 子面板里的 from_call 也要解析（paper13 图4 空图根因：
            # 顶层解析漏了 panels 数组，子面板的 from_call 原样传给引擎）
            if isinstance(spec.get("panels"), list):
                for p in spec["panels"]:
                    if isinstance(p, dict):
                        if isinstance(p.get("series"), list):
                            p["series"] = [_resolve_refs(s) if isinstance(s, dict) else s
                                           for s in p["series"]]
                        if isinstance(p.get("particles"), dict):
                            p["particles"] = _resolve_refs(p["particles"])
            return figures.make_figure(fig_dir, spec, fig_state["n"])
        except Exception as e:
            fig_state["n"] -= 1
            return {"error": True, "message": f"{type(e).__name__}: {e}"}

    def plot_script(args: dict) -> dict:
        """自由绘图：agent 写 matplotlib 代码（受限沙箱执行）。

        代码在预配置好的环境执行（APS rcParams/Viridis 色板/600dpi 已就位），
        数据用 data(N) 取第 N 次工具调用的完整返回（大数据不经上下文），
        或小数据内联（照抄工具返回）。画完的当前 figure 自动保存编号。
        """
        code = str(args.get("code", "")).strip()
        if not code:
            return {"error": True, "message": "code 必填（matplotlib 绘图代码）"}
        # 数据流硬约束（b101 教训→机制：分析/绘图代码禁止手抄大数值数组——
        # 内联数据绕过工具记录，是符号翻转/幽灵点的通道。小常量（≤12 个数值
        # 字面量）放行；数组形态的内联数据拒绝，须用 data(N) 从工具记录取）
        _nums_inline = re.findall(r"-?\d+\.\d+(?:e-?\d+)?", code)
        if len(_nums_inline) > 12:
            return {"error": True,
                    "message": (f"代码内含 {len(_nums_inline)} 个数值字面量（上限 12）——"
                                "数据须用 data(N) 从工具调用记录取（from_call 机制），"
                                "不得手抄进代码。b101 教训：手抄数据曾致 10 处符号翻转。"
                                "如确需常量表（如网格定义），把数值留在一次工具结果里再引用。")}
        # 沙箱纪律：只许画图相关的 import
        import re as _re
        imports = re.findall(r"^\s*(?:from\s+(\w+)|import\s+(\w+))", code, _re.M)
        allowed = {"matplotlib", "numpy", "math", "mpl_toolkits"}
        bad = [a or b for a, b in imports if (a or b).split(".")[0] not in allowed]
        if bad:
            return {"error": True,
                    "message": f"plot_script 沙箱只允许 import {sorted(allowed)}，"
                               f"检测到: {sorted(set(bad))}。数据用 data(N) 取，"
                               "不要读写文件/系统。"}
        for danger in ("os.", "subprocess", "open(", "__import__", "shutil",
                       "pathlib", "exec(", "eval("):
            if danger in code:
                return {"error": True,
                        "message": f"plot_script 禁止使用 {danger}——纯绘图代码，"
                                   "数据用 data(N) 或内联，保存由平台自动完成。"}
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_state["n"] += 1
        n = fig_state["n"]

        def _data(ref: int):
            if bus is None or ref >= len(bus.call_log):
                raise ValueError(f"data({ref}) 超出工具调用记录范围（共 {len(bus.call_log) if bus else 0} 条）")
            return bus.call_log[ref].get("result") or {}

        env = {"plt": figures.plt, "np": figures.np, "data": _data,
               "SINGLE_COL": figures.SINGLE_COL, "DOUBLE_COL": figures.DOUBLE_COL}
        import io
        import contextlib
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<plot_script>", "exec"), env)   # noqa: S102
            fig = figures.plt.gcf()
            if not fig.axes:
                # 无图探查模式放行（#hr-10：agent 用 plot_script 纯 print 探查
                # data(N) 结构——强制有 figure 使探查代码 100% 失败，两场战役
                # 各烧 4-6 步。有 stdout 输出的探查是合法用途）
                printed_probe = buf.getvalue().strip()
                if printed_probe:
                    fig_state["n"] -= 1   # 探查不占图号
                    return {"ok": True, "probe": True,
                            "stdout": printed_probe[:4000],
                            "note": "探查模式（无 figure，仅 stdout）——"
                                    "正式出图请 plt.subplots() 建图"}
                raise ValueError("代码执行后没有可保存的 figure——用 plt.subplots() 建图")
            fig_path = fig_dir / f"fig_{n}.pdf"
            figures._save_both(fig, fig_path)
            figures.plt.close(fig)
        except Exception as e:
            fig_state["n"] -= 1
            figures.plt.close("all")
            return {"error": True,
                    "message": f"绘图代码失败: {type(e).__name__}: {e}。"
                               "常见坑：matplotlib 3.11 删了 basex/basey（用 ax.set_xscale('log',base=10)）；"
                               "mathtext 不支持的命令剥 $ 降级纯文本。"}
        # stdout 捕获进结果（审计 A-1：提示词钦定的派生数值合规通道是
        # "plot_script 打印算式再照抄"——print 内容必须进工具记录，否则
        # 数值审计把合规数值判为孤儿，合规路径反而必被拒）
        printed = buf.getvalue().strip()
        return {"ok": True, "figure": str(fig_path),
                "png": str(fig_dir / f"fig_{n}.png"), "figure_name": f"fig_{n}",
                "stdout": printed[:4000] if printed else "",
                "hint": "rcParams 已预配置（APS 尺寸/Viridis/600dpi）。"
                        "复杂构图建议直接 plt.subplots(nrows, ncols, figsize=(7.0, h))。"
                        "print 输出已记入工具结果（数值审计语料）——派生数值打印后可照抄。"}

    def make_slides(args: dict) -> dict:
        """写汇报幻灯片：slides.tex（Beamer，load_skill("slides-making") 的骨架）。"""
        tex = str(args.get("tex", "")).strip()
        if len(tex) < 500:
            return {"error": True,
                    "message": f"幻灯片源码过短（{len(tex)} 字符）——完整汇报远不止于此"}
        if run_dir is None:
            return {"error": True, "message": "无运行目录"}
        if "\\documentclass" not in tex or "beamer" not in tex:
            return {"error": True, "message": "必须是 Beamer 文档（\\documentclass{beamer}）"}
        issues = slides_tools.slide_structure_audit(tex)
        if issues:
            return {"error": True, "structure_issues": issues,
                    "message": "汇报叙事结构不达标（slides-making 技能的逐帧自检），"
                               "按下列问题修改后重新 make_slides：\n- " + "\n- ".join(issues)}
        (run_dir / "slides.tex").write_text(tex, encoding="utf-8")
        n_frames = tex.count("\\begin{frame}")
        return {"ok": True, "slides": str(run_dir / "slides.tex"), "n_frames": n_frames,
                "message": f"已写入 {n_frames} 帧。用 compile_latex(tex_name='slides.tex') 编译验证。"}

    def compile_slides_tool(args: dict) -> dict:
        if run_dir is None:
            return {"error": True, "message": "无运行目录"}
        return slides_tools.compile_slides(run_dir, args.get("tex_name", "slides.tex"))

    def vision_analyze(args: dict) -> dict:
        image = str(args.get("image", "")).strip()
        question = str(args.get("question", "")).strip()
        if not question:
            return {"error": True, "message": "question 必填"}
        if not image:
            return {"error": True, "message": "image 路径必填（make_figure 返回的 png）"}
        p = Path(image)
        if not p.is_absolute():
            p = (run_dir / image) if run_dir else Path(image)
        try:
            answer = llm.vision_chat(config.MODEL_PRESETS["vision"], str(p), question,
                                     purpose="vision")
        except Exception as e:
            return {"error": True, "message": f"{type(e).__name__}: {e}"}
        return {"ok": True, "image": str(p), "analysis": answer,
                "hint": "视觉分析是定性判读；数值结论仍须以工具返回数据为准"}

    def write_manuscript(args: dict) -> dict:
        tex = str(args.get("tex", "")).strip()
        if len(tex) < 3000:
            return {"error": True,
                    "message": f"手稿过短（{len(tex)} 字符）——完整学术论文（REVTeX）远不止于此。"}
        if run_dir is None:
            return {"error": True, "message": "无运行目录，无法落盘手稿"}
        # 自动注入 graphicspath：图在 figs/ 子目录，tex 里写 fig_1 或 figs/fig_1 均可
        # 须在 graphicx 加载之后（graphicspath 由 graphicx 提供）
        if "graphicspath" not in tex:
            gp = "\\graphicspath{{figs/}}"
            gx = tex.find("usepackage{graphicx}")
            if gx >= 0:
                line_end = tex.find("\n", gx)
                tex = tex[:line_end + 1] + gp + "\n" + tex[line_end + 1:]
            else:
                tex = tex.replace("\\begin{document}", gp + "\n\\usepackage{graphicx}\n\\begin{document}", 1)
        # 引用预检（C-F9 教训→约束：两场战役首稿引用零登记被审稿拒收——
        # 经验库有教训但写作时不强制，"记录器非约束器"。写作阶段即打回）
        _bib_keys = set(re.findall(r"\\bibitem\{([^}]+)}", tex))
        _cited = {k.strip() for grp in re.findall(r"\\cite\{([^}]+)}", tex)
                  for k in grp.split(",")}
        if _bib_keys and bus is not None:
            import json as _json
            _lit_text = " ".join(
                _json.dumps(e.get("result", {}), ensure_ascii=False)
                for e in bus.call_log
                if str(e.get("tool", "")).startswith(("litx_", "lit_", "litf_"))
                or e.get("tool") == "deep_read_paper")
            _unverified = []
            for _bk in _bib_keys:
                _m2 = re.search(
                    r"\\bibitem\{" + re.escape(_bk) + r"\}\s*(.*?)(?=\\bibitem\{|\\end\{thebibliography\})",
                    tex, re.S)
                if not _m2:
                    continue
                _seg = re.sub(r"\[a-zA-Z]+|[{}~]", " ", _m2.group(1))
                _punct = '.,;:()' + chr(0x201c) + chr(0x201d)
                _words = [w.strip(_punct) for w in _seg.split()]
                _words = [w for w in _words if len(w) > 3]
                _probe = _words[:6]
                if _probe:
                    _hits = sum(1 for w in _probe if w.lower() in _lit_text.lower())
                    if _hits < max(3, len(_probe) - 1):
                        _unverified.append(_bk)
            if _unverified:
                return {"error": True,
                        "message": (f"参考文献条目无检索记录支撑: {sorted(_unverified)}"
                                    "——写 bibliography 前必须先用 litx_search_external 检索"
                                    "取得真实元数据（或 litx_cite_register 登记），"
                                    "凭记忆写的书目会被拒。这是写作阶段的硬约束"
                                    "（V1/V2 两场战役首稿同坑）。")}
        path = run_dir / "paper.tex"
        path = run_dir / "paper.tex"
        path.write_text(tex, encoding="utf-8")
        n_sec = tex.count("\\section{")
        # 计数一致性自查（paper12 审稿意见：摘要/图注/结论写 13 工况、正文只枚举 9 个）：
        # 扫描全文同类计数表述，同一限定语搭配不同数字 → 提示作者核对
        counts = re.findall(
            r"(\d+)\s*(?:[- ]?(?:case|工况|点|cases)|个(?:工况|案例|点))", tex)
        from collections import Counter as _Counter
        freq = _Counter(counts)
        multi = {n: c for n, c in freq.items() if c >= 2}
        warn = ""
        if len(multi) >= 2:
            warn = (f"计数一致性自查：文中出现多个高频计数 {dict(multi)}——"
                    "若它们指的是同一对象（如样本数），全文（摘要/图注/结论/正文）"
                    "必须一致且与正文枚举的实际数量相符。逐个核对。")
        result = {"ok": True, "manuscript": str(path), "n_sections": n_sec,
                  "message": f"已写入 {path}（{n_sec} 个 section）。写完/修改后用 compile_latex 编译验证。"}
        if warn:
            result["consistency_warning"] = warn
        return result

    def compile_latex(args: dict) -> dict:
        if run_dir is None:
            return {"error": True, "message": "无运行目录"}
        return latex_tools.compile_latex(run_dir, args.get("tex_name", "paper.tex"))

    def submit_paper(args: dict) -> dict:
        """论文提交：必须先 write_manuscript + compile_latex 成功。"""
        if run_dir is None:
            return {"error": True, "message": "无运行目录"}
        # 假设运行时对账门：未对账报数不得提交（机制 b 的收笔面）
        if hr_enabled and hr_store is not None:
            import granusci.agent.hyp_runtime as _hrt
            blocker = _hrt.pending_blocker(hr_store)
            if blocker:
                return {"error": True, "message": blocker}
        pdf = run_dir / "paper.pdf"
        if not pdf.exists():
            return {"error": True,
                    "message": "尚未编译出 paper.pdf——先 write_manuscript 再 compile_latex，编译通过后提交。"}
        tex = (run_dir / "paper.tex").read_text(encoding="utf-8")
        n_sec = tex.count("\\section{")
        if n_sec < 4:
            return {"error": True,
                    "message": f"论文仅 {n_sec} 个 section，正式论文至少需要 Introduction/Methods/Results/Conclusions 等核心结构。"}
        figs_used = set(re.findall(r"includegraphics\[[^\]]*\]\{(fig_\d+)\}", tex))
        figs_avail = {p.stem for p in fig_dir.glob("fig_*.pdf")} if fig_dir.exists() else set()
        missing = figs_used - figs_avail
        if missing:
            return {"error": True,
                    "message": f"论文引用了不存在的图: {sorted(missing)}。可用图: {sorted(figs_avail)}。"}
        from ..latex_tools import _count_pages  # noqa
        # 引用真实性闸（借鉴 nature-ref-verifier）：\cite 键须在检索记录里出现过
        import re as _re
        cited_keys = set(_re.findall(r"\\cite\{([^}]+)\}", tex))
        cited_keys = {k.strip() for grp in cited_keys for k in grp.split(",")}
        bib_keys = set(_re.findall(r"\\bibitem\{([^}]+)\}", tex))
        dangling = cited_keys - bib_keys
        if dangling:
            return {"error": True,
                    "message": f"正文 \\cite 引用了 thebibliography 中不存在的键: {sorted(dangling)}。补全或修正。"}
        uncited = bib_keys - cited_keys
        if uncited:
            return {"error": True,
                    "message": f"参考文献未被正文引用: {sorted(uncited)}——每条文献必须在正文对应主张处被 \\cite。"}
        # 引用真实性闸 v2（审计 A-8：注释宣称"cite 键须在检索记录里出现过"
        # 但从未实现——完全凭空捏造但正文引用过的 bibliography 能通过双向检查）：
        # 每个 \bibitem 的 title 须出现在 litx/检索类工具记录中（含
        # litx_search_external / lit_search_openalex / litx_cite_register /
        # litx_get_source_doc / deep_read_paper）
        if bus is not None:
            import json as _json
            lit_corpus = []
            for e in bus.call_log:
                t = str(e.get("tool", ""))
                if (t.startswith("litx_") or t.startswith("lit_")
                        or t.startswith("litf_") or t == "deep_read_paper"):
                    lit_corpus.append(_json.dumps(e.get("result", {}), ensure_ascii=False))
            lit_text = " ".join(lit_corpus)
            unverified = []
            for bk in bib_keys:
                # bibitem 后面跟的标题文本（到下一个 \bibitem 或文档尾）
                m2 = _re.search(
                    r"\\bibitem\{" + re.escape(bk) + r"\}\s*(.*?)(?=\\bibitem\{|\\end\{thebibliography\})",
                    tex, _re.S)
                if not m2:
                    continue
                # 标题提取：bibitem 行内最长的非命令文本段
                seg = _re.sub(r"\\[a-zA-Z]+|[{}~]", " ", m2.group(1))
                words = [w.strip('.,;:()“”') for w in seg.split()]
                words = [w for w in words if len(w) > 3]
                # 取前 6 个实质词做语料命中测试（至少 4 个命中才算真实检索过）
                probe = words[:6]
                if not probe:
                    continue
                hits = sum(1 for w in probe if w.lower() in lit_text.lower())
                if hits < max(3, len(probe) - 1):
                    unverified.append(bk)
            if unverified:
                return {"error": True,
                        "message": (f"参考文献条目未在文献工具记录中核验到: {sorted(unverified)}"
                                    "——每条 \\bibitem 须来自一次真实检索（litx_search_external）"
                                    "或登记（litx_cite_register），凭空编造的书目不能过引用门。"
                                    "请检索真实元数据后修正参考文献。")}
        # paper 流走 PAPER_MARKER（独立于 report——report 状态字段是行为报告
        # 专用，若混入论文 tex 会让 report.md 变成论文全文，paper12 死因之一）
        return {PAPER_MARKER: tex, "format": "latex",
                "pdf": str(pdf), "figures": sorted(figs_used),
                "message": "论文已受理（编译通过、结构完整、图表齐备、引用经检索记录核验），进入审稿。"}

    def submit_report(args: dict) -> dict:
        """行为报告：研究全过程的完整记录（非论文翻译）——校验过程记录要素。"""
        # 假设运行时对账门：未对账报数不得提交（机制 b 的收笔面）
        if hr_enabled and hr_store is not None:
            import granusci.agent.hyp_runtime as _hrt
            blocker = _hrt.pending_blocker(hr_store)
            if blocker:
                return {"error": True, "message": blocker}
        report = str(args.get("report", "")).strip()
        if len(report) < 1000:
            return {"error": True,
                    "message": f"行为报告过短（{len(report)} 字）——完整的研究过程记录"
                               "（概览/假设来源/过程时间线/发现/审稿/教训）远不止于此。"
                               "数据源：账本 ledger + trace + 审稿记录。"}
        # 行为报告要素：过程记录的标志词（时间线/决策依据/失败）
        has_process = any(k in report for k in ("过程", "时间线", "阶段", "决策"))
        has_failure = any(k in report for k in ("失败", "绕行", "局限", "教训", "调整"))
        if not (has_process and has_failure):
            return {"error": True,
                    "message": "行为报告缺过程记录要素——须含研究过程叙事（时间线/阶段/决策依据）"
                               "与失败处置记录（失败/绕行/教训）。这不是论文摘要，是过程档案。"}
        return {SUBMIT_MARKER: report, "message": "已受理，进入审稿。"}

    return [
        Tool("kb_search", ("检索本地知识库（领域知识综述、DDDA 方法论、工具契约）。任何领域事实/参数范围/公式定义先查这里。"
                           if blind_kb else
                           "检索本地知识库（本方向博士论文全文、DDDA 方法论、工具契约）。任何领域事实/参数范围/公式定义先查这里。"),
             {"type": "object", "properties": {
                 "query": {"type": "string", "description": "检索问题"},
                 "k": {"type": "integer", "description": "返回条数，默认 5"}},
              "required": ["query"]}, kb_search),
        Tool("load_skill", "加载技能（操作性方法论指引；三层渐进：目录在系统提示词→主体在此→deep 参数展开深层参考）",
             {"type": "object", "properties": {
                 "name": {"type": "string", "description": "技能名"},
                 "deep": {"type": "string", "description": "参考层文档名（可选）——展开该技能的深层完整框架（rubric/范式/模板）"}},
              "required": ["name"]}, load_skill),
        Tool("record_finding", "研究状态账本：记录一条结构化研究发现。每确认一条关键事实当下就记——标度律、关键数值、实验观察、开放问题、重要决策、关键产物路径。账本每轮都在你上下文中且断点续跑后完整（消息历史会被压缩，账本才是权威记忆）；写论文/报告时以账本为数据底稿。记录时数值照抄工具返回。",
             {"type": "object", "properties": {
                 "kind": {"type": "string",
                          "enum": ["law", "observation", "number", "open_question", "decision", "artifact"],
                          "description": "law=标度律/公式; observation=实验观察; number=关键数值; open_question=未解决问题; decision=重要决策及理由; artifact=关键产物（job_id/文件/图）"},
                 "title": {"type": "string", "description": "短标题（≤60字）"},
                 "content": {"type": "string", "description": "正文；law/number 类必须含具体小数数值（照抄工具返回）"},
                 "provenance": {"type": "string", "description": "来源：工具名+关键参数，或截断提示里的 call_ref 编号"},
                 "confidence": {"type": "string", "enum": ["high", "medium", "low"],
                                 "description": "置信度，默认 medium"}},
              "required": ["kind", "title", "content", "provenance"]}, record_finding),
        Tool("hypothesis_register", "假设层：登记一条形式化科学假设（结构化对象，区别于 record_finding 的观察记录）。假设=可证伪的结构性主张（分区/符号/边界/标度/机制），须带可检验预测与数值边界（有则填）。文献型假设须先 litx_cite_register 拿 cite_id。假设将进入生命周期管理（testing→supported/refuted/revised），报告写作时假设生死状态是核心素材。",
             {"type": "object", "properties": {
                 "statement": {"type": "string", "description": "可证伪的假设表述（≥1 句，含变量与方向）"},
                 "kind": {"type": "string", "enum": ["partition", "mechanism", "scaling", "boundary", "sign", "other"],
                          "description": "partition=分区结构; mechanism=机制归因; scaling=标度律; boundary=边界位置; sign=符号/方向"},
                 "prediction": {"type": "string", "description": "可检验预测（实验可证伪的形式）"},
                 "partition_vars": {"type": "array", "items": {"type": "string"},
                                    "description": "主张的分区变量列表"},
                 "boundaries": {"type": "array", "items": {"type": "object"},
                                "description": "数值边界估计（如 [{'var':'gamma','op':'>','value':2.0}]）"},
                 "mechanism": {"type": "string", "description": "机制叙事（哪区什么机制主导）"},
                 "source": {"type": "object",
                            "description": "来源：{'type':'cite','cite_id':'...'} 或 {'type':'reasoning'}"},
                 "confidence": {"type": "string", "enum": ["high", "medium", "low"]}},
              "required": ["statement"]}, _mk_hyp_register()),
        Tool("hypothesis_update", "假设层：迁移假设状态（testing=开始检验 / supported=证据支持 / refuted=被证据推翻 / revised=已修订 / retired=放弃）。refuted 须附证据；死亡位置=新科学问题候选。也可 revise 修订陈述（旧版留档）。",
             {"type": "object", "properties": {
                 "hid": {"type": "string", "description": "hypothesis_register 返回的假设 ID"},
                 "action": {"type": "string", "enum": ["state", "revise"],
                            "description": "state=迁移状态；revise=修订陈述"},
                 "state": {"type": "string", "enum": ["testing", "supported", "refuted", "revised", "retired"]},
                 "evidence": {"type": "string", "description": "证据（工具记录引用；refuted/support 必填）"},
                 "note": {"type": "string"},
                 "new_statement": {"type": "string", "description": "（revise）新表述"},
                 "new_prediction": {"type": "string", "description": "（revise）新预测"},
                 "why": {"type": "string", "description": "（revise）修订原因"}},
              "required": ["hid", "action"]}, _mk_hyp_update()),
        Tool("hypothesis_query", "假设层：查询假设档案（按状态/类型过滤），返回全部历史与生死记录。写报告前查询一次——supported/refuted 的分布是论文核心素材。",
             {"type": "object", "properties": {
                 "state": {"type": "string", "enum": ["draft", "registered", "testing", "supported", "refuted", "revised", "retired", ""]},
                 "kind": {"type": "string", "enum": ["partition", "mechanism", "scaling", "boundary", "sign", "other", ""]}},
              "required": []}, _mk_hyp_query()),
        *([] if not hr_enabled else [Tool(
            "hyp_report_measurement",
            "假设运行时：测量报数（数据落地的结构化声明，报数门强制）。每个工况"
            "跑完并提取出响应量后必须报数——有未报数的已完成工况时新工况构建"
            "被系统拒绝。报数自动入研究状态账本，并触发系统对账（在档假设 × "
            "本批数据）与 DDDA 自动分区分析（矩阵达阈值时，结果附在返回里）。"
            "params 只填本工况的调控参数（变化量），values 填响应量测值，"
            "evidence 须指向产出该测值的工具记录（plot_script 输出/"
            "dem_analyze_frame 结果）。纯结构观察/失败诊断等无数值产出的工况，"
            "只给 job_id + no_measurement_reason 显式销账。",
            {"type": "object", "properties": {
                "job_id": {"type": "string", "description": "对应 build/run 的工况 job_id"},
                "hypothesis_id": {"type": "string", "description": "本工况检验的假设 hid（探索工况可省）"},
                "params": {"type": "object", "description": "本工况调控参数（如 {\"gamma\": 2.0, \"D_over_d\": 6}）"},
                "values": {"type": "object", "description": "响应量测值（如 {\"discharge_rate\": 0.00145}）"},
                "evidence": {"type": "string", "description": "测值的工具记录出处（必填——无出处的测量不被接受）"},
                "no_measurement_reason": {"type": "string", "description": "（无数值产出工况）显式声明无测量产出并销账——纯结构观察/失败诊断用"}},
             "required": ["job_id"]},
            _mk_report_measurement())]),
        Tool("knowledge_deposit", "研究记忆-知识库：把战役中确立的可复用知识入库（标度律/相边界/机制判据），成为后续战役的先验（检索与文献平权）。战役收官前把 supported 假设与关键规律入库。",
             {"type": "object", "properties": {
                 "statement": {"type": "string", "description": "规律陈述"},
                 "kind": {"type": "string", "enum": ["law", "boundary", "mechanism", "pitfall"]},
                 "quantitative": {"type": "string", "description": "定量表述（有则填）"},
                 "evidence": {"type": "string", "description": "证据（账本/工具记录引用）"},
                 "confidence": {"type": "string", "enum": ["high", "medium", "low"]}},
              "required": ["statement", "kind", "evidence"]}, _mk_knowledge()),
        Tool("knowledge_query", "研究记忆-知识库检索：按关键词/类型查历史战役沉淀的规律（研究计划阶段的先验来源之一，与 litx 文献平权）。",
             {"type": "object", "properties": {
                 "keyword": {"type": "string"}, "kind": {"type": "string", "enum": ["law", "boundary", "mechanism", "pitfall", ""]}},
              "required": []}, _mk_knowledge_query()),
        Tool("campaign_log", "战役编排：记录战役阶段推进（screening 粗扫 / mapping 结构图 / targeted 定向加密 / confirmatory 确证 / writing 收笔 / terminated 终止决策）。每次批次转换、预算消耗、终止决策（如提前停机）都记一条——终止决策必须含理由。无参数调用=读战役日志。",
             {"type": "object", "properties": {
                 "phase": {"type": "string", "description": "阶段名（空=只读日志）"},
                 "note": {"type": "string", "description": "决策理由/预算说明"},
                 "budget_used": {"type": "object", "description": "本阶段消耗 {cases:N, tokens:N}"}},
              "required": []}, _mk_campaign()),
        Tool("submit_research_plan", "提交研究计划（研究启动协议的第一步，须在用户批复后方可开始仿真执行）。计划要素：候选方向与文献依据、科学问题与初步假设、可行性边界（诚实）、参数空间与预算、预期产出与风险、需用户拍板的开放问题。",
             {"type": "object", "properties": {
                 "plan": {"type": "string", "description": "研究计划全文（Markdown）"}},
              "required": ["plan"]}, submit_research_plan),
        Tool("lit_search_openalex", "OpenAlex 学术检索（免费开放学术库，元数据级：标题/年份/期刊/DOI/引用数/OA状态）。波尔文献服务不可用时的备用腿；检索发现（找论文）用本工具或 lit_literature_search，精读全文用 litf_acquire。",
             {"type": "object", "properties": {
                 "query": {"type": "string", "description": "检索问题（英文关键词效果更佳）"},
                 "limit": {"type": "integer", "description": "返回条数，默认 10"}},
              "required": ["query"]}, lit_search_openalex),
        Tool("litf_acquire", "获取论文全文并触发 MinerU 解析（本地文献工作台）。输入 DOI（快，推荐）或精确题名。文献发现用 lit_literature_search，本工具用于精读特定论文：拿到 paper_id 后用 litf_read 读全文。",
             {"type": "object", "properties": {
                 "doi": {"type": "string", "description": "论文 DOI，如 10.1103/PhysRevLett.90.014301"},
                 "title": {"type": "string", "description": "精确题名（DOI 未知时用，较慢）"}},
             }, litf_acquire),
        Tool("litf_read", "读取论文全文 markdown（MinerU 解析产物）。⚠️ 只用于精确定位短片段（公式/表格/某节）；整篇通读请用 deep_read_paper（避免全文灌入主上下文）。",
             {"type": "object", "properties": {
                 "paper_id": {"type": "string", "description": "litf_acquire 返回的 paper_id"},
                 "max_chars": {"type": "integer", "description": "本次返回字符数上限，默认 24000"},
                 "offset": {"type": "integer", "description": "阅读偏移（翻页），默认 0"}},
              "required": ["paper_id"]}, litf_read),
        Tool("deep_read_paper", "精读子代理（推荐）：在独立上下文里通读论文全文，返回结构化摘要（方法/关键数值/可引用要点）。主对话只需摘要——整篇通读一律用本工具，不要用 litf_read 逐段灌全文。",
             {"type": "object", "properties": {
                 "paper_id": {"type": "string", "description": "论文ID"},
                 "questions": {"type": "string", "description": "精读要回答的问题（默认：方法/定量结果/机制结论/可引用要点）"}},
              "required": ["paper_id"]}, deep_read_paper),
        Tool("lit_survey", "批量文献调研子代理：围绕一个主题在独立上下文完成多角度检索、选篇、top-k 全文精读与结构化综述。mode=survey（默认）→研究现状综述；**mode=gaps→缺口挖掘**：精读问题自动指向论文 Limitations/Future Work，输出结构化缺口清单（每条含来源论文/作者建议方向/共识缺口标记）——研究假设的来源（配 gap-mining 技能）。比多次 lit_search_openalex 省主上下文；单点查证仍用 lit_search_openalex。",
             {"type": "object", "properties": {
                 "topic": {"type": "string", "description": "调研主题（如 'vibrated granular bed segregation'）"},
                 "questions": {"type": "string", "description": "精读要回答的问题（gaps 模式有默认缺口问题组，可覆盖）"},
                 "mode": {"type": "string", "enum": ["survey", "gaps"], "description": "survey=综述（默认）；gaps=缺口挖掘（假设来源）"},
                 "max_papers": {"type": "integer", "description": "纳入的最大论文数，默认 8"},
                 "deep_read": {"type": "integer", "description": "全文精读篇数上限（有 DOI 优先），默认 2；0=仅元数据综述"}},
              "required": ["topic"]}, lit_survey_tool),
        Tool("make_figure", "生成出版级图表（PDF+PNG，APS 尺寸规范）。图自动编号 fig_1, fig_2...。小数据直接写在 series/particles 里；大数据（快照/场/剖面）用 from_call 引用之前工具调用的完整数据（见工具结果中的 call_ref 提示），如 particles: {\"from_call\": 5} 会取第 5 次调用的返回。",
             {"type": "object", "properties": {
                 "plot_type": {"type": "string", "enum": ["line", "scatter", "loglog", "semilogx", "heatmap", "errorbar", "particles", "panels"],
                                "description": "图类型；particles=颗粒快照渲染（colorby=type/speed）；panels=多面板组图（各面板承担不同论证角色，如 (a)现象(b)机制(c)边界；面板数≤8）"},
                 "xlabel": {"type": "string", "description": "x 轴标签（支持 LaTeX 片段如 $r_s$）"},
                 "ylabel": {"type": "string", "description": "y 轴标签"},
                 "title": {"type": "string", "description": "标题（可选）"},
                 "series": {"type": "array", "description": "数据系列；heatmap 时单系列 {x:[刻度],y:[刻度],z:[[矩阵]]}；数组可换成 {\"from_call\":N,\"x\":\"字段名\",...}。字段名是点路径：嵌套用 'time_series.dh'；对列表里每个元素取字段用 * 通配（'summary.*.gamma' = summary 数组每个元素的 gamma 列；'summary.0.gamma' 只取第 0 个）。不要用 [] 或 [*] 语法",
                             "items": {"type": "object"}},
                 "panels": {"type": "array", "description": "panels 组图的子面板列表，每项是一个完整单面板 spec（plot_type/series/xlabel/ylabel 等）；ncols 指定列数（默认2），sharey=true 统一 y 轴（跨参数面板对比用），panel_labels 自定义面板标号",
                             "items": {"type": "object"}},
                 "ncols": {"type": "integer", "description": "panels 组图列数，默认 2"},
                 "sharey": {"type": "boolean", "description": "panels 组图统一 y 轴范围（同一响应量跨参数组对比时用）"},
                 "particles": {"type": "object", "description": "颗粒快照数据 {x,y,type,radius,speed} 或 {\"from_call\":N}"},
                 "colorby": {"type": "string", "enum": ["type", "speed"], "description": "particles 图着色方式，默认 type"},
                 "width": {"type": "string", "enum": ["single", "double"], "description": "单栏3.4in/双栏7.0in，默认 single"},
                 "diag": {"type": "boolean", "description": "画 y=x 对角参考线（预测 vs 实测对照图用）"},
                 "cbar_label": {"type": "string", "description": "heatmap/speed 色标标签"}},
              "required": ["plot_type", "xlabel", "ylabel"]}, make_figure),
        Tool("plot_script", "自由绘图（matplotlib 代码，受限沙箱执行）：标准图型之外的定制构图用——双 y 轴、inset、分布图、自定义多面板、特殊可视化。写 matplotlib 代码（plt/np 已注入，rcParams 已按 APS 规范配置好：Viridis 色板/字号/600dpi）；大数据用 data(N) 取第 N 次工具调用的完整返回（不经上下文）；小数据照抄工具返回内联。图自动保存编号 fig_N（PDF+PNG）。画图前建议 load_skill('figure-crafting')。数值纪律与 make_figure 相同：数据只来自工具记录。",
             {"type": "object", "properties": {
                 "code": {"type": "string", "description": "matplotlib 绘图代码（plt/np/data/SINGLE_COL/DOUBLE_COL 已注入环境）"}},
              "required": ["code"]}, plot_script),
        Tool("vision_analyze", "视觉模型分析一张图（make_figure 返回的 png，或其他 png 路径）：快照/速度场/浓度场的结构判读（分层、对流环、气隙、团簇等），也可看论文图表提取信息。定性判读用；数值结论以工具数据为准。",
             {"type": "object", "properties": {
                 "image": {"type": "string", "description": "图片路径（make_figure 返回的 png）"},
                 "question": {"type": "string", "description": "要视觉判读的问题，具体明确（如'两类颗粒是否分层？大颗粒位于何处？是否存在底部气隙？'）"}},
              "required": ["image", "question"]}, vision_analyze),
        Tool("write_manuscript", "写入/更新学术论文 LaTeX 手稿（REVTeX 4.2，Physical Review E 格式）到 paper.tex。",
             {"type": "object", "properties": {
                 "tex": {"type": "string", "description": "论文 LaTeX 全文"}},
              "required": ["tex"]}, write_manuscript),
        Tool("compile_latex", "编译 LaTeX（paper.tex 用 pdflatex；slides.tex/beamer 自动切 xelatex）。失败返回错误列表，按错误修复后重编译直至成功。",
             {"type": "object", "properties": {
                 "tex_name": {"type": "string", "description": "paper.tex（论文）或 slides.tex（汇报）"}}},
              compile_latex),
        Tool("submit_paper", "提交学术论文（终态动作）。前置条件：compile_latex 编译出 paper.pdf、结构≥4节、引用的图全部存在。提交后由独立审稿人按期刊标准审计。",
             {"type": "object", "properties": {}}, submit_paper),
        Tool("make_slides", "写入汇报幻灯片 slides.tex（Beamer/XeLaTeX，10pt/16:9）。先 load_skill('slides-making') 拿骨架与密度约束。论文图复用 figs/（graphicspath 已设），但字号要适配投影（轴标签≥18pt 等效——必要时重制简化版）。",
             {"type": "object", "properties": {
                 "tex": {"type": "string", "description": "Beamer LaTeX 全文"}},
              "required": ["tex"]}, make_slides),
        Tool("compile_slides", "编译 slides.tex（XeLaTeX×2）并校验页数（汇报 1 页/1.5-2 分钟）。失败按错误修复。",
             {"type": "object", "properties": {
                 "tex_name": {"type": "string", "description": "默认 slides.tex"}}},
              compile_slides_tool),
        Tool("submit_report", "提交智能体行为报告（Markdown）——研究全过程的完整记录：概览/假设来源（缺口挖掘）/过程时间线（含决策依据与失败绕行）/发现清单/审稿记录/教训。数据源=账本+trace+审稿记录。区别于论文（论证结构）与 run_report.html（机器视图）——这是叙事化的过程档案，供审计与后续 run 复用。",
             {"type": "object", "properties": {
                 "report": {"type": "string", "description": "报告 Markdown 全文"}},
              "required": ["report"]}, submit_report),
    ]


# ================================================================ 数值审计（确定性）
_NUM_RE = re.compile(r"\d+\.\d{2,}")
_CORPUS_NUM_RE = re.compile(r"-?\d+\.\d+")
# 数据管线工具前缀（审计 A-5：Results 数值只认本研究数据管线语料，
# kb/文献类返回的数字属于文献值——只许出现在引言/对照讨论且须标注来源，
# 从 kb 抄数值冒充结果属于"文献数字洗白"，确定性层必须拦）
_DATA_PIPELINE_PREFIXES = ("dem_", "ddda_", "plot_script", "make_figure")
_INT_CONTEXT_RE = re.compile(
    r"(\d+)\s*(?:个工况|工况|个案例|案例|个数据点|数据点|个样本|样本"
    r"|个条件|条件|cases|runs|simulations|particles|颗|个假设)")


def numeric_audit(report: str, tool_log: list[dict],
                  data_only_section: str = "") -> dict:
    """报告中的小数（≥2 位小数位）须能在工具结果中找到出处。

    匹配容忍三种格式差异（均不算孤儿）：
      - 舍入：报告 0.630234 ↔ 工具 0.6302339…（round 后相等即通过）
      - 百分数：报告 13.11 ↔ 工具 0.1311
      - 字面子串：报告值原样出现在工具文本中

    v3 加严（审计 A-5/A-13）：
    - 数据管线分类：Results 类数值（data_only_section 给定时）只认
      dem_/ddda_/plot_script 类工具语料——文献值不得冒充本研究结果
    - 整数计数：正文中的工况/样本/案例计数须与数据管线语料中的
      build/run 调用计数相容（paper12 教训：摘要 13 工况 vs 正文 9 个）
    - 短小数容差收紧：d≤2 位的报告值不再享受区间放行（大语料下
      任意 [x-0.005, x+0.005) 的浮点都能碰撞通过——假阴性高发）
    """
    # 全语料（宽松层：出处存在性）+ 数据管线语料（严格层）
    corpus = json.dumps([e.get("result", {}) for e in tool_log], ensure_ascii=False)
    data_entries = [e.get("result", {}) for e in tool_log
                    if str(e.get("tool", "")).startswith(_DATA_PIPELINE_PREFIXES)]
    data_corpus = json.dumps(data_entries, ensure_ascii=False)
    corpus_floats = [float(x) for x in _CORPUS_NUM_RE.findall(corpus)]
    report_nums = set(_NUM_RE.findall(report))

    def sourced(n: str, c: str, floats: list[float]) -> bool:
        if n in c:
            return True
        try:
            v = float(n)
        except ValueError:
            return False
        d = len(n.split(".")[1])
        for f in floats:
            if round(f, d) == v:            # 舍入一致
                return True
            if round(abs(f), d) == v:       # 绝对值舍入（报告写幅值）
                return True
            if abs(f * 100 - v) < 10 ** (-d) / 2 or abs(f / 100 - v) < 10 ** (-d) / 2:
                return True                 # 百分数换算（报告 13.11 ↔ 语料 0.1311）
        return False

    orphan = sorted(n for n in report_nums
                    if not sourced(n, corpus, corpus_floats))
    out = {"report_decimal_count": len(report_nums), "orphans": orphan}

    # 数据管线洗白检查：Results 段的小数若只在全语料有源、在数据管线
    # 语料无源，标记为 literature_sourced（审稿人核查项，不是硬拒）
    if data_only_section:
        sec_nums = set(_NUM_RE.findall(data_only_section))
        lit_wash = sorted(n for n in sec_nums
                          if sourced(n, corpus, corpus_floats)
                          and not sourced(n, data_corpus, []))
        if lit_wash:
            out["literature_sourced_numbers"] = lit_wash
            out["note_lit_wash"] = ("Results 段以下数值仅文献/KB 语料有源、"
                                    "数据管线无源——若非标注引用的文献对照值，"
                                    "属于文献数字冒充结果，须改写或补数据出处")

    # 整数计数相容性（A-13）：正文计数 vs 数据管线 run/build 计数
    report_counts = {int(m.group(1)) for m in _INT_CONTEXT_RE.finditer(report)}
    if report_counts:
        # 工况创建类工具全集：casegen v2 路径 build/modify（V2r 实况：3 build
        # + 21 modify 派生工况——只数 build 会把 17 个 modify 工况报成计数矛盾）
        # + legacy 设计批 run_simulation。run_case/run 是执行已建 job，不重复计。
        _BUILD_TOOLS = ("dem_build_case", "dem_modify_case", "dem_run_simulation")
        # casegen build 有失败重试，实际成功工况以 result.ok 计
        n_ok = sum(1 for e in tool_log
                   if str(e.get("tool", "")) in _BUILD_TOOLS
                   and isinstance(e.get("result"), dict)
                   and e["result"].get("ok"))
        # 报告声称数 > 成功构建数 → 计数矛盾（声称 < 构建数合法：可能只报告子集）
        mism = sorted(c for c in report_counts if c > max(n_ok, 0) and c > 2)
        if mism:
            out["count_inconsistency"] = {
                "claimed": mism, "successful_builds": n_ok,
                "note": ("正文声称的工况/样本计数超过工具记录中的成功构建数"
                         f"（{mism} > {n_ok}）——检查是否把失败/计划中的工况"
                         "计入了正文（paper12 教训：摘要 13 vs 实际 9）")}
    return out


# ================================================================ LLM 视图压缩（缓存友好架构）
# paratera GLM-5.3 有隐式前缀缓存（实测 99% 命中、寿命≥7 分钟）：请求前缀
# 逐字节稳定时，只有新增部分按全价计——旧滑窗压缩每轮改写历史，缓存全废。
# 新架构两区制：冻结区[0:cut) 一次性压缩 + 追加区[cut:) 原文只追加。
COMPACT_TRIGGER_CHARS = 100_000   # 追加区超过此规模触发一次批量压缩（~5万 token）
COMPACT_KEEP_CHARS = 50_000       # 批量压缩后追加区保留规模（活跃工作集）

_COMPACT_TOOL_TAIL = 250                          # 冻结区 tool 结果保留头部长度
_COMPACT_ASST_HEAD, _COMPACT_ASST_TAIL = 1500, 1000   # 超长 assistant 头尾保留


def _compact_message(m: dict) -> dict:
    """单消息确定性压缩（同样输入永远同样输出——前缀稳定的前提）。"""
    role, c = m.get("role"), m.get("content") or ""
    if role == "tool" and len(c) > 400:
        m = dict(m)
        m["content"] = c[:_COMPACT_TOOL_TAIL] + \
            f" …[已压缩|原{len(c)}字符|关键数值以研究状态账本为准，未记账的用 record_finding 补记]"
    elif role == "assistant" and len(c) > 3000:
        m = dict(m)
        m["content"] = (c[:_COMPACT_ASST_HEAD]
                        + f"\n…[中段压缩{len(c) - _COMPACT_ASST_HEAD - _COMPACT_ASST_TAIL}字符]\n"
                        + c[-_COMPACT_ASST_TAIL:])
    return m


def _advance_compact_cut(messages: list, cut: int) -> int:
    """追加区超阈值时一次性推进冻结边界；平时不动（前缀稳定=缓存命中）。"""
    tail = sum(len(str(m.get("content") or "")) for m in messages[cut:])
    if tail <= COMPACT_TRIGGER_CHARS:
        return cut
    keep, i = 0, len(messages)
    while i > cut and keep < COMPACT_KEEP_CHARS:
        i -= 1
        keep += len(str(messages[i].get("content") or ""))
    return max(cut + 1, i)       # 至少推进 1 条，避免触发后原地踏步


def compress_view(messages: list, compact_cut: int = 0,
                  log_fn: Callable[[str], None] | None = None) -> list:
    """发给 LLM 的历史压缩视图——state 不动（完整历史留给审计/trace）。

    缓存友好两区制：
    - [0:compact_cut) 冻结区：确定性逐消息压缩（只截 content，不动结构与
      tool_call 配对）；user 消息永不压缩（计划、批复、预算告警是关键锚点）
    - [compact_cut:] 追加区：原文全保——历史只追加不改写，视图对同一段
      消息逐字节稳定 → 服务端前缀缓存可命中，每轮只按新增部分计
    - compact_cut 由 agent_node 在追加区超阈值时一次性推进（罕见事件：
      推进那一轮接受缓存失效，其余轮次全命中）
    """
    saved = 0
    out = []
    for i, m in enumerate(messages):
        if i >= compact_cut:
            out.append(m)
            continue
        m2 = _compact_message(m)
        if m2 is not m:
            saved += len(m.get("content") or "") - len(m2.get("content") or "")
        out.append(m2)
    if saved > 20000 and log_fn:
        log_fn(f"[ctx] 冻结区压缩：节省 {saved // 1000}k 字符"
               f"（第 {compact_cut} 条起为追加区原文）")
    return out


# ================================================================ 图构建
def _strip_mechanism_section(skill_body: str) -> str:
    """消融 no_mechanism：剥掉技能里的机制调查节（## 6.5 … 到下一个 ## 之前）。"""
    m = re.search(r"\n## 6\.5 .*?(?=\n## )", skill_body, re.S)
    return skill_body[:m.start()] + skill_body[m.end():] if m else skill_body


def build_graph(agent_model: str, reviewer_model: str, max_steps: int,
                log_fn: Callable[[str], None] = print,
                run_dir: Path | None = None,
                checkpointer=None,
                review_enabled: bool = True,
                ablations: set[str] | None = None,
                excluded_tools: list[str] | None = None,
                blind_kb: bool = False,
                hr_enabled: bool = False):
    ablations = set(ablations or [])
    skills = load_skills()
    if "no_mechanism" in ablations:
        rw = skills.get("research-workflow")
        if rw is not None:
            rw.body = _strip_mechanism_section(rw.body)
    system_prompt = prompts.build_system_prompt(
        skills, experience=cross_memory.load_for_prompt(),
        ablations=ablations, blind_kb=blind_kb, hr_enabled=hr_enabled)
    if excluded_tools:
        # 工具消融（如文献腿对照）：禁用清单进提示词，智能体不浪费步数去试
        system_prompt += ("\n\n# 本配置禁用的工具（实验控制）\n"
                          f"以下工具在本配置中不可用，不要尝试调用："
                          f"{', '.join(excluded_tools)}。"
                          "相关需求改用其余可用工具完成；确实无替代途径时在报告局限中如实说明。")
    # 消融双保险：工具既从 LLM 可见列表移除（openai_tools exclude），
    # 调用时也硬拦截（防幻觉调用照常执行——消融严谨性要求）
    excluded_set = set(excluded_tools or [])
    bus = McpBus(excluded=excluded_set)
    # 分区假设运行时（结构中介制）：进程内可变容器，tools_node 每轮与
    # state.hr 双向同步（工具闭包不得引用 state——hypothesis NameError 教训；
    # state 走 checkpoint，续跑恢复）
    from . import hyp_runtime as _hrt
    from .hyp_runtime import new_hr
    from ..hypothesis import HypothesisStore as _HStore
    hr_store: dict = new_hr()
    # 假设档案查询（tools_node 门校验用；与 _make_local_tools 内实例同路径
    # 同文件——HypothesisStore 走 jsonl 持久化，两个实例读同一档案）
    _hyp_store_rt = _HStore()
    local_tools = _make_local_tools(skills, run_dir=run_dir, bus=bus, log_fn=log_fn,
                                    agent_model=agent_model, blind_kb=blind_kb,
                                    hr_store=hr_store, hr_enabled=hr_enabled)

    # ---------------- agent 节点：一次 LLM 调用（空响应自动重试一次）
    def _ledger_block(state: ResearchState) -> str:
        """账本压缩锚点：随每轮注入，历史压缩后关键数值不丢。"""
        text = _render_ledger(state.get("ledger"))
        return ("\n\n" + text) if text else ""

    async def agent_node(state: ResearchState) -> dict:
        history = [m for m in state["messages"]]
        if not history:
            history = [{"role": "user", "content": state["intent"]}]
        # 软预算（双触发）：①接近步数上限 ②里程碑写作预留——DDDA 定稿（账本有
        # law 条目）后剩余步数的 40% 应留给写作（paper12 教训：写作仅占 6%，
        # 终审 3 条意见全是赶工痕迹）
        n = state.get("n_steps", 0)
        has_law = any(e.get("kind") == "law" for e in (state.get("ledger") or []))
        budget_low = (not state.get("report") and not state.get("paper_tex")
                      and n >= max_steps - 15
                      and not any("预算" in str(m.get("content", ""))[:30] for m in history[-3:]))
        milestone_due = (has_law and n >= max_steps * 0.6
                          and not state.get("paper_tex") and not state.get("report")
                          and not state.get("slides_offered")   # paper13：论文已过审进入交付段后不再触发
                          and not any("写作预留" in str(m.get("content", ""))[:30]
                                      for m in history[-3:]))
        if budget_low:
            history = history + [{"role": "user", "content":
                f"【系统预算告警】已用 {n}/{max_steps} 步。停止一切新的仿真与分析探索，"
                "立即用已有数据收尾：把已有关键结果整理成图（make_figure），"
                "写论文（write_manuscript→compile_latex→submit_paper）。"
                "数据不足的部分在论文局限中如实说明，不得再开新工况。"}]
        elif milestone_due:
            history = history + [{"role": "user", "content":
                f"【写作预留提醒】已用 {n}/{max_steps} 步且标度律已入账（law 条目）——"
                "进入写作阶段：先列论文图集清单（6-8 张，各一句话论证角色），再按清单"
                "出图、写作（write_manuscript→compile_latex→submit_paper）。"
                "确有必要的补充分析从简，把剩余步数的一半以上留给写作与审稿修改；"
                "机制调查已覆盖的部分不再展开新工况。"}]
        # 缓存友好：平时视图只追加（前缀稳定），追加区超阈值才一次性推进冻结边界
        cut = _advance_compact_cut(history, state.get("compact_cut", 0))
        if cut != state.get("compact_cut", 0):
            tail_k = sum(len(str(m.get("content") or "")) for m in history[cut:]) // 1000
            log_fn(f"[ctx] 批量压缩：冻结边界推进到第 {cut} 条消息"
                   f"（追加区约 {tail_k}k 字符原文保留，此后前缀稳定）")
        msgs = [{"role": "system", "content": system_prompt + _ledger_block(state)}] \
            + compress_view(history, compact_cut=cut, log_fn=log_fn)
        import time as _time
        try:
            message = llm.chat(agent_model, msgs, tools=bus.openai_tools(exclude=excluded_set),
                               max_tokens=32000, purpose="agent")
        except llm.LLMError as e:
            # quota 类（团队 token 配额耗尽）：等待重置远超 run 生命周期，直接终止
            # 并提示续跑命令（checkpoint 已保存全部进度——paper12 死因处置）
            if getattr(e, "kind", "") == "quota":
                log_fn(f"[agent] LLM 配额耗尽（等待周期重置），终止: {str(e)[:200]}")
                log_fn(f"[agent] 配额恢复后续跑：python -m granusci.run --resume <本次 run 目录>")
                return {"messages": history, "n_steps": state.get("n_steps", 0) + 1,
                        "final": "llm_quota_exhausted"}
            # permanent 类（鉴权/参数错）重试无意义，直接终止
            if getattr(e, "kind", "") == "permanent":
                log_fn(f"[agent] LLM 永久性错误，终止: {str(e)[:200]}")
                return {"messages": history, "n_steps": state.get("n_steps", 0) + 1,
                        "final": "llm_unavailable"}
            _time.sleep(60)
            try:
                message = llm.chat(agent_model, msgs, tools=bus.openai_tools(exclude=excluded_set),
                                   max_tokens=32000, purpose="agent")
            except llm.LLMError as e2:
                if getattr(e2, "kind", "") == "quota":
                    log_fn(f"[agent] LLM 配额耗尽，终止: {str(e2)[:200]}")
                    log_fn(f"[agent] 配额恢复后续跑：python -m granusci.run --resume <本次 run 目录>")
                    return {"messages": history, "n_steps": state.get("n_steps", 0) + 1,
                            "final": "llm_quota_exhausted"}
                log_fn(f"[agent] LLM 服务不可用，终止: {str(e2)[:200]}")
                return {"messages": history, "n_steps": state.get("n_steps", 0) + 1,
                        "final": "llm_unavailable"}
        if not (message.get("content") or "").strip() and not message.get("tool_calls"):
            # 少数情况下模型对长反馈返回空消息——加一句督促重试
            msgs.append({"role": "assistant", "content": ""})
            msgs.append({"role": "user", "content": "请继续：根据审稿反馈修改报告并重新 submit_report。"})
            message = llm.chat(agent_model, msgs, tools=bus.openai_tools(exclude=excluded_set), purpose="agent")
        if not (message.get("content") or "").strip() and not message.get("tool_calls"):
            return {"messages": history, "n_steps": state.get("n_steps", 0) + 1,
                    "final": "empty_response"}
        # 无行动退出防线：研究未交付（无 report/paper 且非终态）时模型只说话不调工具
        # ——多半是上下文过长或注意力涣散。注入督促消息继续，而不是静默结束整个 run。
        # 交付扩展段收口：论文已过审（slides_offered），agent 回复不带工具调用
        # → 汇报完成或拒绝，run 以 passed 终态
        if (not message.get("tool_calls") and state.get("slides_offered")
                and state.get("review", {}).get("pass")
                and not state.get("final")):
            closing_final = ("submitted_no_review" if not review_enabled
                             else "passed")
            return {"messages": history + [_sanitize_assistant(message)],
                    "n_steps": state.get("n_steps", 0) + 1,
                    "compact_cut": cut, "final": closing_final}
        if (not message.get("tool_calls")
                and not state.get("report") and not state.get("paper_tex")
                and not state.get("final")):
            retry_note = ("你刚才的回复没有调用任何工具，但研究尚未交付（无 submit_paper/submit_report）。"
                          "继续执行研究计划：下一步该调用什么工具就调用；若认为研究已完成，"
                          "走论文流程 make_figure→write_manuscript→compile_latex→submit_paper。")
            log_fn("[agent] 无行动退出拦截，注入督促")
            return {"messages": history + [_sanitize_assistant(message),
                                           {"role": "user", "content": retry_note}],
                    "n_steps": state.get("n_steps", 0) + 1, "compact_cut": cut}
        return {"messages": history + [_sanitize_assistant(message)],
                "n_steps": state.get("n_steps", 0) + 1, "compact_cut": cut}

    # ---------------- tools 节点：执行全部 tool_calls
    loop_state = {"recent": []}   # 连续重复调用检测
    idem_cache: dict = {}         # 幂等工具结果缓存（同参直返，省重复调用）
    merged_bus = {"n": 0}         # 本进程已并入 state.tool_log 的 bus 条数（续跑从 0 起）
    fail_streak: dict[str, int] = {}   # 同名工具连续失败计数（V2r make_slides 17 连败盲区）

    async def tools_node(state: ResearchState) -> dict:
        last = state["messages"][-1]
        tool_calls = last.get("tool_calls") or []
        new_msgs, report, plan, paper_tex = [], None, None, None
        new_ledger: list[dict] = []
        # 假设运行时状态：state（checkpoint 权威）→ 进程内容器（工具闭包读写）
        if hr_enabled and isinstance(state.get("hr"), dict) and state["hr"]:
            hr_store.update(state["hr"])
        if hr_enabled:
            # 报数门依据：从工具日志观测 job 完成态（done 标记持久进 hr.jobs）
            _hrt.mark_done_jobs(hr_store, bus.call_log)
        for tc in tool_calls:
            name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments") or tc["function"].get("args") or "{}"
            try:
                args = json.loads(raw_args)
                if not isinstance(args, dict):
                    args = {"_arg": args}
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            # 启动协议门：计划未批准前拦截仿真执行（结构强制，见 _PLAN_GATE_TOOLS）
            if name in _PLAN_GATE_TOOLS and not state.get("plan_approved"):
                log_fn(f"[plan-gate] 拦截 {name}: 研究计划未批准")
                result = {"error": True, "plan_gate": True,
                          "message": "研究启动协议未完成：研究计划获批前不得开始仿真执行。"
                                     "请先做缺口挖掘（load_skill(\"gap-mining\")，≤15 次调用），"
                                     "再以 submit_research_plan 提交完整计划（候选方向与缺口来源、"
                                     "科学问题与假设、可行性边界、参数空间与预算、开放问题），"
                                     "等待用户批复后再执行。"}
                bus.call_log.append({"tool": name, "args": args, "ts": time.time(),
                                     "duration": 0.0, "ok": False, "result": result})
                new_msgs.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": json.dumps(result, ensure_ascii=False)})
                continue
            # 连续重复调用守卫（防 paper10 式 113 次同参复读死循环）
            # 幂等状态查询豁免（paper12 教训）：wait=false 异步仿真后轮询
            # job_status 属合法等待——同参轮询不是死循环
            sig = (name, raw_args)
            idempotent = name in _IDEMPOTENT_POLL_TOOLS
            if not idempotent:
                loop_state["recent"] = [s for s in loop_state["recent"] if s is not None] + [sig]
            recent = loop_state["recent"]
            dup = sum(1 for s in recent[-4:] if s == sig)
            if dup >= 3 and not idempotent:
                log_fn(f"[loop-guard] 拦截重复调用 {name}（连续 {dup} 次同参数）")
                result = {"error": True, "loop_guard": True,
                          "message": (f"你已连续 {dup} 次用完全相同的参数调用 {name}——"
                                      "这是死循环。停止重复：改用不同参数/翻页参数，"
                                      "或直接基于已有结果继续下一步。")}
                body = json.dumps(result, ensure_ascii=False)
                new_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": body})
                continue
            # 连续失败守卫：同名工具连续 4 次失败（参数各异——loop-guard 签名
            # 盲区）= 策略性失败，不是重试能解决的。停下来诊断，换方法。
            # （V2r 教训：make_slides 17 连败全被放行，白烧 17 步）
            # 上限 3 次拦截：拦截消息也计数，不封顶会死锁（v3hr 实况：agent
            # 若坚持同工具重试被永久拦截；封顶后放行一次给最终出路）
            if 4 <= fail_streak.get(name, 0) < 7:
                log_fn(f"[fail-guard] 拦截 {name}（连续失败 {fail_streak[name]} 次）")
                result = {"error": True, "fail_guard": True,
                          "message": (f"{name} 已连续失败 {fail_streak[name]} 次——"
                                      "这不是重试能解决的（每次参数都不同仍失败说明"
                                      "方法本身有问题）。停止再试：①重读对应技能文件"
                                      "（load_skill）找规范做法；②看最近一次失败返回的"
                                      "具体错误信息，定位根因；③换一条路径达成目标"
                                      "（如手动写文件/降级方案），或如实放弃并在账本"
                                      "记录失败原因。")}
                body = json.dumps(result, ensure_ascii=False)
                new_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": body})
                fail_streak[name] = fail_streak.get(name, 0) + 1  # 拦截也计数，防拦截消息被无视
                continue
            log_fn(f"  -> {name}({json.dumps(args, ensure_ascii=False)[:160]})")
            # 机制 a+b：实验设计工具准入裁决（假设引用/对账门）——拒绝发生在
            # bus.call 之前；放行则剥离 hypothesis_id/exploratory 后转发
            gate_verdict = None
            if hr_enabled and name in _hrt.DESIGN_TOOLS:
                gate_verdict, args = _hrt.gate_design(
                    name, args, hr_store,
                    lambda hid: bool(_hyp_store_rt.lifecycle(hid).get("hypothesis")))
                if not gate_verdict["ok"]:
                    log_fn(f"[hr-gate] 拒绝 {name}: {gate_verdict.get('gated')}")
                    result = {"error": True, "hr_gate": True,
                              **{k: v for k, v in gate_verdict.items() if k != "ok"}}
                    bus.call_log.append({"tool": name, "args": json.loads(raw_args),
                                         "ts": time.time(), "duration": 0.0,
                                         "ok": False, "result": result})
                    body = json.dumps(result, ensure_ascii=False)
                    new_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": body})
                    fail_streak[name] = 0   # 门拒绝不是工具失败
                    continue
            # 结果缓存只对真正幂等的工具（同参同结果）；dem_job_status 等
            # 时变状态工具豁免 loop-guard 但不缓存（审计 A-3：缓存使轮询
            # 永远拿到首个状态，异步收割在代码层被堵死）
            if name in _IDEM_CACHE_TOOLS and sig in idem_cache:
                result = dict(idem_cache[sig])
                result["cached"] = True
                result.setdefault("note", "本工具幂等：返回上次调用的缓存结果，重复调用无新信息")
            else:
                result = await bus.call(name, args)
                if name in _IDEM_CACHE_TOOLS and isinstance(result, dict) and not result.get("error"):
                    idem_cache[sig] = dict(result)
            # 连续失败计数更新（成功清零）
            if isinstance(result, dict) and (result.get("error") or result.get("ok") is False):
                fail_streak[name] = fail_streak.get(name, 0) + 1
            else:
                fail_streak[name] = 0
            # 机制 a 落地：设计工具成功 → 记 job↔假设关联 + 附边界加密建议
            if (hr_enabled and gate_verdict is not None and gate_verdict["ok"]
                    and isinstance(result, dict) and not result.get("error")):
                _hrt.note_job(hr_store, str(result.get("job_id", "")), gate_verdict)
                advice = _hrt.boundary_advice(hr_store)
                if advice:
                    result = dict(result)
                    result.setdefault("message", "")
                    result["message"] = (str(result["message"]) + advice).strip()
                    body_note = advice.strip()
                    log_fn(f"[hr-gate] {name} 附边界建议（{len(body_note)} 字）")
            if SUBMIT_MARKER in result:
                report = result.pop(SUBMIT_MARKER)
            if PAPER_MARKER in result:
                paper_tex = result.pop(PAPER_MARKER)
            if PLAN_MARKER in result:
                plan = result.pop(PLAN_MARKER)
            if LEDGER_MARKER in result:
                entry = result.pop(LEDGER_MARKER)
                base = len(state.get("ledger") or []) + len(new_ledger)
                entry["seq"] = base + 1
                entry["ts"] = round(time.time(), 2)
                new_ledger.append(entry)
            body = json.dumps(result, ensure_ascii=False)
            shown = body[: config.TOOL_RESULT_CHAR_CAP]
            if len(body) > config.TOOL_RESULT_CHAR_CAP:
                call_ref = len(bus.call_log) - 1
                # ACI 修复（paper12：57/65 次绘图失败全是 from_call 盲猜字段）——
                # 截断时附带机器可读的字段清单：agent 无需探查即可写出正确点路径
                shown += (f"\n...[结果过长已截断，共 {len(body)} 字符。完整数据已缓存："
                          f"call_ref={call_ref}。**引用字段清单（直接用，勿猜）**：\n"
                          f"{_field_tree(result, depth=2)}\n"
                          f"make_figure 用 {{\"from_call\": {call_ref}, \"x\": \"<上述字段路径>\", "
                          f"\"y\": \"<...>\"}} 引用（嵌套用点路径，列表元素通配 *）；"
                          f"关键数值请当轮记录使用]")
            new_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": shown})
            entry = bus.call_log[-1]
            log_fn(f"     <- ok={entry['ok']} {entry['duration']}s len={len(body)}")
        update: dict = {"messages": state["messages"] + new_msgs,
                        "tool_log": _merge_tool_log(state.get("tool_log"), bus.call_log,
                                                    merged_bus["n"])}
        merged_bus["n"] = len(bus.call_log)
        if hr_enabled:
            # 假设运行时状态写回（state → checkpoint 持久化；续跑恢复）
            update["hr"] = dict(hr_store)
        if new_ledger:
            ledger = list(state.get("ledger") or []) + new_ledger
            update["ledger"] = ledger
            _dump_ledger_json(run_dir, ledger)   # 消融实验取数 + 崩溃可查
        if report is not None:
            update["report"] = report
        if paper_tex is not None:
            update["paper_tex"] = paper_tex   # 审稿对象；report 字段留给行为报告
        if plan is not None:
            update["plan"] = plan
            update["plan_approved"] = False   # 新计划待批（重交场景）
        return update

    # ---------------- plan_gate 节点：人工批复（interrupt）
    def plan_gate_node(state: ResearchState) -> dict:
        """挂起等用户批复。resume 值即批语（"批准"或修改意见）。"""
        from langgraph.types import interrupt
        feedback = interrupt({"plan": state.get("plan") or ""})
        feedback = str(feedback or "").strip() or "批准"
        return {"plan_feedback": feedback}

    # ---------------- inbox_gate 节点：运行中用户指令注入（每 10 步查一次收件箱）
    def inbox_gate_node(state: ResearchState) -> dict:
        """run_dir/inbox.md 存在且非空时挂起，把内容作为用户指令注入。

        用户随时往 inbox.md 写指令（文件被消费后自动归档为 inbox_consumed.md）。
        分离进程（后台）与交互终端两种运行方式都适用。
        """
        from langgraph.types import interrupt
        if run_dir is None:
            return {}
        inbox = run_dir / "inbox.md"
        if not inbox.exists():
            return {}
        content = inbox.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            return {}
        interrupt({"user_message": content[:4000]})
        # resume 后指令入史并归档收件箱——目标名带时间戳（Windows rename
        # 到已存在目标抛 FileExistsError，审计 A-11：第二条指令必崩）
        _dest = run_dir / f"inbox_consumed_{int(time.time())}.md"
        try:
            inbox.rename(_dest)
        except OSError:
            import os as _os
            _os.replace(str(inbox), str(_dest))
        return {"messages": (state.get("messages") or []) + [
            {"role": "user", "content": f"【用户中途指令】{content[:4000]}"}]}

    # ---------------- no_review gate：消融 no_reviewer 时的交付管线（不依赖审稿事件）
    # paper12 教训（review_off 消融暴露的 harness 缺陷）：交付扩展段（slides 制作、
    # 交付收口）原本寄生在"审稿 PASS"事件上——审稿门被消融时整个后段流程死掉，
    # run 以 aborted 终态结束。修复：剥离审稿的 run 走本节点，用"提交动作+产物
    # 完备性"驱动同样的交付管线。
    async def no_review_gate_node(state: ResearchState) -> dict:
        update: dict = {}
        # 交付收口：slides 已产出（或 run_dir 无交付目录需求）→ 正常终态
        slides_done = run_dir is not None and (run_dir / "slides.pdf").exists()
        if state.get("slides_offered") and slides_done:
            update["final"] = "submitted_no_review"
            update["review"] = {"pass": True, "ablated": True,
                                "critical_issues": [], "suggestions": [],
                                "feedback": "（审稿门已在本配置中剥离）"}
            return update
        # 首次提交（论文或报告）：注入 slides 交付指令，继续 agent 循环
        if not state.get("slides_offered"):
            update["slides_offered"] = True
            update["final"] = ""          # 暂不终态
            update["review"] = {"pass": True, "ablated": True,
                                "critical_issues": [], "suggestions": [],
                                "feedback": "（审稿门已在本配置中剥离，产出未经评审）"}
            update["messages"] = state["messages"] + [{"role": "user", "content":
                "【交付扩展】研究产出已提交（本配置已剥离审稿门，产出未经评审）。"
                "生成工作汇报幻灯片（load_skill('slides-making') → make_slides → "
                "compile_slides），复用论文图表。完成后说明即可；若确认本次不需要"
                "汇报，回复说明后结束。"}]
            log_fn("[no_review] 首次提交，注入 slides 交付指令（审稿门已剥离）")
            return update
        # slides_offered 但 slides 未产出且未拒绝 → 重新提醒（与 reviewer 的
        # 交付提醒分支同语义；拒绝路径由 agent_node 收口逻辑处理）
        update["final"] = ""
        update["messages"] = state["messages"] + [{"role": "user", "content":
            "【交付提醒】汇报幻灯片（slides.pdf）尚未产出——现在完成 "
            "load_skill('slides-making') → make_slides → compile_slides。"
            "若确认本次不需要汇报，直接回复说明即可结束。"}]
        log_fn("[no_review] slides 未产出，重注入交付提醒")
        return update

    # ---------------- reviewer 节点：确定性数值审计 + 视觉图审计 + 独立 LLM 评审
    def _vision_figure_audit(tex: str) -> list[dict]:
        """论文图的视觉审计（paper13 图4 空图逃过全部检查的教训）：
        每张 includegraphics 的图过一次视觉模型——空图/点数与 caption 不符/
        图例缺失/乱码。失败非阻塞（视觉服务故障不拦审稿流程）。
        """
        results = []
        if run_dir is None:
            return results
        # 每个 figure 环境提取 includegraphics + 配对 caption
        # （\s* 容忍 caption 与 label 间的换行——paper13 实测格式）
        for m in re.finditer(
                r"\\includegraphics\[[^\]]*\]\{(fig_\d+)\}.*?\\caption\{(.*?)\}"
                r"\s*\\label", tex, re.S):
            fig_name, caption_raw = m.group(1), m.group(2)
            png = run_dir / "figs" / f"{fig_name}.png"
            if not png.exists():
                continue
            q = ("这是学术论文的一张插图。图的 caption 摘要："
                 f"{caption_raw[:300]}。请检查："
                 "(1) 是否为空图（无数据点/曲线/颗粒，只有空坐标系）？"
                 "(2) 图上数据元素的数量级是否与 caption 声称的相符（如声称 N 个数据点/粒子）？"
                 "(3) 有多种颜色系列但无图例或色标吗？"
                 "(4) 有乱码、方框字符或文字重叠吗？"
                 '只输出 JSON：{"empty": true/false, "issues": ["问题描述", ...]}')
            try:
                raw = llm.vision_chat(config.MODEL_PRESETS.get(
                    "vision", "Qwen3.8-Max"), str(png), q)
                jm = re.search(r"\{.*\}", raw, re.S)
                v = json.loads(jm.group(0)) if jm else {"empty": False,
                                                        "issues": ["视觉判读解析失败"]}
            except Exception as e:
                v = {"empty": False, "issues": [f"视觉审计调用失败: {type(e).__name__}"]}
            v["figure"] = fig_name
            results.append(v)
            log_fn(f"[reviewer] 视觉审计 {fig_name}: empty={v.get('empty')} "
                   f"issues={v.get('issues', [])[:2]}")
        return results

    async def reviewer_node(state: ResearchState) -> dict:
        rounds = state.get("review_rounds", 0)
        # 审稿对象切换时 rounds 重置（审计 A-6：论文 PASS→行为报告提交时
        # rounds 残留 1，报告的首轮审稿被强制执行"二轮协议"只验修复——
        # V2 第五轮马拉松的直接机制根因）。对象标记用 paper_tex 有无识别。
        _obj = "paper" if state.get("paper_tex") else "report"
        _prev_obj = state.get("review_object")
        if _prev_obj and _prev_obj != _obj:
            rounds = 0   # 新对象从第一轮开始，注入上下文由下方 rounds==0 分支处理
        
        # 审稿对象：行为报告或论文 tex（submit_paper 流）
        report = state.get("report") or state.get("paper_tex") or ""
        # Results 段提取（文献洗白检查的范围；找不到该节时退化为全文宽松层）
        _sec_re = re.compile(
            r"\\section\{[^}]*Results[^}]*\}(.*?)(?=\\section\{|\\end\{document\})",
            re.S | re.I)
        _m = _sec_re.search(report)
        _results_sec = _m.group(1)[:20000] if _m else ""
        audit = numeric_audit(report, state["tool_log"],
                              data_only_section=_results_sec)
        log_fn(f"[reviewer] 数值审计: {audit['report_decimal_count']} 个小数, "
               f"{len(audit['orphans'])} 个无出处 {audit['orphans'][:10]}")
        # 视觉图审计（论文流）：每张图过视觉模型，结果作为审稿证据注入
        vision_audit = _vision_figure_audit(report) if state.get("paper_tex") else []
        vision_digest = ""
        if vision_audit:
            lines = [f"- {v['figure']}: {'**空图**' if v.get('empty') else '有数据'}"
                     + (f"；问题: {v.get('issues', [])}" if v.get("issues") else "")
                     for v in vision_audit]
            vision_digest = ("\n\n# 视觉图审计结果（视觉模型逐图检查产出）\n"
                             + "\n".join(lines))
            log_fn(f"[reviewer] 视觉图审计: {len(vision_audit)} 张图, "
                   f"{sum(1 for v in vision_audit if v.get('empty'))} 张空图")
        # 证据摘要总量控制（审计 A-12：无上限时长 run 达 18 万字符撑爆
        # 审稿上下文）——按工具类别加权：数据管线优先保留，检索类截短，
        # 总量封顶 6 万字符（约 3 万 token）
        _DIGEST_CAP = 60_000
        _PRIO_TOOLS = ("dem_", "ddda_", "plot_script", "hypothesis_",
                       "record_finding", "submit_", "litx_cite")
        _prio, _rest = [], []
        for e in state["tool_log"]:
            r = json.dumps(e.get("result", {}), ensure_ascii=False)
            _is_prio = str(e.get("tool", "")).startswith(_PRIO_TOOLS)
            _nl = chr(10)
            _entry = (f"[{e['tool']}] args={json.dumps(e['args'], ensure_ascii=False)[:300]}"
                      + _nl + f"result={r[:3000 if _is_prio else 800]}")
            (_prio if _is_prio else _rest).append(_entry)
        digest = _prio + _rest
        _total = sum(len(d) for d in digest)
        while digest and _total > _DIGEST_CAP:
            _total -= len(digest.pop(0))
        digest = digest[-60:]
        ledger_digest = ""
        if state.get("ledger"):
            entries = "\n".join(
                f"- L{e['seq']} [{e['kind']}|{e.get('confidence', '')}] {e['title']}: "
                f"{str(e['content'])[:180]}（来源: {e.get('provenance', '')}）"
                for e in state["ledger"][-40:])
            ledger_digest = (f"\n\n# 研究状态账本（智能体自记的关键发现——"
                             f"报告中声称的发现应与账本一致）\n{entries}")
        user = (f"# 待审报告\n{report}\n\n# 数值审计结果（确定性程序产出）\n"
                f"报告小数 {audit['report_decimal_count']} 个，其中未在工具记录中找到出处的："
                f"{audit['orphans']}\n\n# 工具调用记录（摘）\n" + "\n---\n".join(digest[-60:])
                + ledger_digest + vision_digest)
        # 二轮 re-review：注入第一轮意见，审稿人只验证修复情况不重开新战场
        if rounds >= 1:
            prev = state.get("review") or {}
            prev_issues = json.dumps(
                {"critical_issues": prev.get("critical_issues", []),
                 "suggestions": prev.get("suggestions", []),
                 "feedback": str(prev.get("feedback", ""))[:2000]},
                ensure_ascii=False)
            user += (f"\n\n# 第一轮审稿意见（本轮是第二轮——按二轮协议逐条验证修复，"
                     f"只在出现修复副作用时提新问题）\n{prev_issues}")
        raw = None
        for _attempt in range(2):   # 审稿调用与 agent 同款重试（审计 A-10：
            # 一次网络抖动曾直接杀 run——agent 有 60s 重试而 reviewer 没有）
            try:
                raw = llm.chat_text(reviewer_model, [
                    {"role": "system", "content": prompts.REVIEWER_SYSTEM_PROMPT},
                    {"role": "user", "content": user}], temperature=0.0, purpose="reviewer")
                break
            except llm.LLMError as e:
                if e.kind in ("quota", "permanent") or _attempt:
                    _llm_err = e
                    break
                await asyncio.sleep(60)
                continue
        if raw is None:
            e = _llm_err
        if raw is None:
            # 配额/服务故障：与 agent_node 同款处置——可恢复终态，续跑从审稿重入
            if getattr(e, "kind", "") in ("quota", "permanent"):
                log_fn(f"[reviewer] LLM {getattr(e, 'kind')} 错误，终止待续跑: {str(e)[:160]}")
                return {"final": "llm_quota_exhausted" if getattr(e, "kind") == "quota"
                        else "llm_unavailable"}
            raise
        try:
            m = re.search(r"\{.*\}", raw, re.S)
            verdict = json.loads(m.group(0)) if m else {"pass": False, "feedback": raw}
        except json.JSONDecodeError:
            verdict = {"pass": False, "feedback": raw}
        passed = bool(verdict.get("pass"))
        if audit["orphans"]:
            passed = False
            verdict.setdefault("critical_issues", []).append(
                f"报告中 {len(audit['orphans'])} 个数值在工具记录中无出处: {audit['orphans'][:15]}")
        # 视觉审计硬失败：空图 = 图文断层的最严重形态，直接拒收（paper13 图4 教训）
        empty_figs = [v["figure"] for v in vision_audit if v.get("empty")]
        if empty_figs:
            passed = False
            verdict.setdefault("critical_issues", []).append(
                f"论文图 {empty_figs} 经视觉审计确认为空图（无数据元素）——"
                f"caption 声称的内容与图上实际呈现完全不符，必须修复绘图后重交")
        verdict["pass"] = passed
        verdict["numeric_audit"] = audit
        if vision_audit:
            verdict["vision_audit"] = vision_audit
        log_fn(f"[reviewer] 判定: {'PASS' if passed else 'FAIL'} (round {rounds + 1}) "
               f"issues={verdict.get('critical_issues', [])[:5]}")
        update: dict = {"review": verdict, "review_rounds": rounds + 1,
                        "review_object": _obj}
        if passed or rounds + 1 >= 2:
            update["final"] = "passed" if passed else "review_failed_twice"
            if not passed:
                # v2_hopper 教训：终态 FAIL 不清审稿对象标记，resume 复活后首个
                # 工具轮直接重审旧稿（四条问题全部"未修复"的假象）
                update["report"] = None
                update["paper_tex"] = None
            # P1 修复：论文过审后的交付扩展段——首次 PASS 注入汇报制作指令
            # （reviewer PASS 即终态会把 slides 的产出路径完全堵死）
            if passed and not state.get("slides_delivered")                     and not state.get("slides_offered"):
                update["slides_offered"] = True
                update["final"] = ""      # 暂不终态
                update["report"] = None   # 清报告标记（防 slides 阶段反复回审稿门）
                update["paper_tex"] = None
                update["messages"] = state["messages"] + [{"role": "user", "content":
                    "【交付扩展】论文已通过审稿。生成工作汇报幻灯片（load_skill("
                    "'slides-making') → make_slides → compile_slides），复用论文图表。"
                    "完成后说明即可；若确认本次不需要汇报，回复说明后结束。"}]
            elif (passed and state.get("slides_offered")
                    and run_dir is not None and not (run_dir / "slides.pdf").exists()):
                # paper13 教训：首次 PASS 注入 slides 指令后，agent 可能先交行为报告——
                # 报告 PASS 不能直接终态把 slides 截断。slides 未产出且未明确拒绝时，
                # 重新提醒而非结束（拒绝路径由 agent_node 收口逻辑处理：无工具调用
                # 的说明性回复 → final=passed）
                update["final"] = ""
                update["report"] = None
                update["paper_tex"] = None
                update["messages"] = state["messages"] + [{"role": "user", "content":
                    "【交付提醒】汇报幻灯片（slides.pdf）尚未产出——此前论文已过审，"
                    "现在完成 load_skill('slides-making') → make_slides → compile_slides"
                    "（复用论文图表）。若确认本次不需要汇报，直接回复说明即可结束。"}]
        else:
            # 回炉路由（Agent Lab 的 Report Refinement 机制移植）：
            # 审稿意见指向数据缺失时，允许回实验补数据——不只是改文字
            feedback = (
                "审稿未通过，须修改后重新 submit_report。\n"
                f"严重问题: {verdict.get('critical_issues', [])}\n"
                f"建议: {verdict.get('suggestions', [])}\n"
                f"修改指令: {verdict.get('feedback', '')}\n\n"
                "【修改路径判定】若上述问题属于『数据/工况缺失』类（如缺少 N 个工况、"
                "需要补充实验、样本不足），且步数预算还有余量（当前 "
                f"{state.get('n_steps', 0)}/{max_steps}），应**回实验阶段补齐数据后再修订"
                "论文**（设计补充工况→仿真→分析→改写相应章节），而不是把声称的数字改小"
                "来迁就现有数据。若预算不足或问题纯属写作层面，才走文本修订路线。")
            update["messages"] = state["messages"] + [{"role": "user", "content": feedback}]
            update["report"] = None
            update["paper_tex"] = None   # 论文流：重交前清标记
        return update

    # ---------------- 边
    def after_agent(state: ResearchState) -> str:
        # 硬上限（软预算收尾宽限 +25%）：防止软预算指令被无视后无限跑
        if state.get("n_steps", 0) > int(max_steps * 1.25) + 5:
            return END
        last = state["messages"][-1]
        if last.get("tool_calls"):
            return "tools"
        # 尾消息是 user（无行动防线注入的督促）→ 回 agent 应答。
        # 历史缺陷：此前注入督促后直接 END，督促消息永远得不到响应。
        # （plan_gate/inbox_gate/reviewer 回传走显式边，不经本路由。）
        if last.get("role") == "user":
            return "agent"
        return END

    def after_tools(state: ResearchState) -> str:
        if state.get("report") is not None or state.get("paper_tex") is not None:
            # 消融 no_reviewer：交付管线不依赖审稿事件（review_off 消融暴露的
            # harness 缺陷——原实现直接 END 把 slides/收口全堵死），走 no_review gate
            if not review_enabled:
                return "no_review_gate"
            return "reviewer"
        if (state.get("plan") is not None
                and not state.get("plan_approved")
                and state.get("plan_feedback") is None):
            return "plan_gate"
        if state.get("n_steps", 0) % 10 == 0 and state.get("n_steps", 0) > 0:
            return "inbox_gate"
        return "agent"

    # plan_gate 之后须把批复注入消息史——用包装节点
    async def plan_gate_exit(state: ResearchState) -> dict:
        feedback = state.get("plan_feedback") or "批准"
        if feedback == "批准":
            content = ("【用户批复】批准。进入执行阶段：以批准的计划为章程开始研究"
                       "（科学问题、参数范围、预算以计划为准；执行中的自主决策由你基于证据做出）。")
            return {"messages": state["messages"] + [{"role": "user", "content": content}],
                    "plan_approved": True, "plan_feedback": None}
        content = f"【用户批复】{feedback}\n请据此修订计划并再次 submit_research_plan。"
        return {"messages": state["messages"] + [{"role": "user", "content": content}],
                "plan": None, "plan_approved": False, "plan_feedback": None}

    builder = StateGraph(ResearchState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("plan_gate", plan_gate_node)
    builder.add_node("plan_gate_exit", plan_gate_exit)
    builder.add_node("inbox_gate", inbox_gate_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("no_review_gate", no_review_gate_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", after_agent,
                                  {"tools": "tools", "agent": "agent", END: END})
    builder.add_conditional_edges("tools", after_tools,
                                  {"agent": "agent", "reviewer": "reviewer",
                                   "no_review_gate": "no_review_gate",
                                   "plan_gate": "plan_gate", "inbox_gate": "inbox_gate",
                                   END: END})
    builder.add_edge("plan_gate", "plan_gate_exit")
    builder.add_edge("plan_gate_exit", "agent")
    builder.add_edge("inbox_gate", "agent")
    builder.add_conditional_edges("reviewer",
                                  lambda s: END if s.get("final") else "agent",
                                  {END: END, "agent": "agent"})
    builder.add_conditional_edges("no_review_gate",
                                  lambda s: END if s.get("final") else "agent",
                                  {END: END, "agent": "agent"})
    graph = builder.compile(checkpointer=checkpointer or MemorySaver())
    return graph, bus, local_tools
