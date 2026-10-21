# -*- coding: utf-8 -*-
"""③ 加固测试：断点续跑组合场景。

核心场景（真实故障的沙盘重演）：
- SqliteSaver checkpoint 崩溃→新进程续跑：state 完整恢复 + tool_log 前后段拼接
- 续跑途中 MCP 服务器"重启"（传输层异常→重连重试成功，paper1 死因）
- 计划门生命周期（paper5/6 批准态缺陷回归：批准后不得再挂起）
- inbox 中途指令 HITL
- 同批工具调用部分失败不影响其余结果与账本（原子性）
- fig 编号续跑防覆盖
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from granusci import config, llm as llm_module
from granusci.agent import graph as g
from granusci.mcp_bus import McpBus, Tool
from granusci.stream_runner import run_with_plan_gate
from granusci.tests.test_mcp_bus import FakeConn


def _init_state(**kw):
    s = {"intent": "研究意图", "messages": [], "tool_log": [], "ledger": [],
         "plan": None, "plan_approved": False, "plan_feedback": None,
         "report": None, "review": None, "review_rounds": 0,
         "n_steps": 0, "final": "", "artifacts_dir": "."}
    s.update(kw)
    return s


def _tc(name, args, i):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": name,
                                         "arguments": json.dumps(args, ensure_ascii=False)}}]}


REPORT_700 = ("本研究围绕振动颗粒体系展开系统考察的过程记录。研究过程中我们记录了"
              "每个阶段的决策与失败处置：试点阶段因数据不支持而调整了方案，"
              "主线阶段的决策依据来自工具返回的数据信号，审稿阶段的修正也如实记录。"
              "时间线完整覆盖了假设来源、实验设计、规律发现与机制调查各阶段；"
              "失败与绕行包括试点批次的数据异常处置与工具调用超时的重试路径。"
              "局限与教训部分总结了本次研究的不足与智能体行为改进点。") * 6  # 1600+字
PLAN_700 = ("本计划拟研究的科学问题是：振动参数如何调控颗粒体系的能量与结构响应。"
            + "候选方向论证、可行性分析、参数空间与预算、风险与开放问题。" * 22)  # ≥600 字含"科学问题"
PASS_VERDICT = {"role": "assistant", "content": json.dumps(
    {"pass": True, "critical_issues": [], "suggestions": [], "feedback": ""}, ensure_ascii=False)}

# 过 slides 结构审计的最小 Beamer（标题页+背景+缺口+问题+方法+结果+结论+Ref）
MINIMAL_GOOD_SLIDES = (
    "\\documentclass[aspectratio=169,10pt]{beamer}\n\\begin{document}\n"
    "\\title{规律发现}\n\\author{A}\n\\date{2026}\n"
    "\\begin{frame}\\titlepage\\end{frame}\n"
    "\\begin{frame}{背景：领域现状与物理现象}背景内容，讲研究对象为什么重要。\\end{frame}\n"
    "\\begin{frame}{缺口：已知A但C未知}已知动能标度与相图，但能量分配未被研究。\\end{frame}\n"
    "\\begin{frame}{本研究：问题与途径}一句话问题加一句话途径，交代工况数与方法。\\end{frame}\n"
    "\\begin{frame}{系统与方法}仿真系统与协议。\\end{frame}\n"
    "\\begin{frame}{结果：发现}结果内容。\\end{frame}\n"
    "\\begin{frame}{结果：验证}验证内容。\\end{frame}\n"
    "\\begin{frame}{结论}1. A 2. B 3. C\\end{frame}\n"
    "\\begin{frame}{References}文献列表。\\end{frame}\n"
    "\\end{document}\n")


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(config, "MCP_SERVERS", {})


@pytest.fixture
def fast_sleep(monkeypatch):
    async def fake_sleep(s):
        return None
    monkeypatch.setattr("asyncio.sleep", fake_sleep)


def _inject_dem_probe(bus: McpBus, script):
    """注册一个走真实 _make_mcp_handler 重连路径的假 DEM 工具。"""
    conn = FakeConn(script)
    bus._conns["dem"] = conn
    bus.tools["dem_probe"] = Tool(
        "dem_probe", "探针", {"type": "object"},
        bus._make_mcp_handler("dem", "probe"))
    return conn


# ================================================================ 核心组合场景
def test_crash_resume_with_mcp_restart(offline, fast_sleep, tmp_path):
    """崩溃→新进程续跑→续跑途中 MCP 重启（重连成功）→交付通过。

    时间线：
      阶段1: agent 记账(L1) + dem_probe(ok) → agent LLM 崩溃（RuntimeError）
      阶段2: 新 bus/新图/同一 checkpoint 续跑：
             agent 调 dem_probe（传输异常→重连→ok，模拟服务器重启）
             → submit_report → reviewer PASS → END
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    responses_p1 = [
        # 一轮同时调两个工具
        {"role": "assistant", "content": "",
         "tool_calls": [
             {"id": "c1", "type": "function", "function": {
                 "name": "record_finding",
                 "arguments": json.dumps({"kind": "number", "title": "δ*",
                                          "content": "δ*=0.0527",
                                          "provenance": "call_ref=0",
                                          "confidence": "high"}, ensure_ascii=False)}},
             {"id": "c2", "type": "function", "function": {
                 "name": "dem_probe", "arguments": "{}"}}]},
    ]
    responses_p2 = [
        _tc("dem_probe", {}, 3),                 # 续跑后首个动作：MCP 已"重启"
        _tc("submit_report", {"report": REPORT_700}, 4),
        PASS_VERDICT,                             # reviewer
    ]
    calls = []

    def phase_chat(responses):
        def fake(model, messages, tools=None, **kw):
            calls.append(kw.get("purpose", "agent"))
            if responses:
                return responses.pop(0)
            return {"role": "assistant", "content": "（兜底）"}
        return fake

    async def run():
        ckpt = AsyncSqliteSaver.from_conn_string(str(tmp_path / "ckpt.sqlite"))
        checkpointer = await ckpt.__aenter__()
        try:
            stream_cfg = {"recursion_limit": 60,
                          "configurable": {"thread_id": "t_resume"}}
            # ---- 阶段 1：跑到崩溃
            def crash_chat(model, messages, tools=None, **kw):
                calls.append(kw.get("purpose", "agent"))
                if responses_p1:
                    return responses_p1.pop(0)
                raise RuntimeError("模拟进程崩溃")
            llm_module.chat = crash_chat
            graph1, bus1, lt1 = g.build_graph("GLM-5.3", "DeepSeek-V4-Pro", 20,
                                              log_fn=lambda m: None, run_dir=tmp_path,
                                              checkpointer=checkpointer)
            await bus1.start(lt1)
            _inject_dem_probe(bus1, [{"ok": True, "phase": 1}])
            with pytest.raises(RuntimeError):
                await run_with_plan_gate(graph1, _init_state(), stream_cfg,
                                         tmp_path, yolo=True, log_fn=lambda m: None)
            await bus1.close()
            # 崩溃现场：checkpoint 已保存 tools 完成后的状态
            assert (tmp_path / "ledger.json").exists()

            # ---- 阶段 2：新进程（新 bus/新图）续跑，MCP 首调传输异常→重连成功
            llm_module.chat = phase_chat(responses_p2)
            graph2, bus2, lt2 = g.build_graph("GLM-5.3", "DeepSeek-V4-Pro", 20,
                                              log_fn=lambda m: None, run_dir=tmp_path,
                                              checkpointer=checkpointer)
            await bus2.start(lt2)
            conn = _inject_dem_probe(bus2, [
                __import__("httpx").RemoteProtocolError("peer closed"),  # 模拟重启中
                {"ok": True, "phase": 2},                                  # 重连后成功
            ])
            final = await run_with_plan_gate(graph2, None, stream_cfg,
                                             tmp_path, yolo=True, log_fn=lambda m: None)
            await bus2.close()
            return final, conn
        finally:
            await ckpt.__aexit__(None, None, None)

    final, conn = asyncio.run(run())
    final = dict(final or {})
    # 终态：审稿通过
    assert final.get("final") == "passed"
    assert final.get("review_rounds") == 1
    # 账本跨进程存活（checkpoint 恢复）
    ledger = final.get("ledger") or []
    assert len(ledger) == 1 and ledger[0]["seq"] == 1 and ledger[0]["title"] == "δ*"
    # tool_log 前后段拼接：阶段1两条 + 阶段2两条（dem_probe + submit_report）
    tools_seq = [e["tool"] for e in final.get("tool_log") or []]
    assert tools_seq == ["record_finding", "dem_probe", "dem_probe", "submit_report"]
    # MCP 重启场景：重连一次后成功
    assert conn.connects == 1
    # 消息史完整：中断点前后消息都在
    roles = [m["role"] for m in final.get("messages") or []]
    assert roles[0] == "user" and roles.count("tool") == 4


# ================================================================ 可恢复终态复活（paper12 二次死因回归）
def test_recoverable_final_revival_on_resume(offline, tmp_path):
    """quota 终态入 checkpoint 后，--resume 必须复活续跑而非秒退。

    时间线：
      阶段1: agent 调一次工具 → 下轮 llm.chat 抛 kind="quota" 的 LLMError
             → agent_node 返回 final=llm_quota_exhausted → 线程 END
      阶段2: 新进程（新 bus/新图/同 checkpoint）续跑：
             应检测可恢复终态 → 注入恢复指令 → 继续 → 交付通过
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    responses_p2 = [
        _tc("submit_report", {"report": REPORT_700}, 1),
        PASS_VERDICT,
    ]

    async def run():
        ckpt = AsyncSqliteSaver.from_conn_string(str(tmp_path / "ckpt_q.sqlite"))
        checkpointer = await ckpt.__aenter__()
        try:
            stream_cfg = {"recursion_limit": 60,
                          "configurable": {"thread_id": "t_quota"}}
            # ---- 阶段 1：正常跑一轮工具，随后撞配额
            responses_p1 = [_tc("kb_search", {"query": "偏析"}, 1)]
            state_p1 = {"fired": False}

            def quota_chat(model, messages, tools=None, **kw):
                if responses_p1:
                    return responses_p1.pop(0)
                state_p1["fired"] = True
                raise llm_module.LLMError(
                    "HTTP 429 配额耗尽（团队 token 上限，等待周期性重置）：...", kind="quota")

            llm_module.chat = quota_chat
            graph1, bus1, lt1 = g.build_graph("GLM-5.3", "DeepSeek-V4-Pro", 20,
                                              log_fn=lambda m: None, run_dir=tmp_path,
                                              checkpointer=checkpointer)
            await bus1.start(lt1)
            final1 = await run_with_plan_gate(graph1, _init_state(), stream_cfg,
                                              tmp_path, yolo=True, log_fn=lambda m: None)
            await bus1.close()
            final1 = dict(final1 or {})
            assert final1.get("final") == "llm_quota_exhausted", \
                f"阶段1应以 quota 终态结束: {final1.get('final')}"
            assert state_p1["fired"]

            # ---- 阶段 2：新进程续跑（配额已恢复）
            def ok_chat(model, messages, tools=None, **kw):
                return responses_p2.pop(0) if responses_p2 \
                    else {"role": "assistant", "content": "结束。"}

            llm_module.chat = ok_chat
            graph2, bus2, lt2 = g.build_graph("GLM-5.3", "DeepSeek-V4-Pro", 20,
                                              log_fn=lambda m: None, run_dir=tmp_path,
                                              checkpointer=checkpointer)
            await bus2.start(lt2)
            final2 = await run_with_plan_gate(graph2, None, stream_cfg,
                                              tmp_path, yolo=True, log_fn=lambda m: None)
            await bus2.close()
            return final2
        finally:
            await ckpt.__aexit__(None, None, None)

    final2 = asyncio.run(run())
    final2 = dict(final2 or {})
    # 复活成功：不是秒退的 llm_quota_exhausted，而是走完交付
    assert final2.get("final") == "passed"
    assert final2.get("review_rounds") == 1
    # 恢复指令确实注入消息史
    msgs = final2.get("messages") or []
    assert any("现已恢复" in str(m.get("content", "")) for m in msgs)
    # 阶段1的工具调用记录未丢
    assert [e["tool"] for e in final2.get("tool_log") or []] == ["kb_search", "submit_report"]


# ================================================================ 计划门生命周期（paper5/6 回归）
def test_plan_gate_lifecycle_no_rehang(offline, tmp_path):
    """计划批准后连续工作不再二次挂起（paper5/6 死因回归），直至交付。"""
    responses = [
        _tc("submit_research_plan", {"plan": PLAN_700}, 1),
        _tc("record_finding", {"kind": "observation", "title": "试点观察",
                               "content": "床层出现分层结构（定性）",
                               "provenance": "dem 快照"}, 2),
        _tc("submit_report", {"report": REPORT_700}, 3),
        PASS_VERDICT,
        {"role": "assistant", "content": "本次不需要汇报，交付完成。"},  # slides 扩展段拒绝
    ]
    interrupts = []
    chat_calls = []

    def fake(model, messages, tools=None, **kw):
        chat_calls.append(kw.get("purpose", "agent"))
        return responses.pop(0) if responses else {"role": "assistant", "content": "结束。"}

    import granusci.llm as L
    orig = L.chat
    L.chat = fake
    try:
        graph, bus, lt = g.build_graph("GLM-5.3", "DeepSeek-V4-Pro", 20,
                                       log_fn=lambda m: None, run_dir=tmp_path)

        async def run():
            await bus.start(lt)
            # 引用门 v2 前置：注入一条 litx 检索记录，使 bibitem 标题可核验
            bus.call_log.append({"tool": "litx_search_external",
                                 "args": {"query": "test"},
                                 "result": {"ok": True, "results": [
                                     {"title": "Title of Test Paper by Author 2024",
                                      "authors": ["Author"], "year": 2024}]}})
            try:
                cfg = {"recursion_limit": 40, "configurable": {"thread_id": "t_plan"}}
                input_ = _init_state()
                while True:
                    hit = None
                    final = None
                    async for chunk in graph.astream(input_, cfg, stream_mode="values"):
                        final = chunk
                        if isinstance(chunk, dict) and chunk.get("__interrupt__"):
                            hit = chunk["__interrupt__"]
                    if hit is None:
                        return final, interrupts
                    interrupts.append(len(interrupts) + 1)
                    from langgraph.types import Command
                    input_ = Command(resume="批准")
            finally:
                await bus.close()

        final, _ = asyncio.run(run())
    finally:
        L.chat = orig

    final = dict(final or {})
    assert len(interrupts) == 1, f"计划门批准后再次挂起（paper5/6 回归）: {len(interrupts)} 次"
    assert final.get("plan_approved") is True
    assert final.get("final") == "passed"
    assert final.get("review_rounds") == 1


# ================================================================ inbox HITL
def test_inbox_midrun_instruction(offline, tmp_path):
    """运行中写入 inbox.md → 第 10 步挂起 → 注入用户指令 → 继续 → 交付。"""
    (tmp_path / "inbox.md").write_text("优先采用对数网格设计。", encoding="utf-8")
    responses = [
        _tc("kb_search", {"query": "偏析机制"}, 1),
        _tc("submit_report", {"report": REPORT_700}, 2),
        PASS_VERDICT,
    ]

    def fake(model, messages, tools=None, **kw):
        return responses.pop(0) if responses else {"role": "assistant", "content": "结束。"}

    import granusci.llm as L
    orig = L.chat
    L.chat = fake
    try:
        graph, bus, lt = g.build_graph("GLM-5.3", "DeepSeek-V4-Pro", 20,
                                       log_fn=lambda m: None, run_dir=tmp_path)

        async def run():
            await bus.start(lt)
            try:
                cfg = {"recursion_limit": 40, "configurable": {"thread_id": "t_inbox"}}
                # n_steps=9 起步：首轮 agent 后到 10 → tools → inbox_gate 挂起
                return await run_with_plan_gate(graph, _init_state(n_steps=9), cfg,
                                                tmp_path, yolo=True, log_fn=lambda m: None)
            finally:
                await bus.close()

        final = asyncio.run(run())
    finally:
        L.chat = orig

    final = dict(final or {})
    assert final.get("final") == "passed"
    msgs = final.get("messages") or []
    injected = [m for m in msgs if "用户中途指令" in str(m.get("content", ""))]
    assert injected, "inbox 指令未注入消息史"
    assert "对数网格" in injected[0]["content"]
    # 归档名带时间戳（审计 A-11 修复：固定名在 Windows 上第二次 rename 必崩）
    assert list(tmp_path.glob("inbox_consumed_*.md")), "收件箱已归档"
    assert not (tmp_path / "inbox.md").exists()


# ================================================================ 工具批原子性
def test_batch_partial_failure_atomic(offline, tmp_path):
    """同批 3 个调用 1 个抛异常：其余 2 个结果照常回传、账本照常入账、run 不崩。"""
    responses = [
        {"role": "assistant", "content": "",
         "tool_calls": [
             {"id": "c1", "type": "function", "function": {
                 "name": "record_finding",
                 "arguments": json.dumps({"kind": "observation", "title": "观察A",
                                          "content": "定性观察无异常",
                                          "provenance": "快照"}, ensure_ascii=False)}},
             {"id": "c2", "type": "function", "function": {
                 "name": "boom", "arguments": "{}"}},
             {"id": "c3", "type": "function", "function": {
                 "name": "kb_search", "arguments": json.dumps({"query": "偏析"})}},
         ]},
        {"role": "assistant", "content": "结束。"},
    ]

    def fake(model, messages, tools=None, **kw):
        return responses.pop(0) if responses else {"role": "assistant", "content": "结束。"}

    import granusci.llm as L
    orig = L.chat
    L.chat = fake
    try:
        graph, bus, lt = g.build_graph("GLM-5.3", "DeepSeek-V4-Pro", 20,
                                       log_fn=lambda m: None, run_dir=tmp_path)

        async def run():
            await bus.start(lt)
            def boom(args):
                raise ValueError("内部错误")
            bus.tools["boom"] = Tool("boom", "", {"type": "object"}, boom)
            try:
                cfg = {"recursion_limit": 200, "configurable": {"thread_id": "t_atomic"}}
                final = None
                async for chunk in graph.astream(_init_state(), cfg, stream_mode="values"):
                    final = chunk
                return final
            finally:
                await bus.close()

        final = asyncio.run(run())
    finally:
        L.chat = orig

    final = dict(final or {})
    tool_msgs = [m for m in final.get("messages") or [] if m.get("role") == "tool"]
    assert len(tool_msgs) == 3                       # 三条应答都在
    boom_msg = json.loads([m for m in tool_msgs
                           if m["tool_call_id"] == "c2"][0]["content"])
    assert boom_msg["error"] is True and "ValueError" in boom_msg["message"]
    # 账本与 tool_log 未被失败调用破坏
    assert len(final.get("ledger") or []) == 1
    assert [e["tool"] for e in final["tool_log"]] == ["record_finding", "boom", "kb_search"]


# ================================================================ fig 编号续跑防覆盖
def test_fig_numbering_continues_after_resume(tmp_path):
    """run_dir 已有 fig_1.pdf（崩溃前所绘）→ 续跑新进程第一张图是 fig_2 不覆盖。"""
    figs = tmp_path / "figs"
    figs.mkdir(parents=True)
    (figs / "fig_1.pdf").write_bytes(b"%PDF-dummy")

    tools = g._make_local_tools({}, run_dir=tmp_path)
    make_fig = next(t for t in tools if t.name == "make_figure")
    result = make_fig.handler({
        "plot_type": "line", "xlabel": "x", "ylabel": "y",
        "series": [{"x": [1, 2, 3], "y": [2, 4, 8], "label": "s"}]})
    assert result.get("ok") or "figure" in result
    assert "fig_2" in json.dumps(result)
    assert (figs / "fig_2.pdf").exists()
    assert (figs / "fig_1.pdf").read_bytes() == b"%PDF-dummy"   # 原图未被覆盖


def test_delivery_extension_slides_then_close(offline, tmp_path):
    """P1 回归：论文过审后注入汇报制作指令，完成 slides 后自然终态。"""
    from granusci import llm as L
    responses = [
        _tc("submit_report", {"report": REPORT_700}, 1),
        PASS_VERDICT,
        _tc("load_skill", {"name": "slides-making"}, 2),
        _tc("make_slides", {"tex": MINIMAL_GOOD_SLIDES}, 3),
        _tc("compile_slides", {}, 4),
        {"role": "assistant", "content": "汇报幻灯片完成，交付完毕。"},
    ]

    def fake(model, messages, tools=None, **kw):
        return responses.pop(0) if responses else {"role": "assistant", "content": "结束。"}

    orig = L.chat
    L.chat = fake
    try:
        graph, bus, lt = g.build_graph("GLM-5.3", "DeepSeek-V4-Pro", 20,
                                       log_fn=lambda m: None, run_dir=tmp_path)
        import asyncio

        async def run():
            await bus.start(lt)
            try:
                cfg = {"recursion_limit": 100, "configurable": {"thread_id": "t_deliver"}}
                final = None
                async for chunk in graph.astream(_init_state(n_steps=0), cfg,
                                                 stream_mode="values"):
                    final = chunk
                return final
            finally:
                await bus.close()

        final = asyncio.run(run())
    finally:
        L.chat = orig
    final = dict(final or {})
    # slides_offered 注入过、终态 passed（交付完成路径）
    assert final.get("slides_offered") is True
    assert final.get("final") == "passed"
    assert (tmp_path / "slides.tex").exists()


def test_submit_paper_does_not_pollute_report_field(offline, tmp_path):
    """paper12 回归：submit_paper 的论文 tex 不得进 report 状态字段。

    report 字段是行为报告专用（report.md 的来源）——此前 submit_paper 把
    LaTeX 全文塞进 report，导致 report.md 变成论文全文，行为报告缺失。
    """
    # 准备一个能通过 submit_paper 校验的最小论文环境（B=chr(92) 防转义地狱）
    B = chr(92)
    tex = (f"{B}documentclass{{article}}{B}begin{{document}}"
           f"{B}section{{A}}{B}section{{B}}{B}section{{C}}{B}section{{D}}"
           f"text {B}cite{{k1}}{B}begin{{thebibliography}}{{1}}"
           f"{B}bibitem{{k1}} Author. Title. 2024.{B}end{{thebibliography}}"
           f"{B}end{{document}}")
    (tmp_path / "paper.tex").write_text(tex, encoding="utf-8")
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-dummy")
    figs = tmp_path / "figs"
    figs.mkdir(exist_ok=True)

    responses = [
        _tc("submit_paper", {}, 1),          # 论文提交 → 审稿
        PASS_VERDICT,                        # reviewer PASS（首轮）
        {"role": "assistant", "content": "不需要汇报，交付完成。"},  # slides 扩展段拒绝
        _tc("submit_report", {"report": REPORT_700}, 2),   # 行为报告补交
        PASS_VERDICT,
        {"role": "assistant", "content": "完成。"},
    ]

    def fake(model, messages, tools=None, **kw):
        return responses.pop(0) if responses else {"role": "assistant", "content": "结束。"}

    import granusci.llm as L
    orig = L.chat
    L.chat = fake
    try:
        graph, bus, lt = g.build_graph("GLM-5.3", "DeepSeek-V4-Pro", 20,
                                       log_fn=lambda m: None, run_dir=tmp_path)
        import asyncio

        async def run():
            await bus.start(lt)
            # 引用门 v2 前置：注入 litx 检索记录使 bibitem 可核验
            bus.call_log.append({"tool": "litx_search_external",
                                 "args": {"query": "test"},
                                 "result": {"ok": True, "results": [
                                     {"title": "Title of Test Paper by Author 2024",
                                      "authors": ["Author"], "year": 2024}]}})

            try:
                cfg = {"recursion_limit": 100, "configurable": {"thread_id": "t_paper"}}
                final = None
                async for chunk in graph.astream(_init_state(), cfg, stream_mode="values"):
                    final = chunk
                return final
            finally:
                await bus.close()

        final = asyncio.run(run())
    finally:
        L.chat = orig
    final = dict(final or {})
    # 核心断言：审稿轮的输入是论文 tex（而非 report 字段被污染后残留 tex）；
    # 最终 PASS 后交付扩展段会清 report（防重复审稿）——这是设计行为。
    # 因此用消息史验证：submit_report 的报告内容确实经过了 tools 节点。
    msgs = final.get("messages") or []
    tool_bodies = [m.get("content", "") for m in msgs if m.get("role") == "tool"]
    assert any("已受理" in b for b in tool_bodies), "submit_report 未被受理"
    assert final.get("final") == "passed"
    # reviewer 的审稿对象曾包含论文（messages 里有论文提交的受理消息）
    assert any("论文已受理" in b for b in tool_bodies)


def test_reviewer_vision_audit_catches_empty_figure(offline, monkeypatch, tmp_path):
    """paper13 图4 回归：审稿门的视觉审计发现空图 → 硬性拒收。

    文本审稿人（mock 为宽松 PASS）+ 视觉审计（mock 一张空图）→ 最终 FAIL。
    """
    # 造一张"论文图"：paper.tex 引用 fig_1 + caption
    B = chr(92)
    tex = (f"{B}documentclass{{article}}{B}begin{{document}}"
           f"{B}section{{A}}{B}section{{B}}{B}section{{C}}{B}section{{D}}"
           f"{B}begin{{figure}}{B}includegraphics[width=1cm]{{fig_1}}"
           f"{B}caption{{231 particles each.}}{B}label{{fig:x}}{B}end{{figure}}"
           f"{B}end{{document}}")
    (tmp_path / "paper.tex").write_text(tex, encoding="utf-8")
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-dummy")
    figs = tmp_path / "figs"
    figs.mkdir()
    (figs / "fig_1.pdf").write_bytes(b"%PDF-dummy")  # submit_paper 校验需要 pdf
    (figs / "fig_1.png").write_bytes(b"PNG-dummy")   # 视觉审计读 png（视觉是 mock）

    responses = [
        _tc("submit_paper", {}, 1),     # 论文提交
        {"role": "assistant", "content": "宽容的审稿人倾向通过。"},  # reviewer 文本：宽松
        {"role": "assistant", "content": "结束。"},
    ]

    def fake_chat(model, messages, tools=None, **kw):
        return responses.pop(0) if responses else {"role": "assistant", "content": "结束。"}

    def fake_vision(model, image_path, question, **kw):
        # 视觉审计 mock：fig_1 判为空图
        return '{"empty": true, "issues": ["三个面板均无颗粒"]}'

    import granusci.llm as L
    monkeypatch.setattr(L, "chat", fake_chat)
    monkeypatch.setattr(L, "vision_chat", fake_vision)
    graph, bus, local_tools = g.build_graph("GLM-5.3", "DeepSeek-V4-Pro", 20,
                                            log_fn=lambda m: None, run_dir=tmp_path)
    import asyncio

    async def run():
        await bus.start(local_tools)
        try:
            cfg = {"recursion_limit": 100, "configurable": {"thread_id": "t_vision"}}
            final = None
            async for chunk in graph.astream(_init_state(), cfg, stream_mode="values"):
                final = chunk
            return final
        finally:
            await bus.close()

    final = asyncio.run(run())
    final = dict(final or {})
    review = final.get("review") or {}
    # 空图硬失败：即使文本审稿人宽松，也必须 FAIL 且 critical_issues 含空图
    assert review.get("pass") is False, "视觉审计发现的空图未拦截（paper13 图4 回归）"
    assert any("空图" in str(i) for i in review.get("critical_issues", []))
    assert any(v.get("empty") for v in review.get("vision_audit", []))
