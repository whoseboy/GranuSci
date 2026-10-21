# -*- coding: utf-8 -*-
"""图级离线集成测试：账本全链路（工具调用→state→注入→落盘）+ 无行动防线修复。

不连 MCP（MCP_SERVERS 置空）、mock llm.chat——验证 LangGraph 真实流转。
"""
from __future__ import annotations

import asyncio
import json

import pytest

from granusci import config, llm as llm_module
from granusci.agent import graph as g

from .test_resume_hardening import MINIMAL_GOOD_SLIDES, REPORT_700


def _init_state():
    return {"intent": "研究意图", "messages": [], "tool_log": [], "ledger": [],
            "plan": None, "plan_approved": False, "plan_feedback": None,
            "report": None, "review": None, "review_rounds": 0,
            "n_steps": 0, "final": "", "artifacts_dir": "."}


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(config, "MCP_SERVERS", {})


def _tool_call_msg(name, args, call_id="c1"):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name,
                                         "arguments": json.dumps(args, ensure_ascii=False)}}]}


def test_ledger_full_flow(offline, monkeypatch, tmp_path):
    """record_finding 工具调用 → state.ledger → 下一轮 system 注入 → ledger.json 落盘。"""
    chat_calls = []
    responses = [
        _tool_call_msg("record_finding", {
            "kind": "law", "title": "动能标度律",
            "content": "e_k*≈3.0·x^0.93, R²=0.976",
            "provenance": "call_ref=5", "confidence": "high"}),
        {"role": "assistant", "content": "记账完成。"},
    ]

    def fake_chat(model, messages, tools=None, **kw):
        chat_calls.append(messages)
        return responses[min(len(chat_calls) - 1, len(responses) - 1)]

    monkeypatch.setattr(llm_module, "chat", fake_chat)
    graph, bus, local_tools = g.build_graph("GLM-5.3", "DeepSeek-V4-Pro", 50,
                                            log_fn=lambda m: None, run_dir=tmp_path)

    async def run():
        await bus.start(local_tools)
        try:
            cfg = {"recursion_limit": 20, "configurable": {"thread_id": "t1"}}
            async for chunk in graph.astream(_init_state(), cfg, stream_mode="values"):
                if len(chat_calls) >= 2:      # 第二轮 agent 已执行（带账本注入）
                    return chunk
            return None
        finally:
            await bus.close()

    state = asyncio.run(run())
    # 账本入 state 且带序号
    assert state and state.get("ledger"), "账本条目未进入 state"
    e = state["ledger"][0]
    assert e["seq"] == 1 and e["kind"] == "law" and "ts" in e
    # 落盘（消融取数）
    disk = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert disk[0]["title"] == "动能标度律"
    # 第二轮 agent 的 system 消息含账本压缩锚点
    sys2 = chat_calls[1][0]["content"]
    assert "L1 [law|high]" in sys2 and "权威记录" in sys2
    # 第一轮（空账本）不注入账本块（行为准则第8条的字样不算）
    assert "L1 " not in chat_calls[0][0]["content"] and "权威记录" not in chat_calls[0][0]["content"]


def test_no_reviewer_delivery_pipeline_not_dead(offline, monkeypatch, tmp_path):
    """no_reviewer 消融：交付管线不依赖审稿事件（review_off 的 aborted 教训）。
    提交报告 → 注入 slides 指令（不 END）→ 完成 slides → 正常终态
    submitted_no_review（不是 aborted）。"""
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-dummy")
    responses = [
        _tool_call_msg("submit_report", {"report": REPORT_700}),
        _tool_call_msg("load_skill", {"name": "slides-making"}),
        _tool_call_msg("make_slides", {"tex": MINIMAL_GOOD_SLIDES}),
        {"role": "assistant", "content": "汇报幻灯片完成，交付完毕。"},
    ]

    def fake_chat(model, messages, tools=None, **kw):
        return responses.pop(0) if responses else {"role": "assistant", "content": "结束。"}

    monkeypatch.setattr(llm_module, "chat", fake_chat)
    # make_slides 直接落文件（绕过审计细节）
    (tmp_path / "slides.pdf").write_bytes(b"%PDF-dummy-slides")

    async def run():
        graph, bus, lt = g.build_graph("M", "R", 50, log_fn=lambda m: None,
                                       run_dir=tmp_path, review_enabled=False)
        await bus.start(lt)
        try:
            cfg = {"recursion_limit": 40, "configurable": {"thread_id": "t_nr"}}
            state = _init_state()
            state["artifacts_dir"] = str(tmp_path)
            final = None
            async for chunk in graph.astream(state, cfg, stream_mode="values"):
                if chunk.get("final"):
                    final = chunk["final"]
            return final
        finally:
            await bus.close()

    assert asyncio.run(run()) == "submitted_no_review"


def test_reviewer_quota_death_is_recoverable(offline, monkeypatch, tmp_path):
    """审稿节点配额死亡 → final=llm_quota_exhausted（可恢复终态），不再未捕获崩溃。
    （lit_none 实战暴露：agent_node 有配额处置、reviewer_node 没有，审稿时配额
    耗尽会炸掉整个 run。）"""
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-dummy")
    tex4 = ("\\documentclass{revtex4-2}\n\\begin{document}\n"
            "\\section{A}\n\\cite{k1}\n\\section{B}\n\\section{C}\n\\section{D}\n"
            "\\begin{thebibliography}{9}\n\\bibitem{k1} X.\n\\end{thebibliography}\n"
            "\\end{document}\n")
    (tmp_path / "paper.tex").write_text(tex4, encoding="utf-8")

    def fake_chat(model, messages, tools=None, **kw):
        return _tool_call_msg("submit_paper", {})

    def fake_chat_text(model, messages, **kw):
        raise llm_module.LLMError("HTTP 429 配额耗尽", kind="quota")

    monkeypatch.setattr(llm_module, "chat", fake_chat)
    monkeypatch.setattr(llm_module, "chat_text", fake_chat_text)

    async def run():
        graph, bus, lt = g.build_graph("M", "R", 50, log_fn=lambda m: None,
                                       run_dir=tmp_path)
        await bus.start(lt)
        try:
            cfg = {"recursion_limit": 12, "configurable": {"thread_id": "t"}}
            state = _init_state()
            state["artifacts_dir"] = str(tmp_path)
            final = None
            async for chunk in graph.astream(state, cfg, stream_mode="values"):
                if chunk.get("final"):
                    final = chunk["final"]
            return final
        finally:
            await bus.close()

    assert asyncio.run(run()) == "llm_quota_exhausted"


def test_no_action_guard_nudges_not_ends(offline, monkeypatch):
    """无行动防线：只说话不调工具时注入督促继续，而非注入后立刻 END（历史缺陷回归测试）。"""
    chat_calls = []

    def fake_chat(model, messages, tools=None, **kw):
        chat_calls.append(messages)
        return {"role": "assistant", "content": "我在思考。"}

    monkeypatch.setattr(llm_module, "chat", fake_chat)
    graph, bus, local_tools = g.build_graph("GLM-5.3", "DeepSeek-V4-Pro", 2,
                                            log_fn=lambda m: None, run_dir=None)

    async def run():
        await bus.start(local_tools)
        try:
            cfg = {"recursion_limit": 40, "configurable": {"thread_id": "t1"}}
            final = None
            async for chunk in graph.astream(_init_state(), cfg, stream_mode="values"):
                final = chunk
            return final
        finally:
            await bus.close()

    final = asyncio.run(run())
    # 督促消息真的被应答过：LLM 被调用多次（此前缺陷：1 次后就 END）
    assert len(chat_calls) >= 3, f"防线未生效，chat 仅调用 {len(chat_calls)} 次"
    # 最终由硬预算上限收口（max_steps=2 → n_steps>7 终止）
    assert len(final["messages"]) > 2
