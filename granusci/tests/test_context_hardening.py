# -*- coding: utf-8 -*-
"""① 加固测试：上下文压缩不破坏（a）LLM 协议的消息配对（b）数值审计链。

图级验证：mock LLM 捕获 agent_node 实际发出的消息（经 compress_view 后的视图），
在长历史（15+ 轮大结果工具调用）下检验配对完整性与审计语料不受影响。
"""
from __future__ import annotations

import asyncio
import json

import pytest

from granusci import config, llm as llm_module
from granusci.agent import graph as g
from granusci.mcp_bus import Tool


ROUNDS = 15          # > keep_full_tool(10)，触发压缩


def _init_state():
    return {"intent": "研究意图", "messages": [], "tool_log": [], "ledger": [],
            "plan": None, "plan_approved": False, "plan_feedback": None,
            "report": None, "review": None, "review_rounds": 0,
            "n_steps": 0, "final": "", "artifacts_dir": "."}


def _tc(name, args, i):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": name,
                                         "arguments": json.dumps(args, ensure_ascii=False)}}]}


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(config, "MCP_SERVERS", {})


def _pairing_valid(msgs) -> tuple[bool, str]:
    """OpenAI 协议配对校验：每个 tool 消息引用先前 assistant 的 tool_call_id；
    每个 assistant tool_call 有紧随其后的 tool 应答。"""
    open_calls: set[str] = set()
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                open_calls.add(tc["id"])
                # 应答必须在下一条消息出现（连续性）
                nxt = msgs[i + 1] if i + 1 < len(msgs) else {}
                if nxt.get("role") != "tool" or nxt.get("tool_call_id") != tc["id"]:
                    # 多 call 时按顺序排列在紧随的 tool 消息块里
                    j = i + 1
                    found = False
                    while j < len(msgs) and msgs[j].get("role") == "tool":
                        if msgs[j].get("tool_call_id") == tc["id"]:
                            found = True
                            break
                        j += 1
                    if not found:
                        return False, f"tool_call {tc['id']} 无应答"
        elif m.get("role") == "tool":
            if m.get("tool_call_id") not in open_calls:
                return False, f"tool 消息 {m.get('tool_call_id')} 无对应调用"
    return True, ""


def test_compression_preserves_pairing_and_audit_chain(offline, monkeypatch, tmp_path):
    """长历史压缩后：发给 LLM 的消息配对完整；tool_log 审计语料完整不受压缩影响。

    缓存友好架构下压缩是阈值触发（默认 100k 字符）——测试里把阈值调小，
    15 轮 800 字符工具结果即可触发批量冻结，检验同样的不变量。
    """
    chat_calls = []
    round_i = {"n": 0}
    monkeypatch.setattr(g, "COMPACT_TRIGGER_CHARS", 3000)
    monkeypatch.setattr(g, "COMPACT_KEEP_CHARS", 1500)

    def fake_chat(model, messages, tools=None, **kw):
        chat_calls.append(messages)
        round_i["n"] += 1
        if round_i["n"] <= ROUNDS:
            # 每轮调用 big_result，参数不同（避 loop guard），返回 800 字符含数值大结果
            return _tc("big_result", {"q": f"第{round_i['n']}轮"}, round_i["n"])
        return {"role": "assistant", "content": "研究记录完毕，结束。"}

    monkeypatch.setattr(llm_module, "chat", fake_chat)
    graph, bus, local_tools = g.build_graph("GLM-5.3", "DeepSeek-V4-Pro", 30,
                                            log_fn=lambda m: None, run_dir=tmp_path)

    async def run():
        await bus.start(local_tools)
        # 注入大结果假工具（模拟返回 800 字符的 MCP 工具）
        async def big_result(args):
            return {"ok": True, "q": args.get("q"),
                    "data": "v" * 800, "delta_star": 0.0527}
        bus.tools["big_result"] = Tool("big_result", "", {"type": "object"}, big_result)
        try:
            cfg = {"recursion_limit": 200, "configurable": {"thread_id": "t1"}}
            final = None
            async for chunk in graph.astream(_init_state(), cfg, stream_mode="values"):
                final = chunk
            return final
        finally:
            await bus.close()

    final = asyncio.run(run())

    # (a) 每一轮发给 LLM 的视图（含压缩后）消息配对完整
    for k, msgs in enumerate(chat_calls):
        ok, why = _pairing_valid(msgs)
        assert ok, f"第 {k+1} 轮视图配对破坏: {why}"

    # (b) 压缩确实发生了（后期视图含压缩标记），但只在视图层
    assert any("已压缩" in m.get("content", "")
               for m in chat_calls[-1] if m.get("role") == "tool")
    # (c) 审计链不受影响：tool_log 全量完整、数值可溯源
    tool_log = final["tool_log"]
    assert len(tool_log) == ROUNDS
    audit = g.numeric_audit("综合误差 δ*=0.0527。", tool_log)
    assert audit["orphans"] == []          # 数值在完整 tool_log 中有出处
    # (d) bus.call_log 保留完整数据（from_call 通道不受压缩影响）
    assert all(len(str(e["result"].get("data", ""))) == 800 for e in tool_log)
    # (e) 冻结边界在 state 里持久化且单调
    assert final.get("compact_cut", 0) > 0


def test_compact_cut_prefix_stability_across_turns(offline, monkeypatch, tmp_path):
    """缓存友好回归（agent 主循环真实路径）：连续两轮请求的公共前缀逐字节一致。

    服务端 KV 缓存按最长公共前缀命中——这是每轮只按新增部分计费的结构保证。
    """
    chat_calls = []
    round_i = {"n": 0}
    monkeypatch.setattr(g, "COMPACT_TRIGGER_CHARS", 10_000_000)  # 不触发压缩

    def fake_chat(model, messages, tools=None, **kw):
        chat_calls.append([dict(m) for m in messages])
        round_i["n"] += 1
        if round_i["n"] <= 5:
            return _tc("big_result", {"q": f"r{round_i['n']}"}, round_i["n"])
        return {"role": "assistant", "content": "结束。"}

    monkeypatch.setattr(llm_module, "chat", fake_chat)
    graph, bus, local_tools = g.build_graph("GLM-5.3", "DeepSeek-V4-Pro", 30,
                                            log_fn=lambda m: None, run_dir=tmp_path)

    async def run():
        await bus.start(local_tools)
        async def big_result(args):
            return {"ok": True, "data": "v" * 600}
        bus.tools["big_result"] = Tool("big_result", "", {"type": "object"}, big_result)
        try:
            cfg = {"recursion_limit": 100, "configurable": {"thread_id": "t_stab"}}
            async for chunk in graph.astream(_init_state(), cfg, stream_mode="values"):
                pass
        finally:
            await bus.close()

    asyncio.run(run())
    # 每轮请求 = 上轮请求 + 新增消息（逐字节前缀一致）
    for prev, cur in zip(chat_calls, chat_calls[1:]):
        assert cur[:len(prev)] == prev, "轮间前缀不稳定——缓存会失效"


def test_compressed_numbers_recoverable_via_ledger_anchor(offline, monkeypatch, tmp_path):
    """压缩后数值的第二通道：账本锚点始终在 system 消息里（不受历史压缩影响）。"""
    chat_calls = []
    round_i = {"n": 0}

    def fake_chat(model, messages, tools=None, **kw):
        chat_calls.append(messages)
        round_i["n"] += 1
        if round_i["n"] == 1:
            return _tc("record_finding", {
                "kind": "number", "title": "δ*", "content": "δ*=0.0527",
                "provenance": "call_ref=0", "confidence": "high"}, 1)
        if round_i["n"] <= 13:
            return _tc("big_result", {"q": f"r{round_i['n']}"}, round_i["n"])
        return {"role": "assistant", "content": "结束。"}

    monkeypatch.setattr(llm_module, "chat", fake_chat)
    graph, bus, local_tools = g.build_graph("GLM-5.3", "DeepSeek-V4-Pro", 30,
                                            log_fn=lambda m: None, run_dir=tmp_path)

    async def run():
        await bus.start(local_tools)
        async def big_result(args):
            return {"ok": True, "data": "w" * 700}
        bus.tools["big_result"] = Tool("big_result", "", {"type": "object"}, big_result)
        try:
            cfg = {"recursion_limit": 200, "configurable": {"thread_id": "t1"}}
            async for chunk in graph.astream(_init_state(), cfg, stream_mode="values"):
                pass
        finally:
            await bus.close()

    asyncio.run(run())
    # 任何一轮（含历史已深度压缩的后期轮）system 消息都带账本锚点
    for k, msgs in enumerate(chat_calls[1:], start=2):
        assert "δ*=0.0527" in msgs[0]["content"], f"第 {k} 轮丢失账本锚点"


def test_loop_guard_blocks_repeated_big_calls(offline, monkeypatch, tmp_path):
    """④ 回归：连续 3 次同参调用被拦（压缩/账本加固过程中不许退化 loop guard）。"""
    chat_calls = []
    round_i = {"n": 0}

    def fake_chat(model, messages, tools=None, **kw):
        chat_calls.append(messages)
        round_i["n"] += 1
        if round_i["n"] <= 6:
            return _tc("big_result", {"q": "same"}, round_i["n"])   # 完全同参
        return {"role": "assistant", "content": "结束。"}

    monkeypatch.setattr(llm_module, "chat", fake_chat)
    graph, bus, local_tools = g.build_graph("GLM-5.3", "DeepSeek-V4-Pro", 30,
                                            log_fn=lambda m: None, run_dir=tmp_path)

    async def run():
        await bus.start(local_tools)
        async def big_result(args):
            return {"ok": True, "data": "z" * 100}
        bus.tools["big_result"] = Tool("big_result", "", {"type": "object"}, big_result)
        try:
            cfg = {"recursion_limit": 100, "configurable": {"thread_id": "t1"}}
            async for chunk in graph.astream(_init_state(), cfg, stream_mode="values"):
                pass
        finally:
            await bus.close()

    asyncio.run(run())
    # 第 3 次同参起被拦截：视图里出现 loop_guard 错误消息
    guarded = [m for msgs in chat_calls for m in msgs
               if m.get("role") == "tool" and "loop_guard" in m.get("content", "")]
    assert guarded, "loop guard 未拦截同参重复调用"


def test_loop_guard_exempts_idempotent_poll(offline, monkeypatch, tmp_path):
    """paper12 回归：dem_job_status 同参轮询（异步仿真的合法等待）不再被
    loop-guard 拦截——幂等状态查询豁免。"""
    chat_calls = []
    round_i = {"n": 0}

    def fake_chat(model, messages, tools=None, **kw):
        round_i["n"] += 1
        if round_i["n"] <= 8:
            return _tc("dem_job_status", {"job_id": "j1"}, round_i["n"])  # 同参轮询 8 轮
        return {"role": "assistant", "content": "结束。"}

    monkeypatch.setattr(llm_module, "chat", fake_chat)
    graph, bus, local_tools = g.build_graph("GLM-5.3", "DeepSeek-V4-Pro", 20,
                                            log_fn=lambda m: None, run_dir=tmp_path)

    async def run():
        await bus.start(local_tools)
        async def job_status(args):
            return {"ok": True, "status": "running", "job_id": args.get("job_id")}
        bus.tools["dem_job_status"] = Tool("dem_job_status", "", {"type": "object"}, job_status)
        try:
            cfg = {"recursion_limit": 100, "configurable": {"thread_id": "t_poll"}}
            final = None
            async for chunk in graph.astream(_init_state(), cfg, stream_mode="values"):
                final = chunk
            return final
        finally:
            await bus.close()

    final = asyncio.run(run())
    # 8 次同参轮询全部真实执行：消息史里 8 条 tool 结果且无 loop_guard 拦截
    msgs = dict(final or {}).get("messages") or []
    tool_bodies = [str(m.get("content", "")) for m in msgs if m.get("role") == "tool"]
    assert len(tool_bodies) == 8, f"轮询未全部执行: {len(tool_bodies)}"
    assert not any("loop_guard" in b for b in tool_bodies), "幂等轮询被 loop-guard 误拦"
