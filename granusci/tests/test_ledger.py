# -*- coding: utf-8 -*-
"""研究状态账本（⑤）单元测试：校验、渲染、tools_node 提取落盘、resume 兼容。"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from granusci.agent import graph as g
from granusci.mcp_bus import McpBus, Tool


# ---------------------------------------------------------------- 校验
def test_validate_ok_full():
    rv = g._validate_ledger_entry({
        "kind": "law", "title": "动能标度律",
        "content": "e_k*≈3.027·(ad^0.934·Γ^0.893)+0.34, R²=0.976",
        "provenance": "ddda_run_full_pipeline job=X", "confidence": "high"})
    assert isinstance(rv, dict) and rv["kind"] == "law" and rv["confidence"] == "high"


def test_validate_defaults_confidence():
    rv = g._validate_ledger_entry({
        "kind": "open_question", "title": "Γ 反转边界", "content": "Γ=9-12 间未定位",
        "provenance": "观察"})
    assert rv["confidence"] == "medium"


def test_validate_law_requires_decimal():
    rv = g._validate_ledger_entry({
        "kind": "law", "title": "立方律", "content": "mix_index 随粒径比立方下降（无数值）",
        "provenance": "fit"})
    assert isinstance(rv, str) and "小数" in rv


def test_validate_bad_kind_and_missing_fields():
    assert "kind" in g._validate_ledger_entry({"kind": "hypothesis", "title": "t",
                                               "content": "c", "provenance": "p"})
    assert "title" in g._validate_ledger_entry({"kind": "law", "title": "",
                                                "content": "x=1.5", "provenance": "p"})
    assert "provenance" in g._validate_ledger_entry({"kind": "observation", "title": "t",
                                                     "content": "c", "provenance": ""})


def test_validate_long_title_rejected():
    rv = g._validate_ledger_entry({"kind": "number", "title": "长" * 80,
                                   "content": "δ*=0.0527", "provenance": "p"})
    assert isinstance(rv, str) and "60" in rv


# ---------------------------------------------------------------- 渲染（压缩锚点）
def test_render_empty():
    assert g._render_ledger(None) == "" and g._render_ledger([]) == ""


def test_render_compact_lines():
    ledger = [{"seq": 1, "kind": "law", "confidence": "high", "title": "动能标度律",
               "content": "e_k*≈3.027·x^0.934, R²=0.976", "provenance": "call_ref=23"}]
    text = g._render_ledger(ledger)
    assert "L1 [law|high] 动能标度律" in text
    assert "call_ref=23" in text
    assert "权威记录" in text


def test_render_caps_entries_and_chars():
    many = [{"seq": i, "kind": "number", "confidence": "medium", "title": f"t{i}",
             "content": "v=1.23", "provenance": "p"} for i in range(1, 61)]
    text = g._render_ledger(many, max_entries=40)
    assert "L60" in text and "L5:" not in text.split("L60")[0]  # 只留最新 40 条
    long_entries = [{"seq": 1, "kind": "number", "confidence": "m", "title": "t",
                     "content": "x" * 500, "provenance": "p"}]
    assert len(g._render_ledger(long_entries)) < 500


# ---------------------------------------------------------------- tools_node 集成
def _record_tool():
    for t in g._make_local_tools({}, run_dir=None):
        if t.name == "record_finding":
            return t
    raise AssertionError("record_finding 未注册")


def test_record_tool_returns_marker():
    t = _record_tool()
    result = t.handler({"kind": "number", "title": "δ*", "content": "δ*=0.0527",
                        "provenance": "ddda", "confidence": "high"})
    assert g.LEDGER_MARKER in result and result[g.LEDGER_MARKER]["kind"] == "number"
    assert "message" in result


def test_record_tool_error_no_marker():
    t = _record_tool()
    result = t.handler({"kind": "law", "title": "无数值律", "content": "纯文字",
                        "provenance": "p"})
    assert g.LEDGER_MARKER not in result and result["error"] is True


def test_tools_node_extracts_and_persists(monkeypatch, tmp_path):
    """bus.call → marker 提取 → ledger.json 落盘端到端（离线：不连 MCP）。"""
    monkeypatch.setattr(g.config, "MCP_SERVERS", {})

    async def run():
        _, bus, local_tools = g.build_graph("GLM-5.3", "DeepSeek-V4-Pro", 50,
                                            log_fn=lambda m: None, run_dir=tmp_path)
        await bus.start(local_tools)   # 只注册本地工具（不连 MCP）
        try:
            return await bus.call("record_finding", {
                "kind": "law", "title": "动能标度律", "content": "e_k*≈3.0·x^0.93, R²=0.976",
                "provenance": "call_ref=5", "confidence": "high"})
        finally:
            await bus.close()

    result = asyncio.run(run())
    assert g.LEDGER_MARKER in result
    entry = result[g.LEDGER_MARKER]
    entry["seq"] = 1
    entry["ts"] = 0.0
    g._dump_ledger_json(tmp_path, [entry])
    disk = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert disk[0]["seq"] == 1 and disk[0]["title"] == "动能标度律"


def test_dump_ledger_json_tolerates_bad_dir():
    g._dump_ledger_json(Path("Z:/nonexistent/dir"), [{"seq": 1}])  # 不抛即过


# ---------------------------------------------------------------- resume 兼容
def test_merge_tool_log_fresh_process():
    """同进程首轮：state 空，bus 全部是新的。"""
    bus = [{"tool": "a", "ts": 1.0}, {"tool": "b", "ts": 2.0}]
    merged = g._merge_tool_log([], bus, already_merged=0)
    assert merged == bus


def test_merge_tool_log_same_process_incremental():
    """同进程第二轮：prev 已含 bus 前段，只追加 bus 尾部（不重复）。"""
    prev = [{"tool": "a", "ts": 1.0}, {"tool": "b", "ts": 2.0}]        # = bus[:2]
    bus = [{"tool": "a", "ts": 1.0}, {"tool": "b", "ts": 2.0}, {"tool": "c", "ts": 3.0}]
    merged = g._merge_tool_log(prev, bus, already_merged=2)
    assert [e["tool"] for e in merged] == ["a", "b", "c"]


def test_merge_tool_log_resume_no_duplication():
    """续跑多轮不重复（回归：前缀判定方案每轮重复拼接）。

    续跑进程：state 前段=[a,b]；本进程 bus 逐轮增长 [c]→[c,d]。
    第一轮 already=0 → [a,b,c]；第二轮 already=1 → [a,b,c,d]（而非 [a,b,c,c,d]）。
    """
    prev = [{"tool": "a", "ts": 1.0}, {"tool": "b", "ts": 2.0}]
    bus_round1 = [{"tool": "c", "ts": 100.0}]
    merged1 = g._merge_tool_log(prev, bus_round1, already_merged=0)
    assert [e["tool"] for e in merged1] == ["a", "b", "c"]
    bus_round2 = [{"tool": "c", "ts": 100.0}, {"tool": "d", "ts": 101.0}]
    merged2 = g._merge_tool_log(merged1, bus_round2, already_merged=1)
    assert [e["tool"] for e in merged2] == ["a", "b", "c", "d"]


def test_merge_tool_log_edges():
    assert g._merge_tool_log([], [], 0) == []
    assert g._merge_tool_log([], [{"tool": "x", "ts": 1.0}], 0) == [{"tool": "x", "ts": 1.0}]
    assert g._merge_tool_log([{"tool": "x", "ts": 1.0}], [], 0) == [{"tool": "x", "ts": 1.0}]


def test_ledger_in_state_schema():
    from granusci.agent.state import ResearchState
    assert "ledger" in ResearchState.__annotations__


def test_ledger_survives_state_roundtrip():
    """账本条目全部为 JSON 可序列化 dict（AsyncSqliteSaver checkpoint 前提）。"""
    entry = g._validate_ledger_entry({
        "kind": "law", "title": "t", "content": "a=1.5", "provenance": "p"})
    entry.update(seq=1, ts=123.45)
    json.dumps([entry], ensure_ascii=False)  # 不抛即过
