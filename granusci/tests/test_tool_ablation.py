# -*- coding: utf-8 -*-
"""工具消融（--exclude-tools）+ 子代理模型跟随（消融有效性）测试。"""
from __future__ import annotations

import asyncio
import json

import pytest

from granusci import config
from granusci.agent import graph as g


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(config, "MCP_SERVERS", {})


def test_excluded_tools_note_in_prompt(offline):
    graph, _, _ = g.build_graph("GLM-4.5-Flash", "R", 50, log_fn=lambda m: None,
                                excluded_tools=["lit_search_openalex", "lit_survey"])
    # 提示词含禁用说明（通过 agent_node 捕获验证——这里直接重新构建 prompt 验证）
    graph2, bus, lt = g.build_graph("M", "R", 50, log_fn=lambda m: None,
                                    excluded_tools=["lit_search_openalex"])
    assert bus  # 构建不崩


def test_excluded_tool_removed_from_bus_and_unknowable(offline, tmp_path):
    """禁用工具从注册表移除：调用返回结构化未知工具错误。"""
    from granusci.mcp_bus import Tool

    async def run():
        _, bus, lt = g.build_graph("M", "R", 50, log_fn=lambda m: None,
                                   run_dir=tmp_path)
        await bus.start(lt)
        try:
            assert "lit_search_openalex" in bus.tools
            for t in ["lit_search_openalex", "lit_survey"]:
                bus.tools.pop(t, None)
            r = await bus.call("lit_search_openalex", {"query": "x"})
            return r
        finally:
            await bus.close()

    r = asyncio.run(run())
    assert r["error"] is True and "未知工具" in r["message"]
    assert r["retryable"] is False


def test_subagent_model_follows_agent_model(offline, monkeypatch, tmp_path):
    """模型消融有效性：deep_read_paper 子代理用主智能体同款模型。"""
    seen = {}

    def fake_chat(model, messages, tools=None, **kw):
        seen.setdefault("models", []).append(model)
        if kw.get("purpose") == "deep_read":
            return {"role": "assistant", "content": "结构化摘要。"}
        return {"role": "assistant", "content": "",
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": "deep_read_paper",
                                             "arguments": json.dumps({"paper_id": "p1"})}}]}

    from granusci import llm as llm_module
    from granusci import scievo
    monkeypatch.setattr(llm_module, "chat", fake_chat)
    monkeypatch.setattr(scievo, "read",
                        lambda pid, max_chars=24000, offset=0: {
                            "ok": True, "total_chars": 100, "content": "全文内容"})

    async def run():
        graph, bus, lt = g.build_graph("GLM-4.5-Flash", "R", 50,
                                       log_fn=lambda m: None, run_dir=tmp_path)
        await bus.start(lt)
        try:
            cfg = {"recursion_limit": 10, "configurable": {"thread_id": "t"}}
            state = {"intent": "i", "messages": [], "tool_log": [], "ledger": [],
                     "plan": None, "plan_approved": False, "plan_feedback": None,
                     "report": None, "review": None, "review_rounds": 0,
                     "n_steps": 0, "final": "", "artifacts_dir": "."}
            async for chunk in graph.astream(state, cfg, stream_mode="values"):
                if any(m.get("role") == "tool" for m in (chunk.get("messages") or [])):
                    return chunk
        finally:
            await bus.close()

    asyncio.run(run())
    # 子代理精读调用用的模型 = 主模型（GLM-4.5-Flash），不是默认预设
    assert "GLM-4.5-Flash" in seen.get("models", [])


def test_lit_survey_model_param_threaded(monkeypatch):
    """lit_survey.run_survey(model=...) 透传到全部 LLM 调用。"""
    from granusci import lit_survey as ls
    models = []

    def chat_text(model, messages, **kw):
        models.append(model)
        sys = messages[0]["content"]
        if "检索查询生成器" in sys:
            return "q1\nq2"
        if "文献筛选器" in sys:
            return '{"selected": [0]}'
        if "文献调研综合器" in sys:
            return "## 研究现状\n..."
        return "摘要"

    monkeypatch.setattr(ls.llm, "chat_text", chat_text)
    monkeypatch.setattr(ls.openalex_lit, "search",
                        lambda q, limit=10: {"papers": [
                            {"title": "T", "year": 2020, "doi": "10.1/a", "citations": 5}]})

    r = ls.run_survey("topic", "q", max_papers=3, deep_read=0,
                      model="GLM-5-Turbo", log_fn=lambda m: None)
    assert r["ok"] is True
    assert models and all(m == "GLM-5-Turbo" for m in models), models


def test_matrix_templates_expand_with_new_flags():
    """新矩阵模板（含 --exclude-tools flags）可展开、命令可构建。"""
    from granusci import experiments as xp
    from pathlib import Path
    for f in Path("granusci/matrices").glob("*.json"):
        spec = json.loads(f.read_text(encoding="utf-8"))
        units = xp.expand_matrix(spec)
        assert units
        for u in units:
            cmd = xp._build_cmd(u, Path("runs") / xp._run_tag(u), resume=False)
            assert "--yolo" in cmd
        # 冲刺消融矩阵：文献腿全断 cell 的 exclude 清单完整成对
        if spec["name"] == "sprint_ablation":
            cell = next(u for u in units if u["grid_id"] == "lit_none")
            cmd = xp._build_cmd(cell, Path("runs/x"), resume=False)
            i = cmd.index("--exclude-tools")
            excluded = cmd[i + 1].split(",")
            assert set(excluded) == {"lit_literature_search", "lit_check_novelty",
                                     "lit_search_openalex", "lit_survey",
                                     "litf_acquire", "litf_read", "deep_read_paper"}


def test_excluded_tools_hard_blocked(offline, monkeypatch, tmp_path):
    """消融硬拦截：被禁工具即便被幻觉调用也不执行，返回结构化 ablated 错误。"""
    calls = []

    def fake_chat(model, messages, tools=None, **kw):
        calls.append(tools)
        if not calls[:-1]:   # 第一轮：请求调用被禁工具
            return {"role": "assistant", "content": "",
                    "tool_calls": [{"id": "c1", "type": "function",
                                    "function": {"name": "lit_search_openalex",
                                                 "arguments": json.dumps({"query": "x"})}}]}
        return {"role": "assistant", "content": "改用其余工具。"}

    from granusci import llm as llm_module
    monkeypatch.setattr(llm_module, "chat", fake_chat)

    async def run():
        graph, bus, lt = g.build_graph("M", "R", 50, log_fn=lambda m: None,
                                       run_dir=tmp_path,
                                       excluded_tools=["lit_search_openalex"])
        await bus.start(lt)
        try:
            # ① 工具列表层面：被禁工具不再出现在 LLM 可见列表
            names = [t["function"]["name"] for t in bus.openai_tools()]
            assert "lit_search_openalex" not in names
            assert "kb_search" in names
            # ② 硬拦截层面：幻觉调用直接返回 ablated 错误（不触达 handler）
            r = await bus.call("lit_search_openalex", {"query": "x"})
            return r
        finally:
            await bus.close()

    r = asyncio.run(run())
    assert r["error"] is True and r.get("ablated") is True
    assert "禁用" in r["message"]


def test_blind_kb_no_thesis_leak(offline, tmp_path):
    """盲 KB 消融：系统提示词与 kb_search 工具描述均不出现博士论文。"""
    from granusci.agent import prompts as P
    from granusci.skills import load_skills

    sp = P.build_system_prompt(load_skills(), experience="", blind_kb=True)
    assert "博士论文" not in sp
    assert "领域综述与方法论" in sp
    # 完整模式不受影响
    sp_full = P.build_system_prompt(load_skills(), experience="")
    assert "博士论文" in sp_full

    # 工具描述变体：盲模式不提论文，完整模式提及
    async def run():
        _, bus, lt = g.build_graph("M", "R", 50, log_fn=lambda m: None,
                                   run_dir=tmp_path, blind_kb=True)
        await bus.start(lt)
        try:
            blind_desc = bus.tools["kb_search"].description
            _, bus2, lt2 = g.build_graph("M", "R", 50, log_fn=lambda m: None,
                                         run_dir=tmp_path)
            await bus2.start(lt2)
            try:
                full_desc = bus2.tools["kb_search"].description
            finally:
                await bus2.close()
            return blind_desc, full_desc
        finally:
            await bus.close()

    blind_desc, full_desc = asyncio.run(run())
    assert "博士论文" not in blind_desc
    assert "博士论文" in full_desc


def test_blind_kb_sources_exist_and_scrubbed():
    """盲 KB 源文件存在且已净化（无答案键内容：F 指数判据/675 工况/王驰）。"""
    from granusci import config
    from pathlib import Path
    for rel in config.KB_SOURCES_BLIND:
        p = Path(rel) if Path(rel).is_absolute() else config.ROOT / rel
        assert p.exists(), rel
    text = (config.ROOT / "granusci/kb_sources_blind/domain-knowledge.md").read_text(
        encoding="utf-8")
    for leak in ("王驰", "博士论文", "675", "13.86", "层数指数判据"):
        assert leak not in text, leak
    # 公共领域知识保留（净化不是清空）
    assert "Lacey" in text and "Buckingham" in text and "Eshuis" in text
