# -*- coding: utf-8 -*-
"""缺口挖掘管线测试：gap-mining 技能内容 + lit_survey gaps 模式 + 提示词契约。"""
from __future__ import annotations

from pathlib import Path

import pytest

from granusci import lit_survey as ls


def test_gap_mining_skill_content():
    """gap-mining 技能含五步管线与工具协商环节。"""
    s = Path("granusci/skills/gap-mining.md").read_text(encoding="utf-8")
    for key in ["Limitations", "Future Work", "FINER", "对抗审查", "工具缺口",
                "查新", "配齐", "不开研究", "open_question"]:
        assert key in s, f"gap-mining 缺: {key}"


def test_research_workflow_routes_new_skills():
    """research-workflow 路由表含新技能 + 失败入账 + VLM 自查。"""
    s = Path("granusci/skills/research-workflow.md").read_text(encoding="utf-8")
    for key in ["gap-mining", "paper-structure", "paper-style", "plot-script",
                "失败路径入账", "vision_analyze", "工具缺口清单"]:
        assert key in s, f"research-workflow 缺: {key}"


def test_system_prompt_gap_mining_contract():
    """提示词研究启动协议引用缺口挖掘与工具缺口协商。"""
    from granusci.agent import prompts
    from granusci.skills import load_skills
    sp = prompts.build_system_prompt(load_skills(), experience="")
    for key in ["缺口挖掘", "gap-mining", "FINER", "工具缺口清单", "未配齐前不开研究"]:
        assert key in sp, f"提示词缺: {key}"
    # 技能目录含全部新技能（catalog 自动生成）
    for name in ["gap-mining", "paper-structure", "paper-style", "plot-script",
                 "figure-crafting", "slides-making"]:
        assert name in sp, f"技能目录缺: {name}"


def test_lit_survey_gaps_mode_defaults(monkeypatch):
    """mode=gaps 时精读问题自动指向 Limitations/Future Work。"""
    seen = {}

    def chat_text(model, messages, **kw):
        sys = messages[0]["content"]
        if "检索策略设计器" in sys:
            return "segregation phenomenon\nconvection mechanism\nscaling law"
        if "文献筛选器" in sys:
            return '{"selected": [0]}'
        if "文献精读子代理" in sys:
            seen["read_q"] = messages[1]["content"][:200]
            return "局限性：(1) 未考虑液桥力；(2) 参数空间未充分探索。未来方向：多场耦合。"
        if "缺口挖掘器" in sys:
            seen["sys_gaps"] = True
            return "## 缺口清单\n**[液桥力对偏析的影响]** | 来源[1] Limitations | 方向：加 Cohesion 模型"
        return "摘要"

    monkeypatch.setattr(ls.llm, "chat_text", chat_text)
    monkeypatch.setattr(ls.openalex_lit, "search",
                        lambda q, limit=10: {"papers": [
                            {"title": "T", "year": 2024, "doi": "10.1/a", "citations": 5}]})
    monkeypatch.setattr(ls.scievo, "acquire",
                        lambda doi=None, title=None: {"ok": True, "paper_id": "p1"})
    monkeypatch.setattr(ls.scievo, "read",
                        lambda pid, max_chars=24000, offset=0: {
                            "ok": True, "total_chars": 5000, "content": "论文正文…" * 100})
    r = ls.run_survey("granular segregation", "", max_papers=3, deep_read=1,
                      mode="gaps", log_fn=lambda m: None)
    assert r["ok"] is True
    # 精读问题被 gap 问题组替换
    assert "Limitations" in seen["read_q"]
    # 综合器切到缺口挖掘角色
    assert seen.get("sys_gaps") is True


def test_lit_survey_survey_mode_unchanged(monkeypatch):
    """默认 survey 模式行为不变（回归）。"""
    def chat_text(model, messages, **kw):
        sys = messages[0]["content"]
        if "检索策略设计器" in sys:
            return "segregation phenomenon\nconvection mechanism\nscaling law"
        if "文献筛选器" in sys:
            return '{"selected": [0]}'
        if "文献精读子代理" in sys:
            return "摘要：用 DEM 研究偏析。"
        return "## 研究现状\n..."

    monkeypatch.setattr(ls.llm, "chat_text", chat_text)
    monkeypatch.setattr(ls.openalex_lit, "search",
                        lambda q, limit=10: {"papers": [
                            {"title": "T", "year": 2024, "doi": "10.1/a", "citations": 5}]})
    r = ls.run_survey("granular segregation", "custom q", max_papers=3, deep_read=0,
                      log_fn=lambda m: None)
    assert r["ok"] is True


def test_tool_schema_has_mode():
    """lit_survey 工具 schema 注册了 mode 参数。"""
    from granusci.agent import graph as g
    tools = {t.name: t for t in g._make_local_tools({}, run_dir=None)}
    assert "mode" in tools["lit_survey"].schema["properties"]
    assert "gaps" in tools["lit_survey"].schema["properties"]["mode"]["enum"]
    assert "缺口挖掘" in tools["lit_survey"].description
