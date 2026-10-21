# -*- coding: utf-8 -*-
"""⑨ run 报告生成器测试：全字段渲染 + 残缺 run 容错。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from granusci import run_report


def _mk_full_run(tmp_path: Path) -> Path:
    d = tmp_path / "20260831_120000_test"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "intent": "研究偏析机制", "agent_model": "GLM-5.3", "reviewer_model": "DeepSeek-V4-Pro",
        "max_steps": 300, "elapsed_s": 4830.2, "final": "passed",
        "n_steps": 87, "tool_calls": 12, "ledger_entries": 2}, ensure_ascii=False), "utf-8")
    trace = [
        {"tool": "lit_search_openalex", "args": {"query": "granular segregation"},
         "ts": 1756622400.0, "duration": 3.2, "ok": True, "result": {"papers": []}},
        {"tool": "dem_run_simulation", "args": {"design_id": "d1", "wait": True},
         "ts": 1756622500.0, "duration": 620.5, "ok": True, "result": {"ok": True}},
        {"tool": "ddda_run_full_pipeline", "args": {"y_col": "e_k_star"},
         "ts": 1756623200.0, "duration": 45.0, "ok": False,
         "result": {"error": True, "message": "数据文件缺少列: ['nope']"}},
    ]
    (d / "trace.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in trace), "utf-8")
    (d / "ledger.json").write_text(json.dumps([
        {"seq": 1, "kind": "law", "confidence": "high", "title": "动能标度律",
         "content": "e_k*≈3.027·(ad^0.934·Γ^0.893·F^-1.253)+0.34", "provenance": "call_ref=8",
         "ts": 1756623300.0},
        {"seq": 2, "kind": "open_question", "confidence": "medium", "title": "反转边界",
         "content": "Γ=9-12 间 BNE→RBNE 边界未定位", "provenance": "观察",
         "ts": 1756623400.0}], ensure_ascii=False, indent=1), "utf-8")
    (d / "llm_usage.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"ts": 1.0, "model": "GLM-5.3", "purpose": "agent", "prompt_tokens": 120000,
         "completion_tokens": 8000, "duration": 30.0, "attempts": 1, "total_tokens": 128000},
        {"ts": 2.0, "model": "GLM-5.3", "purpose": "agent", "prompt_tokens": 130000,
         "completion_tokens": 6000, "duration": 25.0, "attempts": 1, "total_tokens": 136000},
        {"ts": 3.0, "model": "DeepSeek-V4-Pro", "purpose": "reviewer", "prompt_tokens": 40000,
         "completion_tokens": 3000, "duration": 40.0, "attempts": 2, "total_tokens": 43000},
    ]), "utf-8")
    (d / "review.md").write_text(json.dumps({
        "pass": True, "critical_issues": [], "suggestions": ["可加敏感性分析"],
        "feedback": "", "numeric_audit": {"report_decimal_count": 20, "orphans": []}},
        ensure_ascii=False, indent=2), "utf-8")
    (d / "research_plan.md").write_text("# 研究计划\n科学问题：……" + "细节。" * 100, "utf-8")
    return d


def test_full_run_report(tmp_path):
    d = _mk_full_run(tmp_path)
    out = run_report.generate(d)
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    # 概览与统计
    for key in ["20260831_120000_test", "passed", "87", "80.5min",
                "研究偏析机制", "GLM-5.3", "DeepSeek-V4-Pro"]:
        assert key in html, f"缺: {key}"
    # 时间线含失败高亮与错误信息
    assert "ddda_run_full_pipeline" in html and "数据文件缺少列" in html
    # 工具统计
    assert "dem_run_simulation" in html and "10.3min" in html
    # 账本
    assert "动能标度律" in html and "L1" in html and "open_question" in html
    # 审稿
    assert "PASS" in html and "可加敏感性分析" in html
    # 成本
    assert "290.0k" in html          # prompt 总量 120k+130k+40k
    assert "reviewer" in html and "agent" in html


def test_empty_dir_no_crash(tmp_path):
    """崩溃早期（只有空目录）也要出报告，标注缺失。"""
    d = tmp_path / "20260831_130000_crash"
    d.mkdir()
    out = run_report.generate(d)
    html = out.read_text(encoding="utf-8")
    assert "中断/未知" in html
    assert "无 trace.jsonl" in html
    assert "无账本条目" in html and "无 review.md" in html and "无 llm_usage.jsonl" in html


def test_partial_and_corrupt_files(tmp_path):
    """半程崩溃 + 损坏 jsonl 行：能读多少算多少，不崩。"""
    d = tmp_path / "20260831_140000_partial"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(
        {"intent": "能量转化", "final": "review_failed_twice", "n_steps": 30},
        ensure_ascii=False), "utf-8")
    (d / "trace.jsonl").write_text(
        json.dumps({"tool": "kb_search", "args": {"query": "x"}, "ts": 1.0,
                    "duration": 0.5, "ok": True, "result": {}}) + "\n{corrupt line}\n", "utf-8")
    out = run_report.generate(d)
    html = out.read_text(encoding="utf-8")
    assert "kb_search" in html and "review_failed_twice" in html


def test_cli(tmp_path):
    d = _mk_full_run(tmp_path)
    workspace_root = Path(__file__).resolve().parents[2]
    r = subprocess.run([sys.executable, "-m", "granusci.run_report", str(d)],
                       capture_output=True, text=True, cwd=str(workspace_root))
    assert r.returncode == 0, r.stderr
    assert "report.html" in r.stdout


def test_cli_missing_dir(tmp_path):
    r = subprocess.run([sys.executable, "-m", "granusci.run_report",
                        str(tmp_path / "nope")], capture_output=True, text=True)
    assert r.returncode == 2
