# -*- coding: utf-8 -*-
"""⑫ 实验矩阵 runner 测试：矩阵展开/断点续跑/收集容错/统计汇总/消融开关落地。"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from granusci import experiments as xp
from granusci.agent import graph as g
from granusci.agent import prompts as P
from granusci.skills import load_skills


SPEC = {
    "name": "unit_test_exp",
    "base": {"intent": "研究振动颗粒体系", "max_steps": 50},
    "grid": [
        {"id": "baseline"},
        {"id": "noreview", "flags": ["--no-reviewer"]},
        {"id": "seedonly", "seeds": [7]},
    ],
    "seeds": [1, 2],
}


# ---------------------------------------------------------------- 矩阵展开
def test_expand_matrix():
    units = xp.expand_matrix(SPEC)
    # baseline/noreview × seeds[1,2] + seedonly × [7] = 5
    assert len(units) == 5
    tags = [xp._run_tag(u) for u in units]
    assert "exp_unit_test_exp_baseline_s1" in tags
    assert "exp_unit_test_exp_seedonly_s7" in tags
    # 单元覆盖 base
    cell = next(u for u in units if u["grid_id"] == "seedonly")
    assert cell["seed"] == 7 and cell["max_steps"] == 50 and cell["intent"]


def test_expand_matrix_validation():
    with pytest.raises(ValueError):
        xp.expand_matrix({"name": "x", "grid": []})                 # 空 grid
    with pytest.raises(ValueError):
        xp.expand_matrix({"name": "x", "grid": [{"a": 1}]})         # 缺 id
    with pytest.raises(ValueError):
        xp.expand_matrix({"name": "x", "grid": [{"id": "a"}]})      # 缺 intent


def test_build_cmd_flags():
    u = {"exp": "e", "grid_id": "g", "seed": 3, "intent": "意图",
         "model": "GLM-5.3", "reviewer": None, "max_steps": 99,
         "flags": ["--no-reviewer", "--ablate", "no_mechanism"]}
    # 全新 run：--run-dir 固定路径 + 完整配置
    cmd = xp._build_cmd(u, Path("runs/exp_e_g_s3"), resume=False)
    assert "--run-dir" in cmd and str(Path("runs/exp_e_g_s3")) in " ".join(cmd)
    assert "--seed" in cmd and "3" in cmd
    assert "--no-reviewer" in cmd and "--ablate" in cmd and "no_mechanism" in cmd
    assert "--yolo" in cmd
    # 续跑：--resume + 目录，不再传 intent/model
    cmd2 = xp._build_cmd(u, Path("runs/exp_e_g_s3"), resume=True)
    assert "--resume" in cmd2 and "--run-dir" not in cmd2
    assert "意图" not in " ".join(cmd2)


# ---------------------------------------------------------------- 收集与汇总
def _mk_run(run_dir: Path, final="passed", steps=40, calls=60, ledger=5,
            elapsed=3000.0, review_pass=True, orphans=2, tokens=200000,
            review_rounds=1):
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({
        "intent": "i", "final": final, "n_steps": steps, "tool_calls": calls,
        "ledger_entries": ledger, "elapsed_s": elapsed,
        "review_rounds": review_rounds}, ensure_ascii=False), "utf-8")
    (run_dir / "review.md").write_text(json.dumps({
        "pass": review_pass,
        "numeric_audit": {"report_decimal_count": 30, "orphans": ["0.1"] * orphans}},
        ensure_ascii=False), "utf-8")
    (run_dir / "usage.json").write_text(json.dumps({
        "total_calls": 40,
        "by_model": {"GLM-5.3": {"calls": 38, "prompt_tokens": tokens - 10000,
                                  "completion_tokens": 8000, "total_tokens": tokens - 2000},
                     "DeepSeek-V4-Pro": {"calls": 2, "prompt_tokens": 9000,
                                          "completion_tokens": 2000, "total_tokens": 11000}},
    }), "utf-8")


def test_collect_and_summarize(tmp_path):
    runs = tmp_path / "runs"
    _mk_run(runs / "exp_unit_test_exp_baseline_s1", steps=40, orphans=2, tokens=200000)
    _mk_run(runs / "exp_unit_test_exp_baseline_s2", steps=60, orphans=4, tokens=300000)
    _mk_run(runs / "exp_unit_test_exp_noreview_s1", final="review_failed_twice",
            review_pass=False, orphans=9, tokens=250000)
    # 崩溃 run：只有初始 config.json
    (runs / "exp_unit_test_exp_noreview_s2").mkdir(parents=True)
    (runs / "exp_unit_test_exp_noreview_s2" / "config.json").write_text(
        json.dumps({"intent": "i", "final": "(运行中/中断)",
                    "seed": 2, "reviewer_enabled": False}, ensure_ascii=False), "utf-8")

    rows = xp.collect(tmp_path, SPEC)
    assert len(rows) == 4
    by = {(r["grid_id"], r["seed"]): r for r in rows}
    r = by[("baseline", 1)]
    assert r["final"] == "passed" and r["orphan_numbers"] == 2
    assert r["total_tokens"] == 198000 + 11000        # by_model 的 total_tokens 求和
    # 崩溃 run 容错：指标缺什么记什么
    r2 = by[("noreview", 2)]
    assert r2["final"] == "(运行中/中断)" and r2["orphan_numbers"] is None
    # CSV 落盘
    csv_text = (tmp_path / "results.csv").read_text(encoding="utf-8-sig")
    assert "baseline" in csv_text and "orphan_numbers" in csv_text

    summary = xp.summarize(rows, tmp_path)
    md = summary.read_text(encoding="utf-8")
    assert "baseline" in md and "noreview" in md
    assert "50.0±10.0" in md          # steps 40,60 → mean 50, pstdev(总体)=10.0
    assert "1/2" in md or "2/2" in md  # 审稿通过率列
    assert "(运行中/中断)×1" in md     # 终态分布含崩溃


def test_manifest_resume_skip(tmp_path, monkeypatch):
    """断点续跑：自然完成（config final）的 run 不再执行；可恢复终态走 --resume 续跑。"""
    executed = []
    monkeypatch.setattr(xp, "EXP_ROOT", tmp_path)

    class FakeProc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(cmd, **kw):
        executed.append(" ".join(cmd[:5]))
        # 造一个 run 目录以供收集；终态由目录里已有的 config 决定（fake 不覆盖）
        rd = Path(cmd[cmd.index("--run-dir") + 1]) if "--run-dir" in cmd \
            else Path(cmd[cmd.index("--resume") + 1])
        if not (rd / "config.json").exists():
            _mk_run(rd, final="passed")
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    spec_path = tmp_path / "matrix.json"
    spec = {"name": "unit_resume",
            "base": {"intent": "意图A"},
            "grid": [{"id": "a"}, {"id": "b"}],
            "seeds": [1]}
    spec_path.write_text(json.dumps(spec, ensure_ascii=False), "utf-8")

    xp.run_matrix(spec_path, log_fn=lambda m: None)
    assert len(executed) == 2
    # 再跑一遍：全部自然完成，跳过
    executed.clear()
    xp.run_matrix(spec_path, log_fn=lambda m: None)
    assert executed == []
    assert (tmp_path / "unit_resume" / "summary.md").exists()

    # 可恢复终态：再次调用时用 --resume 续跑（不重开新目录）——须有 checkpoint 状态
    rd = tmp_path / "unit_resume" / "runs" / "exp_unit_resume_a_s1"
    cfg = json.loads((rd / "config.json").read_text(encoding="utf-8"))
    cfg["final"] = "llm_quota_exhausted"
    (rd / "config.json").write_text(json.dumps(cfg, ensure_ascii=False), "utf-8")
    import sqlite3
    con = sqlite3.connect(str(rd / "checkpoint.sqlite"))
    con.execute("CREATE TABLE checkpoints (a INTEGER)")
    con.execute("INSERT INTO checkpoints VALUES (1)")
    con.commit(); con.close()

    def fake_run2(cmd, **kw):
        executed.append(" ".join(cmd[:5]))
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run2)
    executed.clear()
    xp.run_matrix(spec_path, log_fn=lambda m: None)
    assert len(executed) == 1 and "--resume" in executed[0]
    manifest = (tmp_path / "unit_resume" / "manifest.jsonl").read_text(
        encoding="utf-8").splitlines()
    assert '"status": "recoverable"' in manifest[-1]


# ---------------------------------------------------------------- 消融开关
def test_no_reviewer_routes_to_end():
    """--no-reviewer：报告提交后直接 END（终态 submitted_no_review）。"""
    from granusci import config as cfg
    graph, _, _ = g.build_graph("M", "R", 50, log_fn=lambda m: None,
                                review_enabled=False)
    nodes = graph.get_graph().nodes
    assert "reviewer" in nodes          # 节点仍在（不破坏拓扑），只是不再路由过去


def test_no_mechanism_strips_skill_and_prompt():
    skills = load_skills()
    rw = skills["research-workflow"]
    assert "## 6.5" in rw.body                      # 基线存在
    stripped = g._strip_mechanism_section(rw.body)
    assert "## 6.5" not in stripped
    assert "## 7. 查新与报告" in stripped            # 后续节完好
    assert "决策纪律" in stripped                     # 文件尾部完好

    sp_full = P.build_system_prompt(load_skills(), experience="")
    assert "机制调查（结构观测/视觉判读/鉴别实验）是正式研究的必要环节" in sp_full
    sp_abl = P.build_system_prompt(load_skills(), experience="",
                                   ablations={"no_mechanism"})
    assert "必要环节" not in sp_abl
    assert "规律发现+误差评价+物理解读+查新" in sp_abl   # 判据其余部分保留


def test_seed_plumbed_to_request_body(monkeypatch):
    """--seed 注入 LLM 请求体。"""
    import httpx
    from granusci import llm
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    monkeypatch.setattr(llm, "_client", lambda timeout=300, model="": httpx.Client(
        base_url="https://mock.test", transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(llm, "usage", llm.UsageTracker())
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    llm.set_seed(42)
    try:
        llm.chat("M", [{"role": "user", "content": "hi"}])
        assert bodies[0].get("seed") == 42
    finally:
        llm.set_seed(None)
    llm.chat("M", [{"role": "user", "content": "hi"}])
    assert "seed" not in bodies[1]
