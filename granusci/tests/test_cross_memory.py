# -*- coding: utf-8 -*-
"""cross_memory.py（⑦ 跨会话经验记忆）单元测试——LLM 调用全部 mock。

重点验证分流纪律：重复的不进 memory、dev-feedback 走工单不注入、
过时条目出 PRUNE 建议、晋升建议进工单。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from granusci import cross_memory, llm


@pytest.fixture
def mem_files(monkeypatch, tmp_path):
    exp = tmp_path / "experience.md"
    dev = tmp_path / "dev_feedback.md"
    monkeypatch.setattr(cross_memory, "MEMORY_FILE", exp)
    monkeypatch.setattr(cross_memory, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(cross_memory, "DEV_FEEDBACK_FILE", dev)
    return type("Files", (), {"exp": exp, "dev": dev})()


def _entry(title, body):
    return f"### [{title}]\n{body}\n\n"


def _distill_output(memory=None, prune=None, promote=None, dev_feedback=None):
    return json.dumps({"memory": memory or [], "prune": prune or [],
                       "promote": promote or [], "dev_feedback": dev_feedback or []})


def test_load_empty_missing_file(mem_files):
    assert cross_memory.load_for_prompt() == ""


def test_load_small_file_full(mem_files):
    mem_files.exp.write_text(
        cross_memory._HEADER + _entry("type:env 2026-08-30 run_a", "服务器有缓存。"),
        encoding="utf-8")
    text = cross_memory.load_for_prompt()
    assert "服务器有缓存" in text


def test_load_oversized_truncates_at_entry_boundary(mem_files):
    """超长时截尾保留最新，且从条目边界开始（不撕破条目）。"""
    parts = [cross_memory._HEADER]
    for i in range(60):
        parts.append(_entry(f"type:method 2026-08-{i:02d} run_{i}", "x" * 300))
    mem_files.exp.write_text("".join(parts), encoding="utf-8")
    text = cross_memory.load_for_prompt(max_chars=4000)
    assert len(text) <= 4100
    assert text.startswith("### [")
    assert "run_59" in text


def test_distill_routes_memory_vs_dev_feedback(mem_files, monkeypatch):
    """分流：行为经验进 experience.md；工具缺陷进 dev_feedback.md 工单。"""
    captured = {}

    def fake_chat_text(model, messages, **kw):
        captured["user"] = messages[-1]["content"]
        captured["purpose"] = kw.get("purpose")
        return _distill_output(
            memory=[{"type": "pitfall", "title": "过度声明",
                     "body": "预测偏差小于δ*的声明须逐位核对。"}],
            dev_feedback=[{"title": "ddda_predict 对偏析目录报 KeyError",
                           "evidence": "本次 3 次调用均失败于固定常数列"}])

    monkeypatch.setattr(llm, "chat_text", fake_chat_text)
    ok = cross_memory.distill(Path("runs/fake_run"), {"final": "passed"}, [],
                              log_fn=lambda m: None)
    assert ok is True
    exp = mem_files.exp.read_text(encoding="utf-8")
    dev = mem_files.dev.read_text(encoding="utf-8")
    # 经验条目带类型与 run 溯源
    assert "type:pitfall" in exp and "fake_run" in exp and "过度声明" in exp
    # 工具缺陷走工单，且明确不注入（experience 里没有）
    assert "DEV-FEEDBACK" in dev and "KeyError" in dev
    assert "KeyError" not in exp
    assert captured["purpose"] == "memory_distill"


def test_distill_nonagent_type_rejected_from_experience(mem_files, monkeypatch):
    """蒸馏器即使输出了 dev-feedback 类型到 memory 数组，也拒绝进经验文件。"""
    monkeypatch.setattr(llm, "chat_text", lambda *a, **kw: _distill_output(
        memory=[{"type": "dev-feedback", "title": "工具bug", "body": "绕坑做法"}]))
    ok = cross_memory.distill(Path("runs/f"), {}, [], log_fn=lambda m: None)
    assert ok is False                       # 无有效产出
    assert not mem_files.exp.exists() or "工具bug" not in mem_files.exp.read_text(encoding="utf-8")


def test_distill_encoded_knowledge_in_prompt(mem_files, monkeypatch):
    """反重复清单必须喂进蒸馏提示词（已编码的知识不许重复沉淀）。"""
    seen = {}

    def fake_chat_text(model, messages, **kw):
        seen["user"] = messages[-1]["content"]
        return _distill_output()

    monkeypatch.setattr(llm, "chat_text", fake_chat_text)
    cross_memory.distill(Path("runs/f"), {}, [], log_fn=lambda m: None,
                         encoded_knowledge="- make_figure: 大数据用 from_call 引用\n- 技能目录…")
    assert "from_call" in seen["user"]
    assert "已编码知识清单" in seen["user"]


def test_distill_prune_and_promote_to_tickets(mem_files, monkeypatch):
    """过时条目与晋升建议都进工单（给开发者），不自动改经验文件。"""
    mem_files.exp.write_text(
        cross_memory._HEADER + _entry("type:method 2026-08-30 run_old", "merge 要分批。"),
        encoding="utf-8")
    monkeypatch.setattr(llm, "chat_text", lambda *a, **kw: _distill_output(
        prune=[{"target": "merge 要分批", "reason": "工具已修复文件名长度"}],
        promote=[{"target": "负值半对数", "to_skill": "ddda-analysis", "reason": "两次运行验证"}]))
    ok = cross_memory.distill(Path("runs/f"), {}, [], log_fn=lambda m: None)
    assert ok is True
    dev = mem_files.dev.read_text(encoding="utf-8")
    assert "PRUNE" in dev and "已修复" in dev
    assert "PROMOTE" in dev and "ddda-analysis" in dev
    # prune 不自动删除（人工审）——经验文件原样
    assert "merge 要分批" in mem_files.exp.read_text(encoding="utf-8")


def test_distill_llm_failure_does_not_crash(mem_files, monkeypatch):
    def boom(*a, **kw):
        raise llm.LLMError("HTTP 500: down", kind="transient")

    monkeypatch.setattr(llm, "chat_text", boom)
    ok = cross_memory.distill(Path("runs/f"), {}, [], log_fn=lambda m: None)
    assert ok is False


def test_distill_non_json_output_skipped(mem_files, monkeypatch):
    monkeypatch.setattr(llm, "chat_text", lambda *a, **kw: "我觉得这次运行挺好的。")
    ok = cross_memory.distill(Path("runs/f"), {}, [], log_fn=lambda m: None)
    assert ok is False
    assert not mem_files.exp.exists()


def test_distill_non_json_retry_recovers(mem_files, monkeypatch):
    """非 JSON 输出 → 纠正重试一次成功（paper11 教训修复）。"""
    calls = {"n": 0}

    def chat_text(model, messages, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return "这次运行发现：工具X很好用，另外 Y 参数要小心。"   # 无 JSON
        return _distill_output(memory=[{"type": "method", "title": "X用法",
                                        "body": "调用 X 时带参数 y。"}])

    monkeypatch.setattr(llm, "chat_text", chat_text)
    ok = cross_memory.distill(Path("runs/f"), {}, [], log_fn=lambda m: None)
    assert ok is True and calls["n"] == 2
    assert "X用法" in mem_files.exp.read_text(encoding="utf-8")


def test_distill_non_json_retry_still_bad_skips(mem_files, monkeypatch):
    """重试仍非 JSON：跳过不崩。"""
    calls = {"n": 0}

    def chat_text(model, messages, **kw):
        calls["n"] += 1
        return "还是纯文本。"

    monkeypatch.setattr(llm, "chat_text", chat_text)
    ok = cross_memory.distill(Path("runs/f"), {}, [], log_fn=lambda m: None)
    assert ok is False and calls["n"] == 2
    assert not mem_files.exp.exists() or "还是" not in mem_files.exp.read_text(encoding="utf-8")


def test_distill_includes_failures_in_prompt(mem_files, monkeypatch):
    seen = {}

    def fake_chat_text(model, messages, **kw):
        seen["user"] = messages[-1]["content"]
        return _distill_output()

    monkeypatch.setattr(llm, "chat_text", fake_chat_text)
    state = {"final": "review_failed_twice",
             "review": {"critical_issues": ["预测偏差0.0653>δ*却称在δ*内"]}}
    call_log = [{"tool": "ddda_predict", "args": {"x": 1}, "ok": False,
                 "result": {"message": "KeyError: 'ad'"}}]
    cross_memory.distill(Path("runs/seg2"), state, call_log, log_fn=lambda m: None)
    assert "ddda_predict" in seen["user"] and "KeyError" in seen["user"]
    assert "预测偏差" in seen["user"]


def test_injection_into_system_prompt(mem_files):
    """端到端：经验文件 → 系统提示词注入。"""
    from granusci.agent import prompts
    from granusci.skills import load_skills
    mem_files.exp.write_text(
        cross_memory._HEADER + _entry("type:env 2026-08-31 run_x", "服务器有床层缓存。"),
        encoding="utf-8")
    sp = prompts.build_system_prompt(load_skills(),
                                     experience=cross_memory.load_for_prompt())
    assert "服务器有床层缓存" in sp
    assert "过往运行经验" in sp
    sp2 = prompts.build_system_prompt(load_skills(), experience="")
    assert "暂无——首次运行" in sp2
