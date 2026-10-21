# -*- coding: utf-8 -*-
"""make_figure 加固测试（paper11 run 后优化项）。"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from granusci import config
from granusci.agent import graph as g
from granusci.mcp_bus import McpBus, Tool


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(config, "MCP_SERVERS", {})


def _bus_with_call_log(monkeypatch, entries):
    """构造带伪造 call_log 的 bus（from_call 解析的数据源）。"""
    bus = McpBus()
    bus.call_log.extend(entries)
    return bus


POSTPROC_RESULT = {
    "ok": True, "job_id": "j1", "n_cases": 2,
    "responses_file": "/work/results/j1/responses.csv",
    "summary": [
        {"case_id": 1, "gamma": 3.0, "seg_dh_star": 0.53, "mix_index": 0.99},
        {"case_id": 2, "gamma": 7.0, "seg_dh_star": 1.19, "mix_index": 0.99},
        {"case_id": 3, "gamma": 30.0, "seg_dh_star": 1.81, "mix_index": 0.98},
    ],
}


def test_field_tree_diagnostic_in_error(offline, tmp_path, monkeypatch):
    """字段解析失败时错误消息带结构树+正确语法示例（而非只列顶层字段名）。"""
    bus = _bus_with_call_log(monkeypatch, [
        {"tool": "dem_postprocess", "args": {}, "ok": True, "duration": 0,
         "result": POSTPROC_RESULT}])
    tools = g._make_local_tools({}, run_dir=tmp_path, bus=bus)
    make_fig = next(t for t in tools if t.name == "make_figure")
    r = make_fig.handler({
        "plot_type": "scatter", "xlabel": "G", "ylabel": "y",
        "series": [{"from_call": 0, "x": "summary[].gamma", "y": "summary[].seg_dh_star"}]})
    assert r["error"] is True
    msg = r["message"]
    # 结构树：标明 summary 是 list of dict 及其字段
    assert "list[3] of dict" in msg and "gamma" in msg and "seg_dh_star" in msg
    # 语法指引：* 通配
    assert "summary.*.gamma" in msg and "不要用 [] 或 [*]" in msg


def test_correct_syntax_still_works(offline, tmp_path, monkeypatch):
    """正确语法 summary.*.gamma 走通（诊断消息不误伤正常路径）。"""
    bus = _bus_with_call_log(monkeypatch, [
        {"tool": "dem_postprocess", "args": {}, "ok": True, "duration": 0,
         "result": POSTPROC_RESULT}])
    tools = g._make_local_tools({}, run_dir=tmp_path, bus=bus)
    make_fig = next(t for t in tools if t.name == "make_figure")
    r = make_fig.handler({
        "plot_type": "scatter", "xlabel": "G", "ylabel": "dh",
        "series": [{"from_call": 0, "x": "summary.*.gamma", "y": "summary.*.seg_dh_star"}]})
    assert r.get("ok") is True and r.get("n_series") == 1


def test_mathtext_bad_label_degrades(offline, tmp_path):
    """裸 $\\Gamma$ 等不支持的 mathtext：剥 $ 降级为纯文本，整图不失败。"""
    tools = g._make_local_tools({}, run_dir=tmp_path)
    make_fig = next(t for t in tools if t.name == "make_figure")
    r = make_fig.handler({
        "plot_type": "scatter", "xlabel": "$\\Gamma$", "ylabel": "value",
        "series": [{"x": [1, 2, 3], "y": [1, 4, 9]}]})
    assert r.get("ok") is True
    # 合法 mathtext 保持原样
    from granusci.figures import _tex
    assert _tex("$r_s$") == "$r_s$"


def test_desc_overrides_applied():
    """工具描述增强层：dem_run_simulation 注册后带异步纪律前缀。"""
    from granusci.mcp_bus import _DESC_OVERRIDES
    assert "dem_run_simulation" in _DESC_OVERRIDES
    assert "job_id" in _DESC_OVERRIDES["dem_run_simulation"]
    assert "dem_postprocess" in _DESC_OVERRIDES
    assert "幂等" in _DESC_OVERRIDES["dem_postprocess"]


def test_skills_async_discipline_and_predmeas():
    """技能文件含异步纪律与 pred/meas 组装路径说明。"""
    dem = Path("granusci/skills/dem-simulation.md").read_text(encoding="utf-8")
    assert "异步纪律" in dem and "record_finding(kind=artifact)" in dem
    assert "幂等" in dem
    ddda = Path("granusci/skills/ddda-analysis.md").read_text(encoding="utf-8")
    assert "没有逐点 pred/meas 数组" in ddda and "手抄" in ddda


# ================================================================ paper12 质检三项修复
def test_plain_log_ticks_integer_labels(tmp_path):
    """窄范围（<2 数量级）对数轴刻度输出纯数字，不出现 10⁰/2×10⁰ 科学计数。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from granusci.figures import _plain_log_ticks
    fig, ax = plt.subplots()
    ax.set_xscale("log"); ax.set_xlim(1, 7)
    _plain_log_ticks(ax)
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert not any("10^0" in l or "×10" in l for l in labels)
    assert any(l in ("1", "2", "3", "5") for l in labels)
    plt.close(fig)


def test_legend_guard_warns_unlabeled_multi_series(tmp_path):
    """多系列无 label → quality_warning 回传 + 图面警告水印（paper12 图1）。"""
    from granusci.figures import make_figure
    r = make_figure(tmp_path, {
        "plot_type": "scatter", "xlabel": "t", "ylabel": "dh*",
        "series": [{"x": [1, 2, 3, 4], "y": [0.1, 0.2, 0.15, 0.3]},
                   {"x": [1, 2, 3, 4], "y": [0.2, 0.1, 0.25, 0.2]}]}, 901)
    assert r["ok"] is True
    assert "无图例" in r.get("quality_warning", "")


def test_legend_guard_silent_when_labeled(tmp_path):
    """带 label 的多系列与单系列都不触发警告（避免误报）。"""
    from granusci.figures import make_figure
    r1 = make_figure(tmp_path, {
        "plot_type": "line", "xlabel": "t", "ylabel": "y",
        "series": [{"x": [1, 2], "y": [1, 2], "label": "a"},
                   {"x": [1, 2], "y": [2, 3], "label": "b"}]}, 902)
    r2 = make_figure(tmp_path, {
        "plot_type": "scatter", "xlabel": "t", "ylabel": "y",
        "series": [{"x": [1, 2, 3], "y": [1, 2, 3]}]}, 903)   # 单系列无 label 合法
    assert "quality_warning" not in r1
    assert "quality_warning" not in r2


def test_legend_guard_panels_aggregates_warnings(tmp_path):
    """panels 组图：无 label 面板的警告聚合到 quality_warning。"""
    from granusci.figures import make_figure
    r = make_figure(tmp_path, {
        "plot_type": "panels", "panels": [
            {"plot_type": "scatter", "xlabel": "t", "ylabel": "y",
             "series": [{"x": [1, 2], "y": [1, 2]}, {"x": [1, 2], "y": [2, 1]}]},
            {"plot_type": "line", "xlabel": "t", "ylabel": "y",
             "series": [{"x": [1, 2], "y": [1, 2], "label": "ok"}]}]}, 904)
    assert r["ok"] is True
    assert "panel(a)" in r.get("quality_warning", "")


# ================================================================ 程序化叠字检测（paper12 fig_76）
def test_collision_audit_detects_overlapping_ticks():
    """密集刻度小图（真重叠）被检出；正常图无误报。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from granusci.figures import _collision_audit
    fig, ax = plt.subplots(figsize=(1.0, 0.8))
    ax.set_xticks(range(0, 101, 5))
    ax.set_xticklabels([str(v) for v in range(0, 101, 5)])
    ax.plot([0, 100], [0, 1])
    r = _collision_audit(ax)
    assert r and "x轴刻度标签重叠" in r
    plt.close(fig)
    fig2, ax2 = plt.subplots(figsize=(3.4, 2.4))
    ax2.plot([1, 2, 3, 4], [1, 4, 9, 16])
    assert _collision_audit(ax2) is None
    plt.close(fig2)


def test_panels_with_particles_subpanels(tmp_path):
    """paper13 图4 回归：panels 组图的 particles 子面板必须真渲染（不空图）。"""
    import numpy as np
    from granusci.figures import make_figure, _particles
    # ① 有数据 → 渲染成功无警告
    rng = np.random.default_rng(0)
    spec = {"plot_type": "panels", "ncols": 2, "panels": [
        {"plot_type": "particles", "colorby": "speed",
         "particles": {"x": rng.uniform(0, 0.1, 30).tolist(),
                       "y": rng.uniform(0, 0.05, 30).tolist(),
                       "radius": [0.002] * 30, "speed": rng.uniform(0, 2, 30).tolist()},
         "title": "(a)", "xlabel": "x", "ylabel": "y"},
        {"plot_type": "particles", "colorby": "type",
         "particles": {"x": [0.02, 0.05], "y": [0.01, 0.02],
                       "radius": [0.002, 0.004], "type": [1, 2]},
         "title": "(b)", "xlabel": "x", "ylabel": ""},
    ]}
    r = make_figure(tmp_path, spec, 911)
    assert r["ok"] and "quality_warning" not in r
    # ② 空数据（from_call 未解析成功场景）→ 警告回传，不再静默空图
    spec2 = {"plot_type": "panels", "panels": [
        {"plot_type": "particles", "particles": {"from_call": 999},
         "title": "(a)"}]}
    r2 = make_figure(tmp_path, spec2, 912)
    assert "缺少 x/y" in r2.get("quality_warning", "")


def test_graph_layer_resolves_panels_from_call(tmp_path):
    """graph 层回归：panels 子面板里的 from_call 必须解析（paper13 图4 根因一）。"""
    from granusci.agent import graph as g
    tools = g._make_local_tools({}, run_dir=tmp_path)
    # 模拟 bus.call_log：用 plot_script 的 data 通道不行——直接造 bus
    # 简化：构造带 from_call 的 panels spec 调 make_figure 工具，无 bus 时
    # from_call 解析应报错（比静默空图好）
    mk = next(t for t in tools if t.name == "make_figure")
    r = mk.handler({"plot_type": "panels", "panels": [
        {"plot_type": "particles", "particles": {"from_call": 0}, "title": "(a)"}]})
    # 无 bus 环境：要么报错要么警告——不允许静默成功
    warned = isinstance(r, dict) and ("缺少" in str(r.get("quality_warning", ""))
                                      or r.get("error"))
    assert warned, f"panels from_call 未解析却静默成功: {r}"
