# -*- coding: utf-8 -*-
"""plot_script 自由绘图沙箱测试（agent 写 matplotlib 代码路线）。"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from granusci.agent import graph as g
from granusci.mcp_bus import McpBus


@pytest.fixture
def env(tmp_path):
    bus = McpBus()
    bus.call_log.append({
        "tool": "dem_get_snapshot", "args": {}, "ok": True, "duration": 0,
        "result": {"x": [0.1, 0.2, 0.3], "y": [0.5, 0.6, 0.4],
                   "type": [1, 2, 1], "speed": [0.01, 0.03, 0.02]}})
    tools = {t.name: t for t in g._make_local_tools({}, run_dir=tmp_path, bus=bus)}
    return tools["plot_script"], tmp_path


def test_basic_double_panel_with_data_ref(env):
    ps, tmp = env
    r = ps.handler({"code": """
import numpy as np
snap = data(0)
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.6))
axes[0].scatter(snap["x"], snap["y"], c=snap["type"])
axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
axes[1].plot(snap["x"], snap["speed"], "o-")
fig.tight_layout()
"""})
    assert r.get("ok") is True
    assert Path(r["png"]).exists() and Path(r["figure"]).exists()
    assert r["figure_name"].startswith("fig_")


def test_inline_data_allowed(env):
    """小数据内联（照抄工具返回）与 make_figure 手抄同级。"""
    ps, tmp = env
    r = ps.handler({"code": "plt.plot([1, 2, 3], [2, 4, 8], 'o-')\nplt.xlabel('n')\nplt.ylabel('v')"})
    assert r.get("ok") is True


def test_sandbox_blocks_dangerous_imports(env):
    ps, _ = env
    for code in ["import os", "import subprocess", "import pathlib",
                 "import sys\nplt.plot([1],[1])"]:
        r = ps.handler({"code": code})
        assert r.get("error") and "沙箱" in r["message"], code


def test_sandbox_blocks_system_calls(env):
    ps, _ = env
    for frag in ["open('x','w')", "os.getcwd()", "exec('1')", "eval('1')", "shutil.copy('a','b')"]:
        r = ps.handler({"code": f"plt.plot([1],[1])\n{frag}"})
        assert r.get("error") and "禁止" in r["message"], frag


def test_exception_returns_structured_error(env):
    ps, _ = env
    r = ps.handler({"code": "x = 1/0"})
    assert r.get("error") and "绘图代码失败" in r["message"] and "ZeroDivisionError" in r["message"]


def test_data_ref_out_of_range(env):
    ps, _ = env
    r = ps.handler({"code": "d = data(99)\nplt.plot([1],[1])"})
    assert r.get("error") and "超出工具调用记录" in r["message"]


def test_no_figure_created(env):
    ps, _ = env
    r = ps.handler({"code": "x = 1"})
    assert r.get("error") and "没有可保存的 figure" in r["message"]


def test_allowed_imports_pass(env):
    ps, _ = env
    r = ps.handler({"code": """
import numpy as np
import matplotlib.pyplot as _p
import math
plt.plot([math.pi, 2*math.pi], [1, 2])
"""})
    assert r.get("ok") is True


def test_figure_crafting_skill_exists():
    """绘图技能拆分后：figure-crafting 设计原则 + plot-script 代码手册。"""
    s = Path("granusci/skills/figure-crafting.md").read_text(encoding="utf-8")
    for key in ["一个 Results 级科学问题", "Viridis", "线型冗余", "期刊图自查清单", "数值保真"]:
        assert key in s, f"figure-crafting 缺: {key}"
    ps = Path("granusci/skills/plot-script.md").read_text(encoding="utf-8")
    for key in ["data(N)", "basex", "沙箱纪律", "inset", "双 y 轴"]:
        assert key in ps, f"plot-script 缺: {key}"


def test_cjk_font_fallback_no_glyph_warnings(env):
    """中英混排标题零字形缺失警告（paper11 乱码修复回归）。"""
    import warnings as _w
    ps, tmp = env
    with _w.catch_warnings(record=True) as ws:
        _w.simplefilter("always")
        r = ps.handler({"code": """
fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
ax.plot([1, 2, 3], [2, 4, 6])
ax.set_title("(2.0, 0.5, Γ=3) 强BNE低Γ 混排")
ax.set_xlabel("位置 (m)")
fig.tight_layout()
"""})
        glyph = [x for x in ws if "missing from font" in str(x.message)]
    assert r.get("ok") is True
    assert len(glyph) == 0, f"CJK 字形缺失: {[str(x.message)[:60] for x in glyph]}"


def test_slides_tools_pipeline(tmp_path):
    """make_slides + compile_slides（XeLaTeX+xeCJK 中文管线）。"""
    from granusci.agent import graph as gg
    tools = {t.name: t for t in gg._make_local_tools({}, run_dir=tmp_path)}
    tex = r"""\documentclass[aspectratio=169,10pt]{beamer}
\usetheme{Madrid}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{xeCJK}
\setCJKmainfont{Microsoft YaHei}
\title{中文标题测试}
\author{GranuSci}
\begin{document}
\begin{frame}
\titlepage
\end{frame}
\begin{frame}{背景：研究对象的领域现状}
颗粒体系是工业与自然界广泛存在的对象，其动力学现象具有重要的科学意义。
\end{frame}
\begin{frame}{缺口：已知规律但未知机制}
已知实验现象与经验规律，但其定量机制尚未被刻画，留下未解决的问题。
\end{frame}
\begin{frame}{本研究：问题与途径}
本研究围绕该缺口展开，途径是仿真与数据分析结合。
\end{frame}
\begin{frame}{测试帧：结果与证据}
$E = mc^2$ 中文内容排版验证。这里添加足量文本以满足 make_slides 的最短源码检查。
此测试验证 Beamer 管线的三个环节：写入、帧计数、编译。中文与数学混排是重点——
XeLaTeX 需要 xeCJK 包才能渲染中文，这条路径在真实 run 中由技能骨架保证。
再补充一些内容使源码超过八百字符阈值：验证内容包括中文标题页的正常显示、
数学公式的上下标排版、以及页码的规范放置。这些是学术汇报幻灯片的基本要素。
\end{frame}
\begin{frame}{结论}
1. 第一条发现；2. 第二条发现；3. 第三条发现。
\end{frame}
\begin{frame}{References}
测试文献条目。
\end{frame}
\end{document}
"""
    r1 = tools["make_slides"].handler({"tex": tex})
    assert r1.get("ok") and r1["n_frames"] >= 2
    assert (tmp_path / "slides.tex").exists()
    # 过短拒绝
    r2 = tools["make_slides"].handler({"tex": "\documentclass{beamer}太短"})
    assert r2.get("error")
