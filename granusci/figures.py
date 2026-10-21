# -*- coding: utf-8 -*-
"""出版级图表生成：make_figure 工具的后端（matplotlib → PDF，APS 尺寸规范）。

所有图存 runs/<run>/figs/fig_<n>.pdf；LaTeX 以 includegraphics 直接引用。
数据一律由调用方（LLM）从工具返回记录中转录传入——本模块不自行取数，
保证图上数值与工具记录一致、可被数值审计追溯。
"""
from __future__ import annotations

import json
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# APS 单栏 3.4in / 双栏 7.0in；字号与 REVTeX 10pt 正文匹配
SINGLE_COL = 3.4
DOUBLE_COL = 7.0
# 感知均匀无障碍色板（Viridis 采样）——多系列默认色序
VIRIDIS = ["#440154", "#3B528B", "#21918C", "#5EC962", "#FDE725"]


def _cjk_font_chain() -> list[str]:
    """字体链：DejaVu（拉丁/数字）优先，其后插入系统 CJK 字体（中文标题/标签不再出方框）。

    paper11 教训：默认链只有 DejaVu Sans（无 CJK 字形），中文 title 渲染成方框。
    """
    import matplotlib.font_manager as fm
    cjk_candidates = ["Microsoft YaHei", "DengXian", "SimHei", "SimSun",
                      "Noto Sans CJK SC", "WenQuanYi Micro Hei"]
    available = {f.name for f in fm.fontManager.ttflist}
    chain = [c for c in cjk_candidates if c in available]
    return chain or ["sans-serif"]


plt.rcParams.update({
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.markersize": 4,
    "figure.dpi": 200,
    "savefig.dpi": 600,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.0,
    # 字体关键坑（三轮 Qwen 视觉审查确诊）：matplotlib 对含 $ 的混排串，
    # 段外文本用 font.family 列表第一个字体且不做逐字符回退——CJK 必须放首位，
    # 否则中文变方框（YaHei 的拉丁/数字字形微软优化过，论文图够用）；
    # 纯文本串才有逐字符回退。数学段用 dejavusans（与 YaHei 无衬线协调）。
    "font.family": _cjk_font_chain()[:1] + ["DejaVu Sans"] + _cjk_font_chain(),
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "axes.prop_cycle": plt.cycler(color=VIRIDIS),
    # 期刊轴风格（Qwen 审查"不像正规论文图"）：开放轴——去顶右 spine
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,           # 网格在数据下层
})

# 线型-标记冗余编码循环（色盲/灰度打印可辨——skill 规范"Never rely on color alone"）
# marker × linestyle 双维度，色是第三维度
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":"]
# marker 与 linestyle 同索引各自循环（Qwen 审查：任一维度单独外层都会退化成单维度）
# → 前4系列即得 ●- ■-- ▲-. ◆:，真正的双维度冗余
STYLE_CYCLE = [(MARKERS[i % len(MARKERS)], LINESTYLES[i % len(LINESTYLES)])
               for i in range(max(len(MARKERS), len(LINESTYLES)) * 4)]


def _collision_audit(ax) -> str | None:
    """程序化叠字检测（nature-figure 的 audit 思路移植，paper12 fig_76 教训：

    3×10⁰ 与 4×10⁰ 刻度标签互相黏连——VLM 审查看不出"重叠"，程序量 bbox
    交叉才是确定性的。返回警告文本或 None。
    """
    fig = ax.figure
    fig.canvas.draw()          # 强制布局，拿到真实 bbox
    warnings = []
    # ① 刻度标签两两交叉检测（沿轴方向的重叠：x 轴查水平、y 轴查垂直）
    for axis_name, horiz in (("xaxis", True), ("yaxis", False)):
        tick_bboxes = []
        for t in getattr(ax, axis_name).get_major_ticks():
            label = t.label1
            if label.get_text() and label.get_visible():
                try:
                    bb = label.get_window_extent()
                    if bb.width > 0:
                        tick_bboxes.append(bb)
                except Exception:
                    pass
        # 沿轴排序（x 轴按 x0，y 轴按 y0——刻度从下往上）
        tick_bboxes.sort(key=lambda b: b.x0 if horiz else b.y0)
        for i in range(len(tick_bboxes) - 1):
            a, b = tick_bboxes[i], tick_bboxes[i + 1]
            if horiz:
                overlap = a.x1 - b.x0        # 水平方向侵入
                ref = a.width
            else:
                overlap = a.y1 - b.y0        # 垂直方向侵入
                ref = a.height
            if overlap > 0.3 * ref:
                warnings.append(f"{axis_name[0]}轴刻度标签重叠（相邻标签间距不足）")
                break
    # ② 图例与轴标签交叉（图例压住轴文字）
    leg = ax.get_legend()
    if leg is not None:
        try:
            leg_bb = leg.get_window_extent()
            for txt in (ax.xaxis.label, ax.yaxis.label):
                bb = txt.get_window_extent()
                if leg_bb.overlaps(bb):
                    warnings.append("图例与轴标签重叠")
                    break
        except Exception:
            pass
    if warnings:
        return "叠字检测：" + "；".join(sorted(set(warnings))) + \
               "——收紧刻度稀疏化或移动图例（loc='best'）后重画"
    return None


def _parse_series(s: dict):
    """series dict -> (x, y, label)。y 支持标量列表或 {values, err}。"""
    x = np.asarray(s["x"], dtype=float)
    y = s["y"]
    if isinstance(y, dict):
        return x, np.asarray(y["values"], dtype=float), np.asarray(y.get("err", []), dtype=float)
    return x, np.asarray(y, dtype=float), None


def _require_data(x, y, ctx="series"):
    if len(x) == 0 or len(y) == 0:
        raise ValueError(f"{ctx} 数据为空（x len={len(x)}, y len={len(y)})——"
                         f"检查 from_call 字段路径是否正确，禁止画空图")


def make_figure(fig_dir: Path, spec: dict, fig_index: int) -> dict:
    """按 spec 生成一张图。

    spec 结构：
      {
        "plot_type": "line"|"scatter"|"loglog"|"semilogx"|"heatmap"|"errorbar"|"particles"|"panels",
        "title": 可选,
        "xlabel": str, "ylabel": str,
        "series": [{"x": [...], "y": [...] 或 {"values": [...], "err": [...]},
                    "label": 可选, "style": 可选("o-"/"s"--"等 matplotlib 风格)}...],
        "width": "single"|"double"（默认 single）,
        "grid": bool（默认 True）,
        "diag": bool（画 y=x 对角参考线，预测对照图用）,
        "cbar_label": heatmap 色标,
      }
    heatmap 时 series[0] 提供 {"x": [...刻度], "y": [...刻度], "z": [[...矩阵]]}
      —— z 为 len(y)×len(x) 矩阵，行对应 y 刻度。
    particles 时提供 {"particles": {"x":[...], "y":[...], "type":[...],
      "radius":[...], "speed":[...]}, "colorby": "type"|"speed"}。
    panels（多面板组图）时提供 {"panels": [子spec...], "ncols": 2, "panel_labels": ["a","b",...]}
      —— 每个子 spec 是普通单面板（除 panels 外字段皆可），共享 width；适合
      (a)(b)(c)(d) 参数扫描面板与机制对照图：各面板承担不同论证角色。
    series/particles 的数组可整体替换为 {"from_call": N, "x": "字段名", ...}
    引用第 N 次工具调用的返回数据（由 graph 层解析，本函数只见纯数组）。
    """
    pt = spec.get("plot_type", "line")
    fig_path = fig_dir / f"fig_{fig_index}.pdf"

    if pt == "panels":
        return _panels(fig_dir, spec, fig_path)
    if pt == "heatmap":
        return _heatmap(fig_dir, spec, fig_path)
    if pt == "particles":
        return _particles(fig_dir, spec, fig_path)

    width = DOUBLE_COL if spec.get("width") == "double" else SINGLE_COL
    fig, ax = plt.subplots(figsize=(width, width * 0.72))
    warn = _draw_single(ax, pt, spec)
    fig.tight_layout(pad=0.4)
    _save_both(fig, fig_path)
    plt.close(fig)
    result = {"ok": True, "figure": str(fig_path),
              "png": str(fig_path).replace(".pdf", ".png"),
              "figure_name": fig_path.stem,
              "n_series": len(spec.get("series", []))}
    if warn:
        result["quality_warning"] = warn   # 图例守卫：回传给 agent 促其补 label
    return result


def _plain_log_ticks(ax):
    """对数轴刻度规范化（paper12 质检发现）：

    matplotlib 默认 LogFormatter 把整数刻度渲染成 10⁰、2×10⁰ 这类
    科学计数——学术图规范：整数直接写数字，只有跨 ≥4 个数量级才用
    10^n 形式。同时收紧主刻度数防窄范围叠字（fig_76 的 3×10⁰/4×10⁰ 黏连）。
    """
    from matplotlib.ticker import FuncFormatter, LogLocator
    for name in ("xaxis", "yaxis"):
        axis = getattr(ax, name)
        if axis.get_scale() != "log":
            continue
        lo, hi = axis.get_view_interval()
        if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0 or hi <= 0:
            continue
        span = np.log10(hi) - np.log10(lo)
        if span < 2.0:
            # 窄范围（<2 个数量级）：整数刻度直接写数字
            axis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0)))
            axis.set_major_formatter(
                FuncFormatter(lambda v, _: f"{v:g}" if v >= 0.01 else f"{v:.0e}"))
        elif span < 4.0:
            # 中范围（2-4 个数量级）：10 的幂用上标形式，其余写数字
            axis.set_major_formatter(
                FuncFormatter(lambda v, _: (
                    f"$10^{{{int(round(np.log10(v)))}}}$"
                    if abs(np.log10(v) - round(np.log10(v))) < 1e-9 and v != 1
                    else f"{v:g}")))
        # ≥4 个数量级：matplotlib 默认科学计数本来就规范，不动


def _legend_guard(ax, spec):
    """图例完备性守卫（paper12 图 1 教训）：图上画了多个可区分系列
    （scatter 多色 / 多条线）但无任何 label → 自动补一个说明性警告水印，
    返回警告文本由调用方带回给 agent（vision 自查前就能发现）。
    """
    series = spec.get("series", [])
    labeled = sum(1 for s in series if s.get("label"))
    if len(series) > 1 and labeled == 0:
        warn = ("多系列无图例：图上有多个系列但都没有 label——读者无法区分。"
                "给每个 series 加 label（或单系列统一颜色）。")
        ax.text(0.98, 0.02, "WARNING: no legend", transform=ax.transAxes,
                fontsize=7, color="red", ha="right", va="bottom", alpha=0.75)
        return warn
    return None


def _draw_single(ax, pt: str, spec: dict):
    """单面板绘图核心（供单图与 panels 组图复用）。

    多系列自动分配 marker×linestyle 冗余编码（skill 规范"Never rely on
    color alone"——灰度打印/色盲必须可辨）；agent 显式传 style 时尊重其选择。
    返回图例守卫警告（无则 None）。
    """
    n_series = len(spec.get("series", []))
    for si, s in enumerate(spec.get("series", [])):
        x, y, err = _parse_series(s)
        _require_data(x, y)
        label = _tex(s.get("label")) if s.get("label") else None
        style = s.get("style")
        # 自动线型分配：多系列且未显式指定时按 marker+linestyle 循环
        if style is None and n_series > 1:
            mk, ls = STYLE_CYCLE[si % len(STYLE_CYCLE)]
            style = mk + ls
        if pt == "errorbar" or err is not None:
            ax.errorbar(x, y, yerr=err if err is not None and len(err) else None,
                        fmt=style or "o", capsize=2, label=label, markersize=4)
        elif pt == "loglog":
            ax.set_xscale("log", base=10)
            ax.set_yscale("log", base=10)
            ax.plot(x, y, style or "o-", label=label)
        elif pt == "semilogx":
            ax.set_xscale("log", base=10)
            ax.plot(x, y, style or "o-", label=label)
        else:
            ax.plot(x, y, style or "o-", label=label) if pt == "line" else \
                ax.scatter(x, y, s=14, label=label)

    ax.set_xlabel(_tex(spec.get("xlabel", "")))
    ax.set_ylabel(_tex(spec.get("ylabel", "")))
    if spec.get("title"):
        ax.set_title(_tex(spec["title"]))
    if spec.get("diag"):
        allv = np.concatenate([np.asarray(s.get("x", []), dtype=float)
                               for s in spec.get("series", [])] or [np.array([])])
        if allv.size:
            lo, hi = float(allv.min()), float(allv.max())
            pad = (hi - lo) * 0.05 if hi > lo else 0.1
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                    color="0.4", linestyle="--", linewidth=0.8,
                    label="_nolegend_", zorder=0)
    if any(s.get("label") for s in spec.get("series", [])):
        ax.legend(frameon=False)
    if spec.get("grid", True):
        ax.grid(True, alpha=0.3, linewidth=0.4)
    # 刻度规范化与图例守卫（paper12 质检三项修复）
    if pt in ("loglog", "semilogx", "semilogy"):
        _plain_log_ticks(ax)
    warns = []
    for w in (_legend_guard(ax, spec), _collision_audit(ax)):
        if w:
            warns.append(w)
    return "；".join(warns) if warns else None


_PANEL_LABELS = "abcdefghij"


def _panels(fig_dir: Path, spec: dict, fig_path: Path) -> dict:
    """多面板组图：每个子 spec 一个面板，各面板承担不同论证角色。"""
    panels = spec.get("panels") or []
    if not panels:
        return {"ok": False, "error": "no_panels",
                "message": "panels 图须提供 panels 数组（每项一个子 spec）"}
    if len(panels) > 8:
        return {"ok": False, "error": "too_many_panels",
                "message": f"面板数 {len(panels)} 超过 8——拆成多张图"}
    ncols = max(1, min(int(spec.get("ncols", 2)), len(panels)))
    nrows = (len(panels) + ncols - 1) // ncols
    width = DOUBLE_COL if spec.get("width") in ("double", None) else SINGLE_COL
    # 组图默认双栏（面板多），除非显式 single
    if spec.get("width") == "single":
        width = SINGLE_COL
    ph = spec.get("panel_height", 2.45)          # 每面板高度（英寸）
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width, ph * nrows),
                             squeeze=False)
    labels = spec.get("panel_labels") or list(_PANEL_LABELS[:len(panels)])
    # 统一 y 轴范围（同响应量跨参数面板直接对比）
    sharey = spec.get("sharey", False)
    if sharey:
        yvals = []
        for p in panels:
            for s in p.get("series", []):
                _, y, _ = _parse_series(s)
                yvals.append(np.asarray(y, dtype=float))
        if yvals:
            allv = np.concatenate(yvals)
            ymin, ymax = float(np.nanmin(allv)), float(np.nanmax(allv))
            marg = (ymax - ymin) * 0.06 or 0.1
            yspan = (ymin - marg, ymax + marg)
    warns = []
    for i, p in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        pt = p.get("plot_type", "line")
        if pt == "particles":
            # 子面板粒子渲染（paper13 图4 空图根因之二：panels 里 particles
            # 无渲染路径，静默画空坐标系）——直接在子 ax 上画
            w = _draw_particles_on_ax(ax, p)
        else:
            w = _draw_single(ax, pt, p)
        if w:
            warns.append(f"panel({labels[i] if i < len(labels) else i}): {w}")
        if sharey:
            ax.set_ylim(*yspan)
        if i < len(labels):
            ax.text(0.03, 0.95, f"({labels[i]})", transform=ax.transAxes,
                    fontsize=10, fontweight="bold", va="top",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="0.8", alpha=0.9, linewidth=0.5))
    # 隐藏多余面板
    for j in range(len(panels), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    if spec.get("suptitle"):
        fig.suptitle(_tex(spec["suptitle"]), fontsize=10)
    fig.tight_layout(pad=0.5)
    _save_both(fig, fig_path)
    plt.close(fig)
    result = {"ok": True, "figure": str(fig_path),
              "png": str(fig_path).replace(".pdf", ".png"),
              "figure_name": fig_path.stem,
              "n_panels": len(panels), "plot_type": "panels"}
    if warns:
        result["quality_warning"] = "；".join(warns)
    return result


def _heatmap(fig_dir: Path, spec: dict, fig_path: Path) -> dict:
    s0 = spec.get("series", [{}])[0]
    xs = np.asarray(s0.get("x", []), dtype=float)
    ys = np.asarray(s0.get("y", []), dtype=float)
    z = np.asarray(s0.get("z", []), dtype=float)
    if z.ndim != 2 or z.shape != (len(ys), len(xs)):
        return {"ok": False, "error": "bad_shape",
                "message": f"z 形状 {z.shape} 须为 ({len(ys)},{len(xs)})（行对应 y 刻度）"}
    width = DOUBLE_COL if spec.get("width") == "double" else SINGLE_COL
    fig, ax = plt.subplots(figsize=(width, width * 0.75))
    im = ax.pcolormesh(xs, ys, z, shading="nearest", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    if spec.get("cbar_label"):
        cbar.set_label(_tex(spec["cbar_label"]))
    ax.set_xlabel(_tex(spec.get("xlabel", "")))
    ax.set_ylabel(_tex(spec.get("ylabel", "")))
    if spec.get("title"):
        ax.set_title(_tex(spec["title"]))
    fig.tight_layout(pad=0.4)
    _save_both(fig, fig_path)
    plt.close(fig)
    return {"ok": True, "figure": str(fig_path),
            "png": str(fig_path).replace(".pdf", ".png"),
            "figure_name": fig_path.stem, "shape": list(z.shape)}


def _draw_particles_on_ax(ax, spec: dict) -> str | None:
    """在给定 ax 上渲染颗粒快照（panels 子面板用）。返回错误消息或 None。

    paper13 图4 空图根因之二：panels 里 particles 子面板没有渲染路径，
    _draw_single 画了空坐标系还返回成功。本函数把 _particles 的核心
    绘制逻辑移植到任意 ax（colorbar 由 fig 级处理，这里画 colorbar 简化为
    标量 colorby 图例省略——panels 场景按 speed 着色时可读性已足够）。
    """
    p = spec.get("particles") or {}
    x = np.asarray(p.get("x", []), dtype=float)
    y = np.asarray(p.get("y", []), dtype=float)
    if not len(x):
        return (f"particles 面板缺少 x/y 数组（from_call 是否解析成功？）"
                f"——本面板画的是空坐标系")
    r = np.asarray(p.get("radius", [0.001] * len(x)), dtype=float)
    colorby = spec.get("colorby", "type")
    fill_alpha = float(spec.get("alpha", 0.55))
    if colorby == "speed":
        c = np.asarray(p.get("speed", [0.0] * len(x)), dtype=float)
        ax.scatter(x, y, s=(r / r.max()) ** 2 * 90, c=c, cmap="inferno",
                   alpha=fill_alpha, edgecolors="k", linewidths=0.3)
    else:
        tp = np.asarray(p.get("type", [1] * len(x)), dtype=int)
        for t_val, color in ((1, "#3B528B"), (2, "#C44E52")):
            m = tp == t_val
            if m.any():
                ax.scatter(x[m], y[m], s=(r[m] / max(r.max(), 1e-12)) ** 2 * 90,
                           c=color, alpha=fill_alpha,
                           edgecolors="k", linewidths=0.3)
    ax.set_xlabel(_tex(spec.get("xlabel") or "x (m)"))
    ax.set_ylabel(_tex(spec.get("ylabel") or "y (m)"))
    if spec.get("title"):
        ax.set_title(_tex(spec["title"]))
    ax.set_aspect("equal", adjustable="datalim")
    return None


def _particles(fig_dir: Path, spec: dict, fig_path: Path) -> dict:
    """颗粒快照渲染（准二维 x-y 平面，圆盘尺寸∝真实半径）。"""
    p = spec.get("particles") or {}
    x = np.asarray(p.get("x", []), dtype=float)
    y = np.asarray(p.get("y", []), dtype=float)
    if not len(x):
        return {"ok": False, "error": "no_particles",
                "message": "particles 缺少 x/y 数组"}
    r = np.asarray(p.get("radius", [0.001] * len(x)), dtype=float)
    colorby = spec.get("colorby", "type")
    width = DOUBLE_COL if spec.get("width") == "double" else 3.0
    span_x = float(x.max() - x.min()) or 1.0
    span_y = float(y.max() - y.min()) or 1.0
    # 纵横比保真：figsize 按数据比例（高度设下限防轴标签占比过大），
    # equal aspect 交给 set_aspect 在 axes 框内自适应
    fig_h = max(1.2, width * (span_y / span_x if span_x > 0 else 1))
    fig, ax = plt.subplots(figsize=(width, fig_h))
    # 面积∝真实半径²——重叠场景半透明填充+深描边（Qwen 视觉审查：实心互叠糊成一片）
    fill_alpha = float(spec.get("alpha", 0.55))
    if colorby == "speed":
        c = np.asarray(p.get("speed", [0.0] * len(x)), dtype=float)
        sc = ax.scatter(x, y, s=(r / r.max()) ** 2 * 90, c=c, cmap="inferno",
                        alpha=fill_alpha, edgecolors="k", linewidths=0.3)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(_tex(spec.get("cbar_label") or "speed (m/s)"))
    else:
        tp = np.asarray(p.get("type", [1] * len(x)), dtype=int)
        # 规范图例：色块+说明（Qwen 审查：纯文字图例压数据且无颜色对应）
        from matplotlib.lines import Line2D
        handles, labels = [], []
        for t_val, color, name in ((1, "#3B528B", "small (type 1)"),
                                   (2, "#C44E52", "large (type 2)")):
            m = tp == t_val
            if m.any():
                # 实际数据层（无 label——图例用示意 handle 表达色块+尺寸）
                ax.scatter(x[m], y[m], s=(r[m] / max(r.max(), 1e-12)) ** 2 * 90,
                           c=color, alpha=fill_alpha,
                           edgecolors="k", linewidths=0.3)
                r_med = float(np.median(r[m]) / max(r.max(), 1e-12))
                handles.append(Line2D([0], [0], marker="o", linestyle="",
                                      markerfacecolor=color, markeredgecolor="k",
                                      markersize=6 * r_med + 3, alpha=0.9, label=name))
        if handles:
            # 图例放图外下方独立一行（Qwen 审查：与 xlabel 打架→锚点更低+更大 bottom）
            fig.legend(handles=handles, loc="upper center", ncol=len(handles),
                       frameon=False, fontsize=7,
                       bbox_to_anchor=(0.5, -0.005))
            fig.subplots_adjust(bottom=0.24)
    ax.set_xlabel(_tex(spec.get("xlabel") or "x (m)"))
    ax.set_ylabel(_tex(spec.get("ylabel") or "y (m)"))
    if spec.get("title"):
        ax.set_title(_tex(spec["title"]))
    # 边界外扩：物理半径 2.5 倍与数据跨度 1.5% 取大者
    # （Qwen 审查：pad 太小时颗粒贴边被视觉切半）
    r_max = float(r.max()) if len(r) else 0.001
    pad_abs = max(r_max * 2.5, 0.015 * max(span_x, span_y))
    ax.set_xlim(x.min() - pad_abs, x.max() + pad_abs)
    ax.set_ylim(y.min() - pad_abs, y.max() + pad_abs)
    ax.set_aspect("equal", adjustable="box")      # 边界含 pad 由我们控制，box 内等比
    fig.tight_layout(pad=0.3)
    _save_both(fig, fig_path)
    plt.close(fig)
    return {"ok": True, "figure": str(fig_path), "png": str(fig_path).replace(".pdf", ".png"),
            "figure_name": fig_path.stem, "n_particles": len(x), "colorby": colorby}


def _save_both(fig, fig_path: Path):
    """PDF（LaTeX 用）+ PNG（视觉分析用）双写。"""
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)


# 常见物理记号的"裸写→mathtext"自动转换（Qwen 审查：字面 r_s/r_ρ/Δh* 平排，下标未排版）
# 模式：变量_下标 / 希腊字母 / Δh* / Γ 等智能体习惯性裸写的记号
_MATHIFY_PATTERNS = [
    # r_s / r_rho / r_ρ / d_s / e_k / E_k 等 变量_下标（下标是字母或希腊名）
    (re.compile(r"\b([A-Za-z]+)_([A-Za-z]+|rho|gamma)\b(?![\w])"),
     r"$\1_{\2}$"),
]


def _mathify_token(tok: str) -> str:
    """单个 token 的数学化：r_s→$r_s$、r_ρ→$r_{\\rho}$、Δh*→$\\Delta h^*$、Γ→$\\Gamma$。"""
    # 希腊字母 Unicode 直接映射
    greek = {"Γ": r"\Gamma", "Δ": r"\Delta", "ρ": r"\rho", "γ": r"\gamma",
             "π": r"\pi", "δ": r"\delta", "θ": r"\theta", "λ": r"\lambda",
             "σ": r"\sigma", "μ": r"\mu", "τ": r"\tau", "ω": r"\omega",
             "Φ": r"\Phi", "Σ": r"\Sigma", "Ω": r"\Omega"}
    # 1) 变量_下标（含希腊下标名的英文写法）
    m = re.match(r"^([A-Za-z]+)_([A-Za-z]+(?:_[A-Za-z]+)*)$", tok)
    if m:
        var, sub = m.group(1), m.group(2)
        sub_map = {"rho": r"\rho", "gamma": r"\gamma", "delta": r"\delta",
                   "pi": r"\pi", "theta": r"\theta", "sigma": r"\sigma",
                   "star": "*", "max": r"\mathrm{max}", "min": r"\mathrm{min}",
                   "eff": r"\mathrm{eff}"}
        sub_tex = sub_map.get(sub, sub)          # 普通字母直接用
        return f"${var}_{{{sub_tex}}}$"
    # 2) Δh* / dh* 型（Δ 是 Unicode 或裸写 d）
    m = re.match(r"^([ΔdD])([a-zA-Z]+)(\*?)$", tok)
    if m:
        pre, body, star = m.groups()
        pre_tex = r"\Delta" if pre in "ΔD" else "d"
        star_tex = "^{*}" if star else ""
        return f"${pre_tex} {body}{star_tex}$" if pre in "ΔD" else f"${pre_tex}{body}{star_tex}$"
    # 3) 希腊字母裸 Unicode
    for uni, tex in greek.items():
        if uni in tok:
            return tok.replace(uni, f"${tex}$") if tok == uni else \
                re.sub(f"([A-Za-z0-9^_]+)?{uni}([A-Za-z0-9^_]+)?",
                       lambda mm: f"${mm.group(0)}$", tok)
    return tok


def _tex(s: str) -> str:
    r"""标签文本 -> mathtext 安全形式（自动数学化 + 失败降级）。

    智能体可直传 LaTeX（$r_s$ 原样过）；裸写记号自动转换：
    r_s→$r_s$ 下标排版、r_ρ→$r_{\rho}$、Δh*→$\Delta h^{*}$、Γ→$\Gamma$
    （Qwen 视觉审查沉淀：字面 r_s 平排不合格，物理记号必须数学排版）。
    paper11 教训：mathtext 不支持的命令触发 ParseException 整图失败——
    解析不过时剥掉 $ 降级纯文本（显示打折也好过整图失败）。
    """
    s = str(s)
    # 整串是单个 $...$ 段才直通（"$r_s$"）；"$r_s$ vs Γ" 混排串走逐 token 数学化
    if s.startswith("$") and s.endswith("$") and s.count("$") == 2:
        try:
            import matplotlib.mathtext as _mt
            _mt.MathTextParser("agg").parse(s)
            return s
        except Exception:
            return s.strip("$")     # 降级：剥 $ 当纯文本
    # 裸文本：先切出已有 $...$ 段保护，其余逐 token 数学化
    try:
        import matplotlib.mathtext as _mt
        parts = re.split(r"(\$[^$]*\$)", s)      # $...$ 段与普通段交替
        out = []
        for part in parts:
            if part.startswith("$") and part.endswith("$") and len(part) > 1:
                out.append(part)                    # 已是 math 段，原样
                continue
            for tok in re.split(r"(\s+|\(|\)|=|,)", part):
                if not tok or tok.strip() in ("(", ")"):
                    out.append(tok)
                    continue
                new = _mathify_token(tok)
                if new != tok:
                    _mt.MathTextParser("agg").parse(new)
                out.append(new)
        return "".join(out)
    except Exception:
        return s                     # 任一环失败：原文返回（至少不比之前差）
