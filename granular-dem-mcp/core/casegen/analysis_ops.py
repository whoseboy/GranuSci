# -*- coding: utf-8 -*-
"""诊断箱 v1 标准算子：从末帧 dump 直接产出标准几何/物理量（A2-c）。

背景（C1 三测教训）：agent 手工从 bbox 推导堆积分数算错（0.00235 vs 真值
≈0.55-0.6，把整域体积当床层体积）。标准物理量应由工具给出，agent 只消费。

算子（全部从 dump 帧计算，输出自动带计算说明供审计）：
  bed_geometry(帧)    床层几何：按重力方向的包围高度（含半径修正）、
                      床层包围体积（bbox 正截面×床高）、堆积分数
                      φ = ΣV_p / V_bed、孔隙率 = 1-φ
  density_profile(帧) 沿重力方向分层密度剖面（N 层，每层局部堆积分数）
                      ——分层/流化/表面结构的诊断素材
  coordination_hint   （v2：配位数——需接触检测，暂缓）
约定：重力轴 = case.fields.gravity 的主分量轴（z-up 世界，默认 [0,0,-9.81]）。
"""
from __future__ import annotations

import math


def _load_frame_rows(path: str):
    rows = []
    in_atoms = False
    col = None  # 列名→位置索引（按 dump 头解析，不假设列序）
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("ITEM: ATOMS"):
                # "ITEM: ATOMS id type x y z vx vy vz radius" → 列名表
                col = {name: i for i, name in enumerate(line.split()[2:])}
                need = {"x", "y", "z", "vx", "vy", "vz"}
                if not need <= set(col):
                    missing = sorted(need - set(col))
                    raise ValueError(
                        f"dump 列头缺必需列 {missing}——实际列：{line.split()[2:]}。"
                        f"case 的 output.dump_columns 须至少包含 "
                        f"id/type/x/y/z/vx/vy/vz/radius（缺速度列时床几何/诊断无法计算）")
                in_atoms = True
                continue
            if line.startswith("ITEM:") or not line:
                if in_atoms:
                    break
                continue
            if in_atoms and col is not None:
                p = line.split()
                if len(p) == len(col):
                    rows.append({
                        "id": int(p[col["id"]]) if "id" in col else len(rows),
                        "type": int(p[col["type"]]) if "type" in col else 1,
                        "x": float(p[col["x"]]), "y": float(p[col["y"]]),
                        "z": float(p[col["z"]]),
                        "vx": float(p[col["vx"]]), "vy": float(p[col["vy"]]),
                        "vz": float(p[col["vz"]]),
                        "r": float(p[col["radius"]]) if "radius" in col
                             else float(p[col.get("r", col["x"])] or 0) or 0.001})
    return rows


def _grav_axis(gravity: list[float]) -> int:
    return max(range(3), key=lambda i: abs(gravity[i]))


def bed_geometry(path: str, gravity: list[float],
                 bed_base: float | None = None) -> dict:
    """床层几何与堆积分数（C1 三测的直接修复算子）。

    床层体积 = 床层平面包围盒（水平两轴的 bbox+粒径修正）× 床高
    （颗粒最高点+半径 - 床底）。φ = 颗粒总体积/床层体积。
    bed_base 不给时取帧内最低点-半径（床底=底壁位置近似）。
    """
    rows = _load_frame_rows(path)
    if not rows:
        return {"ok": False, "error": "no_atoms", "message": "帧内无颗粒"}
    gax = _grav_axis(gravity)
    hax = [i for i in range(3) if i != gax]           # 两个水平轴
    coord = {0: "x", 1: "y", 2: "z"}[gax]

    pos = [r[coord] for r in rows]
    r_mean = sum(r["r"] for r in rows) / len(rows)
    z_min, z_max = min(pos), max(pos)
    base = bed_base if bed_base is not None else z_min - r_mean
    h_top = z_max + r_mean
    bed_h = h_top - base

    # 水平包围（含半径修正——单颗粒半径=自身半径，非均值，避免过度修正）
    span_h = []
    for ax in hax:
        c = {0: "x", 1: "y", 2: "z"}[ax]
        vals = [r[c] for r in rows]
        lo = min(vals) - _r_at(rows, min(vals), c)
        hi = max(vals) + _r_at(rows, max(vals), c)
        span_h.append(hi - lo)

    v_bed = span_h[0] * span_h[1] * bed_h
    v_particles = sum(4 / 3 * math.pi * r["r"] ** 3 for r in rows)
    phi = v_particles / v_bed if v_bed > 0 else float("nan")

    return {
        "ok": True, "n": len(rows), "gravity_axis": coord,
        "bed_height_m": round(bed_h, 5),
        "bed_base_m": round(base, 5), "bed_top_m": round(h_top, 5),
        "bed_footprint_m2": round(span_h[0] * span_h[1], 6),
        "bed_volume_m3": round(v_bed, 8),
        "particle_volume_m3": round(v_particles, 10),
        "packing_fraction": round(phi, 4),
        "porosity": round(1 - phi, 4) if phi == phi else None,
        "sanity": ("随机密堆典型 φ≈0.55-0.64（规则排列更低）；表面单层或极稀"
                   "分布时 φ 会显著更低——若远低于 0.5 且 n 大，检查是否仍在"
                   "空中/域过大")}


def _r_at(rows, val, coord):
    """取处于该坐标极值的颗粒半径（修正用）。"""
    rs = [r["r"] for r in rows if abs(r[coord] - val) < 1e-12]
    return max(rs) if rs else 0.0


def density_profile(path: str, gravity: list[float],
                    n_layers: int = 10) -> dict:
    """沿重力轴的分层局部堆积分数（每层颗粒体积/层体积）。"""
    rows = _load_frame_rows(path)
    if not rows:
        return {"ok": False, "error": "no_atoms"}
    gax = _grav_axis(gravity)
    coord = {0: "x", 1: "y", 2: "z"}[gax]
    hax = [i for i in range(3) if i != gax]

    pos = [r[coord] for r in rows]
    r_max = max(r["r"] for r in rows)
    lo, hi = min(pos) - r_max, max(pos) + r_max
    dz = (hi - lo) / n_layers

    # 层水平截面（按全体颗粒的水平包围——层内稀疏时不缩小截面，保持可比）
    span = []
    for ax in hax:
        c = {0: "x", 1: "y", 2: "z"}[ax]
        vals = [r[c] for r in rows]
        span.append(max(vals) + r_max - (min(vals) - r_max))
    area = span[0] * span[1]

    layers = []
    for k in range(n_layers):
        z0 = lo + k * dz
        members = [r for r in rows if z0 <= r[coord] < z0 + dz]
        v_p = sum(4 / 3 * math.pi * r["r"] ** 3 for r in members)
        v_layer = area * dz
        layers.append({
            "layer": k, "z_center_m": round(z0 + dz / 2, 5),
            "n": len(members),
            "local_phi": round(v_p / v_layer, 4) if v_layer > 0 else None})

    return {"ok": True, "n_layers": n_layers, "gravity_axis": coord,
            "z_range_m": [round(lo, 5), round(hi, 5)],
            "layers": layers,
            "usage": ("局部 φ 的层间变化即密度剖面：底部平台≈密堆、"
                      "顶部渐零=自由表面；中部凹陷=空隙/流化区")}


# ================================================================ v1.1
# A2-d 分析算子库：拟合 / 置信区间 / 模型比较（输出自带溯源口径）
# 设计：输入=工况-响应数据（CSV 字符串或行列表），全部纯 numpy/scipy
# 实现，输出结构含口径说明（供数值审计门核验）。

def _parse_xy(data):
    """接受 [(x...), y] 行列表或 CSV 文本（首行表头自动跳过）。"""
    rows = []
    if isinstance(data, str):
        for line in data.strip().splitlines():
            parts = line.replace(",", " ").split()
            try:
                vals = [float(p) for p in parts]
                rows.append(vals)
            except ValueError:
                continue  # 表头
    else:
        rows = [list(map(float, r)) for r in data]
    if not rows or len(rows[0]) < 2:
        raise ValueError("数据须为每行 [x1...xk, y]")
    X = [r[:-1] for r in rows]
    y = [r[-1] for r in rows]
    return X, y


def fit_power_law(data, n_params: int = 1) -> dict:
    """幂律拟合 y = a * x^b（单变量）或多元 log-log 线性回归。
    返回系数 + 参数标准误（CI 原料）+ R²（log 与物理空间双报）+ 残差。
    """
    import numpy as np
    X, y = _parse_xy(data)
    X = np.asarray(X)
    y = np.asarray(y)
    if np.any(X <= 0) or np.any(y <= 0):
        return {"ok": False, "message": "幂律拟合要求全部 x,y>0（log 变换域）"}
    lx = np.log10(X)
    ly = np.log10(y)
    Xd = np.column_stack([np.ones(len(lx)), lx])
    beta, res, rank, sv = np.linalg.lstsq(Xd, ly, rcond=None)
    n, k = Xd.shape
    resid = ly - Xd @ beta
    s2 = (resid ** 2).sum() / max(n - k, 1)
    cov = s2 * np.linalg.pinv(Xd.T @ Xd)
    se = np.sqrt(np.diag(cov))
    r2_log = 1 - (resid ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    yhat = 10 ** (Xd @ beta)
    ss = ((y - yhat) ** 2).sum()
    r2_phys = 1 - ss / ((y - y.mean()) ** 2).sum()
    import math as _m
    from scipy import stats as _st
    tcrit = _st.t.ppf(0.975, max(n - k, 1))
    ci = [[float(b - tcrit * e), float(b + tcrit * e)] for b, e in zip(beta, se)]
    return {
        "ok": True, "n": n,
        "coefficients": [float(b) for b in beta],
        "coef_names": (["const", "log10(x)"] if k == 2
                       else ["const"] + [f"log10(x{i+1})" for i in range(k - 1)]),
        "std_err": [float(e) for e in se],
        "ci95": ci,
        "r2_log": round(float(r2_log), 4),
        "r2_physical": round(float(r2_phys), 4),
        "note": ("口径：log10 空间最小二乘；R² 双报（log/物理），"
                 "CI=t(0.975, n-k)×SE；物理空间 R² 通常显著低于 log 空间，"
                 "报告时两者都要给（C1 三测教训）"),
        "model": ("y = 10^{:.4g} * x^{:.4g}".format(beta[0], beta[1])
                  if k == 2 else "log10 y = " + " + ".join(
                      f"{b:.4g}*{nm}" for b, nm in zip(beta[1:], [""]*0) or []) or "multivariate"),
    }


def model_compare(data) -> dict:
    """模型比较：幂律 vs 线性 vs 二次多项式（单变量），AIC/BIC + 各自 R²。
    用于"该用哪个函数形式"的裁决（假设层 scaling 类检验的配套）。
    """
    import numpy as np
    X, y = _parse_xy(data)
    x = np.asarray([r[0] for r in X])
    y = np.asarray(y)
    n = len(y)
    results = {}

    def _fit(name, Xd):
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        r = y - Xd @ beta
        rss = (r ** 2).sum()
        k = Xd.shape[1]
        aic = n * np.log(max(rss, 1e-300) / n) + 2 * k
        bic = n * np.log(max(rss, 1e-300) / n) + k * np.log(n)
        r2 = 1 - rss / ((y - y.mean()) ** 2).sum()
        results[name] = dict(k=k, rss=float(rss), AIC=float(aic), BIC=float(bic),
                             R2=float(r2))

    _fit("linear", np.column_stack([np.ones(n), x]))
    _fit("quadratic", np.column_stack([np.ones(n), x, x ** 2]))
    # 幂律：在物理空间拟合（y = a·x^b 最小二乘），保证与 linear/quadratic
    # 的 AIC/BIC 同响应空间可比。旧实现把 log 空间 RSS 与物理空间 RSS 直接
    # 比 BIC——log 变换天然压缩残差，幂律被系统性偏爱（审计 B-4）。
    power_fitted = False
    if np.all(x > 0) and np.all(y > 0):
        from scipy.optimize import curve_fit
        try:
            def _pw(t, a, b):
                return a * t ** b
            popt, _ = curve_fit(_pw, x, y, p0=[y[0] / max(x[0], 1e-12), 1.0],
                                maxfev=20000)
            r = y - _pw(x, *popt)
            rss = float((r ** 2).sum())
            k = 2
            results["power_law"] = dict(
                k=k, rss=rss,
                AIC=float(n * np.log(max(rss, 1e-300) / n) + 2 * k),
                BIC=float(n * np.log(max(rss, 1e-300) / n) + k * np.log(n)),
                R2=float(1 - rss / ((y - y.mean()) ** 2).sum()),
                coef=[float(popt[0]), float(popt[1])],
                model=f"y = {popt[0]:.4g} * x^{popt[1]:.4g}",
                note="物理空间非线性最小二乘（curve_fit）——与其他模型同空间，BIC 可比")
            power_fitted = True
        except Exception:  # noqa: BLE001  曲线拟合不收敛（含负值/发散）
            pass
    if not power_fitted:
        results["power_law"] = dict(
            k=2, rss=None, AIC=None, BIC=None, R2=None,
            note="不可拟合（x 或 y 含非正值）")
    comparable = [k_ for k_ in results if results[k_].get("BIC") is not None]
    best = min(comparable, key=lambda k_: results[k_]["BIC"]) if comparable else None
    return {"ok": True, "n": n, "models": results,
            "best_by_BIC": best,
            "note": "BIC 越小越好；三模型统一物理空间拟合（幂律为非线性最小"
                    "二乘），BIC 跨模型可比。幂律不收敛或数据含非正值时不参与"
                    "裁决。"}


# ================================================================ v1.2
# B1 观察工具集（门 2b 发现的直接转化）：假设引出前的结构化观察接口。
# 门 2b 结论：裸数值表是假设感知的最不利工况（0.684）；残差/剖面视图下
# 0.90+。本节把这些视图做成标准算子——agent 提假设前必须先过观察工具。

def residual_view(data) -> dict:
    """全局幂律残差视图（门 2b 实验的残差呈现臂）。
    拟合全局 log-log 幂律后，返回按各变量排序的残差表——跳变/分区结构
    在残差列上被放大（原始数值表中信号弱）。
    data: 每行 [x1,...,xk, y]。
    """
    import numpy as np
    X, y = _parse_xy(data)
    X = np.asarray(X)
    y = np.asarray(y)
    if np.any(X <= 0) or np.any(y <= 0):
        return {"ok": False, "message": "残差视图要求 x,y>0"}
    lx = np.log10(X)
    ly = np.log10(y)
    Xd = np.column_stack([np.ones(len(lx)), lx])
    beta, *_ = np.linalg.lstsq(Xd, ly, rcond=None)
    ratio = 10 ** (ly - Xd @ beta)          # y/y_fit（门 2b 同口径）
    # 按每个变量排序的残差序列（分变量呈现=结构检测）
    views = {}
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j])
        views[f"x{j+1}_sorted_ratio"] = [
            {"x": float(X[o, j]), "ratio": round(float(ratio[o]), 3)}
            for o in order]
    return {
        "ok": True, "n": len(y),
        "global_beta": [float(b) for b in beta],
        "residual_ratio_views": views,
        "usage": ("ratio≈1=符合全局幂律；ratio 的系统性漂移/跳变=分区或"
                  "状态切换信号（按变量排序找台阶）。假设引出纪律：先看本视图"
                  "再下结构主张（门 2b：裸表感知 0.684 → 残差视图 0.90+）"),
    }


def diff_view(data, var_index: int = 0) -> dict:
    """差分视图：按某变量排序后的一阶差分（检测断点/状态切换）。
    门 2b：M3 型跳变需主动差分观察才能发现。
    """
    import numpy as np
    X, y = _parse_xy(data)
    X = np.asarray(X)
    y = np.asarray(y)
    order = np.argsort(X[:, var_index])
    xs = X[order, var_index]
    ys = y[order]
    dx = np.diff(xs)
    dy = np.diff(ys)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(np.abs(dx) > 0, dy / np.maximum(np.abs(dx), 1e-12), np.nan)
    # 断点检测（双判据，任一命中）：斜率超中位 3 倍；或 dy 偏离
    # 平滑预期（|dy - median(dy)| > 3×MAD）
    med = np.nanmedian(np.abs(slope)) or 1e-12
    dy_med = np.median(dy)
    dy_mad = np.median(np.abs(dy - dy_med)) or 1e-12
    breaks = []
    for i in range(len(slope)):
        slope_hit = np.isfinite(slope[i]) and abs(slope[i]) > 3 * med
        jump_hit = abs(dy[i] - dy_med) > 3 * dy_mad
        if slope_hit or jump_hit:
            breaks.append(int(i))
    return {
        "ok": True, "var": f"x{var_index+1}", "n_points": len(y),
        "break_candidates": breaks,
        "note": (f"按 x{var_index+1} 排序的一阶差分；疑似断点 "
                 f"{len(breaks)} 处（斜率超中位 5 倍）——断点=分区边界候选，"
                 "配 hypothesis_register(boundaries) 使用"),
    }
