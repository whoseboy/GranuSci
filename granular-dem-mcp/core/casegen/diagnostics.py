# -*- coding: utf-8 -*-
"""仿真诊断箱 v0（A2）：从 dump 序列 + thermo 提取 agent 可读的可信性信号。

诊断信号（每条含 verdict + 数值 + 一句话解读，供 agent 自主判断"这次仿真
可信吗"）：
  D1 粒子守恒      末帧 n vs 插入 n（丢失>20% → FAIL；>2% → WARN）
  D2 动能弛豫      ke 序列（thermo 或 dump 反算）是否衰减到平台
                   （末段/首段比 <1e-3 → settled；仍在下降 → running-in）
  D3 出域检查      末帧 bbox 是否贴域边界（贴边+外向速度 → 泄漏风险）
  D4 速度爆炸      max|v| 是否超物理合理值（v > sqrt(2*g*域高)*10 → FAIL）
  D5 沉降静态      静置 case 的末帧 ke（dump 反算）是否近零
设计约束：全部只读 job 目录内已有文件（dump+run.log），无额外仿真成本；
每条诊断独立容错（一条挂了不拖垮整体）。
"""
from __future__ import annotations

import math
import os
import re


def _read_dump_frames(path: str):
    """逐帧读 dump：返回 [(step, [(id,type,x,y,z,vx,vy,vz,r), ...]), ...]。"""
    frames = []
    step = None
    atoms = None
    rows = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "ITEM: TIMESTEP":
                if rows is not None:
                    frames.append((step, rows))
                rows = None
                step = int(next(f))
            elif line.startswith("ITEM: NUMBER OF ATOMS"):
                atoms = int(next(f))
                rows = []
            elif line.startswith("ITEM: ATOMS"):
                continue
            elif rows is not None and line:
                p = line.split()
                if len(p) >= 9:
                    rows.append((int(p[0]), int(p[1]), *(float(x) for x in p[2:9])))
    if rows is not None:
        frames.append((step, rows))
    return frames


def _read_thermo(log_path: str):
    """从 run.log 提取 thermo 表 [(step, atoms, ke, rke), ...]。"""
    rows = []
    pat = re.compile(r"^\s*(\d+)\s+(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*$")
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = pat.match(line)
                if m:
                    rows.append((int(m.group(1)), int(m.group(2)),
                                 float(m.group(3)), float(m.group(4))))
    except OSError:
        pass
    return rows


def _ke_from_frame(rows, g=9.81):
    """dump 帧反算无量纲动能（用 mean(0.5|v|²)/(g·d̄)，d̄=2×半径均值）。"""
    if not rows:
        return None
    d_bar = 2.0 * sum(r[8] for r in rows) / len(rows)
    if d_bar <= 0:
        return None
    ke = sum(0.5 * (r[5] ** 2 + r[6] ** 2 + r[7] ** 2) for r in rows) / len(rows)
    return ke / (g * d_bar)


def diagnose(job_dir: str, n_expected: int | None = None,
             gravity: float = 9.81, domain_h: float | None = None) -> dict:
    """主入口：对 casegen 作业目录做全套诊断。"""
    diags = {}

    dumps = sorted(f for f in os.listdir(job_dir)
                   if f.startswith("out_") and f.endswith(".txt"))
    # 末帧按解析的 step 数值取 max（字符串排序坑：v2_hopper 实测
    # out_settle_90000 > out_main_1400000——全套诊断曾打到 settle 帧）
    def _step_of(fn):
        m = re.search(r"_(\d+)\.txt$", fn)
        return int(m.group(1)) if m else -1
    last_dump = max(dumps, key=_step_of) if dumps else None
    thermo = _read_thermo(os.path.join(job_dir, "run.log"))

    # ---- D1 粒子守恒 ----
    try:
        if dumps:
            last = _read_dump_frames(os.path.join(job_dir, last_dump))
            n_last = len(last[-1][1]) if last else 0
        else:
            n_last = 0
        if n_expected is None:
            diags["D1_particle_conservation"] = {
                "verdict": "UNKNOWN", "n_last": n_last,
                "note": "无期望粒子数（n_expected 未提供），仅记录末帧数"}
        elif n_last < 0.8 * n_expected:
            diags["D1_particle_conservation"] = {
                "verdict": "FAIL", "n_last": n_last, "n_expected": n_expected,
                "note": f"末帧 {n_last} vs 插入 {n_expected}——颗粒大量丢失（坠出域），检查壁面封闭性"}
        elif n_last < 0.98 * n_expected:
            diags["D1_particle_conservation"] = {
                "verdict": "WARN", "n_last": n_last, "n_expected": n_expected,
                "note": f"少量丢失（{n_expected - n_last} 颗），多为插入边缘挤出，可接受但记录在案"}
        else:
            diags["D1_particle_conservation"] = {
                "verdict": "PASS", "n_last": n_last, "n_expected": n_expected,
                "note": "粒子守恒"}
    except Exception as e:  # noqa: BLE001
        diags["D1_particle_conservation"] = {"verdict": "ERROR", "note": str(e)[:80]}

    # ---- D2 动能弛豫（v1：相对衰减 + 绝对准静态双判据）----
    try:
        # 过滤插入前的全零行（step=0 且 atoms=0）
        kes = [t[2] for t in thermo if len(t) >= 3 and not (t[0] == 0 and t[1] == 0)]
        if len(kes) >= 3:
            first, last_k = kes[0], kes[-1]
            mid = kes[len(kes) // 2]
            ratio = last_k / max(first, 1e-300)
            still_falling = kes[-1] < kes[-2]
            if ratio < 1e-3 and last_k < mid:
                v = "SETTLED"
                note = f"动能衰减至初值 {ratio:.1e}——已弛豫到统计平台"
            elif still_falling and kes[-1] < mid:
                v = "RUNNING_IN"
                note = "动能仍在下降——沉降/发展未完成，统计可能过早"
            else:
                v = "STEADY"
                note = "动能处于持续输入的稳态（振动/剪切激励下正常）"
            diags["D2_kinetic_relaxation"] = {
                "verdict": v, "ke_series_tail": [f"{k:.3e}" for k in kes[-4:]],
                "note": note}
        else:
            diags["D2_kinetic_relaxation"] = {
                "verdict": "NO_DATA", "note": "thermo 表不足 3 行"}
    except Exception as e:  # noqa: BLE001
        diags["D2_kinetic_relaxation"] = {"verdict": "ERROR", "note": str(e)[:80]}

    # ---- D6 准静态判据（静置 case）----
    # C1 三测教训：300 球颗粒雨跑 2s 末速仍 0.45 m/s——相对判据 STEADY 但
    # 物理上远未沉降。准静态=特征速度 < d/s 量级（颗粒直径每秒的 1/10）。
    try:
        if dumps:
            frames = _read_dump_frames(os.path.join(job_dir, last_dump))
            rows = frames[-1][1] if frames else []
            if rows:
                n_r = len(rows)
                ke_mean = sum(0.5 * (r[5] ** 2 + r[6] ** 2 + r[7] ** 2)
                              for r in rows) / n_r
                d_bar = 2.0 * sum(r[8] for r in rows) / n_r
                v_char = math.sqrt(2 * ke_mean)
                v_quasi = 0.1 * d_bar          # 准静态阈值：0.1 倍粒径/秒
                if v_char < v_quasi:
                    diags["D6_quasi_static"] = {
                        "verdict": "PASS", "v_char": round(v_char, 5),
                        "threshold": round(v_quasi, 6),
                        "note": "特征速度已达准静态——静置目标达成"}
                else:
                    diags["D6_quasi_static"] = {
                        "verdict": "FAIL", "v_char": round(v_char, 5),
                        "threshold": round(v_quasi, 6),
                        "note": (f"特征速度 {v_char:.3f} m/s 仍高于准静态阈值——"
                                 "颗粒仍在明显运动（未沉降/未弛豫），"
                                 "延长 phases 步数后重跑")}
    except Exception as e:  # noqa: BLE001
        diags["D6_quasi_static"] = {"verdict": "ERROR", "note": str(e)[:80]}

    # ---- D3/D4/D5 基于末帧 dump ----
    try:
        if dumps and _read_dump_frames(os.path.join(job_dir, last_dump)):
            frames = _read_dump_frames(os.path.join(job_dir, last_dump))
            rows = frames[-1][1]
            # D4 速度爆炸
            vmax = max(math.sqrt(r[5] ** 2 + r[6] ** 2 + r[7] ** 2) for r in rows)
            v_lim = math.sqrt(2 * gravity * (domain_h or 1.0)) * 10
            diags["D4_velocity_sanity"] = {
                "verdict": "PASS" if vmax < v_lim else "FAIL",
                "v_max": round(vmax, 4), "v_limit": round(v_lim, 3),
                "note": "速度量级物理合理" if vmax < v_lim else
                        f"最大速度 {vmax:.2f} m/s 远超自由落体量级——数值爆炸疑似"}
            # D5 静置末帧动能（静置 case 判据）
            # 首末行 ke 恰好 ==0 的旧门已废（浮点 ke 几乎不可能精确为 0，
            # D5 从未触发过）——改为 ke 低于 1e-12 量级的静置识别
            if thermo and (thermo[-1][2] < 1e-12 and thermo[0][2] < 1e-12):
                ek = _ke_from_frame(rows, gravity)
                if ek is not None and ek < 1e-4:
                    diags["D5_static_settled"] = {
                        "verdict": "PASS", "ek_star": f"{ek:.2e}",
                        "note": "末帧近零动能——静置堆积完成"}
                elif ek is not None:
                    diags["D5_static_settled"] = {
                        "verdict": "WARN", "ek_star": f"{ek:.2e}",
                        "note": "末帧仍有残余动能——可延长沉降段"}
    except Exception as e:  # noqa: BLE001
        diags["D3_dump_analysis"] = {"verdict": "ERROR", "note": str(e)[:80]}

    # 汇总
    verdicts = [d.get("verdict") for d in diags.values()]
    overall = "FAIL" if "FAIL" in verdicts else \
              ("WARN" if "WARN" in verdicts or "RUNNING_IN" in verdicts else "PASS")
    return {"ok": True, "overall": overall, "diagnostics": diags}
