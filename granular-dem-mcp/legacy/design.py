# -*- coding: utf-8 -*-
"""实验设计生成 —— 移植自原 make_design.py（公式已用原 design.csv 数值验证）。

换算公式：
  振幅 a = (a/d) * d
  周期 T = 2*pi*sqrt(a / (Gamma*g))
  save   = round(T / dt / 8)          # 每 1/8 振动周期 dump 一次
  run_dev  = save * dev_periods * 8   # 发展段
  run_stat = save * stat_periods * 8  # 统计段
  颗粒数 n = F * round(L/d)           # L=0.1m -> 每层 25 颗（d=4mm）
"""
import json
import math
import os
import time

G = 9.81
DT = 1e-5
DUMP_PER_PERIOD = 8
L_BOX = 0.1  # 槽长 m（容器几何）


def _case(idx, ad, gamma, F, d, dev_periods, stat_periods):
    a = ad * d
    period = 2 * math.pi * math.sqrt(a / (gamma * G))
    save = int(round(period / DT / DUMP_PER_PERIOD))
    per_layer = int(round(L_BOX / d))
    return {
        "case_id": idx,
        "ad": ad, "gamma": gamma, "F": F, "d": d,
        "n_particles": int(F * per_layer),
        "amplitude": round(a, 6),
        "period": round(period, 6),
        "save": save,
        "run_dev": save * dev_periods * DUMP_PER_PERIOD,
        "run_stat": save * stat_periods * DUMP_PER_PERIOD,
    }


RHO_S = 2500.0  # 小颗粒密度（容器默认颗粒材料，玻璃珠量级）


def _case_binary(idx, ad, gamma, F, d_s, r_s, r_rho, x_large,
                 dev_periods, stat_periods):
    """二元混合工况：d=d_s（小颗粒直径，a/d 与 F 均以小颗粒计）。

    r_s     = d_l/d_s  粒径比
    r_rho   = rho_l/rho_s 密度比
    x_large = 大颗粒体积分数（混合分布质量权重由体积分数+密度换算）
    """
    c = _case(idx, ad, gamma, F, d_s, dev_periods, stat_periods)
    rho_s, rho_l = RHO_S, RHO_S * r_rho
    w_s = (1.0 - x_large) * rho_s
    w_l = x_large * rho_l
    c.update({
        "study": "segregation",
        "r_s": r_s, "r_rho": r_rho, "x_large": x_large,
        "radius_s": round(d_s / 2, 6),
        "radius_l": round(d_s * r_s / 2, 6),
        "rho_s": rho_s, "rho_l": rho_l,
        "w1": round(w_s / (w_s + w_l), 6),
        "w2": round(w_l / (w_s + w_l), 6),
    })
    return c


def generate_design(work_dir, ad_list, gamma_list, F_list, d,
                    dev_periods, stat_periods,
                    study="energy",
                    r_s_list=None, r_rho_list=None, x_large_list=None):
    missing = [n for n, v in (("ad_list", ad_list), ("gamma_list", gamma_list),
                               ("F_list", F_list)) if not v]
    if missing:
        return {"ok": False, "error": "missing_params",
                "message": "必填参数缺失: %s（实验设计须由调用方显式给定参数水平）" % missing}
    study = (study or "energy").strip().lower()
    if study not in ("energy", "segregation"):
        return {"ok": False, "error": "bad_study",
                "message": "study 须为 energy 或 segregation"}

    if not d:
        return {"ok": False, "error": "missing_params",
                "message": "d（参考颗粒直径，米）必填"}
    d = float(d)
    if study == "segregation":
        for n, v in (("r_s_list", r_s_list), ("r_rho_list", r_rho_list),
                     ("x_large_list", x_large_list)):
            if not v:
                return {"ok": False, "error": "missing_params",
                        "message": "segregation 设计的 %s 必填" % n}
        r_s_list = [float(v) for v in r_s_list]
        r_rho_list = [float(v) for v in r_rho_list]
        x_large_list = [float(v) for v in x_large_list]

    ranges = [("ad", ad_list, 0.05, 10), ("gamma", gamma_list, 0.5, 500),
              ("F", F_list, 1, 64)]
    if study == "segregation":
        ranges += [("r_s", r_s_list, 1.0, 2.0),        # 大颗粒不得超过槽宽
                   ("r_rho", r_rho_list, 0.25, 4.0),
                   ("x_large", x_large_list, 0.05, 0.6)]
    for name, lst, lo, hi in ranges:
        for v in lst or []:
            if not (lo <= float(v) <= hi):
                return {"ok": False, "error": "param_out_of_range",
                        "message": "%s=%s 超出建议范围 [%s, %s]" % (name, v, lo, hi)}
    if dev_periods < 10 or stat_periods < 5:
        return {"ok": False, "error": "param_out_of_range",
                "message": "dev_periods>=10 且 stat_periods>=5 才能保证统计可信"}

    cases = []
    idx = 1
    if study == "segregation":
        for ad in ad_list:
            for gamma in gamma_list:
                for F in F_list:
                    for r_s in r_s_list:
                        for r_rho in r_rho_list:
                            for x_large in x_large_list:
                                cases.append(_case_binary(
                                    idx, float(ad), float(gamma), float(F), d,
                                    float(r_s), float(r_rho), float(x_large),
                                    dev_periods, stat_periods))
                                idx += 1
    else:
        for ad in ad_list:
            for gamma in gamma_list:
                for F in F_list:
                    cases.append(_case(idx, float(ad), float(gamma), float(F), d,
                                       dev_periods, stat_periods))
                    idx += 1

    design_id = "design_%s_%03d" % (time.strftime("%Y%m%d_%H%M%S"), int(time.time()*1000)%1000)
    design_dir = os.path.join(work_dir, "designs")
    os.makedirs(design_dir, exist_ok=True)
    payload = {"design_id": design_id, "d": d, "n_cases": len(cases),
               "study": study,
               "dev_periods": dev_periods, "stat_periods": stat_periods,
               "cases": cases, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(design_dir, design_id + ".json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    extra = ("；二元参数 r_s×r_rho×x_large 全排列" if study == "segregation" else "")
    return {"ok": True, "design_id": design_id, "n_cases": len(cases),
            "study": study, "d": d,
            "preview": cases[:5],
            "hint": ("segregation 设计的 ad/F 以小颗粒直径 d_s 计；"
                     "全排列 = ad×gamma×F(×r_s×r_rho×x_large)%s" % extra),
            "est_seconds_per_case": "分钟级（视工况周期与颗粒数）"}
