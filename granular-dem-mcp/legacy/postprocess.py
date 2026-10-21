# -*- coding: utf-8 -*-
"""后处理：解析 LIGGGHTS 文本 dump，计算无量纲响应量。

  E_k*  = <0.5*|U|^2> / (g*d)          体系平均动能（式2.16无量纲化，质量约掉）
  T_y*  = T_y_bar / (d*g)               纵向平均温度（式2.20-2.22，2d 网格胞元）
  h_m*  = h_m_bar / d                   质心高度（式2.23-2.24，2d 网格胞元）

dump 列：id type x y z vx vy vz omegax omegay omegaz radius
输出 responses.csv（物理空间原始值，DDDA-MCP run_full_pipeline(space="physical") 可直传）：
  case_id, ad, gamma, F, n_particles, e_k_star, t_y_star, h_m_star
"""
import csv
import json
import os
import time

import numpy as np

G = 9.81
GRID_CELL = 2.0  # 网格尺寸 = 2d
SLOT_WIDTH = 0.0045  # 准二维槽宽（胞元体积用）


def _case_dir_of(jobs, job_id, case_id):
    try:
        job = jobs._read_job(job_id)
    except FileNotFoundError:
        return None, None, "找不到作业 %s" % job_id
    case = next((c for c in job["cases"] if int(c["case_id"]) == int(case_id)), None)
    if case is None:
        return None, None, "作业 %s 无工况 %s" % (job_id, case_id)
    cdir = os.path.join(jobs._job_dir(job_id), "case_%s" % case_id)
    if not os.path.isdir(cdir):
        return case, None, "工况目录缺失（可能未运行或已清理）"
    return case, cdir, None


def _dump_files(cdir):
    return sorted((f for f in os.listdir(cdir)
                   if f.startswith("dmp_") and f.endswith(".txt")),
                  key=lambda f: int(f[4:-4]) if f[4:-4].isdigit() else 0)


def get_snapshot(jobs, job_id, case_id, frame=-1):
    """提取单帧颗粒快照（供渲染与视觉分析）：位置/速度/类型/半径。"""
    case, cdir, err = _case_dir_of(jobs, job_id, case_id)
    if err:
        return {"ok": False, "error": "snapshot_failed", "message": err}
    files = _dump_files(cdir)
    if not files:
        return {"ok": False, "error": "no_dumps", "message": "该工况无 dump 文件"}
    idx = int(frame)
    fn = files[idx]
    fr = _parse_dump(os.path.join(cdir, fn))
    if fr is None:
        return {"ok": False, "error": "parse_failed", "message": "dump 解析失败"}
    col = {c: k for k, c in enumerate(fr["cols"])}
    atoms = fr["atoms"]
    order = np.argsort(atoms[:, col["id"]])
    atoms = atoms[order]
    r5 = lambda a: np.round(a, 5).tolist()
    return {
        "ok": True, "job_id": job_id, "case_id": int(case_id),
        "frame_index": idx if idx >= 0 else len(files) + idx,
        "n_frames": len(files),
        "step": int(fn[4:-4]) if fn[4:-4].isdigit() else 0,
        "n_particles": len(atoms),
        "x": r5(atoms[:, col["x"]]), "y": r5(atoms[:, col["y"]]),
        "vx": r5(atoms[:, col["vx"]]), "vy": r5(atoms[:, col["vy"]]),
        "speed": r5(np.sqrt(atoms[:, col["vx"]] ** 2 + atoms[:, col["vy"]] ** 2 + atoms[:, col["vz"]] ** 2)),
        "type": atoms[:, col["type"]].astype(int).tolist(),
        "radius": r5(atoms[:, col["radius"]]),
        "params": {k: case.get(k) for k in
                   ("ad", "gamma", "F", "d", "study", "r_s", "r_rho", "x_large")},
        "note": "颗粒数据已整备（x/y/vx/vy/speed/type/radius 平行数组），供渲染与视觉分析",
    }


def get_profiles(jobs, job_id, case_id, max_frames=300):
    """时间平均剖面与场（机制观测）：

    - large_fraction(y)：大颗粒体积浓度沿高度剖面（二元体系；一元体系返回 None）
    - vx/vy 场：(x,y) 网格上的时间平均速度（对流环诊断）
    - solid_fraction 场：局部体积分数（气隙/流化诊断）
    - time_series：每帧物种高度差（收敛性检查）
    """
    case, cdir, err = _case_dir_of(jobs, job_id, case_id)
    if err:
        return {"ok": False, "error": "profile_failed", "message": err}
    files = _dump_files(cdir)
    if not files:
        return {"ok": False, "error": "no_dumps", "message": "该工况无 dump 文件"}
    if len(files) > max_frames:
        step = max(1, len(files) // max_frames)
        files = files[::step]

    d = case["d"]
    bin_x = bin_y = d
    binary = case.get("study") == "segregation"
    x_lo = x_hi = y_lo = y_hi = None
    yb_small = yb_large = None
    cell_cnt = cell_vx = cell_vy = cell_vol = None
    xs_ys = {"steps": [], "dh": [] if binary else None, "mean_y": [] if not binary else None}

    for fn in files:
        fr = _parse_dump(os.path.join(cdir, fn))
        if fr is None:
            continue
        atoms = fr["atoms"]
        col = {c: k for k, c in enumerate(fr["cols"])}
        x, y = atoms[:, col["x"]], atoms[:, col["y"]]
        vx, vy = atoms[:, col["vx"]], atoms[:, col["vy"]]
        tp = atoms[:, col["type"]].astype(int)
        rad = atoms[:, col["radius"]]
        if x_lo is None:
            x_lo, x_hi = float(x.min()), float(x.max())
            y_lo, y_hi = float(y.min()), float(y.max())
        else:
            x_lo = min(x_lo, float(x.min())); x_hi = max(x_hi, float(x.max()))
            y_lo = min(y_lo, float(y.min())); y_hi = max(y_hi, float(y.max()))
        xs_ys["steps"].append(int(fn[4:-4]) if fn[4:-4].isdigit() else 0)
        if binary:
            nl = int((tp == 2).sum())
            if nl > 0 and nl < len(tp):
                xs_ys["dh"].append(round(float(y[tp == 2].mean() - y[tp != 2].mean()), 6))
            else:
                xs_ys["dh"].append(None)
        else:
            xs_ys["mean_y"].append(round(float(y.mean()), 6))

        if yb_small is None:
            nx = max(2, int(np.ceil((x_hi - x_lo) / bin_x)))
            ny = max(2, int(np.ceil((y_hi - y_lo) / bin_y)))
            yb_small = np.zeros(ny); yb_large = np.zeros(ny)
            cell_cnt = np.zeros((nx, ny)); cell_vx = np.zeros((nx, ny))
            cell_vy = np.zeros((nx, ny)); cell_vol = np.zeros((nx, ny))
        ix = np.clip(((x - x_lo) / bin_x).astype(int), 0, cell_cnt.shape[0] - 1)
        iy = np.clip(((y - y_lo) / bin_y).astype(int), 0, cell_cnt.shape[1] - 1)
        for a, b, t_, v_x, v_y, r_ in zip(ix, iy, tp, vx, vy, rad):
            a, b = int(a), int(b)
            cell_cnt[a, b] += 1
            cell_vx[a, b] += v_x
            cell_vy[a, b] += v_y
            if binary and t_ == 2:
                yb_large[b] += 1
            else:
                yb_small[b] += 1

        # 体积分数：颗粒体积按其包围盒与各胞元的重叠比例展布（守恒且不溢出胞元）
        xg = x_lo + np.arange(cell_cnt.shape[0] + 1) * bin_x
        yg = y_lo + np.arange(cell_cnt.shape[1] + 1) * bin_y
        for x_, y_, r_ in zip(x, y, rad):
            vp = 4.0 / 3.0 * np.pi * r_ ** 3
            i0 = max(0, int(np.floor((x_ - r_ - x_lo) / bin_x)))
            i1 = min(cell_cnt.shape[0] - 1, int(np.floor((x_ + r_ - x_lo) / bin_x)))
            j0 = max(0, int(np.floor((y_ - r_ - y_lo) / bin_y)))
            j1 = min(cell_cnt.shape[1] - 1, int(np.floor((y_ + r_ - y_lo) / bin_y)))
            if i1 < i0 or j1 < j0:
                continue
            ox = np.minimum(xg[i0 + 1:i1 + 2], x_ + r_) - np.maximum(xg[i0:i1 + 1], x_ - r_)
            oy = np.minimum(yg[j0 + 1:j1 + 2], y_ + r_) - np.maximum(yg[j0:j1 + 1], y_ - r_)
            span = 2 * r_
            for ii, wx_ in enumerate(np.maximum(ox, 0) / span):
                for jj, wy_ in enumerate(np.maximum(oy, 0) / span):
                    cell_vol[i0 + ii, j0 + jj] += vp * wx_ * wy_

    n_frames = len(xs_ys["steps"])
    with np.errstate(divide="ignore", invalid="ignore"):
        large_frac = np.where(yb_small + yb_large > 0, yb_large / (yb_small + yb_large), np.nan)
        vx_f = np.where(cell_cnt > 0, cell_vx / cell_cnt, np.nan)
        vy_f = np.where(cell_cnt > 0, cell_vy / cell_cnt, np.nan)
        cell_volume = bin_x * bin_y * SLOT_WIDTH
        sf_f = np.where(cell_cnt > 0, cell_vol / (cell_volume * max(n_frames, 1)), np.nan)

    def _rl(a):
        return np.round(a.astype(float), 4).tolist()

    return {
        "ok": True, "job_id": job_id, "case_id": int(case_id),
        "n_frames_used": n_frames, "bin_size_m": round(bin_x, 5),
        "x_bin_centers": _rl(np.arange(cell_cnt.shape[0]) * bin_x + bin_x / 2 + x_lo),
        "y_bin_centers": _rl(np.arange(cell_cnt.shape[1]) * bin_y + bin_y / 2 + y_lo),
        "large_fraction": _rl(large_frac) if binary else None,
        "vx_field": _rl(vx_f), "vy_field": _rl(vy_f),
        "solid_fraction_field": _rl(sf_f),
        "time_series": xs_ys,
        "params": {k: case.get(k) for k in
                   ("ad", "gamma", "F", "d", "study", "r_s", "r_rho", "x_large")},
        "note": ("fields 为 [ix][iy] 网格（行=x列）；solid_fraction 诊断气隙/流化，"
                 "vy/vx 场诊断对流环；large_fraction(y) 为偏析剖面；time_series 为逐帧演化"),
    }


def _parse_dump(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    data = {}
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i].strip()
        if ln.startswith("ITEM: NUMBER OF ATOMS"):
            data["natoms"] = int(lines[i + 1])
            i += 2
        elif ln.startswith("ITEM: BOX"):
            i += 4
        elif ln.startswith("ITEM: ATOMS"):
            cols = ln.split()[2:]
            rows = int(data.get("natoms", 0))
            arr = np.array([lines[i + 1 + k].split() for k in range(rows)], dtype=float)
            data["cols"], data["atoms"] = cols, arr
            i += 1 + rows
        else:
            i += 1
    if data.get("atoms") is None or data["atoms"].size == 0:
        return None
    return data


def analyze_case(case_dir, d, case=None):
    binary = bool(case and case.get("study") == "segregation")
    files = sorted(f for f in os.listdir(case_dir)
                   if f.startswith("dmp_") and f.endswith(".txt"))
    if not files:
        return None

    cell = GRID_CELL * d
    grid = {}                 # (ix,iy) -> [count, sum_uy, sum_uy2]
    ek_sum, ep_sum, ek_frames = 0.0, 0.0, 0
    seg_dh_sum, mix_sum, bin_frames = 0.0, 0.0, 0
    n_small = n_large = 0
    n_last = 0
    x_lo = x_hi = y_lo = y_hi = None

    for fn in files:
        fr = _parse_dump(os.path.join(case_dir, fn))
        if fr is None:
            continue
        atoms = fr["atoms"]
        col = {c: k for k, c in enumerate(fr["cols"])}
        x, y = atoms[:, col["x"]], atoms[:, col["y"]]
        vx, vy, vz = atoms[:, col["vx"]], atoms[:, col["vy"]], atoms[:, col["vz"]]
        n_last = len(atoms)

        ek_sum += float(np.mean(0.5 * (vx ** 2 + vy ** 2 + vz ** 2)))
        ep_sum += float(np.mean(y - float(y.min())))
        ek_frames += 1

        if x_lo is None:
            x_lo, x_hi = float(x.min()), float(x.max())
            y_lo, y_hi = float(y.min()), float(y.max())
        else:
            x_lo = min(x_lo, float(x.min())); x_hi = max(x_hi, float(x.max()))
            y_lo = min(y_lo, float(y.min())); y_hi = max(y_hi, float(y.max()))

        if binary:
            tp = atoms[:, col["type"]].astype(int)
            is_large = tp == 2
            n_small, n_large = int((~is_large).sum()), int(is_large.sum())
            if n_large > 0 and n_small > 0:
                # 物种高度差（正=大/重颗粒在上），以小颗粒直径无量纲化
                seg_dh_sum += float(y[is_large].mean() - y[~is_large].mean()) / d
                # Lacey 混合指数：胞元大颗粒浓度方差 vs 完全偏析/完全随机基准
                p = n_large / (n_small + n_large)
                ix = np.floor((x - x_lo) / cell).astype(int)
                iy = np.floor((y - y_lo) / cell).astype(int)
                key = ix.astype(np.int64) * 100000 + iy
                uk = np.unique(key)
                n_i = np.array([ (key == k).sum() for k in uk ], dtype=float)
                c_i = np.array([ is_large[key == k].mean() for k in uk ])
                s2 = float((n_i * (c_i - p) ** 2).sum() / n_i.sum())
                s0 = p * (1 - p)
                sr = s0 / max(n_i.mean(), 1.0)
                if s0 > 1e-12:
                    mix = (s0 - s2) / max(s0 - sr, 1e-12)
                    mix_sum += float(np.clip(mix, 0.0, 1.0))
                bin_frames += 1

        ix = np.floor((x - x_lo) / cell).astype(int)
        iy = np.floor((y - y_lo) / cell).astype(int)
        for a, b, uy in zip(ix, iy, vy):
            key = (int(a), int(b))
            st = grid.get(key)
            if st is None:
                st = grid[key] = [0, 0.0, 0.0]
            st[0] += 1
            st[1] += float(uy)
            st[2] += float(uy * uy)

    n_tot = sum(st[0] for st in grid.values())
    if n_tot == 0:
        return None
    ty_w, hm_w = 0.0, 0.0
    for (a, b), (cnt, suy, suy2) in grid.items():
        ty_i = suy2 / cnt - (suy / cnt) ** 2     # 胞元纵向温度（长时间平均）
        h_i = y_lo + (b + 0.5) * cell            # 胞元中心高度
        ty_w += ty_i * cnt
        hm_w += h_i * cnt

    out = {
        "n_particles": int(n_last),
        "n_frames": ek_frames,
        "e_k_star": round(ek_sum / max(ek_frames, 1) / (G * d), 6),
        "t_y_star": round(ty_w / n_tot / (d * G), 6),
        "h_m_star": round(hm_w / n_tot / d, 6),
        "ep_star": round(ep_sum / max(ek_frames, 1) / d, 6),
    }
    if binary:
        out.update({
            "n_small": n_small, "n_large": n_large,
            "seg_dh_star": round(seg_dh_sum / max(bin_frames, 1), 6),
            "mix_index": round(mix_sum / max(bin_frames, 1), 6),
        })
    return out


def postprocess_job(jobs, job_id, work_dir, host_work_dir, rows_limit=5):
    try:
        job = jobs._read_job(job_id)
    except FileNotFoundError:
        return {"ok": False, "error": "job_not_found", "message": "找不到作业 %s" % job_id}
    if job["status"] == "running":
        return {"ok": False, "error": "job_running",
                "message": "作业仍在运行（%s/%s），完成后再后处理" % (job["done"], job["total"])}

    results = []
    for case in job["cases"]:
        cid = case["case_id"]
        cdir = os.path.join(jobs._job_dir(job_id), "case_%s" % cid)
        r = analyze_case(cdir, case["d"], case) if os.path.isdir(cdir) else None
        row = {"case_id": cid, "ad": case["ad"], "gamma": case["gamma"], "F": case["F"]}
        if case.get("study") == "segregation":
            row.update({"r_s": case["r_s"], "r_rho": case["r_rho"],
                        "x_large": case["x_large"]})
        if r:
            row.update({k: r[k] for k in ("n_particles", "e_k_star", "t_y_star", "h_m_star", "ep_star")})
            if case.get("study") == "segregation":
                row.update({k: r[k] for k in ("seg_dh_star", "mix_index")})
        else:
            row.update({"n_particles": "", "e_k_star": "", "t_y_star": "", "h_m_star": ""})
        results.append(row)

    out_dir = os.path.join(work_dir, "results", job_id)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "responses.csv")
    keys = ["case_id", "ad", "gamma", "F",
            "r_s", "r_rho", "x_large",
            "n_particles", "e_k_star", "t_y_star", "h_m_star", "ep_star",
            "seg_dh_star", "mix_index"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for row in results:
            w.writerow([row.get(k, "") for k in keys])

    n_ok = sum(1 for r in results if r["e_k_star"] != "")
    return {
        "ok": True, "job_id": job_id,
        "n_cases": len(results), "n_valid": n_ok,
        "responses_file": jobs.host_path(csv_path),
        "summary": results[:max(1, min(int(rows_limit or 5), 500))],
        "hint": ("responses.csv 为物理空间原始值；将其路径传给 DDDA-MCP 的 "
                 "run_full_pipeline（input_file=<该路径>, space=\"physical\", "
                 "x_cols=[ad,gamma,F], y_col=e_k_star）"),
    }


def merge_responses(work_dir, host_work_dir, job_ids, rows_limit=5):
    """合并多个作业的 responses.csv 为一个文件（自适应多轮采样用）。

    去重规则：同一参数组合（energy: ad,gamma,F；segregation: 加 r_s,r_rho,x_large）
    只保留一次（后出现的覆盖，即最新一轮的同参数结果优先）。
    """
    import csv as _csv

    def _key(row):
        base = (row["ad"], row["gamma"], row["F"])
        if row.get("r_s"):
            base += (row["r_s"], row["r_rho"], row["x_large"])
        return base

    rows = {}
    order = []
    n_dedup = 0
    missing = []
    for jid in job_ids:
        path = os.path.join(work_dir, "results", jid, "responses.csv")
        if not os.path.exists(path):
            missing.append(jid)
            continue
        with open(path, encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                key = _key(row)
                if key in rows:
                    n_dedup += 1
                else:
                    order.append(key)
                rows[key] = row  # 覆盖=最新优先
    if not rows:
        return {"ok": False, "error": "no_data",
                "message": "未找到任何可合并数据（作业: %s）" % job_ids}

    # 短目录名（多个 job_id 直接拼接会超过文件系统 255 字节上限），
    # 完整来源清单写入 manifest.json
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(work_dir, "results", "merged_%s" % stamp)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"source_jobs": [j for j in job_ids if j not in missing],
                   "missing_jobs": missing, "n_dedup": n_dedup}, f, indent=1)
    csv_path = os.path.join(out_dir, "responses.csv")
    keys = ["case_id", "ad", "gamma", "F", "r_s", "r_rho", "x_large",
            "n_particles", "e_k_star", "t_y_star", "h_m_star", "ep_star",
            "seg_dh_star", "mix_index"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(keys)
        for i, key in enumerate(order, 1):
            r = dict(rows[key])
            r["case_id"] = i
            w.writerow([r.get(k, "") for k in keys])

    n_valid = sum(1 for k in order if rows[k].get("e_k_star") not in ("", None))
    host = os.path.join(host_work_dir, "results",
                        "merged_%s" % stamp, "responses.csv")
    cap = max(1, min(int(rows_limit or 5), 500))
    rows_out = []
    for i, key in enumerate(order[:cap], 1):
        r = dict(rows[key])
        r["case_id"] = i
        rows_out.append(r)
    return {"ok": True, "merged_file": host, "n_total": len(order),
            "rows": rows_out,
            "n_valid": n_valid, "n_dedup": n_dedup,
            "jobs": [j for j in job_ids if j not in missing],
            "missing_jobs": missing,
            "hint": "合并后的文件传给 DDDA run_full_pipeline 的 input_file；来源作业清单见 merged 目录下 manifest.json"}
