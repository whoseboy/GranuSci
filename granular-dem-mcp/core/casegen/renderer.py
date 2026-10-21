# -*- coding: utf-8 -*-
"""casegen v1：结构化 case 描述 → LIGGGHTS-PUBLIC 3.8 脚本渲染器。

架构依据：C1 可行性判决（论文ICML2027/C1判决_*.md）——LLM 直写脚本
0/10（LAMMPS 语法污染），带文档人工剥错 7 层可通；故语法层全部封死在
本渲染器（唯一语法出口），LLM 只产 SCHEMA.md 定义的 JSON。

安全与护栏：渲染前 validate() 强制全部物理校验（C1 七层错误+已知
物理下限）；v1 硬顶粒子 5000/步数 2e6。
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

# LIGGGHTS random.cpp 的合法种子表（素数>10000，报错提示实测提取；
# insert/pack、particletemplate、particledistribution 共用此约束）
VALID_SEEDS = [15485863, 15485867, 32452843, 32452867, 49979687,
               49979693, 67867967, 67867979, 86028121, 86028157]


def _pick_seed(user_seed: int | None, slot: int) -> int:
    """把任意用户种子映射到合法素数（slot 区分同 case 内不同 fix 的种子）。"""
    if user_seed in VALID_SEEDS:
        base = VALID_SEEDS.index(user_seed)
    else:
        base = (user_seed or 0) % len(VALID_SEEDS)
    return VALID_SEEDS[(base + slot) % len(VALID_SEEDS)]


# ---------------------------------------------------------------- 校验


@dataclass
class ValidationError:
    field: str
    message: str


RAYLEIGH_COEFF = 0.176  # sqrt 下系数（G=ρ/... 保守常数的工程近似，见校验消息）


def validate(case: dict) -> list[ValidationError]:
    errs: list[ValidationError] = []

    def err(field, msg):
        errs.append(ValidationError(field, msg))

    # 必填段
    for sec in ("meta", "domain", "particles", "material", "fields",
                "numerics", "phases", "output"):
        if sec not in case:
            err(sec, f"缺少必填段 {sec}")
    if errs:
        return errs

    d = case["domain"]
    f = case["fields"]
    box = d.get("box")
    if (not isinstance(box, list) or len(box) != 6
            or box[0] >= box[1] or box[2] >= box[3] or box[4] >= box[5]):
        err("domain.box", "box 须为 [xmin,xmax,ymin,ymax,zmin,zmax] 且各向 min<max")
    for i, w in enumerate(d.get("walls", [])):
        k = w.get("kind")
        if k not in ("xplane", "yplane", "zplane", "xcylinder", "ycylinder",
                     "zcylinder", "mesh"):
            err(f"domain.walls[{i}]", f"未知壁面 kind: {k}")
        elif k.endswith("plane") and "pos" not in w:
            err(f"domain.walls[{i}]", "plane 壁缺 pos")
        elif k.endswith("cylinder") and not {"radius", "c1", "c2"} <= set(w):
            err(f"domain.walls[{i}]", "cylinder 壁缺 radius/c1/c2")
        elif k == "mesh":
            stl = w.get("stl") or {}
            if not (stl.get("gen") or stl.get("file")):
                err(f"domain.walls[{i}]", "mesh 壁的 stl 缺 gen 或 file")
            elif stl.get("gen") and isinstance(box, list):
                # mesh bbox+offset 须在 simbox 内（v2 服务器实测教训：
                # offset 推出域 → 面元被切 → ERROR mesh elements lost）
                from .stl_gen import GENERATORS
                if stl["gen"] in GENERATORS:
                    import inspect as _insp
                    sig = _insp.signature(GENERATORS[stl["gen"]])
                    params = {kk: vv.default for kk, vv in sig.parameters.items()
                              if vv.default is not _insp.Parameter.empty}
                    params.update(stl.get("params", {}))
                    # 粗略 bbox：用生成器内部点采样太重——直接生成并解析
                    import tempfile as _tf, os as _os
                    with _tf.TemporaryDirectory() as _td:
                        from .stl_gen import generate as _gen
                        g = _gen(stl["gen"], stl.get("params", {}),
                                 _os.path.join(_td, "m.stl"))
                        if g.get("ok"):
                            off = stl.get("offset", [0, 0, 0])
                            bb = g["bbox"]
                            lo = [bb[a][0] + off[a] for a in range(3)]
                            hi = [bb[a][1] + off[a] for a in range(3)]
                            boxlo, boxhi = box[0::2], box[1::2]
                            viol = [f"{chr(120+ax)}: STL+offset [{lo[ax]:.3f},{hi[ax]:.3f}] "
                                              f"vs simbox [{boxlo[ax]:.3f},{boxhi[ax]:.3f}]"
                                             for ax in range(3)
                                             if lo[ax] < boxlo[ax] - 1e-9 or hi[ax] > boxhi[ax] + 1e-9]
                            if viol:
                                err(f"domain.walls[{i}]",
                                    "mesh 壁越界（一次报全部轴向）：" + "; ".join(viol)
                                    + f"；STL bbox+offset 全貌 = x[{lo[0]:.3f},{hi[0]:.3f}] "
                                      f"y[{lo[1]:.3f},{hi[1]:.3f}] z[{lo[2]:.3f},{hi[2]:.3f}]"
                                      "——扩 domain.box 或调 offset（注意 gen 参数语义："
                                      "incline 的 length 沿 x 向、高差沿 z 向、width 沿 y 向；所有 gen 均为 z-up 约定，开口/法向朝 +z）")

    # 颗粒
    p = case["particles"]
    types = p.get("types", [])
    if not (1 <= len(types) <= 3):
        err("particles.types", "v1 支持 1-3 种颗粒")
    ids = [t.get("id") for t in types]
    if sorted(ids) != list(range(1, len(types) + 1)):
        err("particles.types", "type id 须为 1..N 连续")
    for t in types:
        if not (1e-5 <= t.get("diameter", 0) <= 0.5):
            err("particles.types", f"type {t.get('id')} diameter 超合理范围")
    ins = p.get("insertion", {})
    ireg = ins.get("region")
    if (not isinstance(ireg, list) or len(ireg) != 6):
        err("particles.insertion.region", "插入区须为 6 元箱体")
    else:
        if box:
            for lo, hi, blo, bhi in ((ireg[0], ireg[1], box[0], box[1]),
                                     (ireg[2], ireg[3], box[2], box[3]),
                                     (ireg[4], ireg[5], box[4], box[5])):
                if lo < blo - 1e-12 or hi > bhi + 1e-12:
                    err("particles.insertion.region", "插入区超出模拟域")
                    break
        # 堆积分数上限
        vol = ((ireg[1] - ireg[0]) * (ireg[3] - ireg[2]) * (ireg[5] - ireg[4]))
        need = 0.0
        comp = ins.get("composition", [])
        byid = {t["id"]: t for t in types}
        n_total = 0
        for c in comp:
            t = byid.get(c.get("type"))
            if not t:
                err("particles.insertion.composition", f"未知 type {c.get('type')}")
                continue
            n = c.get("count", 0)
            n_total += n
            need += n * math.pi / 6.0 * t["diameter"] ** 3
        if vol > 0 and need / vol > 0.55:
            err("particles.insertion", "插入区过小（堆积分数上限 0.55）")
        # 插入区每维 ≥ 最大粒径×2（all_in yes 的物理下限，C1 二测 T3/T4 实测）
        d_max = max((t["diameter"] for t in types), default=0)
        if d_max > 0:
            dims = [(ireg[1] - ireg[0]), (ireg[3] - ireg[2]), (ireg[5] - ireg[4])]
            for ax, w in zip("xyz", dims):
                if w < 2 * d_max:
                    err("particles.insertion.region",
                        f"插入区 {ax} 向厚度 {w*1000:.1f}mm < 2×最大粒径 {2*d_max*1000:.1f}mm"
                        f"（all_in yes 要求区内容纳整球）")
        if n_total > 5000:
            err("particles.insertion", f"粒子总数 {n_total} 超 v1 上限 5000")

    # 材料（C1 第 7 层）
    m = case["material"]
    if m.get("youngs_modulus", 0) < 5e6:
        err("material.youngs_modulus", "SI 单位下杨氏模量须 ≥5e6（LIGGGHTS 硬检查）")

    # 数值：Rayleigh 时间步上限
    n = case["numerics"]
    dt = n.get("timestep", 0)
    if types:
        d_min = min(t["diameter"] for t in types)
        rho = min(t.get("density", 2500) for t in types)
        E = m.get("youngs_modulus", 5e7)
        # Rayleigh: T_R = pi*R*sqrt(rho/G) ；G≈E/(2(1+nu))，保守取 0.2 T_R
        G = E / (2 * (1 + m.get("poissons_ratio", 0.45)))
        t_r = math.pi * (d_min / 2) * math.sqrt(rho / max(G, 1e6))
        # 0.3*T_R：与 legacy 战役实测一致（dt=1e-5 = 0.26 T_R 稳定运行 62 万步）
        if dt > 0.3 * t_r:
            err("numerics.timestep",
                f"时间步 {dt:.1e} 超 Rayleigh 上限（≤{0.3*t_r:.1e}）")

    # 步数
    total_steps = sum(ph.get("steps", 0) for ph in case["phases"])
    if total_steps > 2_000_000:
        err("phases", f"总步数 {total_steps} 超 v1 上限 2e6")

    # 物理护栏：重力指向的方向必须有底壁（否则颗粒坠出域被静默删除——
    # e2e 验证实测的失败模式）
    g = f.get("gravity") or [0, 0, 0]
    g_ax = max(range(3), key=lambda i: abs(g[i]))
    g_neg = g[g_ax] < 0  # 重力向负 → 颗粒落向 box 低端 → 底壁在低位
    kinds = {"x": "xplane", "y": "yplane", "z": "zplane"}
    ax_name = "xyz"[g_ax]
    has_floor = False
    box = d.get("box") or [0] * 6
    for w in d.get("walls", []):
        if w.get("kind") == kinds[ax_name]:
            pos = w.get("pos", 0)
            near_low = abs(pos - box[2 * g_ax]) < abs(box[2 * g_ax + 1] - box[2 * g_ax]) / 2
            if near_low == g_neg:  # 低位壁当且仅当重力向负
                has_floor = True
    has_mesh_wall = any(w.get("kind") == "mesh" for w in d.get("walls", []))
    if abs(g[g_ax]) > 1.0 and not has_floor and not has_mesh_wall:
        err("domain.walls",
            f"重力沿 -{ax_name} 但 {ax_name} 向无底壁——颗粒将坠出模拟域被静默删除；"
            f"请添加 {{'kind':'{kinds[ax_name]}','pos':{box[2*g_ax]}}} 或调整重力方向")

    # 激励
    e = case.get("excitation")
    if e:
        if e.get("type") == "vibrate_domain":
            if not e.get("frequency_hz", 0) > 0:
                err("excitation.frequency_hz", "振动频率须 >0")
            amp = e.get("amplitude", [0, 0, 0])
            if not any(abs(a) > 0 for a in amp):
                err("excitation.amplitude", "振动幅值全零")
        elif e.get("type") == "wall_shear":
            if e.get("shear_wall_index", -1) >= len(d.get("walls", [])):
                err("excitation.shear_wall_index", "剪切壁下标越界")
            if e.get("shear_dim") not in ("x", "y", "z"):
                err("excitation.shear_dim", "shear_dim 须 x/y/z")
        elif e.get("type") == "mesh_motion":
            mi = e.get("mesh_wall_index", -1)
            if mi < 0 or mi >= len(d.get("walls", [])):
                err("excitation.mesh_wall_index", "mesh 壁下标越界")
            elif d["walls"][mi].get("kind") != "mesh":
                err("excitation.mesh_wall_index", "mesh_motion 须指向 mesh 壁")
            if e.get("motion") not in ("wiggle", "linear"):
                err("excitation.motion", "motion 须 wiggle|linear")
            if e.get("motion") == "wiggle" and not (e.get("frequency_hz", 0) > 0):
                err("excitation.frequency_hz", "wiggle 须 frequency_hz>0")

    return errs


# ---------------------------------------------------------------- 渲染

def _material_fixes(n_types: int, m: dict) -> list[str]:
    """C1 第 2 层的封死：property/global 四件套精确格式。"""
    n = n_types
    return [
        f"fix  mat1 all property/global youngsModulus peratomtype {' '.join(str(m['youngs_modulus']) for _ in range(n))}",
        f"fix  mat2 all property/global poissonsRatio peratomtype {' '.join(str(m['poissons_ratio']) for _ in range(n))}",
        f"fix  mat3 all property/global coefficientRestitution peratomtypepair {n} {' '.join(str(m['restitution']) for _ in range(n * n))}",
        f"fix  mat4 all property/global coefficientFriction peratomtypepair {n} {' '.join(str(m['friction']) for _ in range(n * n))}",
        f"fix  mat5 all property/global coefficientRollingFriction peratomtypepair {n} {' '.join(str(m.get('rolling_friction', 0.0)) for _ in range(n * n))}",
    ]


def _wall_line(w: dict, m: dict, exc: dict | None) -> str:
    """C1 第 4/5 层的封死：model 链 + primitive 必选段 + 可选 shear。"""
    def _vec(v) -> str:
        # v3hr 教训：列表直接 f-string 输出 "[0, 0, -0.095]"（带逗号），
        # LIGGGHTS fix_wall_gran 解析炸——向量展开为空格分隔
        return " ".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)
    k = w["kind"]
    if k.endswith("plane"):
        prim = f"{k} {_vec(w['pos'])}"
    else:
        prim = f"{k} {_vec(w['radius'])} {_vec(w['c1'])} {_vec(w['c2'])}"
    shear = ""
    if exc and exc.get("type") == "wall_shear" and \
            exc.get("shear_wall_index") is not None and k in (
                "xplane", "yplane", "zplane"):
        # shear 只加在被指定的壁上（由调用方传 index 判断）
        shear = ""
    return (f"fix  wall_{w.get('_id', k)} all wall/gran model hertz "
            f"tangential history primitive type 1 {prim}{shear}")


def render(case: dict, out_dir: str = ".") -> str:
    """schema → in.liggghts 全文。唯一语法出口：调用前必须 validate 通过。"""
    d = case["domain"]
    p = case["particles"]
    m = case["material"]
    f = case["fields"]
    num = case["numerics"]
    exc = case.get("excitation")
    out = case["output"]

    n_types = len(p["types"])
    b = d["box"]
    L = []
    A = L.append

    A("# 由 casegen v1 渲染（结构化描述 → 固定骨架；语法层零自由度）")
    A(f"# case: {case['meta']['name']} — {case['meta'].get('purpose','')}")
    A("")
    A("echo            both")
    A("log             log.liggghts")
    A("")
    A("atom_style      granular")
    A("atom_modify     map array")
    A(f"boundary        {f.get('boundary', 'fff').replace('', ' ').strip()}")
    A("newton          off")
    A("")
    A("communicate     single vel yes")
    A("")
    A("units           si")
    A("")
    A(f"region  simbox block {b[0]} {b[1]} {b[2]} {b[3]} {b[4]} {b[5]} units box")
    A(f"create_box      {n_types} simbox")
    A("")
    for fx in _material_fixes(n_types, m):
        A(fx)
    roll = m.get("rolling_friction", 0.0)
    rolling = " rolling_friction epsd2 torsionTorque on" if roll else ""
    A(f"pair_style      gran model hertz tangential history{rolling}")
    A("pair_coeff      * *")
    A("")
    A(f"timestep        {num['timestep']}")
    A(f"neighbor        {num.get('neighbor_skin', 0.002)} bin")
    A("neigh_modify    delay 0 every 1 check yes")
    A("")
    g = f.get("gravity", [0, 0, -9.81])
    A(f"fix  grav all gravity {math.hypot(*g) if False else abs(g[1]) if False else math.sqrt(sum(x*x for x in g))} vector {g[0]} {g[1]} {g[2]}")
    A("fix  integr all nve/sphere")
    A("")
    # 壁面（v2：原语壁 + mesh 壁——mesh 先 mesh/surface 再 wall/gran mesh）
    for i, w in enumerate(d.get("walls", [])):
        if w.get("kind") == "mesh":
            stl = w["stl"]
            if stl.get("gen"):
                from .stl_gen import generate as _stl_gen
                fname = f"mesh_{i}.stl"
                fname = os.path.join(out_dir, fname)
                gres = _stl_gen(stl["gen"], stl.get("params", {}), fname)
                if not gres.get("ok"):
                    raise ValueError(f"mesh 壁 {i} 生成失败: {gres.get('message')}")
                src = fname
            else:
                src = stl["file"]
            off = stl.get("offset", [0, 0, 0])
            rel = os.path.basename(src)          # 脚本里相对名（运行 cwd=job 目录）
            if any(abs(o) > 0 for o in off):
                A(f"fix  mesh_{i} all mesh/surface file {rel} type 1 "
                  f"move {off[0]} {off[1]} {off[2]}")
            else:
                A(f"fix  mesh_{i} all mesh/surface file {rel} type 1")
            A(f"fix  wall_{i} all wall/gran model hertz tangential history "
              f"mesh n_meshes 1 meshes mesh_{i}")
        else:
            wl = _wall_line(dict(w, _id=i), m, exc)
            if exc and exc.get("type") == "wall_shear" and                     exc.get("shear_wall_index") == i:
                wl += f" shear {exc['shear_dim']} {exc['shear_velocity']}"
            A(wl)
    A("")
    # mesh 壁运动激励（v2）：fix move/mesh wiggle|linear（legacy 战役同款机制）
    if exc and exc.get("type") == "mesh_motion":
        mi = exc["mesh_wall_index"]
        if exc.get("motion") == "wiggle":
            amp = exc["amplitude"]
            period = 1.0 / exc["frequency_hz"]
            A(f"fix  movemesh_{mi} all move/mesh mesh mesh_{mi} wiggle "
              f"amplitude {amp[0]} {amp[1]} {amp[2]} period {period}")
        else:
            v = exc["velocity"]
            A(f"fix  movemesh_{mi} all move/mesh mesh mesh_{mi} linear "
              f"{v[0]} {v[1]} {v[2]}")
        A("")
    # 激励（整域振动）——fix move wiggle 语法：Ax Ay Az period（位置参数，无关键字）
    if exc and exc.get("type") == "vibrate_domain":
        amp = exc["amplitude"]
        period = 1.0 / exc["frequency_hz"]
        A(f"fix  vib all move wiggle {amp[0]} {amp[1]} {amp[2]} {period}")
        A("")
    # 插入（pack：C1 安全语法）
    ins = p["insertion"]
    ir = ins["region"]
    A(f"region  insreg block {ir[0]} {ir[1]} {ir[2]} {ir[3]} {ir[4]} {ir[5]} units box")
    for t in p["types"]:
        A(f"fix  pt{t['id']} all particletemplate/sphere {_pick_seed(ins.get('seed'), t['id'])} "
          f"radius constant {t['diameter'] / 2.0} density constant {t.get('density', 2500)}")
    # 离散分布：v1 按体积比混合（pack 单次插入全部 type）
    fracs = []
    byid = {t["id"]: t for t in p["types"]}
    comp = ins.get("composition", [])
    vols = {c["type"]: c.get("count", 0) * math.pi / 6.0 * byid[c["type"]]["diameter"] ** 3
            for c in comp}
    vtot = sum(vols.values()) or 1.0
    for c in comp:
        fracs.append((c["type"], vols[c["type"]] / vtot))
    dis = " ".join(f"pt{t} {f_}" for t, f_ in fracs)
    A(f"fix  pdist all particledistribution/discrete {_pick_seed(ins.get('seed'), 3)} {len(fracs)} {dis}")
    # 质量目标（pack 的 mass_in_region 用总质量）
    total_mass = sum(c.get("count", 0) * math.pi / 6.0 * byid[c["type"]]["diameter"] ** 3
                     * byid[c["type"]].get("density", 2500) for c in comp)
    A(f"fix  pins all insert/pack seed {_pick_seed(ins.get('seed'), 0)} distributiontemplate pdist "
      f"maxattempt 100000 insert_every once overlapcheck yes all_in yes region insreg "
      f"mass_in_region {total_mass}")
    A("")
    # 计算/thermo（C1 第 6 层封死：只允许验证过的关键字）
    A("compute  rke all erotate/sphere")
    A("thermo_style  custom step atoms ke c_rke")
    A("thermo        10000")
    A("thermo_modify lost ignore norm no")
    A("")
    # 分段运行 + dump
    for ph in case["phases"]:
        name = ph["name"]
        cols = " ".join(out.get("dump_columns",
                                 ["id", "type", "x", "y", "z", "vx", "vy", "vz", "radius"]))
        A(f"# ---- phase: {name} ----")
        A(f"dump  dmp_{name} all custom {out['dump_every']} ./out_{name}_*.txt {cols}")
        A(f"run   {ph['steps']}")
        A(f"undump dmp_{name}")
        A("")
    if out.get("final_state"):
        A("write_restart   final_state.restart")
    A("")
    return "\n".join(L)


def render_case(case: dict, out_dir: str = ".") -> tuple[str | None, list[dict]]:
    """validate + render 入口。返回 (script_text|None, errors)。"""
    errs = validate(case)
    if errs:
        return None, [e.__dict__ for e in errs]
    return render(case, out_dir=out_dir), []
