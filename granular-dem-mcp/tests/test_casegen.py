# -*- coding: utf-8 -*-
"""casegen v1 单元测试：渲染器校验规则 + 作业管理器（不跑 LIGGGHTS）。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.casegen.renderer import render_case, validate  # noqa: E402

BASE_CASE = {
    "meta": {"name": "t", "purpose": "unit"},
    "domain": {
        "box": [0, 0.1, 0, 0.15, 0, 0.0045],
        "walls": [
            {"kind": "zplane", "pos": 0.0},
            {"kind": "xplane", "pos": 0.0},
            {"kind": "xplane", "pos": 0.1},
            {"kind": "yplane", "pos": 0.0},
        ],
        "margin": 0.005,
    },
    "particles": {
        "types": [{"id": 1, "diameter": 0.002, "density": 2500}],
        "insertion": {
            "method": "pack",
            "region": [0.002, 0.098, 0.005, 0.12, 0.0002, 0.0043],
            "composition": [{"type": 1, "count": 100}],
            "seed": 32452843,
        },
    },
    "material": {"youngs_modulus": 5.0e7, "poissons_ratio": 0.45,
                 "restitution": 0.95, "friction": 0.3, "rolling_friction": 0.03},
    "fields": {"gravity": [0, -9.81, 0], "boundary": "fff"},
    "excitation": None,
    "numerics": {"timestep": 1.0e-5, "neighbor_skin": 0.002},
    "phases": [{"name": "settle", "steps": 10000, "thermo_every": 1000}],
    "output": {"dump_columns": ["id", "type", "x", "y", "z", "vx", "vy", "vz", "radius"],
               "dump_every": 1000, "final_state": True},
}


def _clone():
    return json.loads(json.dumps(BASE_CASE))


def test_valid_case_renders():
    script, errs = render_case(BASE_CASE)
    assert not errs, errs
    assert "communicate     single vel yes" in script
    assert "primitive type 1 zplane 0.0" in script
    assert "insert/pack" in script and "mass_in_region" in script
    assert "rolling_friction epsd2" in script
    assert "write_restart" in script


def test_youngs_modulus_floor():
    c = _clone()
    c["material"]["youngs_modulus"] = 1e6
    errs = validate(c)
    assert any("杨氏模量" in e.message for e in errs)


def test_rayleigh_timestep():
    c = _clone()
    c["numerics"]["timestep"] = 5e-5  # 远超上限
    errs = validate(c)
    assert any("Rayleigh" in e.message for e in errs)


def test_insertion_region_too_thin():
    c = _clone()
    c["particles"]["types"][0]["diameter"] = 0.006
    c["particles"]["insertion"]["region"] = [0.01, 0.09, 0.01, 0.11, 0.001, 0.0035]
    errs = validate(c)
    assert any("2×最大粒径" in e.message for e in errs)


def test_packing_fraction():
    c = _clone()
    c["particles"]["insertion"]["composition"] = [{"type": 1, "count": 4500}]
    c["particles"]["insertion"]["region"] = [0.02, 0.06, 0.005, 0.04, 0.0002, 0.0043]
    errs = validate(c)
    assert any("堆积分数" in e.message or "上限 5000" in e.message for e in errs)


def test_particle_cap():
    c = _clone()
    c["particles"]["insertion"]["composition"] = [{"type": 1, "count": 6000}]
    errs = validate(c)
    assert any("5000" in e.message for e in errs)


def test_boundary_wall_type():
    c = _clone()
    c["domain"]["walls"].append({"kind": "sphere", "pos": 0})
    errs = validate(c)
    assert any("未知壁面" in e.message for e in errs)


def test_excitation_wiggle_render():
    c = _clone()
    c["excitation"] = {"type": "vibrate_domain",
                       "amplitude": [0, 0.002, 0], "frequency_hz": 10.0}
    script, errs = render_case(c)
    assert not errs, errs
    assert "fix  vib all move wiggle 0 0.002 0 0.1" in script


def test_excitation_shear_render():
    c = _clone()
    c["excitation"] = {"type": "wall_shear", "shear_wall_index": 0,
                       "shear_dim": "x", "shear_velocity": 0.2}
    script, errs = render_case(c)
    assert not errs, errs
    assert "shear x 0.2" in script


def test_two_type_material_matrix():
    c = _clone()
    c["domain"]["box"][5] = 0.011
    c["particles"]["types"].append({"id": 2, "diameter": 0.004, "density": 2500})
    c["particles"]["insertion"]["composition"] = [
        {"type": 1, "count": 80}, {"type": 2, "count": 20}]
    c["particles"]["insertion"]["region"] = [0.002, 0.098, 0.005, 0.14, 0.0002, 0.0093]
    script, errs = render_case(c)
    assert not errs, errs
    # peratomtypepair 2 → 4 个系数值
    assert "peratomtypepair 2 " in script
    # 两个 particletemplate
    assert script.count("particletemplate/sphere") == 2
    # 分布两个模板
    assert "2 pt1 " in script and " pt2 " in script


def test_job_manager_build(tmp_path):
    from core.casegen.jobs import CaseJobManager
    jm = CaseJobManager(str(tmp_path), "liggghts")
    r = jm.build(BASE_CASE)
    assert r["ok"], r
    assert (tmp_path / "casegen_jobs" / r["job_id"] / "case.in").exists()
    assert (tmp_path / "casegen_jobs" / r["job_id"] / "case.json").exists()
    # 状态
    st = jm.status(r["job_id"])
    assert st["status"] == "built"
    # 未知 job
    assert not jm.status("cg_nonexistent").get("ok", True)


def test_job_manager_rejects_over_limit(tmp_path):
    from core.casegen.jobs import CaseJobManager
    jm = CaseJobManager(str(tmp_path), "liggghts")
    c = _clone()
    c["particles"]["insertion"]["composition"] = [{"type": 1, "count": 9999}]
    r = jm.build(c)
    assert not r["ok"]


# ---------------- v2：mesh 壁 ----------------

def test_mesh_wall_render(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = _clone()
    # z-up 语义（v2.1）：box_tray 开口朝 +z，lx 沿 x、lz 沿 y、h 沿 z
    c["domain"]["walls"] = [{"kind": "mesh", "stl": {
        "gen": "box_tray", "params": {"lx": 0.1, "lz": 0.0045, "h": 0.15}}}]
    c["domain"]["box"] = [-0.06, 0.06, -0.005, 0.005, -0.01, 0.16]
    c["particles"]["insertion"]["region"] = [-0.04, 0.04, -0.002, 0.002, 0.005, 0.1]
    script, errs = render_case(c)
    assert not errs, errs
    assert "mesh/surface file mesh_0.stl" in script
    assert "wall/gran model hertz tangential history mesh n_meshes 1 meshes mesh_0" in script
    import os
    assert os.path.exists("mesh_0.stl")


def test_mesh_motion_wiggle_render(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = _clone()
    c["domain"]["walls"] = [{"kind": "mesh", "stl": {
        "gen": "box_tray", "params": {"lx": 0.1, "lz": 0.0045, "h": 0.15}}}]
    c["domain"]["box"] = [-0.06, 0.06, -0.005, 0.005, -0.01, 0.16]
    c["particles"]["insertion"]["region"] = [-0.04, 0.04, -0.002, 0.002, 0.005, 0.1]
    c["excitation"] = {"type": "mesh_motion", "mesh_wall_index": 0,
                       "motion": "wiggle", "amplitude": [0, 0, 0.001],
                       "frequency_hz": 10.0}
    script, errs = render_case(c)
    assert not errs, errs
    assert "move/mesh mesh mesh_0 wiggle amplitude 0 0 0.001 period 0.1" in script


def test_mesh_bbox_out_of_box_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = _clone()
    c["domain"]["walls"] = [{"kind": "mesh", "stl": {
        "gen": "box_tray", "params": {"lx": 0.1, "lz": 0.0045, "h": 0.15},
        "offset": [-0.05, 0, 0.05]}}]
    c["domain"]["box"] = [-0.06, 0.06, -0.005, 0.005, -0.01, 0.16]
    c["particles"]["insertion"]["region"] = [-0.04, 0.04, -0.002, 0.002, 0.005, 0.1]
    errs = validate(c)
    assert any("越界" in e.message for e in errs)


def test_mesh_motion_needs_mesh_wall():
    c = _clone()
    c["excitation"] = {"type": "mesh_motion", "mesh_wall_index": 0,
                       "motion": "wiggle", "amplitude": [0, 0.001, 0],
                       "frequency_hz": 10.0}
    errs = validate(c)
    assert any("mesh 壁" in e.message for e in errs)
