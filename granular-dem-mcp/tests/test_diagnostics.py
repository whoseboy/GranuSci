# -*- coding: utf-8 -*-
"""诊断箱 v0 单元测试：各诊断信号的触发/通过/边界情形。"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.casegen.diagnostics import diagnose  # noqa: E402


def _mk(d, thermo, dump_rows, n_atoms):
    with open(os.path.join(d, "run.log"), "w") as f:
        f.write(thermo)
    with open(os.path.join(d, "out_s_1000.txt"), "w") as f:
        f.write(f"ITEM: TIMESTEP\n1000\nITEM: NUMBER OF ATOMS\n{n_atoms}\n"
                "ITEM: BOX BOUNDS f f f\n0 0.05\n0 0.08\n0 0.006\n"
                "ITEM: ATOMS id type x y z vx vy vz radius\n")
        for r in dump_rows:
            f.write(" ".join(str(x) for x in r) + "\n")
    return d


def test_conservation_pass_and_fail():
    d = tempfile.mkdtemp()
    rows = [(1, 1, 0.01, 0.001, 0.001, 0, 0, 0, 0.001)] * 50
    _mk(d, "0 50 1 1\n1000 50 0.001 0.0001\n", rows, 50)
    r = diagnose(d, n_expected=50)
    assert r["diagnostics"]["D1_particle_conservation"]["verdict"] == "PASS"
    r2 = diagnose(d, n_expected=100)
    assert r2["diagnostics"]["D1_particle_conservation"]["verdict"] == "FAIL"
    r3 = diagnose(d, n_expected=53)  # 丢 3 颗（5.7%）= WARN 区
    assert r3["diagnostics"]["D1_particle_conservation"]["verdict"] == "WARN"


def test_relaxation_states():
    d = tempfile.mkdtemp()
    rows = [(1, 1, 0.01, 0.001, 0.001, 0, 0, 0, 0.001)] * 10
    _mk(d, "0 10 1 1\n500 10 0.5 0.1\n1000 10 0.9 0.2\n", rows, 10)  # 上升=稳态
    r = diagnose(d)
    assert r["diagnostics"]["D2_kinetic_relaxation"]["verdict"] in ("STEADY", "RUNNING_IN")
    d2 = tempfile.mkdtemp()
    _mk(d2, "0 10 1 1\n500 10 0.001 0.0001\n1000 10 0.0000001 0.00000001\n", rows, 10)
    r2 = diagnose(d2)
    assert r2["diagnostics"]["D2_kinetic_relaxation"]["verdict"] == "SETTLED"


def test_velocity_explosion():
    d = tempfile.mkdtemp()
    # v=100 m/s 远超自由落体量级
    rows = [(1, 1, 0.01, 0.001, 0.001, 70, 70, 0, 0.001)] * 5
    _mk(d, "0 5 1 1\n1000 5 1 1\n", rows, 5)
    r = diagnose(d, n_expected=5, domain_h=0.08)
    assert r["diagnostics"]["D4_velocity_sanity"]["verdict"] == "FAIL"
    assert r["overall"] == "FAIL"


def test_overall_aggregation():
    d = tempfile.mkdtemp()
    rows = [(1, 1, 0.01, 0.001, 0.001, 0, 0, 0, 0.001)] * 10
    _mk(d, "0 10 1 1\n1000 10 0.5 0.1\n", rows, 10)  # RUNNING_IN → WARN
    r = diagnose(d, n_expected=10)
    assert r["overall"] in ("WARN", "PASS")
