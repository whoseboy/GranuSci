# -*- coding: utf-8 -*-
"""分区假设运行时（结构中介制四机制）测试——阶段二核心实验的回归防线。

覆盖：机制 a（设计门）/ b（报数+对账+提交阻断）/ c（死亡后继）/
d（DDDA 自动触发）的纯逻辑与工具接线。接线测试用 hyptest_root 隔离
假设档案（不污染 memory/research 正式档案）。
"""
import asyncio
import json
from pathlib import Path

import pytest


# ---------------- 纯逻辑：机制 a 设计门 ----------------

def _hr():
    from granusci.agent.hyp_runtime import new_hr
    return new_hr()


def test_gate_requires_reference():
    from granusci.agent.hyp_runtime import gate_design
    hr = _hr()
    v, cleaned = gate_design("dem_build_case", {"case": "{}"}, hr,
                             lambda hid: True)
    assert not v["ok"] and v["gated"] == "no_reference"
    assert "hypothesis_id" in v["message"]
    # 拦截也计数（验收统计：builds_gated）
    assert hr["enforced"]["builds_gated"] == 1


def test_gate_strips_and_passes():
    from granusci.agent.hyp_runtime import gate_design
    hr = _hr()
    v, cleaned = gate_design(
        "dem_build_case",
        {"case": "{}", "hypothesis_id": "H123", "exploratory": False},
        hr, lambda hid: hid == "H123")
    assert v["ok"] and v["hid"] == "H123"
    # 服务器 schema 不感知的两个字段必须剥离后再转发
    assert "hypothesis_id" not in cleaned and "exploratory" not in cleaned
    assert cleaned == {"case": "{}"}


def test_gate_exploratory_passes_without_hid():
    from granusci.agent.hyp_runtime import gate_design
    hr = _hr()
    v, cleaned = gate_design("dem_modify_case", {"job_id": "j1", "exploratory": True},
                             hr, lambda hid: True)
    assert v["ok"] and v["exploratory"]


def test_gate_rejects_fictional_hid():
    from granusci.agent.hyp_runtime import gate_design
    hr = _hr()
    v, _ = gate_design("dem_build_case", {"hypothesis_id": "H999"}, hr,
                       lambda hid: False)
    assert not v["ok"] and v["gated"] == "bad_hid"


def test_gate_blocks_on_unreconciled():
    """机制 b 硬约束：未对账报数阻断一切新工况（探索也不行）。"""
    from granusci.agent.hyp_runtime import gate_design
    hr = _hr()
    hr["pending"] = ["M1"]
    v, _ = gate_design("dem_build_case", {"exploratory": True}, hr, lambda hid: True)
    assert not v["ok"] and v["gated"] == "unreconciled"


def test_gate_ignores_non_design_tools():
    from granusci.agent.hyp_runtime import gate_design
    hr = _hr()
    v, cleaned = gate_design("dem_run_case", {"job_id": "j1"}, hr, lambda hid: True)
    assert v["ok"] and cleaned == {"job_id": "j1"}


# ---------------- 纯逻辑：机制 b 报数与对账 ----------------

def test_measurement_validation():
    from granusci.agent.hyp_runtime import record_measurement
    hr = _hr()
    rec, sug, _w = record_measurement(hr, {"job_id": ""}, lambda: [])
    assert not rec.get("mid")
    rec, sug, _w = record_measurement(hr, {"job_id": "j1", "params": {"gamma": 1}},
                                  lambda: [])
    assert not rec.get("mid")            # 缺 values
    rec, sug, _w = record_measurement(
        hr, {"job_id": "j1", "params": {"gamma": 1}, "values": {"q": 1.0}},
        lambda: [])
    assert not rec.get("mid")            # 缺 evidence


def test_measurement_pends_when_hypotheses_active():
    from granusci.agent.hyp_runtime import record_measurement, clear_pending
    hr = _hr()
    actives = lambda: [{"hid": "H1", "state": "testing",
                        "statement": "q∝D^2.5", "prediction": "n=2.5"}]
    rec, sug, _w = record_measurement(
        hr, {"job_id": "j1", "params": {"gamma": 2.0},
             "values": {"q": 1.45e-3}, "evidence": "plot_script 输出"},
        actives)
    assert rec["mid"] and hr["pending"] == [rec["mid"]]
    assert sug and sug[0]["hid"] == "H1" and "n=2.5" in sug[0]["prediction"]
    # 裁决清账
    cleared = clear_pending(hr, "H1")
    assert cleared == [rec["mid"]] and hr["pending"] == []
    assert hr["enforced"]["reconciliations"] == 1


def test_measurement_exploratory_no_pending():
    from granusci.agent.hyp_runtime import record_measurement
    hr = _hr()
    rec, sug, _w = record_measurement(
        hr, {"job_id": "j1", "params": {"gamma": 2.0},
             "values": {"q": 1.45e-3}, "evidence": "x"},
        lambda: [])   # 无在档假设 → 不挂对账账目
    assert rec["mid"] and hr["pending"] == [] and sug == []


def test_pending_blocker():
    from granusci.agent.hyp_runtime import pending_blocker
    hr = _hr()
    assert pending_blocker(hr) == ""
    hr["pending"] = ["M1", "M2"]
    assert "2 批" in pending_blocker(hr) and "hypothesis_update" in pending_blocker(hr)


# ---------------- 纯逻辑：机制 c 死亡后继 ----------------

def test_on_refuted_successor_entry():
    from granusci.agent.hyp_runtime import on_refuted
    rec = {"statement": "Beverloo 指数 n=2.5 适用于本几何",
           "prediction": "n=2.5"}
    entry = on_refuted("H7", rec, "实测 n=0.73-0.87（M12-M17）")
    assert entry["kind"] == "open_question" and entry["ref_hid"] == "H7"
    assert "n=2.5" in entry["content"] and "0.73" in entry["content"]
    assert on_refuted("H8", None, "") is None


# ---------------- 纯逻辑：机制 d DDDA 自动触发 ----------------

def _fill_measurements(hr, n, params_fn, _counter=[0]):
    from granusci.agent.hyp_runtime import record_measurement
    for i in range(n):
        _counter[0] += 1
        record_measurement(
            hr, {"job_id": f"j{_counter[0]}", "params": params_fn(i),
                 "values": {"q": 0.001 + 0.0001 * i}, "evidence": "x"},
            lambda: [])


def test_ddda_request_thresholds():
    from granusci.agent.hyp_runtime import (ddda_request, absorb_ddda,
                                            DDDA_MIN_ROWS)
    hr = _hr()
    assert ddda_request(hr) is None          # 空矩阵
    _fill_measurements(hr, DDDA_MIN_ROWS - 2, lambda i: {"gamma": 1.0 + i})
    assert ddda_request(hr) is None          # 行数不足
    _fill_measurements(hr, 2, lambda i: {"gamma": 1.0 + i})
    req = ddda_request(hr)                   # 达行数且新增≥2
    assert req is not None
    assert req["x_cols"] == ["gamma"] and req["y_cols"] == ["q"]
    assert req["columns"] == ["gamma", "q"] and len(req["rows"]) == DDDA_MIN_ROWS
    # absorb（吸收成功）后无新增 → 不再触发
    absorb_ddda(hr, "q", {"ok": True, "fitting": {"delta_star": 0.03,
                                                   "fitting_results": []}})
    assert ddda_request(hr) is None
    # 新增 2 行 → 再次触发
    _fill_measurements(hr, 2, lambda i: {"gamma": 5.0 + i})
    assert ddda_request(hr) is not None


def test_ddda_request_drops_constant_columns():
    from granusci.agent.hyp_runtime import ddda_request, DDDA_MIN_ROWS
    hr = _hr()
    _fill_measurements(hr, DDDA_MIN_ROWS,
                       lambda i: {"gamma": 1.0 + i, "fixed_param": 7})
    req = ddda_request(hr)
    # 常数参数列覆盖完整则保留（分区维度）；稀疏列（大量缺失）被剔除
    assert "gamma" in req["x_cols"]
    # v3hr_incline 回归：整列缺失（None）的参数不进矩阵
    _fill_measurements(hr, 3, lambda i: {"gamma": 1.0 + i, "sparse": None})
    req2 = ddda_request(hr)
    assert "sparse" not in req2["x_cols"]


def test_absorb_ddda_ok_and_fail():
    from granusci.agent.hyp_runtime import absorb_ddda
    hr = _hr()
    ok_result = {"ok": True, "fitting": {"delta_star": 0.03,
                                         "fitting_results": [
                                             {"effective_exponents": {"gamma": 0.5},
                                              "r2": 0.99}]}}
    text = absorb_ddda(hr, "q", ok_result)
    assert "R²=0.990" in text and "γ" not in text  # 摘要含拟合质量
    assert hr["last_partition"]["y_col"] == "q"
    assert hr["ddda_runs"][-1]["ok"] is True
    # 失败路径：结构化提示 agent 手动诊断
    text2 = absorb_ddda(hr, "q", {"ok": False, "message": "列不存在"})
    assert "失败" in text2 and hr["ddda_runs"][-1]["ok"] is False


def test_boundary_advice():
    from granusci.agent.hyp_runtime import boundary_advice, absorb_ddda
    hr = _hr()
    assert boundary_advice(hr) == ""         # 无分区结果 → 无建议
    absorb_ddda(hr, "q", {"ok": True, "fitting": {"delta_star": 0.03,
                                                   "fitting_results": []}})
    adv = boundary_advice(hr)
    assert "边界" in adv and "裁决权" in adv


# ---------------- 工具接线（隔离档案 + mock bus） ----------------

class _MockBus:
    """hyp_report_measurement 的 DDDA 自动调用 mock。"""

    def __init__(self, ddda_result=None):
        self.calls = []
        self.ddda_result = ddda_result or {
            "ok": True, "fitting": {"delta_star": 0.03, "fitting_results": [
                {"effective_exponents": {"gamma": 0.52}, "r2": 0.98}]}}

    async def call(self, name, args):
        self.calls.append((name, args))
        return dict(self.ddda_result)


def _tools(tmp_path, bus, hr_store):
    import granusci.agent.graph as g
    return g._make_local_tools(
        {}, run_dir=tmp_path / "20260924_101911_v2r_hopper", bus=bus,
        hr_store=hr_store, hr_enabled=True,
        hyptest_root=tmp_path / "hyp")


def test_report_measurement_tool_full_chain(tmp_path):
    """报数 → ledger marker → 对账建议 → DDDA 自动触发（机制 b+d 端到端）。"""
    from granusci.agent.hyp_runtime import new_hr, DDDA_MIN_ROWS
    hr = new_hr()
    bus = _MockBus()
    tools = _tools(tmp_path, bus, hr)

    reg = next(t for t in tools if t.name == "hypothesis_register")
    rep = next(t for t in tools if t.name == "hyp_report_measurement")
    assert rep is not None                # hr_enabled 时注册

    r = reg.handler({"statement": "q 随 Γ 非单调", "kind": "scaling",
                     "prediction": "极小在 Γ≈2-4"})
    hid = r["hid"]

    # 低于阈值报数：无 DDDA 调用；有在档假设 → 挂对账
    for i in range(DDDA_MIN_ROWS - 1):
        out = asyncio.run(rep.handler({
            "job_id": f"j{i}", "hypothesis_id": hid,
            "params": {"gamma": 1.0 + i}, "values": {"q": 0.001 + 0.0001 * i},
            "evidence": "plot_script M 拟合"}))
        assert not out.get("error"), out
        assert out["__LEDGER_ENTRY__"]["kind"] == "measurement"
    assert bus.calls == []                 # 阈值未到，DDDA 未触发
    assert len(hr["pending"]) == DDDA_MIN_ROWS - 1

    # 第 DDDA_MIN_ROWS 个报数：触发 DDDA 自动分析（系统调用，非 agent）
    out = asyncio.run(rep.handler({
        "job_id": "j9", "hypothesis_id": hid,
        "params": {"gamma": 3.0}, "values": {"q": 0.0013},
        "evidence": "plot_script M 拟合"}))
    assert len(bus.calls) == 1
    name, payload = bus.calls[0]
    assert name == "ddda_run_full_pipeline"
    assert payload["y_col"] == "q" and payload["x_cols"] == ["gamma"]
    assert len(payload["data"]["rows"]) == DDDA_MIN_ROWS
    assert "DDDA 自动分区分析" in out["message"]
    assert out["__LEDGER_ENTRY__"]["kind"] == "measurement"


def test_hyp_update_refuted_successor(tmp_path):
    """机制 c 接线：refuted → open_question ledger 条目 + 对账清账。"""
    from granusci.agent.hyp_runtime import new_hr, record_measurement
    hr = new_hr()
    tools = _tools(tmp_path, _MockBus(), hr)
    reg = next(t for t in tools if t.name == "hypothesis_register")
    upd = next(t for t in tools if t.name == "hypothesis_update")

    hid = reg.handler({"statement": "n=2.5", "kind": "scaling",
                       "prediction": "n=2.5"})["hid"]
    upd.handler({"hid": hid, "action": "state", "state": "testing"})
    # 挂一笔待对账测量
    record_measurement(hr, {"job_id": "j1", "hid": hid,
                            "params": {"gamma": 1.0}, "values": {"n": 0.8},
                            "evidence": "x"},
                       lambda: [{"hid": hid, "state": "testing",
                                 "statement": "n=2.5", "prediction": "n=2.5"}])
    assert hr["pending"]

    out = upd.handler({"hid": hid, "action": "state", "state": "refuted",
                       "evidence": "M1 实测 n=0.8≠2.5"})
    assert not out.get("error"), out
    assert out["__LEDGER_ENTRY__"]["kind"] == "open_question"
    assert out["__LEDGER_ENTRY__"]["ref_hid"] == hid
    assert "open_question" in out["message"]
    assert hr["pending"] == []              # 裁决清账
    assert hr["enforced"]["successors"] == 1


def test_hr_tools_absent_when_disabled(tmp_path):
    """OFF 配置（阶段三对照前提）：工具不注册，行为与历史版本一致。"""
    import granusci.agent.graph as g
    tools = g._make_local_tools(
        {}, run_dir=tmp_path / "20260924_101911_v2r_hopper",
        hr_store=None, hr_enabled=False, hyptest_root=tmp_path / "hyp")
    assert all(t.name != "hyp_report_measurement" for t in tools)


# ---------------- 报数门（v3hr run1 教训：报数靠 LLM 自觉 100% 缺席） ----------------

def test_mark_and_gate_unreported_done_jobs():
    from granusci.agent.hyp_runtime import (new_hr, gate_design,
                                            mark_done_jobs,
                                            unreported_done_jobs)
    hr = new_hr()
    hr["jobs"]["cg_a"] = {"hid": "H1", "exploratory": False, "ts": 0}
    hr["jobs"]["cg_b"] = {"hid": "", "exploratory": True, "ts": 0}
    # bus 日志观测：cg_a done（case_result），cg_b 仍在跑
    bus_log = [
        {"tool": "dem_case_result", "args": {"job_id": "cg_a"},
         "result": {"job_id": "cg_a", "status": "done"}},
        {"tool": "dem_run_case", "args": {"job_id": "cg_b"},
         "result": {"job_id": "cg_b", "status": "running"}},
    ]
    mark_done_jobs(hr, bus_log)
    assert unreported_done_jobs(hr) == ["cg_a"]
    # 报数门拦截（在 no_reference 之前——done 未报数优先）
    v, _ = gate_design("dem_build_case", {"hypothesis_id": "H1"}, hr,
                       lambda hid: True)
    assert not v["ok"] and v["gated"] == "unreported"
    # cg_a 报数后放行
    from granusci.agent.hyp_runtime import record_measurement
    record_measurement(hr, {"job_id": "cg_a", "params": {"x": 1.0},
                            "values": {"q": 2.0}, "evidence": "t"}, lambda: [])
    assert unreported_done_jobs(hr) == []
    v2, _ = gate_design("dem_build_case", {"hypothesis_id": "H1"}, hr,
                        lambda hid: True)
    assert v2["ok"]


def test_no_measurement_reason_escape():
    from granusci.agent.hyp_runtime import (new_hr, record_measurement,
                                            unreported_done_jobs)
    hr = new_hr()
    hr["jobs"]["cg_x"] = {"hid": "", "exploratory": True, "ts": 0, "done": True}
    assert unreported_done_jobs(hr) == ["cg_x"]
    rec, _, _w = record_measurement(
        hr, {"job_id": "cg_x", "no_measurement_reason": "纯结构观察，无数值产出"},
        lambda: [])
    assert rec.get("no_measurement") and "销账" in rec["message"]
    assert unreported_done_jobs(hr) == []
    # 不在设计门登记的 job 不能销账
    rec2, _, _w2 = record_measurement(
        hr, {"job_id": "cg_ghost", "no_measurement_reason": "x"}, lambda: [])
    assert not rec2.get("no_measurement") and not rec2.get("mid")


def test_done_flag_survives_via_hr():
    """done 标记在 hr.jobs 里持久（resume 后 bus 日志清空仍生效）。"""
    from granusci.agent.hyp_runtime import new_hr, mark_done_jobs, unreported_done_jobs
    hr = new_hr()
    hr["jobs"]["cg_a"] = {"hid": "H1", "exploratory": False, "ts": 0}
    mark_done_jobs(hr, [{"tool": "dem_case_result", "args": {},
                         "result": {"job_id": "cg_a", "status": "done"}}])
    # 模拟 resume：新进程 hr 从 checkpoint 恢复，bus 日志为空
    hr2 = {"jobs": hr["jobs"], "measurements": [], "pending": [],
           "reported_none": {}}
    assert unreported_done_jobs(hr2) == ["cg_a"]
