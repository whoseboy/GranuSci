# -*- coding: utf-8 -*-
"""假设层与战役编排单元测试（A4+B1+B6）。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from granusci.hypothesis import HypothesisStore, STATES, TRANSITIONS  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return HypothesisStore(tmp_path)


def test_register_requires_statement(store):
    r = store.register({"kind": "sign"})
    assert not r["ok"]


def test_register_rejects_cite_without_id(store):
    r = store.register({"statement": "x", "source": {"type": "cite"}})
    assert not r["ok"]
    assert "cite_id" in r["message"]


def test_lifecycle_register_to_refuted(store):
    r = store.register({
        "statement": "偏析符号由 (F,r_s) 决定，r_s=1.2 时几乎全 RBNE",
        "kind": "sign", "prediction": "r_s=1.2 全部 seg<0",
        "partition_vars": ["F", "r_s"], "source": {"type": "reasoning"}})
    assert r["ok"]
    hid = r["hid"]
    assert store.update_state(hid, "testing", evidence="加密轮")["ok"]
    r2 = store.update_state(hid, "refuted", evidence="Γ=0.97 反例 4/6")
    assert r2["ok"] and "新科学问题候选" in r2["note"]


def test_illegal_transition_rejected(store):
    r = store.register({"statement": "x"})
    hid = r["hid"]
    assert not store.update_state(hid, "supported")["ok"]  # registered→supported 非法


def test_revise_keeps_history(store):
    r = store.register({"statement": "v1 陈述"})
    hid = r["hid"]
    store.update_state(hid, "testing", evidence="t")
    r2 = store.revise(hid, "v2 陈述", "数据推翻 v1")
    assert r2["ok"]
    life = store.lifecycle(hid)["hypothesis"]
    assert life["statement"] == "v2 陈述"
    assert life["versions"][0]["statement"] == "v1 陈述"
    assert life["state"] == "revised"


def test_query_filters(store):
    store.register({"statement": "a", "kind": "sign"})
    store.register({"statement": "b", "kind": "partition"})
    assert store.query(kind="sign")["n"] == 1
    assert store.query()["n"] == 2


def test_campaign_log_append(store):
    store.campaign_phase("screening", note="60 工况", budget_used={"cases": 60})
    store.campaign_phase("targeted", note="边界加密", budget_used={"cases": 40})
    log = store.campaign_log()
    assert log["n"] == 2
    assert log["phases"][0]["phase"] == "screening"


def test_states_machine_completeness():
    # 状态机完整性：所有状态有迁移定义（retired 除外）
    for s in STATES:
        assert s in TRANSITIONS
    assert TRANSITIONS["retired"] == set()
