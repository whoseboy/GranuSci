# -*- coding: utf-8 -*-
"""假设层工具接线回归（v1_dryrun 教训：闭包引用不存在的 state 变量，
hypothesis_register 运行时 NameError，单测只测了 HypothesisStore 类漏网）"""
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures()


def _find_tool(tools, name):
    for t in tools:
        if t.name == name:
            return t
    raise AssertionError(f"工具 {name} 不存在")


def test_hyp_register_and_lifecycle(tmp_path):
    import granusci.agent.graph as g
    tools = g._make_local_tools({}, run_dir=tmp_path / "20260923_170519_v1_dryrun",
                                hyptest_root=tmp_path / "hyp")
    reg = _find_tool(tools, "hypothesis_register")
    upd = _find_tool(tools, "hypothesis_update")

    r = reg.handler({"statement": "测试假设：q 随 v_s 单调增", "kind": "mechanism",
                     "prediction": "q>0 当 v_s>0"})
    assert not r.get("error"), r
    hid = r["hid"]
    assert hid

    r2 = upd.handler({"hid": hid, "action": "state", "state": "testing"})
    assert not r2.get("error"), r2

    r3 = upd.handler({"hid": hid, "action": "state", "state": "supported",
                      "evidence": "80 工况 DDDA 拟合"})
    assert not r3.get("error"), r3


def test_knowledge_deposit(tmp_path):
    import granusci.agent.graph as g
    tools = g._make_local_tools({}, run_dir=tmp_path / "20260923_170519_v1_dryrun")
    dep = _find_tool(tools, "knowledge_deposit")
    r = dep.handler({"statement": "测试知识条目", "kind": "law",
                     "evidence": "单测"})
    assert not r.get("error"), r
