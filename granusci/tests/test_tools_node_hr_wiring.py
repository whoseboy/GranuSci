# -*- coding: utf-8 -*-
"""tools_node 四机制接线冒烟（NameError 教训：闭包变量运行时才炸，
单测只测纯函数/本地工具会漏接线层——这里直调编译后图的 tools 节点）。"""
import asyncio
import json
import tempfile
from pathlib import Path


def test_tools_node_hr_wiring():
    from granusci.agent.graph import build_graph

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "20260925_000000_hr_smoke"
        run_dir.mkdir(parents=True)
        graph, bus, tools = build_graph(
            "test-model", "test-model", 10, log_fn=lambda m: None,
            run_dir=run_dir, checkpointer=None, review_enabled=False,
            hr_enabled=True)
        assert any(t.name == "hyp_report_measurement" for t in tools)

        # langgraph 编译节点包装：取原始可调用（.func 或节点本身）
        node = getattr(graph.nodes["tools"], "func", None) or graph.nodes["tools"]
        call = node.ainvoke if hasattr(node, "ainvoke") else node

        # ① 机制 a：无引用的 build 被拦截（不触达 bus——mock 掉 bus.call）
        async def _fake_call(name, args):
            return {"ok": True, "job_id": "cg_test_0001", "params_echo": {}}
        bus.call = _fake_call
        bus.call_log = []

        async def _run():
            out = await call({"messages": [{"role": "assistant", "content": "",
                                            "tool_calls": [{"id": "t1", "type": "function",
                                                            "function": {"name": "dem_build_case",
                                                                         "arguments": json.dumps({"case": "{}"})}}]}],
                              "tool_log": [], "ledger": [], "plan_approved": True, "hr": {}})
            result = json.loads(out["messages"][-1]["content"])
            assert result.get("hr_gate") and result.get("gated") == "no_reference"
            assert out["hr"]["enforced"]["builds_gated"] == 1

            # ② 机制 a 放行：exploratory → 剥离转发，job 关联落账
            out2 = await call({"messages": [{"role": "assistant", "content": "",
                                             "tool_calls": [{"id": "t2", "type": "function",
                                                             "function": {"name": "dem_build_case",
                                                                          "arguments": json.dumps({"case": "{}", "exploratory": True})}}]}],
                               "tool_log": [], "ledger": [], "plan_approved": True, "hr": out["hr"]})
            result2 = json.loads(out2["messages"][-1]["content"])
            assert result2.get("ok") is True
            assert "cg_test_0001" in out2["hr"]["jobs"]

            # ③ 机制 b：pending 非空 → 新工况被对账门拒绝
            out2["hr"]["pending"] = ["M1"]
            out3 = await call({"messages": [{"role": "assistant", "content": "",
                                             "tool_calls": [{"id": "t3", "type": "function",
                                                             "function": {"name": "dem_modify_case",
                                                                          "arguments": json.dumps({"job_id": "cg_test_0001", "exploratory": True})}}]}],
                               "tool_log": [], "ledger": [], "plan_approved": True, "hr": out2["hr"]})
            result3 = json.loads(out3["messages"][-1]["content"])
            assert result3.get("hr_gate") and result3.get("gated") == "unreconciled"

        asyncio.run(_run())
        asyncio.run(bus.close())
