# -*- coding: utf-8 -*-
"""启动协议结构强制回归：计划未批准前仿真类工具必须被拦（webplan 教训）。"""
import asyncio
import json
import tempfile
from pathlib import Path


def test_design_tools_blocked_before_plan_approval():
    from granusci.agent.graph import build_graph

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "20261012_000000_plangate_smoke"
        run_dir.mkdir(parents=True)
        graph, bus, _tools = build_graph(
            "test-model", "test-model", 10, log_fn=lambda m: None,
            run_dir=run_dir, checkpointer=None, review_enabled=False)
        node = getattr(graph.nodes["tools"], "func", None) or graph.nodes["tools"]
        call = node.ainvoke if hasattr(node, "ainvoke") else node

        async def _run():
            await bus.start([])
            try:
                out = await call({"messages": [{"role": "assistant", "content": "",
                                                "tool_calls": [{"id": "t1", "type": "function",
                                                                "function": {"name": "dem_build_case",
                                                                             "arguments": json.dumps({"case": "{}"})}}]}],
                                  "tool_log": [], "ledger": [],
                                  "plan_approved": False, "plan": None})
            finally:
                await bus.close()
            return out

        out = asyncio.run(_run())
        result = json.loads(out["messages"][-1]["content"])
        assert result.get("plan_gate") is True
        assert result.get("error") is True
        # 拦截留痕但不触达真实服务（ok=False 的单条记录）
        assert len(bus.call_log) == 1 and bus.call_log[0]["ok"] is False
