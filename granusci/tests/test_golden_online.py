# -*- coding: utf-8 -*-
"""⑪ 在线 golden 基线：真实 MCP 服务器上的端到端正确性锚点。

与离线 golden 的分工：离线测契约结构，在线测**引擎真的算得对**。

- DDDA 合成幂律 golden：喂已知指数的无噪声人工数据（ad^0.8·Γ^0.5·F^0），
  引擎必须恢复指数（容差 0.12——平滑偏导对网格边界有系统性偏差，
  本 golden 防的是引擎级损坏/幻觉级错误，不是数值漂移）。
  "非摆拍"的自动化常驻版（真伪验证实验的回归化）。
- DEM 单工况 golden：设计→仿真(wait)→后处理全链，一个工况跑通且响应量有限。

服务器不可达时整体 skip（本地离线开发不阻塞）。
"""
from __future__ import annotations

import asyncio
import json
import math
import socket

import pytest

from granusci import config


def _reachable(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _host_port(url: str) -> tuple[str, int]:
    from urllib.parse import urlparse
    u = urlparse(url)
    return u.hostname, u.port or 80


DDDA_UP = _reachable(*_host_port(config.MCP_SERVERS["ddda"]))
DEM_UP = _reachable(*_host_port(config.MCP_SERVERS["dem"]))

pytestmark = pytest.mark.skipif(not (DDDA_UP and DEM_UP),
                                reason="MCP 服务器不可达（离线环境跳过在线 golden）")


async def _call(server: str, tool: str, args: dict) -> dict:
    from contextlib import AsyncExitStack
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    stack = AsyncExitStack()
    try:
        # 与 mcp_bus 同源修复：Clash 系统代理劫持内网流量（httpx 读注册表代理），
        # 测试连接必须 trust_env=False，否则 192.168.x 被 502
        import httpx as _httpx

        def _client_factory(**kwargs):
            kwargs.setdefault("timeout", _httpx.Timeout(120))
            return _httpx.AsyncClient(trust_env=False, **kwargs)

        headers = None
        if server == "dem" and getattr(config, "DEM_MCP_TOKEN", ""):
            headers = {"x-mcp-token": config.DEM_MCP_TOKEN}
        cm = streamablehttp_client(config.MCP_SERVERS[server], timeout=120,
                                   sse_read_timeout=600, headers=headers,
                                   httpx_client_factory=_client_factory)
        reads, writes, _ = await stack.enter_async_context(cm)
        session = await stack.enter_async_context(ClientSession(reads, writes))
        await session.initialize()
        res = await session.call_tool(tool, arguments=args)
        texts = [b.text for b in (res.content or []) if getattr(b, "type", "") == "text"]
        return json.loads("\n".join(texts))
    finally:
        try:
            await stack.aclose()
        except Exception:
            pass


def test_ddda_synthetic_powerlaw_golden():
    """已知真值回归：无噪声幂律 y=2.5·ad^0.8·Γ^0.5·F^0，指数恢复误差 ≤0.12。"""
    rows = []
    for ad in (0.5, 0.8, 1.2, 1.8, 2.5):
        for gamma in (2.0, 4.0, 8.0, 16.0):
            for F in (3.0, 6.0, 12.0):
                rows.append([ad, gamma, F, 2.5 * ad ** 0.8 * gamma ** 0.5])
    result = asyncio.run(_call("ddda", "run_full_pipeline", {
        "data": {"columns": ["ad", "gamma", "F", "y"], "rows": rows},
        "x_cols": ["ad", "gamma", "F"], "y_col": "y",
        "space": "physical", "n_clusters": 1}))
    assert result.get("ok"), f"管线失败: {str(result)[:300]}"
    fitting = result["fitting"]
    delta_star = fitting["delta_star"]
    assert delta_star is not None and delta_star < 0.05, f"无噪声数据 δ*={delta_star} 过大"
    fr = fitting["fitting_results"][0]
    exp = fr["effective_exponents"]
    for name, truth in (("ad", 0.8), ("gamma", 0.5), ("F", 0.0)):
        got = exp[name]
        assert abs(got - truth) <= 0.12, f"指数 {name} 恢复失败: {got} vs 真值 {truth}"
    assert fr.get("r2", 0) > 0.99


def test_ddda_error_path_structured():
    """在线错误路径：坏列名返回结构化错误（ok=False + error/message），不裸崩。"""
    result = asyncio.run(_call("ddda", "run_full_pipeline", {
        "data": {"columns": ["ad", "gamma", "y"], "rows": [[1.0, 2.0, 1.0]]},
        "x_cols": ["ad", "nope"], "y_col": "y", "space": "physical"}))
    assert result.get("ok") is False
    assert result.get("error") or result.get("message"), "错误须带说明"


def test_dem_single_case_golden():
    """DEM 全链 golden：单工况设计→仿真(wait)→后处理，响应量有限值。"""
    design = asyncio.run(_call("dem", "generate_design", {
        "ad_list": [1.0], "gamma_list": [5.0], "F_list": [5.0], "d": 0.004,
        "dev_periods": 20, "stat_periods": 20}))
    assert design.get("ok"), f"设计失败: {str(design)[:200]}"
    assert design["n_cases"] == 1

    sim = asyncio.run(_call("dem", "run_simulation", {
        "design_id": design["design_id"], "wait": True, "wait_timeout": 600}))
    assert sim.get("ok"), f"仿真失败: {str(sim)[:300]}"
    assert sim["status"] == "done" and sim["failed"] == [], f"工况未完成: {sim}"

    post = asyncio.run(_call("dem", "postprocess", {"job_id": sim["job_id"]}))
    assert post.get("ok"), f"后处理失败: {str(post)[:300]}"
    assert post["n_valid"] == 1
    case = post["summary"][0]
    for key in ("e_k_star", "ep_star"):
        v = case.get(key)
        assert v is not None and math.isfinite(v) and abs(v) < 1e4, f"{key} 非有限: {v}"
