# -*- coding: utf-8 -*-
"""mcp_bus.py 重连退避分级（⑩）单元测试——mock 连接对象，不起真服务器。"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from granusci.mcp_bus import McpBus


class FakeConn:
    """可编程假连接：按脚本抛异常/返回结果，记录 connect 次数。"""

    def __init__(self, script):
        self.script = list(script)   # 每次调用弹出一项：Exception 实例或返回值
        self.connects = 0

    async def connect(self):
        self.connects += 1

    async def call_raw(self, name, args):
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


@pytest.fixture
def fast_sleep(monkeypatch):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("asyncio.sleep", fake_sleep)
    return sleeps


async def _run(script):
    bus = McpBus()
    conn = FakeConn(script)
    bus._conns["dem"] = conn
    handler = bus._make_mcp_handler("dem", "run_simulation")
    return conn, await handler({"d": 0.004})


def test_transport_error_reconnect_retry(fast_sleep):
    """传输层异常：退避重连后同参重试成功。"""
    conn, result = asyncio.run(_run([
        httpx.RemoteProtocolError("peer closed"),
        {"ok": True, "job_id": "job_1"},
    ]))
    assert result == {"ok": True, "job_id": "job_1"}
    assert conn.connects == 1
    assert fast_sleep == [2.0]          # 第一次重连前退避 2s


def test_retry_exhausted_retryable_flag(fast_sleep):
    """重连重试全耗尽：返回结构化错误 + retryable=True（智能体可稍后重试）。"""
    conn, result = asyncio.run(_run([
        httpx.ConnectError("refused"),
        httpx.ConnectError("refused"),
        httpx.ConnectError("refused"),
    ]))
    assert result["error"] is True
    assert result["retryable"] is True
    assert "稍等后重试" in result["message"]
    assert conn.connects == 2           # 重连 2 次均失败
    assert fast_sleep == [2.0, 4.0]     # 指数退避


def test_cancel_error_reconnects(fast_sleep):
    """anyio 取消（BaseException 家族）也走重连——paper1 死因防线。"""
    class FakeCancel(BaseException):
        pass

    conn, result = asyncio.run(_run([
        FakeCancel("cancelled"),
        {"ok": True},
    ]))
    assert result == {"ok": True}
    assert conn.connects == 1


def test_keyboard_interrupt_not_swallowed(fast_sleep):
    """KeyboardInterrupt/SystemExit 必须向上抛，不许吞。"""
    bus = McpBus()
    conn = FakeConn([KeyboardInterrupt()])
    bus._conns["dem"] = conn
    handler = bus._make_mcp_handler("dem", "t")
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(handler({}))


def test_unknown_tool_not_retryable():
    bus = McpBus()
    result = asyncio.run(bus.call("nope_tool", {}))
    assert result["error"] is True and result["retryable"] is False


def test_local_tool_exception_not_retryable():
    """本地工具抛异常：结构化错误 + retryable=False（换法子而非重试）。"""
    from granusci.mcp_bus import Tool

    def boom(args):
        raise ValueError("bad args")

    bus = McpBus()
    bus.tools["local_boom"] = Tool("local_boom", "", {"type": "object"}, boom)
    result = asyncio.run(bus.call("local_boom", {}))
    assert result["error"] is True and result["retryable"] is False
    assert "ValueError" in result["message"]
    assert len(bus.call_log) == 1 and bus.call_log[0]["ok"] is False
