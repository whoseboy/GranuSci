# -*- coding: utf-8 -*-
"""llm.py 重试分级（⑩）+ 用量核算（⑥）单元测试——全部离线（httpx MockTransport）。"""
from __future__ import annotations

import json

import httpx
import pytest

from granusci import llm


def _ok_body(content="hello", prompt=10, completion=5):
    return {"choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


def _empty_body():
    return {"choices": [{"message": {"role": "assistant", "content": ""}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 0}}


@pytest.fixture
def env(monkeypatch, tmp_path):
    """隔离环境：假睡眠 + 假客户端 + 新用量台账。"""
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
    tracker = llm.UsageTracker()
    tracker.configure(tmp_path)
    monkeypatch.setattr(llm, "usage", tracker)
    requests = []

    class Env:
        @staticmethod
        def mount(handler):
            def make_client(timeout=300, model=""):
                return httpx.Client(base_url="https://mock.test",
                                    transport=httpx.MockTransport(handler))
            monkeypatch.setattr(llm, "_client", make_client)

    e = Env()
    e.requests = requests
    e.sleeps = sleeps
    e.tracker = tracker
    e.tmp = tmp_path
    return e


def test_transient_429_backoff_then_success(env, monkeypatch):
    env.requests.append(0)

    def handler(request):
        env.requests[0] += 1
        if env.requests[0] <= 2:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json=_ok_body())

    env.mount(handler)
    msg = llm.chat("GLM-5.3", [{"role": "user", "content": "hi"}], purpose="agent")
    assert msg["content"] == "hello"
    assert env.requests[0] == 3                       # 2 次 429 + 1 次成功
    assert len(env.sleeps) == 2                       # 每次瞬时失败退避一次
    # 指数退避+抖动：2^1×[0.5,1.5] 与 2^2×[0.5,1.5]
    assert 1.0 <= env.sleeps[0] <= 3.0
    assert 2.0 <= env.sleeps[1] <= 6.0
    # 用量已记录：attempts 含重试，token 数取成功响应的 usage
    rec = env.tracker.calls[-1]
    assert rec["purpose"] == "agent" and rec["attempts"] == 3
    assert rec["prompt_tokens"] == 10 and rec["completion_tokens"] == 5


def test_permanent_400_fast_fail(env):
    env.requests.append(0)

    def handler(request):
        env.requests[0] += 1
        return httpx.Response(400, text='{"error": "bad request"}')

    env.mount(handler)
    with pytest.raises(llm.LLMError) as ei:
        llm.chat("GLM-5.3", [{"role": "user", "content": "hi"}])
    assert ei.value.kind == "permanent"
    assert env.requests[0] == 2          # 原始 + 确认重试 1 次，不再退避轰炸
    assert env.sleeps == [1]             # permanent 只短暂等 1s
    assert env.tracker.calls == []       # 失败调用不计用量


def test_transient_exhausted(env):
    env.requests.append(0)

    def handler(request):
        env.requests[0] += 1
        return httpx.Response(502, text="bad gateway")

    env.mount(handler)
    with pytest.raises(llm.LLMError) as ei:
        llm.chat("GLM-5.3", [{"role": "user", "content": "hi"}])
    assert ei.value.kind == "transient"
    assert env.requests[0] == 5          # 1 + transient_retries(4)
    assert len(env.sleeps) == 4


def test_nonjson_200_retried_as_protocol(env):
    """200 但返回 HTML 错误页（隧道瞬断假象）：按协议类瞬时故障退避重试。"""
    env.requests.append(0)

    def handler(request):
        env.requests[0] += 1
        if env.requests[0] == 1:
            return httpx.Response(200, text="<html>bad gateway page</html>")
        return httpx.Response(200, json=_ok_body("ok"))

    env.mount(handler)
    msg = llm.chat("GLM-5.3", [{"role": "user", "content": "hi"}])
    assert msg["content"] == "ok"
    assert env.requests[0] == 2


def test_network_error_retry(env):
    env.requests.append(0)

    def handler(request):
        env.requests[0] += 1
        if env.requests[0] == 1:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json=_ok_body("recovered"))

    env.mount(handler)
    msg = llm.chat("GLM-5.3", [{"role": "user", "content": "hi"}])
    assert msg["content"] == "recovered"
    assert env.requests[0] == 2


def test_empty_response_retry_then_valid(env):
    env.requests.append(0)

    def handler(request):
        env.requests[0] += 1
        if env.requests[0] <= 2:
            return httpx.Response(200, json=_empty_body())
        return httpx.Response(200, json=_ok_body("finally"))

    env.mount(handler)
    msg = llm.chat("GLM-5.3", [{"role": "user", "content": "hi"}])
    assert msg["content"] == "finally"
    assert env.requests[0] == 3


def test_empty_exhausted_returns_empty_message(env):
    """空响应重试耗尽：原样返回空消息（graph 有 empty_response 防线）。"""
    def handler(request):
        return httpx.Response(200, json=_empty_body())

    env.mount(handler)
    msg = llm.chat("GLM-5.3", [{"role": "user", "content": "hi"}])
    assert msg["content"] == ""
    assert env.tracker.calls[-1]["completion_tokens"] == 0


def test_usage_jsonl_incremental(env):
    """用量逐调用增量落盘——崩溃时已发生的调用不丢。"""
    def handler(request):
        return httpx.Response(200, json=_ok_body())

    env.mount(handler)
    llm.chat("GLM-5.3", [{"role": "user", "content": "a"}], purpose="agent")
    lines = (env.tmp / "llm_usage.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["purpose"] == "agent"
    llm.chat("DeepSeek-V4-Pro", [{"role": "user", "content": "b"}], purpose="reviewer")
    lines = (env.tmp / "llm_usage.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2


def test_usage_summary_and_dump(env):
    def handler(request):
        return httpx.Response(200, json=_ok_body(prompt=100, completion=50))

    env.mount(handler)
    llm.chat("GLM-5.3", [{"role": "user", "content": "a"}], purpose="agent")
    llm.chat("GLM-5.3", [{"role": "user", "content": "b"}], purpose="agent")
    llm.chat("DeepSeek-V4-Pro", [{"role": "user", "content": "c"}], purpose="reviewer")
    s = env.tracker.summary()
    assert s["total_calls"] == 3
    assert s["by_model"]["GLM-5.3"]["total_tokens"] == 300
    assert s["by_purpose"]["agent"]["calls"] == 2
    path = env.tracker.dump(env.tmp)
    disk = json.loads(path.read_text(encoding="utf-8"))
    assert disk["by_model"]["DeepSeek-V4-Pro"]["calls"] == 1


def test_quota_429_fast_fail(env):
    """配额型 429：立即失败（kind=quota），不做退避重试。"""
    env.requests.append(0)

    def handler(request):
        env.requests[0] += 1
        return httpx.Response(429, json={
            "error": {"message": "Rate limit exceeded for team: X. "
                                 "Limit type: tokens. Current limit: 5000000, "
                                 "Remaining: 0"}})

    env.mount(handler)
    with pytest.raises(llm.LLMError) as ei:
        llm.chat("GLM-5.3", [{"role": "user", "content": "hi"}])
    assert ei.value.kind == "quota"
    assert env.requests[0] == 1          # 不重试——单次请求即抛
    assert env.sleeps == []              # 无退避等待


def test_transient_429_still_retries(env):
    """瞬时限流（无 limit/quota 字样）仍走退避重试——回归保护。"""
    env.requests.append(0)

    def handler(request):
        env.requests[0] += 1
        if env.requests[0] == 1:
            return httpx.Response(429, text="too many requests")
        return httpx.Response(200, json=_ok_body())

    env.mount(handler)
    msg = llm.chat("GLM-5.3", [{"role": "user", "content": "hi"}])
    assert msg["content"] == "hello"
    assert env.requests[0] >= 2          # 有重试
