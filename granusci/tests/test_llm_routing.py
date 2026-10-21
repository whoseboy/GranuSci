# -*- coding: utf-8 -*-
"""llm 多端点路由测试：国外模型走 Higress，国产走 paratera。"""
from __future__ import annotations

import json

import httpx
import pytest

from granusci import config, llm


def test_client_routes_foreign_models_to_higress():
    c = llm._client(model="gpt-5.6-sol")
    assert str(c.base_url).rstrip("/") == config.HIGRESS_BASE_URL.rstrip("/")
    c.close()
    c2 = llm._client(model="GLM-5.3")
    assert str(c2.base_url).rstrip("/") == config.PARATERA_BASE_URL.rstrip("/")
    c2.close()
    # 无模型参数（embed 等本地路径）默认 paratera
    c3 = llm._client()
    assert str(c3.base_url).rstrip("/") == config.PARATERA_BASE_URL.rstrip("/")
    c3.close()


def test_foreign_model_chat_reaches_higress(monkeypatch):
    """gpt-5.6-sol 的 chat 请求实际打到 Higress 端点（mock transport 捕获）。"""
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2}})

    _real_client = httpx.Client

    def fake_client(**kw):
        kw["transport"] = httpx.MockTransport(handler)
        return _real_client(**kw)

    monkeypatch.setattr(llm.httpx, "Client", fake_client)

    m = llm.chat("gpt-5.6-sol", [{"role": "user", "content": "hi"}])
    assert (m.get("content") or "").strip() == "ok"
    assert config.HIGRESS_BASE_URL.rstrip("/") in seen["url"]
    assert seen["body"]["model"] == "gpt-5.6-sol"
