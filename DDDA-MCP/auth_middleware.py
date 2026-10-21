# -*- coding: utf-8 -*-
"""Token 鉴权 ASGI 中间件（三个 MCP 共用同一实现）。

- 环境变量 MCP_AUTH_TOKEN 设置后启用鉴权; 未设置则放行
- 请求须携带 x-mcp-token 或 Authorization: Bearer
- 例外放行: /healthz 健康检查; MCP initialize 握手(平台客户端握手不带token)
- 未授权返回 401 JSON
"""
from __future__ import annotations

import json
import os


class TokenAuth:
    def __init__(self, app):
        self.app = app
        self.token = os.getenv("MCP_AUTH_TOKEN", "").strip()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.token:
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if path.endswith("/healthz"):
            await self.app(scope, receive, send)
            return
        # 读body判断是否initialize握手(读后需回放)
        body = b""
        if scope.get("method") == "POST":
            more = True
            while more:
                msg = await receive()
                if msg["type"] == "http.request":
                    body += msg.get("body", b"")
                    more = msg.get("more_body", False)
                else:
                    more = False
            # 回放body
            sent_body = body
            async def replay():
                yield {"type": "http.request", "body": sent_body, "more_body": False}
                yield {"type": "http.disconnect"}
            _orig_receive = receive
            receive = replay().__anext__
        try:
            # 只拦截 tools/call(有计算副作用的调用)。
            # initialize/notifications/initialized/tools-list/ping等协议流量
            # 全部放行: 平台客户端在这些阶段不带token(实测), 拒绝任何一个
            # 都会导致平台放弃整个MCP连接。
            is_call = b'"tools/call"' in body or b"'tools/call'" in body
        except Exception:
            is_call = False
        if not is_call:
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers", [])
        supplied = None
        for k, v in headers:
            if k == b"x-mcp-token":
                supplied = v.decode("latin-1").strip()
            elif k == b"authorization":
                val = v.decode("latin-1").strip()
                if val.lower().startswith("bearer "):
                    supplied = supplied or val[7:].strip()
        if supplied == self.token:
            await self.app(scope, receive, send)
            return
        resp_body = json.dumps({
            "jsonrpc": "2.0", "id": None,
            "error": {"code": -32001, "message": "unauthorized: missing or bad x-mcp-token"}})
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [(b"content-type", b"application/json"),
                        (b"www-authenticate", b"Bearer")],
        })
        await send({"type": "http.response.body", "body": resp_body.encode("utf-8")})
