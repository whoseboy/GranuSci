# -*- coding: utf-8 -*-
"""MCP 工具总线：连接三个服务器 MCP，统一注册为智能体可调用工具。

- 工具名加服务器前缀（dem_/ddda/lit_），全名即 LLM 看到的函数名。
- 本地工具（kb_search / load_skill / submit_report）在同一注册表中注册，
  智能体对所有工具一视同仁。
- 每服务器维持长连接会话；调用遇传输层异常（服务重启/网络抖动）自动重建会话
  并重试一次。
"""
from __future__ import annotations

import inspect
import json
import time
from contextlib import AsyncExitStack
from typing import Callable

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from . import config


class Tool:
    def __init__(self, name: str, description: str, schema: dict,
                 handler: Callable[[dict], dict]):
        self.name = name
        self.description = description
        self.schema = schema          # JSON Schema of parameters
        self.handler = handler        # (args) -> dict result（同步或异步）


class _ServerConn:
    """单个 MCP 服务器的可重连会话。

    重连时只新建连接并入 stack，不显式关闭旧连接——anyio 的 cancel scope
    要求上下文在创建它的 task 内退出，跨 task 关闭会抛 RuntimeError；
    死连接交给进程退出统一清理。
    """

    def __init__(self, prefix: str, url: str, stack: AsyncExitStack):
        self.prefix = prefix
        self.url = url
        self._stack = stack
        self.session: ClientSession | None = None

    async def connect(self):
        # MCP 服务器全部是内网直连地址：httpx 客户端禁用环境/注册表代理
        # （Clash 系统代理会把 192.168 流量劫持成 502/挂起——NO_PROXY 环境
        # 变量实测间歇性失效，trust_env=False 是代码级根治）
        import httpx as _httpx

        def _client_factory(**kwargs):
            kwargs.setdefault("timeout", _httpx.Timeout(60))
            return _httpx.AsyncClient(trust_env=False, **kwargs)

        headers = None
        if self.prefix == "dem" and config.DEM_MCP_TOKEN:
            headers = {"x-mcp-token": config.DEM_MCP_TOKEN}
        cm = streamablehttp_client(
            self.url, timeout=60, sse_read_timeout=config.MCP_READ_TIMEOUT,
            headers=headers, httpx_client_factory=_client_factory)
        reads, writes, _ = await self._stack.enter_async_context(cm)
        session = await self._stack.enter_async_context(ClientSession(reads, writes))
        await session.initialize()
        self.session = session

    async def call_raw(self, raw_name: str, args: dict):
        res = await self.session.call_tool(raw_name, arguments=args)
        texts = []
        for block in (res.content or []):
            if getattr(block, "type", "") == "text":
                texts.append(block.text)
        raw = "\n".join(texts)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                if res.isError:
                    parsed.setdefault("error", True)
                return parsed
            return {"result": parsed}
        except (json.JSONDecodeError, TypeError):
            return {"text": raw[: config.FULL_RESULT_CHAR_CAP],
                    "is_error": bool(res.isError)}


# 客户端侧工具描述增强（服务器描述不动，注册时追加）。
# 来源：paper11 run 后复盘工单——异步提交返回处理薄弱（16 提交仅 4 落成 job，
# 靠同步重提兜底浪费 ~30 轮）。这里把关键纪律直接写进 LLM 看到的描述。
_DESC_OVERRIDES = {
    "dem_run_simulation": (
        "⚠️ 使用纪律：①max_parallel 建议 32（服务器 128 核工况级并行，单工况单核）；"
        "②wait=false 异步提交后必须立刻用 record_finding 记下返回的 job_id（历史会被压缩，"
        "丢了 job_id 就只能重新提交）；③批量提交多个设计时逐个记录 design_id→job_id 映射，"
        "全部提交完后统一轮询收割；④异步返回后当场检查 job_id 存在，空/报错当场重试。"),
    "dem_postprocess": (
        "⚠️ 幂等提示：本工具对同一 job 重复调用返回相同结果，不产生新信息——重复调用"
        "是浪费。收到 responses_file 路径后立即用 record_finding(kind=artifact) 记入账本"
        "（防历史压缩丢失），之后直接用该路径即可。"),
}


class McpBus:
    """一次研究运行的工具面。"""

    def __init__(self, excluded: set[str] | None = None):
        self.tools: dict[str, Tool] = {}
        self._stack = AsyncExitStack()
        self._conns: dict[str, _ServerConn] = {}
        self.call_log: list[dict] = []   # [{tool,args,ts,duration,result}, ...]
        # 工具消融（--exclude-tools）：调度层硬拦截——被禁工具即便被幻觉调用
        # 也不执行（openai_tools 的 exclude 只管 LLM 可见列表，这里是执行防线）
        self.excluded: set[str] = set(excluded or ())

    # ------------------------------------------------------------ lifecycle
    async def start(self, local_tools: list[Tool] | None = None):
        for t in local_tools or []:
            self.tools[t.name] = t
        for prefix, url in config.MCP_SERVERS.items():
            conn = _ServerConn(prefix, url, self._stack)
            # 单服务器故障不拖死整个 bus（同 AsyncExitStack 下第三个连接
            # 挂起的实测问题——lit-mcp 曾因此阻塞 start，dem/ddda 也全废）。
            # initialize/list_tools 设 20s 上限，超时/异常降级为缺该服务器。
            import asyncio as _aio
            try:
                await _aio.wait_for(conn.connect(), timeout=20)
            except BaseException as e:  # noqa: BLE001  wait_for 超时抛 CancelledError
                # （BaseException 子类——except Exception 接不住，曾使降级路径
                # 本身崩溃、拖死整个 bus）
                print(f"[bus] MCP 服务器 {prefix} 连接失败（{type(e).__name__}），"
                      f"本 run 缺该服务器工具: {str(e)[:80]}", flush=True)
                continue
            self._conns[prefix] = conn
            listed = await conn.session.list_tools()
            for t in listed.tools:
                full = f"{prefix}_{t.name}"
                desc = (t.description or full).strip()
                if full in _DESC_OVERRIDES:
                    desc = desc + "\n" + _DESC_OVERRIDES[full]
                self.tools[full] = Tool(
                    full, desc, t.inputSchema or {"type": "object", "properties": {}},
                    self._make_mcp_handler(prefix, t.name),
                )

    async def close(self):
        try:
            await self._stack.aclose()
        except Exception:
            pass

    def _make_mcp_handler(self, prefix: str, raw_name: str):
        async def handler(args: dict) -> dict:
            conn = self._conns[prefix]
            try:
                return await conn.call_raw(raw_name, args)
            except BaseException as e:
                # 传输层异常（服务重启/隧道抖动）——含 anyio 的 CancelledError
                # （BaseException，不走 except Exception），须一并捕获重连，
                # 否则会穿透到 LangGraph 掐死整个 run。
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                # 重连退避分级：2s、4s，最多 2 次重建会话（覆盖滚动重启窗口）
                import asyncio
                last = e
                for i in range(2):
                    await asyncio.sleep(2.0 * (i + 1))
                    try:
                        await conn.connect()
                        return await conn.call_raw(raw_name, args)
                    except BaseException as e2:
                        if isinstance(e2, (KeyboardInterrupt, SystemExit)):
                            raise
                        last = e2
                return {"error": True, "is_error": True, "retryable": True,
                        "message": (f"工具 {prefix}_{raw_name} 调用失败且重连重试无效"
                                    f"（{type(last).__name__}: {str(last)[:200]}）。"
                                    f"服务可能正在重启——可稍等后重试同一调用；"
                                    f"连续失败则改用其他工具或在报告中如实记录。")}
        return handler

    # ------------------------------------------------------------ dispatch
    async def call(self, name: str, args: dict) -> dict:
        if name in self.excluded:
            result = {"error": True, "ablated": True,
                      "message": (f"工具 {name} 已在本配置中禁用（实验控制）。"
                                  "不要再次调用；相关需求改用其余可用工具完成，"
                                  "确实无替代途径时在报告局限中如实说明。")}
            self.call_log.append({"tool": name, "args": args, "ts": time.time(),
                                  "duration": 0.0, "ok": False, "result": result})
            return result
        tool = self.tools.get(name)
        if tool is None:
            return {"error": True, "retryable": False, "message": f"未知工具: {name}"}
        t0 = time.time()
        try:
            rv = tool.handler(args or {})
            result = await rv if inspect.isawaitable(rv) else rv
            ok = not (isinstance(result, dict) and result.get("error"))
            entry = {"tool": name, "args": args, "ts": time.time(),
                     "duration": round(time.time() - t0, 2), "ok": ok,
                     "result": result}
        except Exception as e:  # 工具异常也走消息回传，智能体可自行处置
            entry = {"tool": name, "args": args, "ts": time.time(),
                     "duration": round(time.time() - t0, 2), "ok": False,
                     "result": {"error": True, "retryable": False,
                            "message": f"{type(e).__name__}: {e}"}}
        self.call_log.append(entry)
        return entry["result"]

    # ------------------------------------------------------------ LLM 侧
    def openai_tools(self, exclude: set[str] = frozenset()) -> list[dict]:
        from .llm import tool_schema
        hidden = set(exclude) | self.excluded
        return [tool_schema(t.name, t.description, t.schema)
                for n, t in sorted(self.tools.items()) if n not in hidden]
