# -*- coding: utf-8 -*-
"""paratera LLM 客户端：chat（含 tool calling）+ embedding + 用量核算。

直接走 OpenAI 兼容协议，不引 langchain 模型层——完整控制超时、重试与消息格式。

重试分级（⑩）：
- transient（408/429/5xx、网络异常、协议解析异常）：指数退避重试，最多 transient_retries 次
- permanent（400/401/403/404 等参数与鉴权类）：快速失败——仅确认重试 1 次
  （个别网关偶发误报 400），再失败即抛 kind="permanent"
- 空响应（content 与 tool_calls 均空）：短暂退避重试 empty_retries 次后原样返回，
  由调用方防线处置（graph 的 empty_response 终态）

用量核算（⑥）：UsageTracker 单例记录每次成功调用的 token 用量，增量落盘
run_dir/llm_usage.jsonl（崩溃也保留已发生部分），run 结束写 usage.json 汇总。
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import httpx

from . import config

# 瞬时故障状态码：网关抖动/限流/上游闪断——值得退避重试
_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 524}


class LLMError(RuntimeError):
    """kind: transient（可退避重试）| permanent（换法子，重试无意义）
    | empty（空响应）| protocol（响应结构异常）"""

    def __init__(self, message: str, kind: str = "unknown"):
        super().__init__(message)
        self.kind = kind


# ================================================================ 用量核算
class UsageTracker:
    """LLM 调用用量台账：逐调用记录 + 增量落盘 + 汇总。

    单进程单 run 使用（run.py 里 configure(run_dir) 绑定产物目录）；
    未绑定时只留在内存，run 结束仍可 summary()。
    """

    def __init__(self):
        self.calls: list[dict] = []
        self._fh = None
        self._path: Path | None = None

    def configure(self, run_dir: Path | str):
        """绑定 run 目录，开启增量落盘（追加模式，断点续跑不覆盖）。"""
        self._path = Path(run_dir) / "llm_usage.jsonl"
        self._fh = self._path.open("a", encoding="utf-8")

    def record(self, model: str, purpose: str, usage: dict | None,
               duration: float, attempts: int):
        entry = {
            "ts": round(time.time(), 2),
            "model": model,
            "purpose": purpose,       # agent / reviewer / deep_read / vision / embedding / memory_distill
            "prompt_tokens": (usage or {}).get("prompt_tokens", 0) or 0,
            "completion_tokens": (usage or {}).get("completion_tokens", 0) or 0,
            # 前缀缓存命中（paratera GLM 隐式缓存：命中部分按折扣计）——
            # 缓存友好压缩架构的收益直接可观测
            "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {})
                             .get("cached_tokens", 0) or 0,
            "duration": round(duration, 2),
            "attempts": attempts,     # 含退避重试在内的总请求次数（成本可观性的一部分）
        }
        entry["total_tokens"] = entry["prompt_tokens"] + entry["completion_tokens"]
        self.calls.append(entry)
        if self._fh is not None:
            try:
                self._fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self._fh.flush()
            except OSError:
                pass  # 落盘失败不拦主流程

    def summary(self) -> dict:
        by_model: dict[str, dict] = {}
        by_purpose: dict[str, dict] = {}
        for e in self.calls:
            for key, agg in (("model", by_model), ("purpose", by_purpose)):
                k = e[key]
                d = agg.setdefault(k, {"calls": 0, "prompt_tokens": 0,
                                       "completion_tokens": 0, "total_tokens": 0,
                                       "cached_tokens": 0})
                d["calls"] += 1
                for f in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"):
                    d[f] += e.get(f, 0)
        return {"total_calls": len(self.calls), "by_model": by_model,
                "by_purpose": by_purpose}

    def dump(self, run_dir: Path | str) -> Path:
        """run 收尾时写汇总 usage.json（finally 里调用，崩溃路径也覆盖）。"""
        path = Path(run_dir) / "usage.json"
        path.write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    def close(self):
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None


usage = UsageTracker()   # 模块级单例：一次 run 一个进程

# 实验 seed（⑫ 消融矩阵用）：设置后注入每次请求体（服务器支持则生效，
# 不支持则被忽略——波动仍来自服务侧采样不确定性）
_seed: int | None = None


def set_seed(seed: int | None):
    global _seed
    _seed = int(seed) if seed is not None else None


# ================================================================ 客户端
def _client(timeout: float = 300, model: str = "") -> httpx.Client:
    """按模型名选端点：国外模型走 Higress 网关，其余走 paratera。"""
    if model and model in config.FOREIGN_MODELS:
        return httpx.Client(
            base_url=config.HIGRESS_BASE_URL,
            headers={"Authorization": f"Bearer {config.HIGRESS_API_KEY}"},
            timeout=httpx.Timeout(timeout, connect=30),
        )
    return httpx.Client(
        base_url=config.PARATERA_BASE_URL,
        headers={"Authorization": f"Bearer {config.PARATERA_API_KEY}"},
        timeout=httpx.Timeout(timeout, connect=30),
    )


def _backoff(attempt: int) -> float:
    """指数退避 + 抖动：2s, 4s, 8s ... 上限 30s。attempt 从 1 起。"""
    return min(30.0, 2.0 ** attempt) * (0.5 + random.random())


def chat(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.1,
    max_tokens: int = 16000,
    purpose: str = "chat",
    transient_retries: int = 4,
    permanent_retries: int = 1,
    empty_retries: int = 2,
) -> dict:
    """一次 chat completion。返回完整 message 对象（含 tool_calls 或 content）。

    空响应（content 与 tool_calls 均空，偶发于超长 tool-call 参数生成中断）
    退避重试 empty_retries 次；仍空则原样返回（调用方有防线）。
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = tools
    if _seed is not None:
        body["seed"] = _seed

    t0 = time.time()
    attempts = transient_seen = permanent_seen = empty_seen = 0
    # GLM-5.3 思考型模型：关思考（2026-09-23 实测 thinking:{type:disabled} 为
    # 唯一有效开关；enable_thinking 系参数无效）。带工具调用的 agent 主循环
    # 生成量大，思考不关则单次 600-1000s 且非流式易挂（C1 一测教训）。
    if "glm-5" in model.lower():
        body["thinking"] = {"type": "disabled"}
    while True:
        attempts += 1
        try:
            with _client(model=model) as c:
                r = c.post("/chat/completions", json=body)
            if r.status_code >= 400:
                # 配额型 429（团队 token 上限耗尽）：重试无意义——等待窗口远超
                # 单次 run 生命周期。与瞬时限流区分（paper12 死因：瞬时限流的
                # 退避策略对配额型只产生 4 次快速重试然后 agent 反复撞墙）
                _t = r.text.lower()
                is_quota = (r.status_code == 429 and (
                    "limit type" in _t or "quota" in _t
                    or "remaining: 0" in _t))
                kind = "quota" if is_quota else (
                    "transient" if r.status_code in _TRANSIENT_STATUS else "permanent")
                if kind == "quota":
                    raise LLMError(
                        f"HTTP 429 配额耗尽（团队 token 上限，等待周期性重置）："
                        f"{r.text[:300]}", kind="quota")
                if kind == "transient":
                    if transient_seen < transient_retries:
                        transient_seen += 1
                        time.sleep(_backoff(transient_seen))
                        continue
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:500]}",
                                   kind="transient")
                # permanent：确认重试一次（个别网关偶发误报 400），再失败即抛
                if permanent_seen < permanent_retries:
                    permanent_seen += 1
                    time.sleep(1)
                    continue
                raise LLMError(f"HTTP {r.status_code}: {r.text[:500]}",
                               kind="permanent")
            data = r.json()
            message = data["choices"][0]["message"]
            if not (message.get("content") or "").strip() and not message.get("tool_calls"):
                empty_seen += 1
                if empty_seen <= empty_retries:
                    time.sleep(2 * empty_seen)
                    continue
                # 空响应重试耗尽：原样返回，调用方防线处置
                usage.record(model, purpose, data.get("usage"),
                             time.time() - t0, attempts)
                return message
            usage.record(model, purpose, data.get("usage"), time.time() - t0, attempts)
            return message
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as e:
            # 网络/超时/响应结构异常——一律按 transient 分级退避
            if transient_seen < transient_retries:
                transient_seen += 1
                time.sleep(_backoff(transient_seen))
                continue
            kind = "protocol" if not isinstance(e, httpx.HTTPError) else "transient"
            raise LLMError(f"chat failed after {attempts} attempts: {e}", kind=kind)


def chat_text(model: str, messages: list[dict], **kw) -> str:
    """便捷封装：只要正文文本。"""
    m = chat(model, messages, **kw)
    return (m.get("content") or "").strip()


def embed(texts: list[str], batch: int = 32) -> list[list[float]]:
    """批量 embedding（GLM-Embedding-3, 2048 维）。"""
    out: list[list[float]] = []
    t0 = time.time()
    with _client(timeout=120) as c:
        for i in range(0, len(texts), batch):
            part = texts[i : i + batch]
            r = c.post("/embeddings", json={"model": config.EMBED_MODEL, "input": part})
            r.raise_for_status()
            data = r.json()
            out.extend(d["embedding"] for d in data["data"])
            usage.record(config.EMBED_MODEL, "embedding", data.get("usage"),
                         time.time() - t0, 1)
    return out


def vision_chat(model: str, image_path: str, question: str,
                max_tokens: int = 4000, purpose: str = "vision") -> str:
    """视觉模型问答：看一张图回答问题（OpenAI 兼容 image_url 格式）。"""
    import base64
    p = Path(image_path)
    if not p.exists():
        raise LLMError(f"图片不存在: {image_path}")
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode()
    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": question},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    t0 = time.time()
    attempts = 0
    last_err: Exception | None = None
    while attempts < 3:   # 瞬时故障最多 3 次尝试；permanent 立即抛
        attempts += 1
        try:
            with _client(timeout=300, model=model) as c:
                r = c.post("/chat/completions", json=body)
            if r.status_code >= 400:
                if r.status_code not in _TRANSIENT_STATUS:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:400]}",
                                   kind="permanent")
                last_err = LLMError(f"HTTP {r.status_code}: {r.text[:400]}",
                                    kind="transient")
                time.sleep(_backoff(1))
                continue
            data = r.json()
            usage.record(model, purpose, data.get("usage"), time.time() - t0, attempts)
            return (data["choices"][0]["message"].get("content") or "").strip()
        except LLMError:
            raise
        except Exception as e:  # 网络层
            last_err = e
            time.sleep(_backoff(1))
    raise LLMError(f"vision_chat failed after {attempts} attempts: {last_err}",
                   kind="transient")


def tool_schema(name: str, description: str, parameters: dict) -> dict:
    """组装 OpenAI function-call 工具描述。"""
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }
