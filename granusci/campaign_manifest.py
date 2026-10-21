# -*- coding: utf-8 -*-
"""战役 manifest 制度化（A6/V1 前置）：新系统战役的复现性档案。

每个战役（dry run 起）在启动时写 manifest，收笔时补完——它回答
"这次战役在什么系统状态下发生"：
  - 系统版本：granusci/granular-dem-mcp/lit-mcp 的关键文件哈希
  - 模型：agent/reviewer/vision 模型名 + 关键 LLM 参数
  - 服务端点：MCP 服务器地址
  - 数据锚点：设计数据/响应数据的关键文件哈希（收笔时）
  - 假设档案锚点：hypotheses.jsonl 的战役段哈希
写入 runs/<tag>/manifest.json；run 收尾时由 runner 调 finalize。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

WS = Path(__file__).parent.parent   # 工作区根

TRACKED_FILES = [
    # granusci 本体
    "granusci/agent/graph.py",
    "granusci/agent/prompts.py",
    "granusci/hypothesis.py",
    "granusci/llm.py",
    "granusci/config.py",
    # 工具面（本地源码=服务器部署源）
    "granular-dem-mcp/server.py",
    "granular-dem-mcp/core/casegen/renderer.py",
    "granular-dem-mcp/core/casegen/jobs.py",
    "granular-dem-mcp/core/casegen/diagnostics.py",
    "granular-dem-mcp/core/casegen/stl_gen.py",
    "lit-mcp/server.py",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def create(tag: str, run_dir: Path) -> dict:
    """战役启动时写 manifest。"""
    files = {}
    for rel in TRACKED_FILES:
        p = WS / rel
        if p.exists():
            files[rel] = _sha(p)
    import granusci.config as cfg
    manifest = {
        "campaign": tag,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": {
            "file_hashes": files,
            "models": dict(cfg.MODEL_PRESETS),
            "mcp_servers": dict(cfg.MCP_SERVERS),
        },
        "data_anchors": {},       # finalize 时填
        "hypothesis_anchor": None,
        "status": "open",
    }
    out = run_dir / "manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    return manifest


def finalize(run_dir: Path, data_files: list[str] | None = None) -> dict:
    """战役收笔：数据锚点+状态关闭。"""
    out = run_dir / "manifest.json"
    if not out.exists():
        return {"ok": False, "message": "manifest 不存在（未 create？）"}
    m = json.loads(out.read_text(encoding="utf-8"))
    anchors = {}
    for rel in data_files or []:
        p = Path(rel) if Path(rel).is_absolute() else WS / rel
        if p.exists():
            anchors[str(rel)] = _sha(p)
    hyp = WS / "granusci" / "memory" / "research" / "hypotheses.jsonl"
    if hyp.exists():
        m["hypothesis_anchor"] = _sha(hyp)
    m["data_anchors"] = anchors
    m["status"] = "closed"
    m["closed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    return m
