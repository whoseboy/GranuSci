# -*- coding: utf-8 -*-
"""sci-evo-extract 本地文献工作台客户端：论文全文获取（PDF→MinerU→markdown）。

服务边界：HTTP API（http://127.0.0.1:8000/api），不直接读 SQLite。
用法定位：文献发现用波尔检索（快）；本工具负责"已知要精读哪篇（DOI/题名）
→ 获取全文 markdown"。MinerU 解析为后台 job，acquire 后需轮询 read。
"""
from __future__ import annotations

import json
import urllib.request

BASE = "http://127.0.0.1:8000/api"
ACQUIRE_TIMEOUT = 300      # acquisitions 内部查外部源+注册，慢
READ_TIMEOUT = 60


def _req(path: str, body: dict | None = None, timeout: float = 60) -> dict:
    if body is not None:
        req = urllib.request.Request(
            BASE + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
    else:
        req = urllib.request.Request(BASE + path)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:400]
        return {"error": True, "status": "http_error", "http_status": e.code,
                "detail": detail}
    except Exception as e:
        return {"error": True, "status": "request_failed",
                "detail": f"{type(e).__name__}: {e}"}


def acquire(doi: str | None = None, title: str | None = None) -> dict:
    """获取一篇论文并触发 MinerU 处理（异步 job）。

    返回 paper_id 与处理状态；处理完成前 read 会报 pending。
    """
    if not doi and not title:
        return {"error": True, "message": "需要 doi 或 title 之一"}
    body = {"process": True}
    if doi:
        body["doi"] = doi
    else:
        body["title"] = title
    r = _req("/library/acquisitions", body, timeout=ACQUIRE_TIMEOUT)
    if r.get("error"):
        return r
    return {
        "ok": True,
        "paper_id": r.get("paper_id") or r.get("paper", {}).get("paper_id"),
        "status": r.get("status"),
        "readiness": r.get("readiness") or r.get("paper", {}).get("readiness"),
        "message": "已注册并触发处理。MinerU 解析为后台任务，稍后用 litf_read(paper_id) 取全文；若返回 pending 请等待后再试。",
    }


def read(paper_id: str, max_chars: int = 24000, offset: int = 0) -> dict:
    """取论文 MinerU markdown 全文（截断窗口，可翻页）。

    未处理完成时返回 pending 与当前 readiness 状态。
    """
    arts = _req(f"/library/papers/{paper_id}/artifacts", timeout=READ_TIMEOUT)
    if arts.get("error"):
        return arts
    items = arts.get("items") or []
    # 优先 mineru_extract_dir 中的 .md；退化到 sciverse / pdf
    md_file, art_id = None, None
    for a in items:
        if a.get("kind") != "mineru_extract_dir":
            continue
        for f in (a.get("manifest") or {}).get("files", []):
            if f.get("relative_path", "").endswith(".md"):
                md_file, art_id = f["relative_path"], a["artifact_id"]
                break
        if md_file:
            break
    if not md_file:
        ready = next((a for a in items if a.get("kind") == "pdf" and a.get("status") == "ready"), None)
        if ready is None:
            return {"ok": False, "state": "no_assets",
                    "message": "该论文尚无 PDF/解析资产——先 litf_acquire 触发获取与处理"}
        return {"ok": False, "state": "pending_mineru",
                "message": "PDF 已就绪但 MinerU 解析未完成，稍后重试 litf_read"}
    url = f"{BASE}/library/artifacts/{art_id}/files/{md_file}"
    try:
        with urllib.request.urlopen(url, timeout=READ_TIMEOUT) as r:
            text = r.read().decode("utf-8", errors="replace")
    except Exception as e:
        return {"error": True, "message": f"取文件失败: {e}"}
    total = len(text)
    chunk = text[offset:offset + max_chars]
    return {
        "ok": True, "paper_id": paper_id,
        "total_chars": total, "offset": offset,
        "returned_chars": len(chunk),
        "has_more": offset + max_chars < total,
        "next_offset": offset + max_chars if offset + max_chars < total else None,
        "content": chunk,
        "hint": "分段阅读：用 offset 翻页；读完后引用须标注论文 DOI/题名",
    }
