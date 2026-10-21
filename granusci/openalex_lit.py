# -*- coding: utf-8 -*-
"""OpenAlex 文献检索（经 sci-evo-extract discovery 管道）——波尔的免费备用腿。

返回论文列表（标题/作者年份venue/DOI/引用数/OA），AI 综合由智能体自己完成。
"""
from __future__ import annotations

import json
import urllib.request

BASE = "http://127.0.0.1:8000/api/library/discovery/search"


def search(query: str, limit: int = 10) -> dict:
    req = urllib.request.Request(
        BASE, data=json.dumps({"query": query, "limit": max(limit * 3, 20)}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return {"error": True, "message": f"openalex 检索失败: {type(e).__name__}: {e}"}

    seen, papers = set(), []
    for c in data.get("source_candidates") or []:
        t = c.get("title")
        if not t or t.lower() in seen:
            continue
        seen.add(t.lower())
        papers.append({
            "title": t, "year": c.get("year"), "venue": c.get("venue"),
            "doi": c.get("normalized_doi"), "citations": c.get("citation_count"),
            "is_oa": c.get("is_oa"),
        })
        if len(papers) >= limit:
            break
    if not papers:
        return {"message": "openalex 无结果", "papers": []}
    return {"papers": papers,
            "hint": "结果为元数据（无摘要）。综合判断由你自己完成：按标题/期刊/引用数"
                    "评估相关性；需精读的用 litf_acquire(doi) 取全文。"}
