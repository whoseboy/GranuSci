# -*- coding: utf-8 -*-
"""lit-mcp：统一文献层 MCP 服务（A3 重构核心件）。

架构（2026-09-23 定案）：
- 外部检索（主通道）：OpenAlex/Crossref API 直查——每条结果带完整书目元数据
  （DOI/标题/作者/年份/venue），引用可溯源；
- 本地文献库（已处理文献）：代理 sci-evo-extract（:8005）HTTP API——
  检索库内已处理论文与其 evidence units；
- 项目源文档：王驰博士论文等（外部检索不可达的内部资料）——带规范书目
  条目登记于本服务，引用同样可溯源；
- 引用登记（引用门）：cite_register——正式引用先登记拿 cite_id，报告审稿
  时核对 cite_id → 元数据链。

设计纪律：
- 所有返回书目信息的工具，其返回结构即"审计证据"（granusci 数值审计门
  按 cite_id 回查）——b101 引用幻觉问题的结构性修复；
- 不做 LLM 检索代理（那是 lit_survey 子代理的职责），本服务只管元数据与
  检索通道的确定性部分。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from mcp.server.fastmcp import FastMCP

SCIEVO_BASE = os.getenv("SCIEVO_BASE", "http://localhost:8005")
# Sciverse（穹宇，上海AI实验室）——主检索腿（2026-09-24 起，用户提供 API key）。
# agentic-search 语义证据检索 + meta-search 精确元数据；OpenAlex 降为无 key 时兜底。
SCIVERSE_BASE = "https://api.sciverse.space"
SCIVERSE_KEY = os.getenv("SCIVERSE_KEY", "")
OPENALEX_BASE = "https://api.openalex.org"
CROSSREF_BASE = "https://api.crossref.org/works"
# 引用登记台账（审计 B-3：cite_register 曾无持久化、无上游核验——
# 幻觉引用随便补个 title 就能"洗白"成已登记。台账 + 上游比对双门）
REGISTRY_PATH = Path(os.getenv("LIT_REGISTRY_PATH",
                               os.path.expanduser("~/mcp/lit-mcp/registry.jsonl")))

mcp = FastMCP("lit-mcp", host="0.0.0.0", port=int(os.getenv("LIT_MCP_PORT", "8006")))

# 上游检索记录（本进程生命周期内）：search_external/library_search 的
# 返回元数据进本缓存，cite_register 时核验"登记条目确实来自一次真实检索"
_UPSTREAM_HITS: list[dict] = []


def _record_upstream(results: list[dict]) -> None:
    """检索返回的元数据记录进上游缓存（供 cite_register 核验）。"""
    for r in results:
        if isinstance(r, dict) and r.get("title"):
            _UPSTREAM_HITS.append({
                "title": str(r.get("title", ""))[:300],
                "doi": str(r.get("doi", "") or ""),
                "ts": time.time()})


def _upstream_match(bib: dict) -> dict:
    """登记条目与上游检索记录比对：DOI 精确命中 or 标题前 60 字符命中。"""
    doi = str(bib.get("doi", "") or "").lower().strip()
    title = str(bib.get("title", "") or "")[:60].lower()
    for h in _UPSTREAM_HITS:
        if doi and h["doi"].lower() == doi:
            return {"matched": True, "via": "doi"}
        if title and h["title"][:60].lower() == title:
            return {"matched": True, "via": "title"}
    return {"matched": False}


def _append_registry(entry: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------- 源文档登记
SOURCE_DOCS = {
    "wangchi2024": {
        "cite_id": "wangchi2024",
        "type": "thesis",
        "title": "数据驱动量纲分析方法及其在颗粒物质研究中的应用",
        "authors": ["王驰"],
        "year": 2024,
        "venue": "博士学位论文，中国矿业大学（北京）",
        "note": "DDDA 管线协议来源（subclust+FCM/附录A.1.4/式2.17-2.19）；"
                "全文检索经 granusci kb_search（论文章节块）",
        "source": "project_source_document",
    },
    "wangchi_ch2": {
        "cite_id": "wangchi2024_ch2",
        "type": "thesis_chapter",
        "title": "数据驱动量纲分析（第2章）",
        "authors": ["王驰"],
        "year": 2024,
        "venue": "博士学位论文第2章，中国矿业大学（北京）",
        "container": "wangchi2024",
        "note": "DDDA 核心章",
        "source": "project_source_document",
    },
}


@mcp.tool(
    name="search_external",
    description=(
        "外部文献检索（Sciverse 为主，OpenAlex/Crossref 兜底）。两种模式："
        "mode=meta（默认）——精确元数据检索（标题/DOI/作者/年份/venue/被引数，"
        "适合查具体文献与引用登记前取证）；mode=agentic——语义证据检索（返回"
        "文献正文片段+doc_id+页码，适合'谁报道过 X 现象'类问题与缺口挖掘）。"
        "正式引用前必先用本工具（或 cite_register）取得可溯源元数据。"
    ),
)
def search_external(query: str, limit: int = 8, mode: str = "meta") -> dict:
    """Sciverse 主检索（agentic/meta 双模式）→ OpenAlex → Crossref 三级回退。"""
    # ---- 主腿：Sciverse ----
    if SCIVERSE_KEY:
        try:
            if mode == "agentic":
                r = httpx.post(
                    f"{SCIVERSE_BASE}/agentic-search",
                    json={"query": query, "top_k": min(limit, 50)},
                    headers={"Authorization": f"Bearer {SCIVERSE_KEY}"},
                    timeout=40)
                r.raise_for_status()
                hits = r.json().get("hits", [])
                out = []
                for h in hits[:limit]:
                    out.append({
                        "title": h.get("title", ""),
                        "authors": h.get("author", [])[:6],
                        "year": h.get("publication_published_year"),
                        "venue": h.get("publication_venue_name_unified", ""),
                        "doi": (h.get("doi") or ""),
                        "doc_id": h.get("doc_id", ""),
                        "cited_by_count": h.get("citation_count", 0),
                        "chunk": (h.get("chunk") or "")[:400],
                        "page_no": h.get("page_no"),
                    })
                _record_upstream(out)
                return {"ok": True, "source": "sciverse-agentic",
                        "n": len(out), "results": out,
                        "note": "语义证据检索——chunk 为正文片段可作主张证据；"
                                "doc_id 可传给 read_fulltext 读原文"}
            else:
                r = httpx.post(
                    f"{SCIVERSE_BASE}/meta-search",
                    json={"query": query, "page": 1, "page_size": min(limit, 20),
                          "fields": ["title", "doi", "publication_published_year",
                                     "publication_venue_name_unified",
                                     "citation_count", "author"]},
                    headers={"Authorization": f"Bearer {SCIVERSE_KEY}"},
                    timeout=30)
                r.raise_for_status()
                items = r.json().get("results", [])
                out = []
                for it in items[:limit]:
                    out.append({
                        "title": it.get("title", ""),
                        "authors": (it.get("author") or [])[:6],
                        "year": it.get("publication_published_year"),
                        "venue": it.get("publication_venue_name_unified", ""),
                        "doi": it.get("doi", ""),
                        "doc_id": it.get("doc_id", ""),
                        "cited_by_count": it.get("citation_count", 0),
                    })
                _record_upstream(out)
                return {"ok": True, "source": "sciverse-meta",
                        "n": len(out), "results": out}
        except Exception as e:  # noqa: BLE001  Sciverse 失败→OpenAlex 兜底
            _sciverse_err = str(e)[:120]
        else:
            _sciverse_err = ""
    else:
        _sciverse_err = "no key"
    # ---- 兜底腿 1：OpenAlex ----
    try:
        r = httpx.get(
            f"{OPENALEX_BASE}/works",
            params={"search": query, "per-page": min(limit, 15), "mailto": "research@example.org"},
            timeout=20)
        r.raise_for_status()
        works = r.json().get("results", [])
        out = []
        for w in works:
            venue = ""
            if w.get("primary_location") and w["primary_location"].get("source"):
                venue = w["primary_location"]["source"].get("display_name", "") or ""
            out.append({
                "title": w.get("display_name") or "",
                "authors": [a["author"]["display_name"]
                            for a in (w.get("authorships") or [])[:6]],
                "year": w.get("publication_year"),
                "venue": venue,
                "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                "openalex_id": (w.get("id") or "").replace("https://openalex.org/", ""),
                "cited_by_count": w.get("cited_by_count", 0),
            })
        _record_upstream(out)
        return {"ok": True, "source": "openalex", "n": len(out), "results": out,
                **({"sciverse_fallback_reason": _sciverse_err} if _sciverse_err else {})}
    except Exception as e:  # noqa: BLE001
        # Crossref 兜底
        try:
            r = httpx.get(f"{CROSSREF_BASE}",
                          params={"query.bibliographic": query,
                                  "rows": min(limit, 10)}, timeout=20)
            r.raise_for_status()
            items = r.json().get("message", {}).get("items", [])
            out = []
            for it in items:
                out.append({
                    "title": (it.get("title") or [""])[0],
                    "authors": [f"{a.get('given','')} {a.get('family','')}".strip()
                                for a in (it.get("author") or [])[:6]],
                    "year": (it.get("issued", {}).get("date-parts") or [[None]])[0][0],
                    "venue": (it.get("container-title") or [""])[0],
                    "doi": it.get("DOI", ""),
                    "source": "crossref",
                })
            _record_upstream(out)
            return {"ok": True, "source": "crossref", "n": len(out), "results": out}
        except Exception as e2:  # noqa: BLE001
            return {"ok": False, "message": f"OpenAlex/Crossref 均失败: {e} / {e2}"}


@mcp.tool(
    name="library_search",
    description=(
        "检索本地已处理文献库（sci-evo-extract）：已入库论文的元数据/处理状态/"
        "证据单元。本地库是'精读过'的文献（有 evidence units），适合机制对比"
        "与深度引用；外部检索（search_external）是'发现'通道。"
    ),
)
def library_search(query: str = "", limit: int = 10) -> dict:
    try:
        r = httpx.get(f"{SCIEVO_BASE}/api/library/papers",
                      params={"q": query, "limit": limit}, timeout=15)
        r.raise_for_status()
        _j = r.json()
        if isinstance(_j.get("papers"), list):
            _record_upstream(_j["papers"])
        elif isinstance(_j.get("results"), list):
            _record_upstream(_j["results"])
        return {"ok": True, **_j}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": f"sci-evo-extract 不可达: {e}"}


@mcp.tool(
    name="library_evidence",
    description=(
        "取本地库中某篇已处理论文的证据单元（Evidence Units：正文/表格/图的"
        "结构化证据，含页码与出处）——深度引用与机制对比的素材。"
    ),
)
def library_evidence(paper_id: str) -> dict:
    try:
        r = httpx.get(f"{SCIEVO_BASE}/api/library/papers/{paper_id}/evidence",
                      timeout=15)
        r.raise_for_status()
        return {"ok": True, **r.json()}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": str(e)[:150]}


@mcp.tool(
    name="get_source_doc",
    description=(
        "取项目源文档的书目条目（外部检索不可达的内部资料：王驰博士论文等）。"
        "引用这些文档必须用本工具返回的 cite_id 与条目。"
    ),
)
def get_source_doc(cite_id: str = "") -> dict:
    if cite_id:
        doc = SOURCE_DOCS.get(cite_id)
        return {"ok": bool(doc), "doc": doc}
    return {"ok": True, "docs": list(SOURCE_DOCS.values())}


@mcp.tool(
    name="read_fulltext",
    description=(
        "读取文献原文（Sciverse content 接口）：用 search_external 返回的 doc_id "
        "分段读取全文文本。offset/limit 按 Unicode 字符计（默认读 4000 字符/次，"
        "长文用返回的 next_offset 续读）。适合引用核验（'文献被赋予的主张它真的"
        "包含吗'）与精确定位公式/表格/数字。返回含 doc_id 与 offset 范围——"
        "引用原文片段时须保留这些来源信息。"
    ),
)
def read_fulltext(doc_id: str, offset: int = 0, limit: int = 4000) -> dict:
    try:
        r = httpx.get(
            f"{SCIVERSE_BASE}/content",
            params={"doc_id": doc_id, "offset": offset, "limit": limit},
            headers={"Authorization": f"Bearer {SCIVERSE_KEY}"},
            timeout=40)
        r.raise_for_status()
        d = r.json()
        text = d.get("content") or d.get("text") or ""
        return {"ok": True, "doc_id": doc_id, "offset": offset,
                "chars": len(text), "next_offset": d.get("next_offset"),
                "more": d.get("more"), "text": text}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": f"content 读取失败: {e}"}


@mcp.tool(
    name="cite_register",
    description=(
        "引用登记（引用门核心）：提交一条拟正式引用的书目元数据，校验完整性、"
        "核验上游来源、写入持久台账，返回 cite_id。**上游核验（B-3 修复）**：登记"
        "条目必须能对上一次真实检索记录（search_external/library_search 的返回）"
        "或源文档条目（get_source_doc）——DOI 或标题前 60 字符匹配。完全凭空"
        "编造的书目（幻觉引用）无法通过本工具登记。报告中的每个 \\cite 必须能"
        "对应到一次本工具登记——无登记的引用会被审稿门拒收。"
    ),
)
def cite_register(bib: dict | str) -> dict:
    try:
        if isinstance(bib, str):
            bib = json.loads(bib)
        required = ("title", "authors", "year")
        missing = [k for k in required if not bib.get(k)]
        if missing:
            return {"ok": False, "message": f"书目缺必填字段: {missing}"}
        # 上游核验：真实检索过才许登记（幻觉引用拦截）
        up = _upstream_match(bib)
        if not up["matched"]:
            # 源文档豁免：get_source_doc 的条目允许直接登记（内部资料无外部检索）
            src_hit = SOURCE_DOCS.get(bib.get("cite_id", "")) or any(
                str(d.get("title", ""))[:60].lower() == str(bib.get("title", ""))[:60].lower()
                for d in SOURCE_DOCS.values())
            if not src_hit:
                return {
                    "ok": False,
                    "message": ("上游核验失败：本条书目未出现在任何检索记录"
                                "（search_external/library_search）或源文档中——"
                                "幻觉引用不得登记。请先检索取得真实元数据，"
                                "再以检索返回原样登记。")}
        cite_id = bib.get("doi") or bib.get("cite_id") or \
            (bib["title"][:24].replace(" ", "_").lower())
        entry = {"cite_id": cite_id, "bib": bib, "ts": time.time(),
                 "upstream": up}
        _append_registry(entry)
        return {"ok": True, "cite_id": cite_id,
                "registered": bib,
                "note": "登记成功（已核验上游来源并写入持久台账）——报告引用处使用本 cite_id"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": str(e)[:150]}


@mcp.tool(
    name="cite_lookup",
    description=(
        "引用台账查询：按 cite_id 查登记条目，或列全部登记（cite_id 不给时）。"
        "审稿侧核对 \\cite → 台账链用；也供 agent 写 bibliography 前自查。"
    ),
)
def cite_lookup(cite_id: str = "") -> dict:
    try:
        if not REGISTRY_PATH.exists():
            return {"ok": True, "entries": []}
        entries = []
        for line in REGISTRY_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                if not cite_id or e.get("cite_id") == cite_id:
                    entries.append(e)
        if cite_id and not entries:
            return {"ok": False, "message": f"cite_id 未登记: {cite_id}"}
        return {"ok": True, "n": len(entries), "entries": entries}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": str(e)[:150]}


if __name__ == "__main__":
    uvicorn.run(mcp.streamable_http_app(), host="0.0.0.0",
                port=int(os.getenv("LIT_MCP_PORT", "8006")))
