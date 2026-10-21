# -*- coding: utf-8 -*-
"""知识库：王驰博士论文 + DDDA 方法论 + 工具契约 的 RAG 检索。

构建：python -m granusci.kb build
查询：kb_search 工具（智能体自主调用）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import chromadb

from . import config, llm

COLLECTION = "granusci_kb"
CHUNK_CHARS = 900       # 目标块长（字符）
OVERLAP = 120


# ---------------------------------------------------------------- chunking
def _split_markdown(text: str) -> list[dict]:
    """按标题层级切块，保留章节路径作为元数据。"""
    lines = text.splitlines()
    chunks: list[dict] = []
    heading_stack: list[tuple[int, str]] = []   # (level, title)
    buf: list[str] = []
    buf_heading = ""

    def flush():
        nonlocal buf
        body = "\n".join(buf).strip()
        buf = []
        if len(body) < 40:
            return
        # 过长块二次切分（带重叠）
        for i in range(0, len(body), CHUNK_CHARS - OVERLAP):
            piece = body[i : i + CHUNK_CHARS]
            if len(piece) < 60:
                break
            chunks.append({"section": buf_heading, "text": piece})

    for line in lines:
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, m.group(2).strip()))
            buf_heading = " > ".join(h for _, h in heading_stack)
        else:
            buf.append(line)
    flush()
    return chunks


# ---------------------------------------------------------------- build
def build(blind: bool = False) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sources = config.KB_SOURCES_BLIND if blind else config.KB_SOURCES
    kb_dir = config.KB_DIR_BLIND if blind else config.KB_DIR
    docs: list[dict] = []
    for rel in sources:
        path = config.ROOT / rel
        text = path.read_text(encoding="utf-8")
        # MinerU 输出常夹图片占位与连续空行，先清一遍
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        pieces = _split_markdown(text)
        print(f"[kb] {rel}: {len(pieces)} chunks")
        for p in pieces:
            docs.append({"source": rel, "section": p["section"], "text": p["text"]})

    print(f"[kb] embedding {len(docs)} chunks ...")
    vectors = llm.embed([d["text"] for d in docs])

    client = chromadb.PersistentClient(path=str(kb_dir))
    try:
        client.delete_collection(COLLECTION)   # 全量重建
    except Exception:
        pass                                   # 首次构建时不存在
    col = client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"})
    for i, (d, v) in enumerate(zip(docs, vectors)):
        col.add(
            ids=[f"chunk_{i}"],
            embeddings=[v],
            documents=[d["text"]],
            metadatas=[{"source": d["source"], "section": d["section"]}],
        )
    print(f"[kb] done: {col.count()} chunks in {kb_dir}")


# ---------------------------------------------------------------- query
def _collection():
    client = chromadb.PersistentClient(path=str(config.KB_DIR))
    return client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"})


def search(query: str, k: int = 5) -> list[dict]:
    vec = llm.embed([query])[0]
    res = _collection().query(query_embeddings=[vec], n_results=min(k, 32))
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, "source": meta["source"],
                     "section": meta["section"], "distance": round(float(dist), 3)})
    return hits


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "build":
        build(blind=len(sys.argv) >= 3 and sys.argv[2] == "blind")
    else:
        print("usage: python -m granusci.kb build [blind]")
