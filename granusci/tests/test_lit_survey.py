# -*- coding: utf-8 -*-
"""② 批量文献调研子代理测试：全 mock（LLM/OpenAlex/sci-evo），验证隔离性与降级。"""
from __future__ import annotations

import json

import pytest

from granusci import lit_survey as ls


PAPERS_A = [
    {"title": "Size segregation of granular material", "year": 2020,
     "venue": "Powder Tech", "doi": "10.1/a", "citations": 50},
    {"title": "Brazil nut effect review", "year": 2019,
     "venue": "Granular Matter", "doi": "10.1/b", "citations": 200},
]
PAPERS_B = [
    {"title": "Brazil nut effect review", "year": 2019,          # 与 A 重复（DOI 相同）
     "venue": "Granular Matter", "doi": "10.1/b", "citations": 200},
    {"title": "Convection rolls in vibrated beds", "year": 2022,
     "venue": "PRE", "doi": "10.1/c", "citations": 10},
]


@pytest.fixture
def mocks(monkeypatch):
    calls = {"chat": [], "search": [], "acquire": [], "read": []}

    def chat_text(model, messages, **kw):
        calls["chat"].append(messages[0]["content"][:20])
        sys = messages[0]["content"]
        if "检索策略设计器" in sys:
            return "segregation phenomenon vibrated beds\nconvection mechanism granular\nscaling law size ratio\nreview brazil nut effect\ndiscrete element method segregation"
        if "文献筛选器" in sys:
            return '{"selected": [0, 1, 3]}'      # 池内序号（含跨查询去重后的第4篇）
        if "文献精读子代理" in sys:
            return "摘要：本篇用 DEM 研究偏析，mix_index 收敛判据 L=0.05。"
        if "文献调研综合器" in sys:
            return "## 研究现状\n[1] 研究了…\n## 研究空白\n尚无人做过 X。"
        return "?"

    def search(query, limit=10):
        calls["search"].append(query)
        return {"papers": PAPERS_A if query == "segregation phenomenon vibrated beds" else PAPERS_B}

    def acquire(doi=None, title=None):
        calls["acquire"].append(doi or title)
        if doi == "10.1/b":
            return {"error": True, "detail": "acquisition service unavailable"}
        return {"ok": True, "paper_id": f"pid_{doi[-1]}", "status": "done"}

    def read(paper_id, max_chars=24000, offset=0):
        calls["read"].append((paper_id, offset))
        if offset == 0:
            return {"ok": True, "total_chars": 30000, "content": "正文第一段……"}
        return {"ok": True, "total_chars": 30000, "content": "正文第二段……"}

    monkeypatch.setattr(ls.llm, "chat_text", chat_text)
    monkeypatch.setattr(ls.openalex_lit, "search", search)
    monkeypatch.setattr(ls.scievo, "acquire", acquire)
    monkeypatch.setattr(ls.scievo, "read", read)
    return calls


def test_full_survey_flow(mocks):
    r = ls.run_survey("granular segregation", "BNE 机制与判据",
                      max_papers=4, deep_read=2, log_fn=lambda m: None)
    assert r["ok"] is True
    # 检索按 LLM 生成的查询变体执行（去重后池 3 篇）
    assert mocks["search"][0] == "segregation phenomenon vibrated beds"
    assert r["pool_size"] == 3
    # 选篇走了 LLM 选择
    assert r["n_selected"] <= 3
    # 精读：按 DOI 优先取 2 篇，其中 10.1/b 获取失败如实记录
    assert len(mocks["acquire"]) == 2
    assert any("失败" in (f.get("note") or "") or f.get("note") for f in r["deep_read_failed"])
    # 综述为最终综合产物
    assert "研究现状" in r["survey"]
    # 主上下文经济性：结果里没有原始检索结果/全文（只有综述+元数据清单）
    body = json.dumps(r, ensure_ascii=False)
    assert "正文第一段" not in body


def test_deep_read_zero(mocks):
    r = ls.run_survey("topic", "q", max_papers=4, deep_read=0, log_fn=lambda m: None)
    assert r["ok"] is True
    assert mocks["acquire"] == []            # 不触发任何全文获取
    assert r["deep_read_ok"] == []
    assert "研究现状" in r["survey"]         # 元数据级综述仍然生成


def test_all_search_fail(monkeypatch):
    monkeypatch.setattr(ls.llm, "chat_text",
                        lambda *a, **kw: "q1\nq2")
    monkeypatch.setattr(ls.openalex_lit, "search",
                        lambda q, limit=10: {"error": True, "message": "服务未启动"})
    r = ls.run_survey("t", "q", log_fn=lambda m: None)
    assert r.get("ok") is False and "sci-evo-extract" in r["message"]


def test_selection_llm_fail_fallback_citation_sort(mocks, monkeypatch):
    """选篇 LLM 失败 → 按引用数退化排序，调研不中断。"""
    real = ls.llm.chat_text

    def chat_text(model, messages, **kw):
        sys = messages[0]["content"]
        if "文献筛选器" in sys:
            raise ls.llm.LLMError("HTTP 500", kind="transient")
        return real(model, messages, **kw)

    monkeypatch.setattr(ls.llm, "chat_text", chat_text)
    r = ls.run_survey("t", "q", max_papers=2, deep_read=0, log_fn=lambda m: None)
    assert r["ok"] is True
    # 引用数最高的排前：Brazil nut effect review (200) 在第一位
    assert r["papers"][0]["title"] == "Brazil nut effect review"


def test_synthesis_fail_degrades_to_metadata(mocks, monkeypatch):
    real = ls.llm.chat_text

    def chat_text(model, messages, **kw):
        sys = messages[0]["content"]
        if "文献调研综合器" in sys:
            raise ls.llm.LLMError("HTTP 502", kind="transient")
        return real(model, messages, **kw)

    monkeypatch.setattr(ls.llm, "chat_text", chat_text)
    r = ls.run_survey("t", "q", max_papers=3, deep_read=0, log_fn=lambda m: None)
    assert r["ok"] is True
    assert "元数据清单" in r["survey"]      # 降级而非失败
    assert len(r["papers"]) >= 1


def test_read_pending_recorded(mocks, monkeypatch):
    """全文未就绪（MinerU pending）：记录 paper_id 供后续 litf_read 重试。"""
    real = mocks  # noqa
    monkeypatch.setattr(ls.scievo, "read",
                        lambda pid, max_chars=24000, offset=0:
                        {"ok": False, "state": "pending_mineru",
                         "message": "MinerU 解析未完成"})
    r = ls.run_survey("t", "q", max_papers=3, deep_read=1, log_fn=lambda m: None)
    assert r["ok"] is True
    failed = r["deep_read_failed"]
    assert failed and "litf_read" in failed[0]["note"]


def test_bohrium_gaps_from_running_event_loop(monkeypatch, capsys):
    """paper12 实跑回归：波尔腿从运行中的事件循环内调用不得炸
    RuntimeError: asyncio.run() cannot be called from a running event loop。
    （死端口即可：走完线程路径，连接失败→优雅返回 None。）
    """
    import asyncio
    from granusci import config
    monkeypatch.setattr(config, "MCP_SERVERS",
                        {"lit": "http://127.0.0.1:1/mcp"})

    async def call_in_loop():
        return ls._bohrium_gaps("topic", "q1 q2")

    r = asyncio.run(call_in_loop())
    err = capsys.readouterr().err
    assert r is None                       # 连接失败→降级 None（不抛出）
    assert "running event loop" not in err  # 旧 bug 的特征串不得再现
    # 可再次调用（线程池正常收放）
    r2 = asyncio.run(call_in_loop())
    assert r2 is None
