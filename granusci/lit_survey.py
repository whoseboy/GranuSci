# -*- coding: utf-8 -*-
"""② 批量文献调研子代理：lit_survey 工具的后端。

与 deep_read_paper 同一设计哲学——隔离上下文。文献调研天然是多轮爆炸的：
一次调研 = 3~5 次检索 × 每次几 KB 结果 + N 篇选读。全放主上下文就是
paper10 式污染的温和版。本子代理在一次性会话里完成：

  LLM① 生成 3~5 个检索查询变体（不同关键词角度）
  →   OpenAlex 逐查询检索（直接函数调用，不经主上下文）
  →   LLM② 从合并去重结果中选出最相关 max_papers 篇（JSON）
  →   top deep_read 篇逐个 litf_acquire + 分段读 + LLM③ 每篇结构化摘要
  →   LLM④ 综合成结构化调研报告（含每篇 DOI，供引用与查新）

主上下文只收最终综述。全链失败时诚实降级（返回元数据综述或失败说明）。
"""
from __future__ import annotations

import json
from typing import Callable

from . import config, llm, openalex_lit, scievo

READ_CHUNK = 20000


def _q_variants(topic: str, questions: str, model: str | None) -> list[str]:
    """LLM①：生成检索查询变体（多角度检索策略）。失败时退化为确定性变体。

    paper11 教训：单一主题直译+机械后缀导致候选池里没有核心领域文献——
    检索面必须按研究线拆开（现象/机制/定量/方法各一路独立关键词）。
    fallback 也不用"主题+后缀"（paper12 观察到 fallback 仍走旧模式）：
    长主题截核心词，各角度用领域惯用独立关键词组合。
    """
    # 长主题截核心：取前 8 个词作锚（检索引擎对长句召回差）
    core = " ".join(topic.split()[:8])
    fallback = [core,
                f"granular materials segregation mechanism",
                f"binary mixture segregation scaling law",
                f"vibrated granular bed review",
                f"discrete element simulation segregation"]
    try:
        raw = llm.chat_text(model or config.MODEL_PRESETS["agent"], [
            {"role": "system",
             "content": "你是学术文献检索策略设计器。针对研究主题生成 5 个互补的英文检索查询，"
                        "每个查询走不同的研究线，用该线惯用的英文关键词（不要都从主题句直译）：\n"
                        "1. 现象/问题本身（领域惯用术语）\n"
                        "2. 机制/理论解释线\n"
                        "3. 定量关系/标度律/模型线\n"
                        "4. 综述或经典文献线\n"
                        "5. 方法/实验技术线\n"
                        "每行一个查询，不要编号。"},
            {"role": "user", "content": f"主题：{topic}\n关注问题：{questions}"}],
            temperature=0.3, max_tokens=400, purpose="lit_survey")
        qs = [q.strip() for q in raw.splitlines() if q.strip()][:6]
        # 质量闸：若所有查询都近似同一前缀（生成器退化），换 fallback
        if qs and len({q.split()[0].lower() for q in qs}) >= 3:
            return qs
        return fallback
    except Exception:
        return fallback


def _select_papers(topic: str, questions: str, pool: list[dict],
                   max_papers: int, model: str | None) -> list[dict]:
    """LLM②：相关性选篇。失败时按引用数退化排序。"""
    if not pool:
        return []
    meta = [{"i": i, "title": p.get("title"), "year": p.get("year"),
             "venue": p.get("venue"), "doi": p.get("doi"),
             "citations": p.get("citations")} for i, p in enumerate(pool)]
    try:
        raw = llm.chat_text(model or config.MODEL_PRESETS["agent"], [
            {"role": "system",
             "content": "你是文献筛选器。从候选列表中选出与主题**真正同领域**的论文（"
                        "按相关性降序）。领域判定优先于引用数——纳米材料/生物/化学等"
                        "仅分享关键词的跨领域论文一律排除，即使被引很高。"
                        "没有足够相关的就少选甚至不选（宁缺毋滥）。"
                        "输出 JSON: {\"selected\": [候选序号 i, ...]}，最多 "
                        f"{max_papers} 篇。只输出 JSON。"},
            {"role": "user",
             "content": f"主题：{topic}\n关注问题：{questions}\n\n候选：\n"
                        + json.dumps(meta, ensure_ascii=False)}],
            temperature=0.1, max_tokens=500, purpose="lit_survey")
        m = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        idx = [int(i) for i in (m.get("selected") or [])
               if isinstance(i, (int, str)) and str(i).lstrip("-").isdigit()
               and 0 <= int(i) < len(pool)]
        return [pool[i] for i in idx][:max_papers]
    except Exception:
        return sorted(pool, key=lambda p: -(p.get("citations") or 0))[:max_papers]


def _read_one_paper(paper: dict, questions: str, model: str | None) -> dict:
    """单篇获取+精读摘要。任何失败都如实记录，不中断调研。"""
    doi = paper.get("doi")
    title = paper.get("title") or ""
    acq = scievo.acquire(doi=doi, title=None if doi else title)
    if not acq.get("ok"):
        return {"paper": paper, "read": False,
                "note": f"获取失败: {str(acq.get('message') or acq.get('detail'))[:150]}"}
    pid = acq.get("paper_id")
    first = scievo.read(pid, max_chars=READ_CHUNK, offset=0)
    if not first.get("ok"):
        return {"paper": paper, "read": False, "paper_id": pid,
                "note": f"全文未就绪（{str(first.get('message') or first.get('state'))[:100]}）——"
                        f"稍后可用 litf_read(paper_id) 重试"}
    total = first.get("total_chars", 0)
    n_seg = max(1, (total + READ_CHUNK - 1) // READ_CHUNK)
    conv = [
        {"role": "system",
         "content": "你是文献精读子代理。分段接收论文全文，最后按问题输出结构化摘要"
                    "（含关键数值与其所在章节）。接收分段时不要输出。"},
        {"role": "user", "content": f"精读问题：{questions}\n\n[分段 1/{n_seg}]\n{first['content']}"}]
    for k in range(1, min(n_seg, 12)):          # 精读上限 12 段（~240KB）
        seg = scievo.read(pid, max_chars=READ_CHUNK, offset=k * READ_CHUNK)
        if not seg.get("ok"):
            break
        conv.append({"role": "user", "content": f"[分段 {k + 1}/{n_seg}]\n{seg['content']}"})
    conv.append({"role": "user",
                 "content": "全文发送完毕（或达段数上限）。输出结构化摘要回答精读问题（中文，含关键数值）。"})
    try:
        summary = llm.chat_text(model or config.MODEL_PRESETS["agent"], conv,
                                temperature=0.1, purpose="lit_survey")
    except Exception as e:
        return {"paper": paper, "read": False, "paper_id": pid,
                "note": f"摘要生成失败: {type(e).__name__}: {str(e)[:120]}"}
    return {"paper": paper, "read": True, "paper_id": pid, "summary": summary}


def _bohrium_gaps(topic: str, questions: str) -> dict | None:
    """波尔文献检索（gaps 模式的第二腿）：AI 综合回答 + 相关论文。

    波尔有光子时返回 answer（机制/定量结论的综述性回答）——比 OpenAlex
    纯元数据多一层"领域已知道什么"的语义，是缺口挖掘的关键信息源。
    失败返回 None（降级到纯 OpenAlex 池）。
    """
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    from contextlib import AsyncExitStack
    import asyncio
    from . import config as _cfg

    async def _call():
        stack = AsyncExitStack()
        try:
            # 波尔 AI 综合检索慢（分钟级）——read timeout 给足
            cm = streamablehttp_client(_cfg.MCP_SERVERS["lit"], timeout=120,
                                       sse_read_timeout=600)
            reads, writes, _ = await stack.enter_async_context(cm)
            session = await stack.enter_async_context(ClientSession(reads, writes))
            await session.initialize()
            res = await session.call_tool("literature_search", arguments={
                "mode": "auto", "question": questions[:400] or topic})
            texts = [b.text for b in (res.content or []) if getattr(b, "type", "") == "text"]
            import json as _json
            return _json.loads(texts[0] if len(texts) == 1 else "\n".join(texts))
        finally:
            try:
                await stack.aclose()
            except Exception:
                pass

    try:
        # 本函数从同步工具 handler 调用，而调用方处于运行中的事件循环里——
        # 直接 asyncio.run 会炸（RuntimeError: cannot be called from a running
        # event loop，paper12 实跑暴露）。挪到独立线程跑（线程内无事件循环）。
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, _call()).result(timeout=700)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        # anyio 取消以 CancelledError（BaseException，绕过 except Exception）
        # 传播——与 McpBus 同款坑，必须捕 BaseException
        import sys as _sys
        print(f"[lit_survey] 波尔腿异常: {type(e).__name__}: {str(e)[:120]}",
              file=_sys.stderr)
        return None


GAP_QUESTIONS = (
    "1) 该论文自述的局限性（Limitations）逐条列出，含原文依据 "
    "2) 作者提出的未来方向（Future Work / Outlook）是什么？ "
    "3) 哪些结论带'需要进一步验证/仅在 XX 条件下成立'的限定？ "
    "4) 作者认为哪些现象缺少解释或机制不明？")


def run_survey(topic: str, questions: str, max_papers: int = 8,
               deep_read: int = 2, model: str | None = None,
               mode: str = "survey",
               log_fn: Callable[[str], None] | None = None) -> dict:
    """批量文献调研主流程。任何一层失败都降级，不抛出。

    model：子代理使用的 LLM（默认全局 agent 预设）。模型消融实验时须与
    主智能体同模型——否则子代理仍是 SOTA，消融被污染。
    mode="gaps"：缺口挖掘模式——精读问题自动指向 Limitations/Future Work，
    综述结构改为缺口清单（gap-mining 技能的管线后端）。
    """
    if mode == "gaps":
        if not questions or questions == topic:
            questions = GAP_QUESTIONS

    def log(m):
        (log_fn or print)(f"[lit_survey] {m}")

    queries = _q_variants(topic, questions, model)
    log(f"检索查询 {len(queries)} 个: {queries}")
    pool, seen = [], set()
    search_errors = []
    for q in queries:
        try:
            r = openalex_lit.search(q, limit=10)
        except Exception as e:
            search_errors.append(f"{q}: {type(e).__name__}")
            continue
        if r.get("error"):
            search_errors.append(f"{q}: {str(r.get('message'))[:80]}")
            continue
        for p in r.get("papers") or []:
            key = (p.get("doi") or p.get("title") or "").lower()
            if key and key not in seen:
                seen.add(key)
                pool.append(p)
    if not pool and search_errors:
        return {"ok": False, "error": True,
                "message": "全部检索失败: " + "; ".join(search_errors[:3])
                           + "。检查本地文献工作台（sci-evo-extract）是否在运行，"
                             "或改用 lit_literature_search（波尔）。"}
    log(f"候选池 {len(pool)} 篇（去重后）")

    # 波尔腿（gaps 模式优先）：AI 综合回答是缺口挖掘的关键语义源——
    # 含机制/定量结论的综述性回答，比 OpenAlex 纯元数据多一层"领域已知道什么"
    bohrium_answer = None
    if mode == "gaps":
        log("波尔 AI 综合检索中…")
        br = _bohrium_gaps(topic, questions)
        if br and br.get("ok") and br.get("answer"):
            bohrium_answer = str(br["answer"])
            log(f"波尔回答 {len(bohrium_answer)} 字符")
            for p in (br.get("papers") or []):
                key = (p.get("doi") or p.get("title") or "").lower()
                if key and key not in {pp.get("doi") or pp.get("title") or "" for pp in pool}:
                    pool.append({"title": p.get("title"), "year": p.get("year"),
                                 "venue": p.get("journal"), "doi": p.get("doi"),
                                 "citations": p.get("citations")})
            log(f"波尔论文并入后 {len(pool)} 篇")
        else:
            log("波尔不可用（无光子/服务断），降级纯 OpenAlex 池")

    # 领域硬预过滤（paper12 前实测：选篇 LLM 仍会被高被引跨领域论文锚定——
    # ZnO 纳米综述被引 2455 排第一进了精读。标题/期刊关键词级筛除兜底）
    def _domain_filter(p, topic):
        t = (str(p.get("title") or "") + " " + str(p.get("venue") or "")).lower()
        # 主题核心词（任一命中即保留候选资格）
        topic_words = [w for w in topic.lower().split() if len(w) > 3][:4]
        if not any(w in t for w in topic_words):
            # 核心词全不命中 → 看领域黑名单词
            blacklist = ["nanoparticle", "nano-material", "zinc oxide", "chromatin",
                         "biolog", "medical", "clinical", "drug", "cancer", "gene",
                         "polymer synthesis", "catalys", "electrode", "battery"]
            if any(b in t for b in blacklist):
                return False
        return True
    pool = [p for p in pool if _domain_filter(p, topic)]
    log(f"领域过滤后 {len(pool)} 篇")

    selected = _select_papers(topic, questions, pool, max_papers, model)
    if not selected:
        return {"ok": False, "error": True, "message": "候选池为空，无法选篇"}

    # 精读 top-k（有 DOI 的优先——acquire 快）
    deep = []
    with_doi = [p for p in selected if p.get("doi")]
    for p in with_doi[:max(0, deep_read)]:
        log(f"精读: {p.get('title', '')[:60]}")
        deep.append(_read_one_paper(p, questions, model))

    read_ok = [d for d in deep if d.get("read")]
    read_fail = [d for d in deep if not d.get("read")]

    # LLM④ 综合（失败重试一次——paper11 教训：静默失败只剩元数据兜底）
    meta_brief = "\n".join(
        f"- {p.get('title')} ({p.get('year')}) [{p.get('venue')}] "
        f"doi:{p.get('doi')} 被引{p.get('citations')}" for p in selected)
    deep_brief = "\n\n".join(
        f"【{d['paper'].get('title')}】\n{d['summary']}" for d in read_ok) or "（无全文精读成功）"
    survey = None
    for attempt in range(2):
        try:
            sys_prompt = ("你是文献缺口挖掘器。基于全文精读摘要，输出结构化缺口清单（中文）："
                          "## 缺口清单\\n每条格式：**[gap 一句话]** | 来源论文[n]与位置（Limitations"
                          "/Discussion/Outlook）| 作者建议方向 | 是否多篇重复出现。"
                          "## 多篇论文重复承认的共识缺口（领域级，权重最高）\\n"
                          "## 与本研究的关联（哪些缺口现有 DEM 仿真+量纲分析工具可研究）。"
                          "不得编造列表之外的论文或数值。"
                          if mode == "gaps" else
                          "你是文献调研综合器。基于元数据列表与全文精读摘要，输出结构化调研报告"
                          "（中文）：## 研究现状（按主题脉络组织，引用论文须标 [n] 对应编号）\\n"
                          "## 与本研究的关联（哪些可对比/引用/借鉴方法）\\n"
                          "## 明显的研究空白或争议。不得编造列表之外的论文或数值。")
            survey = llm.chat_text(model or config.MODEL_PRESETS["agent"], [
                {"role": "system",
                 "content": sys_prompt
                 + ("（上次调用失败，这是重试）" if attempt else "")},
                {"role": "user",
                 "content": f"主题：{topic}\n关注问题：{questions}\n\n"
                            + (f"# 波尔 AI 综合回答（领域已知概览——从这里判断哪些已被解决）\n"
                               f"{bohrium_answer[:3000]}\n\n" if bohrium_answer else "")
                            + f"# 候选论文元数据（编号即 [n]）\n{meta_brief}\n\n"
                            f"# 全文精读摘要\n{deep_brief}"}],
                temperature=0.1, max_tokens=4000, purpose="lit_survey")
            if survey and survey.strip():
                break
            survey = None      # 空响应视为失败
        except Exception as e:
            log(f"综述调用异常: {type(e).__name__}: {str(e)[:150]}")
            survey = None
            continue

    if survey is None:
        # 降级：至少返回结构化元数据清单
        survey = ("（综述生成失败，以下为元数据清单）\n" + meta_brief)
        log(f"综述生成两次失败，降级为元数据清单（deep_brief {len(deep_brief)} 字符）")

    return {
        "ok": True, "topic": topic,
        "queries": queries, "pool_size": len(pool),
        "n_selected": len(selected),
        "papers": [{"title": p.get("title"), "year": p.get("year"),
                    "doi": p.get("doi"), "venue": p.get("venue"),
                    "citations": p.get("citations")} for p in selected],
        "deep_read_ok": [d["paper"].get("title") for d in read_ok],
        "deep_read_failed": [{"title": d["paper"].get("title"), "note": d.get("note")}
                             for d in read_fail],
        "survey": survey,
        "search_errors": search_errors,
        "hint": "调研在子代理上下文完成，主对话只需本综述。引用具体数值前用 litf_read "
                "回查原文；查新结论须基于本综述列出的论文。",
    }
