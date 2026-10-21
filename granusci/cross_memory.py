# -*- coding: utf-8 -*-
"""跨会话经验记忆：run 之间复用的教训与工具使用经验。

设计（⑦，借鉴 Claude Code memory 的纪律——"repo 已记录的不存、错了的要删"）：

**分流路由**（distill 时分类，记忆不是垃圾桶，什么都沉淀=上下文污染）：
- env      环境/服务的时变事实（缓存行为、延迟特性）→ experience.md，注入智能体
- method   已验证做法且未被技能/工具描述覆盖       → experience.md，注入；
           第 2 次被验证 → distill 输出晋升建议进工单（由开发者晋升到技能）
- pitfall  智能体行为陷阱且无现有防线（提示词/审稿门/loop guard 之外）→ 注入
- dev-feedback 疑似工具/架构缺陷 → dev_feedback.md 工单，**永不注入智能体**——
           由开发者修复代码闭环，而不是让智能体记着绕坑
- 与"已编码知识清单"（工具描述+技能目录）重复、或已被代码修复过时的 → 拒绝写入，
  并输出 PRUNE 建议进工单

存储：
- memory/experience.md  注入智能体的条目（env/method/pitfall）
- memory/dev_feedback.md 开发者工单（缺陷报告+晋升/清理建议）

条目格式：

    ### [type:env 2026-08-31 <run_tag>] 标题
    正文 1~3 句，可操作。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable

from . import config, llm

MEMORY_DIR = config.PKG / "memory"
MEMORY_FILE = MEMORY_DIR / "experience.md"
DEV_FEEDBACK_FILE = MEMORY_DIR / "dev_feedback.md"

_HEADER = ("# GranuSci 跨会话经验记忆（仅 env/method/pitfall；dev-feedback 走工单不进本文件）\n\n"
           "<!-- 条目: ### [type:类型 日期 run_tag] 标题 + 正文；run 收尾蒸馏自动追加 -->\n\n")

_DEV_HEADER = ("# GranuSci 开发者工单（蒸馏器自动追加；修复/晋升后请手动勾掉）\n\n"
               "<!-- 两类：缺陷报告（疑似工具/架构问题）与建议（晋升技能/清理过时条目）。"
               "本文件内容永不注入智能体提示词。 -->\n\n")

# 注入智能体的条目类型（dev-feedback 永不注入）
AGENT_TYPES = {"env", "method", "pitfall"}

MAX_PROMPT_CHARS = 4000   # 注入提示词的经验上限（尾部最新优先）


def load_for_prompt(max_chars: int = MAX_PROMPT_CHARS) -> str:
    """读取经验记忆用于提示词注入；只留 agent 类型条目，超长从条目边界截尾。"""
    if not MEMORY_FILE.exists():
        return ""
    text = MEMORY_FILE.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    cut = tail.find("\n### ")
    return tail[cut + 1:].strip() if cut >= 0 else tail


def load_dev_feedback() -> str:
    if not DEV_FEEDBACK_FILE.exists():
        return "(空)"
    text = DEV_FEEDBACK_FILE.read_text(encoding="utf-8").strip()
    return text or "(空)"


def _run_summary(run_dir: Path, state: dict, call_log: list[dict]) -> str:
    """组装本次运行复盘素材：终态、失败调用、审稿结论。"""
    cfg_path = run_dir / "config.json"
    cfg = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    intent = cfg.get("intent") or state.get("intent") or "(未知)"
    final = state.get("final") or cfg.get("final") or "(中断)"
    n_steps = state.get("n_steps") or cfg.get("n_steps") or 0
    lines = [
        f"研究意图: {intent}",
        f"终态: {final} | 步数: {n_steps} | 工具调用: {len(call_log)}",
    ]
    failed = [e for e in call_log if not e.get("ok")]
    if failed:
        lines.append(f"\n失败的工具调用（{len(failed)} 次，摘前 20 条）:")
        for e in failed[:20]:
            msg = str((e.get("result") or {}).get("message", ""))[:200]
            lines.append(f"- {e['tool']}({json.dumps(e.get('args', {}), ensure_ascii=False)[:80]}): {msg}")
    review = state.get("review")
    if review:
        issues = review.get("critical_issues") or []
        if issues:
            lines.append("\n审稿驳回的严重问题:")
            for i in issues[:10]:
                lines.append(f"- {str(i)[:200]}")
    ledger = state.get("ledger") or []
    if ledger:
        # 只给 kind+标题——给蒸馏器看研究覆盖了什么，不给全文（防科学内容混入经验记忆）
        lines.append(f"\n研究状态账本（{len(ledger)} 条，标题级）:")
        for e in ledger[:25]:
            lines.append(f"- L{e.get('seq', '?')} [{e.get('kind', '?')}] {e.get('title', '')}")
    plan = state.get("plan")
    if plan:
        lines.append(f"\n研究计划开头: {plan[:400]}")
    return "\n".join(lines)


DISTILL_SYSTEM = """你是 GranuSci 科研智能体的运行复盘器，负责把一次运行的经验**分流**到正确的通道。

# 分流规则（核心纪律：重复沉淀=上下文污染）
1. **与"已编码知识清单"重复的经验一律不进 memory**——工具描述/技能/系统提示词已经写了的东西，智能体本来就知道。
2. **疑似工具或架构缺陷 → dev_feedback**（不是让智能体记着绕坑，是让开发者修代码）。判据：同类错误重复出现、错误信息指向工具内部逻辑、或某 workaround 已成惯例。
3. **已被代码修复的过时经验 → prune**（现有经验里教智能体绕一个已不存在的坑的条目）。
4. env=环境/服务的时变事实；method=已验证且清单未覆盖的做法；pitfall=无现有防线的行为陷阱。经验条目须通用可操作，不要复述本次研究的科学内容。
5. 现有经验中同主题 method 条目再次被验证 → 在 promote 里建议晋升到技能（由开发者执行）。
6. 确实没有新东西：全空列表。

# 输出 JSON（不要其他文字）
{"memory": [{"type": "env|method|pitfall", "title": "短标题", "body": "1~3句可操作正文"}],
 "prune": [{"target": "现有经验条目标题前缀", "reason": "为何过时/重复"}],
 "promote": [{"target": "经验条目标题前缀", "to_skill": "技能名", "reason": "证据"}],
 "dev_feedback": [{"title": "缺陷/建议短标题", "evidence": "本次运行中的证据"}]}"""

DISTILL_USER = """# 本次运行记录
{summary}

# 已编码知识清单（这些智能体已经知道——重复的不许进 memory）
{encoded}

# 现有经验记忆（尾部）
{existing}

# 任务
按分流规则输出 JSON。"""


def _append_entries(path: Path, header: str, blocks: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(header, encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(blocks) + "\n\n")


def distill(run_dir: Path | str, state: dict, call_log: list[dict],
            model: str | None = None, encoded_knowledge: str = "",
            log_fn: Callable[[str], None] = print) -> bool:
    """run 收尾蒸馏：LLM 分流本次经验。best-effort，绝不阻塞收尾。"""
    run_dir = Path(run_dir)
    try:
        summary = _run_summary(run_dir, state, call_log)
        existing = load_for_prompt()
        msgs = [
            {"role": "system", "content": DISTILL_SYSTEM},
            {"role": "user", "content": DISTILL_USER.format(
                summary=summary,
                encoded=encoded_knowledge.strip() or "(未提供)",
                existing=existing or "(空)")}]

        def _parse(raw):
            """输出→dict；解析失败返回 None。"""
            m = re.search(r"\{.*\}", raw or "", re.S)
            if not m:
                return None
            try:
                d = json.loads(m.group(0))
                return d if isinstance(d, dict) else None
            except json.JSONDecodeError:
                return None

        raw = llm.chat_text(model or config.MODEL_PRESETS["agent"], msgs,
                            temperature=0.1, max_tokens=2000,
                            purpose="memory_distill")
        data = _parse(raw)
        if data is None:
            # paper11 教训：长 run 复盘输出偶尔非 JSON——附错误反馈纠正重试一次
            log_fn("[memory] 蒸馏输出非 JSON，纠正重试一次")
            msgs.append({"role": "assistant", "content": str(raw or "")[:2000]})
            msgs.append({"role": "user", "content":
                         "上面的输出不是合法 JSON（或不是 JSON 对象）。"
                         "重新输出——只输出一个 JSON 对象，"
                         '{"memory": [...], "prune": [...], "promote": [...], '
                         '"dev_feedback": [...]}，不要任何其他文字。'})
            raw = llm.chat_text(model or config.MODEL_PRESETS["agent"], msgs,
                                temperature=0.1, max_tokens=2000,
                                purpose="memory_distill")
            data = _parse(raw)
        if data is None:
            log_fn("[memory] 蒸馏输出两次非 JSON，跳过（不影响本次 run）")
            return False

        date, tag = time.strftime("%Y-%m-%d"), run_dir.name
        # memory 条目
        blocks = []
        for e in data.get("memory") or []:
            if not isinstance(e, dict):
                continue
            typ = str(e.get("type", "")).strip().lower()
            if typ not in AGENT_TYPES:      # dev-feedback 等类型不进经验文件
                continue
            title = str(e.get("title", "")).strip()[:60]
            body = str(e.get("body", "")).strip()
            if title and body:
                blocks.append(f"### [type:{typ} {date} {tag}] {title}\n{body}")
        if blocks:
            _append_entries(MEMORY_FILE, _HEADER, blocks)
            log_fn(f"[memory] 沉淀 {len(blocks)} 条新经验 -> {MEMORY_FILE}")
        # 工单：prune / promote / dev_feedback（给开发者，不注入智能体）
        ticket_lines = []
        for kind, items in (("PRUNE", data.get("prune")), ("PROMOTE", data.get("promote")),
                            ("DEV-FEEDBACK", data.get("dev_feedback"))):
            for it in items or []:
                if isinstance(it, dict) and (it.get("target") or it.get("title")):
                    detail = " | ".join(f"{k}: {v}" for k, v in it.items()
                                        if k in ("target", "to_skill", "reason", "evidence", "title"))
                    ticket_lines.append(f"### [{kind} {date} {tag}] {detail[:300]}")
        if ticket_lines:
            _append_entries(DEV_FEEDBACK_FILE, _DEV_HEADER, ticket_lines)
            log_fn(f"[memory] 开发者工单 +{len(ticket_lines)} 条 -> {DEV_FEEDBACK_FILE}")
        if not blocks and not ticket_lines:
            log_fn("[memory] 本次运行无可沉淀项")
        return bool(blocks or ticket_lines)
    except Exception as e:
        log_fn(f"[memory] 经验蒸馏失败（不影响本次 run）: {type(e).__name__}: {str(e)[:150]}")
        return False
