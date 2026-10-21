# -*- coding: utf-8 -*-
"""研究运行状态。"""
from __future__ import annotations

from typing import Any, TypedDict


class ResearchState(TypedDict):
    """LangGraph 全局状态。

    messages:  主循环消息史（OpenAI 格式 dict，节点显式追加——不用 add_messages，
               避免 LangChain 消息对象转换）
    tool_log:  全部工具调用记录（tools 节点每次以完整快照覆盖——bus 是唯一事实源）
    plan:      研究计划全文（提案阶段产出，人工门批准后作为执行章程）
    plan_approved: 计划是否已获批（True 后不再进 plan_gate）
    plan_feedback: 人工门对计划的批语（批准="批准"；否则为修改意见）
    report:    submit_report 提交的行为报告全文（report.md 的唯一来源——
               submit_paper 的论文 tex 走 paper_tex，不得混入本字段）
    paper_tex: submit_paper 提交的论文 LaTeX 全文（审稿对象之一）
    review:    最近一次审稿结论 {pass, critical_issues, feedback, numeric_audit}
    review_rounds: 已审稿轮数
    n_steps:   主循环已用 LLM 轮数
    compact_cut: 上下文压缩冻结边界（[0:cut) 一次性压缩、[cut:) 原文只追加——
                 前缀稳定使服务端缓存命中；追加区超阈值时一次性推进）
    slides_offered:  论文过审后已注入过汇报制作指令（防重复注入）
    final:     终态标记（passed / review_failed_twice / empty_response）
    artifacts_dir: 本次运行的产物目录
    """
    intent: str
    messages: list
    tool_log: list
    ledger: list         # 研究状态账本：结构化发现条目（record_finding 产出，
                         # checkpoint 持久化——续跑后完整；ledger.json 同步落盘）
    plan: Any            # None | str
    plan_approved: bool
    plan_feedback: Any   # None | str
    report: Any          # None | str
    paper_tex: Any       # None | str
    review: Any          # None | dict
    review_rounds: int
    review_object: str   # "" | "paper" | "report"——对象切换时 rounds 重置（A-6）
    n_steps: int
    compact_cut: int
    slides_offered: bool
    final: str
    artifacts_dir: str
    hr: dict            # 分区假设运行时状态（--hypothesis-runtime 开启时由
                         # tools_node 与进程内 hr_store 双向同步；结构见
                         # agent/hyp_runtime.py 模块注释）
