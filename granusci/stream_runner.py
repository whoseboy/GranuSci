# -*- coding: utf-8 -*-
"""运行循环：interrupt 处理（计划批复 / 用户中途指令）。

模式：每次 astream 要么跑到 END（无 interrupt），要么停在 interrupt；
停在 interrupt 时按载荷类型处理（计划→索要批复；用户指令→注入后自动继续），
用 Command(resume=...) 开启新一轮流。绝不以 None 作为输入重复调用 astream。
"""
from __future__ import annotations

from typing import Callable

from langgraph.types import Command


# 可恢复终态：LLM 侧故障（配额耗尽/服务不可用/空响应）非研究本身的问题，
# checkpoint 里有全部进度，故障恢复后 --resume 应当继续而不是秒退
# （paper12 二次死因：quota 终态被 checkpoint 持久化，续跑路由直接走 END）。
# aborted（无正式终态的退出——如 review_off 消融暴露的旧 harness 缺陷）同样可复活。
RECOVERABLE_FINALS = {"llm_quota_exhausted", "llm_unavailable", "empty_response",
                      "aborted"}


async def run_with_plan_gate(
    graph, init_state: dict | None, stream_cfg: dict, run_dir,
    yolo: bool, log_fn: Callable[[str], None],
):
    """执行图直到 END。

    init_state=None 表示从 checkpoint 恢复（断点续跑）。
    """
    final_state = None
    input_ = init_state
    if init_state is None:
        # 断点续跑：若上次死于 LLM 故障型终态（线程已 END），复活——
        # 清终态 + 注入恢复指令（尾消息变 user → after_agent 路由回 agent）。
        # final 为空也算：图跑完却没形成正式终态（run.py 收尾时标 aborted 的情形，
        # 如旧 harness 的 no-reviewer 流程缺陷）——同样属于"未正式结束"，可复活。
        snap = await graph.aget_state(stream_cfg)
        _fin = (snap.values or {}).get("final") if snap is not None else None
        if (snap is not None and not snap.next
                and (_fin in RECOVERABLE_FINALS or not _fin)
                and (snap.values or {}).get("messages")):
            vals = snap.values or {}
            msgs = list(vals.get("messages") or [])
            msgs.append({"role": "user", "content":
                         "【系统】运行此前因 LLM 服务故障（配额耗尽/服务不可用）中断，"
                         "现已恢复。从中断处继续研究：回顾上方消息史与账本锚点，"
                         "接着执行下一步工具调用。"
                         if _fin else
                         "【系统】运行此前异常退出（未形成正式终态），现已恢复。"
                         "检查当前研究进度：若论文/报告已提交而交付未完成（如汇报"
                         "幻灯片未生成），先完成交付；否则从中断处继续研究。"})
            await graph.aupdate_state(stream_cfg, {
                "final": "", "messages": msgs,
                "n_steps": vals.get("n_steps", 0),
            }, as_node="agent")
            log_fn(f"[resume] 检测到可复活状态（final={_fin!r}），已复活续跑")
            input_ = None
    while True:
        hit_interrupt = None
        async for chunk in graph.astream(input_, stream_cfg, stream_mode="values"):
            final_state = chunk
            if isinstance(chunk, dict) and chunk.get("__interrupt__"):
                hit_interrupt = chunk["__interrupt__"]
        if hit_interrupt is None:
            return final_state
        # interrupt 载荷分类处理
        try:
            payload = hit_interrupt[0].value if hasattr(hit_interrupt[0], "value") \
                else hit_interrupt[0]
        except Exception:
            payload = {}
        if isinstance(payload, dict) and "user_message" in payload:
            # 用户中途指令（inbox.md）：展示后注入继续（interrupt 节点已把消息入史）
            msg = payload["user_message"]
            log_fn(f"[inbox] 收到用户中途指令: {msg[:150]}")
            if not yolo:
                print(f"\n[inbox] 用户指令: {msg[:300]}\n")
            input_ = Command(resume="ok")
            continue
        # 计划批复
        plan = payload.get("plan", "") if isinstance(payload, dict) else str(payload)
        if not plan:
            plan = (final_state or {}).get("plan", "") if isinstance(final_state, dict) else ""
        (run_dir / "research_plan.md").write_text(plan, encoding="utf-8")

        if yolo:
            log_fn("[plan_gate] --yolo：自动批准")
            feedback = "批准"
        else:
            print("\n" + "=" * 70)
            print("【研究计划】（已存 research_plan.md）")
            print("=" * 70)
            print(plan)
            print("=" * 70)
            try:
                feedback = input("你的批复（回车=批准；或写修改意见）: ").strip()
            except EOFError:
                log_fn("[plan_gate] 无终端输入，自动批准")
                feedback = "批准"
            feedback = feedback or "批准"
            log_fn(f"[plan_gate] 用户批复: {feedback[:100]}")
        input_ = Command(resume=feedback)
