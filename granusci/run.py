# -*- coding: utf-8 -*-
"""GranuSci 运行入口。

用法：
  python -m granusci.run "研究意图" [--model GLM-5.3] [--reviewer DeepSeek-V4-Pro]
                            [--max-steps 80] [--tag 备注]

产物：granusci/runs/<时间戳>_<tag>/
  report.md        最终报告
  review.md        审稿结论
  trace.jsonl      全部工具调用记录
  messages.json    主循环消息史
  config.json      本次运行配置
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# MCP 服务器可能是内网地址：绕过系统代理（系统代理把内网流量劫持后 502/
# 挂起——httpx 读系统代理设置，环境变量无代理时也中招。NO_PROXY 使 httpx
# 对 MCP 主机直连）。在导入 mcp/httpx 前设置才生效。
import os as _os
from urllib.parse import urlparse as _urlparse
for _name, _default in (("DEM_MCP_URL", "http://localhost:8001/mcp"),
                         ("DDDA_MCP_URL", "http://localhost:8000/mcp"),
                         ("LITX_MCP_URL", "http://localhost:8006/mcp")):
    _host = _urlparse(_os.environ.get(_name) or _default).hostname or ""
    if not _host:
        continue
    for _var in ("NO_PROXY", "no_proxy"):
        _cur = _os.environ.get(_var, "")
        if _host not in _cur:
            _os.environ[_var] = (_cur + "," if _cur else "") + _host

from . import config, cross_memory, llm
from .agent.graph import build_graph


def ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _has_checkpoint_state(run_dir: Path) -> bool:
    """run 目录的 checkpoint 里有无实际状态（区分'跑过'与'启动即崩'）。"""
    ckpt = run_dir / "checkpoint.sqlite"
    if not ckpt.exists():
        return False
    try:
        import sqlite3
        con = sqlite3.connect(str(ckpt))
        try:
            n = con.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        finally:
            con.close()
        return n > 0
    except Exception:
        return False


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("intent", nargs="?", default=None, help="研究意图（自然语言；--resume 时可省略）")
    ap.add_argument("--model", default=config.MODEL_PRESETS["agent"])
    ap.add_argument("--reviewer", default=config.MODEL_PRESETS["reviewer"])
    ap.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--yolo", action="store_true",
                    help="跳过研究计划人工批复门（自动批准；用于批量实验/测试）")
    ap.add_argument("--resume", default=None, metavar="RUN_DIR",
                    help="从指定运行目录断点续跑（checkpoint.sqlite 恢复全部状态）")
    ap.add_argument("--run-dir", default=None, metavar="RUN_DIR",
                    help="指定运行目录（实验矩阵用；目录已有 config.json 时自动按续跑处理）")
    ap.add_argument("--seed", type=int, default=None,
                    help="实验种子（注入 LLM 请求体，消融矩阵方差控制用）")
    ap.add_argument("--no-reviewer", action="store_true",
                    help="消融：跳过审稿门（提交报告直接终态）")
    ap.add_argument("--ablate", action="append", default=[],
                    choices=["no_mechanism"],
                    help="消融开关，可重复（no_mechanism=剥离机制调查引导）")
    ap.add_argument("--exclude-tools", default="",
                    help="逗号分隔的禁用工具名（消融：文献腿等，如 lit_search_openalex,lit_survey）")
    ap.add_argument("--blind-kb", action="store_true",
                    help="消融：盲知识库（kb_search 指向剔除博士论文的净化库，自主性验证）")
    ap.add_argument("--hypothesis-runtime", action="store_true",
                    help="分区假设运行时（结构中介制四机制：设计门/报数对账/死亡后继/DDDA 自动触发）。"
                         "阶段二 ON 态配置；默认 off（阶段三 OFF 对照即历史档案）")
    args = ap.parse_args()

    if args.seed is not None:
        llm.set_seed(args.seed)
    excluded_tools = [t.strip() for t in args.exclude_tools.split(",") if t.strip()]

    resuming = bool(args.resume)
    if resuming:
        run_dir = Path(args.resume)
        if not run_dir.is_dir():
            print(f"[run] --resume 目录不存在: {run_dir}")
            return
        # 续跑沿用原 run 的全部实验控制变量（config.json 是权威）——
        # --model 未显式传时不落回默认值（gpt56sol 教训），消融/盲库/工具
        # 剥离/步数上限同理（审计 A-2：resume 曾静默解除全部消融控制，
        # 前后半程处于不同实验条件，消融数据作废且无告警）
        try:
            _prev = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            _restored = []
            if args.model == config.MODEL_PRESETS["agent"] and _prev.get("agent_model"):
                args.model = _prev["agent_model"]
                _restored.append("agent_model")
            if args.reviewer == config.MODEL_PRESETS["reviewer"] and _prev.get("reviewer_model"):
                args.reviewer = _prev["reviewer_model"]
                _restored.append("reviewer_model")
            if _prev.get("max_steps") and args.max_steps == config.MAX_STEPS:
                args.max_steps = int(_prev["max_steps"])
                _restored.append("max_steps")
            if _prev.get("ablations") and not args.ablate:
                args.ablate = list(_prev["ablations"])
                _restored.append(f"ablations={args.ablate}")
            if _prev.get("excluded_tools") and not args.exclude_tools:
                args.exclude_tools = _prev["excluded_tools"]
                _restored.append(f"excluded_tools={args.exclude_tools}")
            if _prev.get("blind_kb") and not args.blind_kb:
                args.blind_kb = True
                _restored.append("blind_kb")
            if _prev.get("hypothesis_runtime") and not args.hypothesis_runtime:
                args.hypothesis_runtime = True
                _restored.append("hypothesis_runtime")
            if _restored:
                print(f"[run] resume 恢复实验控制变量: {', '.join(_restored)}")
        except Exception:
            pass
        if not _has_checkpoint_state(run_dir):
            # 启动即崩的 run（config 有、状态无）：有意图就转全新运行，否则退出
            if args.intent:
                print("[run] --resume 目录无 checkpoint 状态，转为全新运行")
                resuming = False
            else:
                print("[run] --resume 目录无 checkpoint 状态且未提供意图，无法续跑")
                return
        elif args.intent:
            print("[run] --resume 模式忽略新意图，续跑原研究")
    else:
        if not args.intent:
            ap.error("新运行必须提供研究意图（或用 --resume 续跑）")
        if args.run_dir:
            # 实验矩阵模式：目录固定（无时间戳前缀）。续跑判定看 checkpoint 里有无
            # 实际状态——只有 config.json 不够（build_graph 前崩溃的 run 会留下
            # config 但无状态，astream(None) 会抛 EmptyInputError）
            run_dir = Path(args.run_dir)
            resuming = _has_checkpoint_state(run_dir)
            if resuming:
                print("[run] --run-dir 检测到 checkpoint 状态，按续跑处理（忽略新意图）")
        else:
            tag = args.tag or args.intent[:12].replace(" ", "_")
            run_dir = config.RUNS_DIR / f"{ts()}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 战役 manifest（复现性档案：系统版本哈希/模型/服务端点）
    try:
        from granusci import campaign_manifest
        # resume 路径不重写 manifest（原 run 已有；tag 变量也只在新建分支赋值）
        if not resuming:
            campaign_manifest.create(tag or "untagged", run_dir)
    except Exception as _me:  # noqa: BLE001  manifest 不阻塞主流程
        print(f"[run] manifest 写入失败（不阻塞）: {_me}", flush=True)

    def log(msg: str):
        print(msg, flush=True)

    # 成本核算：LLM 用量增量落盘 run_dir/llm_usage.jsonl（崩溃路径也保留）
    llm.usage.configure(run_dir)

    log(f"[run] dir={run_dir}{' [RESUME]' if resuming else ''}")
    log(f"[run] agent={args.model} reviewer={args.reviewer} max_steps={args.max_steps}")

    # 实验元数据先落盘（崩溃路径收集器也能读到配置；成功后终版覆盖）
    (run_dir / "config.json").write_text(json.dumps({
        "intent": args.intent, "agent_model": args.model, "reviewer_model": args.reviewer,
        "max_steps": args.max_steps, "final": "(运行中/中断)",
        "seed": args.seed, "reviewer_enabled": not args.no_reviewer,
        "ablations": args.ablate, "excluded_tools": excluded_tools,
        "blind_kb": args.blind_kb, "hypothesis_runtime": args.hypothesis_runtime,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 盲 KB 消融：kb_search 切到净化库（kb.py 在调用时读 config.KB_DIR）
    if args.blind_kb:
        config.KB_DIR = config.KB_DIR_BLIND

    # 持久化 checkpoint：崩溃/中止后可 --resume 断点续跑
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    ckpt_conn = AsyncSqliteSaver.from_conn_string(str(run_dir / "checkpoint.sqlite"))
    checkpointer = await ckpt_conn.__aenter__()

    graph, bus, local_tools = build_graph(args.model, args.reviewer, args.max_steps,
                                          log_fn=log, run_dir=run_dir,
                                          checkpointer=checkpointer,
                                          review_enabled=not args.no_reviewer,
                                          ablations=set(args.ablate),
                                          excluded_tools=excluded_tools,
                                          blind_kb=args.blind_kb,
                                          hr_enabled=args.hypothesis_runtime)
    if args.hypothesis_runtime:
        log("[run] 分区假设运行时 ON（结构中介制：设计门/报数对账/死亡后继/DDDA 自动触发）")
    init_state = None if resuming else {
        "intent": args.intent, "messages": [], "tool_log": [], "ledger": [],
        "plan": None, "plan_approved": False, "plan_feedback": None,
        "report": None, "paper_tex": None, "review": None, "review_rounds": 0,
        "n_steps": 0, "compact_cut": 0, "slides_offered": False,
        "final": "", "artifacts_dir": str(run_dir),
        "hr": {},   # 假设运行时状态（hr_enabled 时 tools_node 维护；checkpoint 持久化）
    }

    def dump_trace():
        # 续跑时追加（bus.call_log 是新进程的，覆盖会丢前半程记录）
        mode = "a" if resuming else "w"
        with (run_dir / "trace.jsonl").open(mode, encoding="utf-8") as f:
            for e in bus.call_log:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    t0 = time.time()
    state: dict = {}
    try:
        await bus.start(local_tools)
        # 工具消融：从注册表移除（LLM 工具列表与调用面都不再有）
        for t in excluded_tools:
            bus.tools.pop(t, None)
        if excluded_tools:
            log(f"[run] 工具消融，已禁用: {excluded_tools}")
        if args.hypothesis_runtime:
            # 设计门字段进 schema（v3hr 教训：不在 schema 里 agent 加不进参数，
            # 门陷入拒绝-重试死循环）
            from granusci.agent.hyp_runtime import augment_design_tool_schemas
            augment_design_tool_schemas(bus)
        stream_cfg = {"recursion_limit": int(args.max_steps * 1.35) * 4,  # 软预算上限=125%+余量
                      "configurable": {"thread_id": run_dir.name}}
        from .stream_runner import run_with_plan_gate
        final_state = await run_with_plan_gate(
            graph, init_state, stream_cfg, run_dir, args.yolo, log)
        state = final_state if isinstance(final_state, dict) else dict(final_state or {})
    finally:
        dump_trace()          # 崩溃也保留已发生的调用记录
        # 成本核算汇总 + 跨会话经验蒸馏（均 best-effort，不因 LLM 故障阻塞收尾）
        try:
            usage_path = llm.usage.dump(run_dir)
            s = llm.usage.summary()
            log(f"[usage] LLM 调用 {s['total_calls']} 次 -> {usage_path}")
            for m, d in sorted(s["by_model"].items()):
                log(f"[usage]   {m}: {d['calls']} 次, "
                    f"prompt={d['prompt_tokens']:,} completion={d['completion_tokens']:,} tokens")
        except Exception as e:
            log(f"[usage] 汇总失败: {e}")
        try:
            # 已编码知识清单（工具描述+技能目录）喂给蒸馏器做反重复过滤
            from .skills import catalog_text, load_skills
            encoded = "\n".join(
                f"- {t.name}: {t.description[:150]}"
                for t in bus.tools.values()) + "\n\n# 技能目录\n" + catalog_text(load_skills())
            cross_memory.distill(run_dir, state, bus.call_log, model=args.model,
                                 encoded_knowledge=encoded, log_fn=log)
        except Exception:
            pass
        llm.usage.close()
        # run 报告（⑨）：崩溃路径也生成，残缺 run 标注缺失不崩
        try:
            from . import run_report
            html_path = run_report.generate(run_dir)
            log(f"[report] run 摘要 -> {html_path}")
        except Exception as e:
            log(f"[report] 生成失败（不影响 run）: {type(e).__name__}: {e}")
        await bus.close()
        try:
            await ckpt_conn.__aexit__(None, None, None)
        except Exception:
            pass

    elapsed = round(time.time() - t0, 1)
    final = state.get("final") or ("max_steps" if state.get("n_steps", 0) > args.max_steps else "aborted")
    log(f"[run] finished in {elapsed}s, final={final}, steps={state.get('n_steps')}, "
        f"tool_calls={len(state.get('tool_log', []))}")

    report = state.get("report")
    if report:
        (run_dir / "report.md").write_text(report, encoding="utf-8")
    elif state.get("paper_tex") or (run_dir / "paper.pdf").exists():
        # paper 流未提交行为报告（paper12 缺陷）：report.md 不得为空也不得
        # 写论文全文——落一份账本驱动的过程骨架，明确标注"未由智能体撰写"
        ledger = state.get("ledger") or []
        lines = [f"# 行为报告（骨架——智能体本次未提交行为报告，此为账本自动生成）", "",
                 f"- 终态：{final}，步数 {state.get('n_steps')}，"
                 f"工具调用 {len(state.get('tool_log', []))} 次",
                 f"- 研究计划：{(run_dir / 'research_plan.md').exists() and 'research_plan.md' or '未存档'}",
                 f"- 审稿记录：{(run_dir / 'review.md').exists() and 'review.md' or '无'}", "",
                 "## 账本摘要（完整版 ledger.json）", ""]
        for e in ledger:
            lines.append(f"- L{e.get('seq')} [{e.get('kind')}] {e.get('title', '')}: "
                         f"{str(e.get('content', ''))[:200]}")
        lines += ["", "（注：论文见 paper.tex / paper.pdf；智能体行为报告的撰写与提交"
                  "应在 submit_paper 前完成——submit_report）"]
        (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    plan = state.get("plan")
    if plan and not (run_dir / "research_plan.md").exists():
        (run_dir / "research_plan.md").write_text(plan, encoding="utf-8")
    if state.get("review"):
        (run_dir / "review.md").write_text(
            json.dumps(state["review"], ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "messages.json").write_text(
        json.dumps(state.get("messages", []), ensure_ascii=False, indent=1), encoding="utf-8")
    # manifest finalize（两战役连现 status=open：数据锚点+假设档案哈希收口）
    try:
        from granusci import campaign_manifest
        campaign_manifest.finalize(run_dir)
    except Exception as _fe:  # noqa: BLE001  finalize 不阻塞收尾
        print(f"[run] manifest finalize 失败（不阻塞）: {_fe}", flush=True)
    (run_dir / "config.json").write_text(json.dumps({
        "intent": args.intent, "agent_model": args.model, "reviewer_model": args.reviewer,
        "max_steps": args.max_steps, "elapsed_s": elapsed, "final": final,
        "n_steps": state.get("n_steps"), "tool_calls": len(state.get("tool_log", [])),
        "ledger_entries": len(state.get("ledger") or []),
        # 三件套存在性（交付完备性统计）
        "paper_pdf": (run_dir / "paper.pdf").exists(),
        "slides_pdf": (run_dir / "slides.pdf").exists(),
        # 实验矩阵元数据（⑫ 收集器读）——含 excluded_tools/blind_kb
        # （审计 A-2：收尾版曾丢这两个字段，resume 恢复实验控制时读不到）
        "seed": args.seed, "reviewer_enabled": not args.no_reviewer,
        "ablations": args.ablate, "excluded_tools": excluded_tools,
        "blind_kb": args.blind_kb, "hypothesis_runtime": args.hypothesis_runtime,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[run] artifacts -> {run_dir}")
    if report:
        log(f"[run] report -> {run_dir / 'report.md'}")


if __name__ == "__main__":
    asyncio.run(main())
