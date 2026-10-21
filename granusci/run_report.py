# -*- coding: utf-8 -*-
"""⑨ Run 报告生成器：run 目录 → 自包含 HTML 摘要。

定位（可观测性基建，兼作论文实验记录）：
- 每次 run 收尾自动生成 run_dir/report.html（崩溃路径也生成——残缺 run 照样出报告，
  缺什么标注什么，绝不因数据残缺而崩）
- 手动：python -m granusci.run_report RUN_DIR [--out path.html]

数据源（全部可选、容错读取）：
  config.json     意图/模型/终态/步数          llm_usage.jsonl  逐调用 LLM 用量
  trace.jsonl     工具调用时间线               ledger.json      研究状态账本
  review.md       审稿结论                     research_plan.md 研究计划

输出结构：概览卡 → 时间线（含失败高亮）→ 工具统计 → 账本 → 审稿 → 成本。
零外部依赖（stdlib + 内联 CSS），单文件自包含。
"""
from __future__ import annotations

import html
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------- 数据读取（容错）
def _read_json(path: Path):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return out


def _read_text(path: Path, limit: int = 2000) -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        pass
    return ""


def _fmt_duration(sec: float) -> str:
    if sec is None:
        return "-"
    sec = float(sec)
    if sec < 90:
        return f"{sec:.0f}s"
    if sec < 5400:
        return f"{sec / 60:.1f}min"
    return f"{sec / 3600:.1f}h"


def _fmt_tokens(n: int) -> str:
    n = int(n or 0)
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}k"
    return str(n)


# ---------------------------------------------------------------- 统计
def _tool_stats(trace: list[dict]) -> list[dict]:
    agg: dict[str, dict] = defaultdict(lambda: {"n": 0, "fail": 0, "time": 0.0})
    for e in trace:
        t = e.get("tool", "?")
        d = agg[t]
        d["n"] += 1
        d["time"] += float(e.get("duration") or 0)
        if not e.get("ok", True):
            d["fail"] += 1
    return sorted(({"tool": k, **v} for k, v in agg.items()),
                  key=lambda x: -x["time"])


def _usage_summary(usage_rows: list[dict]) -> dict:
    by_model: dict[str, dict] = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0})
    by_purpose: dict[str, dict] = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0})
    for r in usage_rows:
        m = by_model[r.get("model", "?")]
        m["calls"] += 1
        m["prompt"] += r.get("prompt_tokens", 0) or 0
        m["completion"] += r.get("completion_tokens", 0) or 0
        p = by_purpose[r.get("purpose", "?")]
        p["calls"] += 1
        p["prompt"] += r.get("prompt_tokens", 0) or 0
        p["completion"] += r.get("completion_tokens", 0) or 0
    return {"by_model": dict(by_model), "by_purpose": dict(by_purpose),
            "total": sum(v["prompt"] + v["completion"] for v in by_model.values()),
            "llm_time": sum(r.get("duration", 0) or 0 for r in usage_rows)}


# ---------------------------------------------------------------- HTML 渲染
_CSS = """
:root { color-scheme: light; }
body { font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; margin: 0;
       background: #f6f7f9; color: #1c2430; font-size: 14px; line-height: 1.55; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 24px 20px 60px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 28px 0 10px; border-left: 4px solid #3567a8;
     padding-left: 8px; }
.meta { color: #667; font-size: 12.5px; }
.cards { display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0 4px; }
.card { background: #fff; border: 1px solid #e3e7ec; border-radius: 8px;
        padding: 10px 14px; min-width: 118px; }
.card .v { font-size: 19px; font-weight: 600; }
.card .k { font-size: 11.5px; color: #778; }
table { border-collapse: collapse; width: 100%; background: #fff;
        border: 1px solid #e3e7ec; border-radius: 8px; overflow: hidden; }
th { background: #eef2f7; text-align: left; font-size: 12px; padding: 7px 10px;
     border-bottom: 1px solid #dde3ea; }
td { padding: 6px 10px; border-bottom: 1px solid #eef1f5; font-size: 12.5px;
     vertical-align: top; }
tr:last-child td { border-bottom: none; }
.fail { color: #b3261e; font-weight: 600; }
.ok { color: #1b7f3b; font-weight: 600; }
.tag { display: inline-block; background: #e8eef7; border-radius: 4px;
       padding: 1px 6px; font-size: 11px; margin-right: 4px; }
.tl { background: #fff; border: 1px solid #e3e7ec; border-radius: 8px;
      padding: 6px 0; max-height: 430px; overflow-y: auto; }
.tlrow { display: flex; gap: 10px; padding: 4px 14px; font-size: 12.5px;
         border-bottom: 1px solid #f0f2f5; }
.tlrow:last-child { border-bottom: none; }
.tlrow .t { color: #889; min-width: 118px; font-variant-numeric: tabular-nums; }
.tlrow .d { min-width: 52px; color: #556; text-align: right;
            font-variant-numeric: tabular-nums; }
.tlrow .n { min-width: 190px; font-family: Consolas, monospace; font-size: 12px; }
.tlrow.bad { background: #fdf3f2; }
.tlrow.bad .n { color: #b3261e; }
.mono { font-family: Consolas, monospace; font-size: 12px; }
pre { background: #fff; border: 1px solid #e3e7ec; border-radius: 8px;
      padding: 12px; overflow-x: auto; font-size: 12px; white-space: pre-wrap; }
.note { color: #987; font-size: 12px; }
"""


def _esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def generate(run_dir: Path | str, out_path: Path | str | None = None) -> Path:
    run_dir = Path(run_dir)
    out_path = Path(out_path) if out_path else run_dir / "report.html"

    cfg = _read_json(run_dir / "config.json") or {}
    trace = _read_jsonl(run_dir / "trace.jsonl")
    usage_rows = _read_jsonl(run_dir / "llm_usage.jsonl")
    ledger = _read_json(run_dir / "ledger.json") or []
    review_raw = _read_text(run_dir / "review.md", 4000)
    plan_head = _read_text(run_dir / "research_plan.md", 1200)
    t0 = trace[0]["ts"] if trace else None

    parts: list[str] = []
    A = parts.append

    # ---------- 头部
    final = cfg.get("final") or "(中断/未知)"
    final_cls = "ok" if final in ("passed",) else "fail" if "fail" in str(final) else ""
    A(f"<h1>GranuSci Run 报告 <span class='meta'>{_esc(run_dir.name)}</span></h1>")
    A(f"<div class='meta'>生成于 {datetime.now():%Y-%m-%d %H:%M} · "
      f"数据目录 {_esc(run_dir)}</div>")
    A("<div class='cards'>")
    A(f"<div class='card'><div class='v {final_cls}'>{_esc(final)}</div>"
      f"<div class='k'>终态</div></div>")
    A(f"<div class='card'><div class='v'>{cfg.get('n_steps', '-')}</div>"
      f"<div class='k'>LLM 步数</div></div>")
    A(f"<div class='card'><div class='v'>{len(trace)}</div>"
      f"<div class='k'>工具调用</div></div>")
    fails = sum(1 for e in trace if not e.get("ok", True))
    fail_cls = "fail" if fails else ""
    A(f"<div class='card'><div class='v {fail_cls}'>{fails}</div>"
      f"<div class='k'>失败调用</div></div>")
    A(f"<div class='card'><div class='v'>{len(ledger)}</div>"
      f"<div class='k'>账本条目</div></div>")
    A(f"<div class='card'><div class='v'>{_fmt_duration(cfg.get('elapsed_s'))}</div>"
      f"<div class='k'>总时长</div></div>")
    A("</div>")
    A(f"<div class='meta'>意图：{_esc(cfg.get('intent') or '(无 config.json——崩溃早期或 resume 片段)')}<br>"
      f"模型：agent={_esc(cfg.get('agent_model', '?'))} · reviewer={_esc(cfg.get('reviewer_model', '?'))}"
      f" · max_steps={_esc(cfg.get('max_steps', '?'))}</div>")

    # ---------- 研究计划（头部摘录）
    if plan_head:
        A("<h2>研究计划（摘）</h2>")
        A(f"<pre>{_esc(plan_head)}{'…' if len(plan_head) >= 1200 else ''}</pre>")

    # ---------- 时间线
    A("<h2>工具调用时间线</h2>")
    if not trace:
        A("<div class='note'>无 trace.jsonl 记录（崩溃于首次工具调用前，或文件缺失）</div>")
    else:
        A("<div class='tl'>")
        for e in trace:
            ts = datetime.fromtimestamp(e.get("ts", 0)).strftime("%H:%M:%S") if e.get("ts") else "?"
            ok = e.get("ok", True)
            args_s = json.dumps(e.get("args", {}), ensure_ascii=False)
            if len(args_s) > 110:
                args_s = args_s[:110] + "…"
            msg = ""
            if not ok:
                m = str((e.get("result") or {}).get("message")
                        or (e.get("result") or {}).get("error") or "")
                msg = f" — {_esc(m[:130])}"
            A(f"<div class='tlrow{' bad' if not ok else ''}'>"
              f"<span class='t'>{ts}</span>"
              f"<span class='d'>{_fmt_duration(e.get('duration'))}</span>"
              f"<span class='n'>{_esc(e.get('tool', '?'))}({_esc(args_s)}){msg}</span>"
              f"</div>")
        A("</div>")

    # ---------- 工具统计
    stats = _tool_stats(trace)
    if stats:
        A("<h2>工具统计（按累计耗时）</h2><table><tr><th>工具</th><th>次数</th>"
          "<th>失败</th><th>累计耗时</th></tr>")
        for s in stats:
            fail_cell = f"<span class='fail'>{s['fail']}</span>" if s["fail"] else "0"
            A(f"<tr><td class='mono'>{_esc(s['tool'])}</td><td>{s['n']}</td>"
              f"<td>{fail_cell}</td><td>{_fmt_duration(s['time'])}</td></tr>")
        A("</table>")

    # ---------- 账本
    A("<h2>研究状态账本</h2>")
    if not ledger:
        A("<div class='note'>无账本条目（record_finding 未被调用或 ledger.json 缺失）</div>")
    else:
        A("<table><tr><th>#</th><th>类型</th><th>置信</th><th>标题</th>"
          "<th>内容（截）</th><th>来源</th></tr>")
        for e in ledger:
            content = str(e.get("content", ""))
            if len(content) > 220:
                content = content[:220] + "…"
            A(f"<tr><td>L{e.get('seq', '?')}</td>"
              f"<td><span class='tag'>{_esc(e.get('kind', '?'))}</span></td>"
              f"<td>{_esc(e.get('confidence', '?'))}</td>"
              f"<td><b>{_esc(e.get('title', ''))}</b></td>"
              f"<td>{_esc(content)}</td>"
              f"<td class='meta'>{_esc(e.get('provenance', ''))[:80]}</td></tr>")
        A("</table>")

    # ---------- 审稿
    A("<h2>审稿结论</h2>")
    if review_raw:
        try:
            verdict = json.loads(review_raw)
            passed = verdict.get("pass")
            issues = verdict.get("critical_issues") or []
            A(f"<p>判定：<b class='{'ok' if passed else 'fail'}'>"
              f"{'PASS' if passed else 'FAIL'}</b> · "
              f"严重问题 {len(issues)} 条</p>")
            if issues:
                A("<ul>" + "".join(f"<li>{_esc(str(i))[:250]}</li>" for i in issues) + "</ul>")
            sugg = verdict.get("suggestions") or []
            if sugg:
                A("<p class='meta'>建议：" + "；".join(_esc(str(s))[:120] for s in sugg[:5]) + "</p>")
        except json.JSONDecodeError:
            A(f"<pre>{_esc(review_raw)}</pre>")
    else:
        A("<div class='note'>无 review.md（未到审稿阶段或运行中断）</div>")

    # ---------- 成本
    A("<h2>LLM 用量与成本</h2>")
    if not usage_rows:
        A("<div class='note'>无 llm_usage.jsonl（旧版本 run 或文件缺失）</div>")
    else:
        u = _usage_summary(usage_rows)
        prompt_total = sum(v["prompt"] for v in u["by_model"].values())
        completion_total = sum(v["completion"] for v in u["by_model"].values())
        A(f"<p>总 token <b>{_fmt_tokens(u['total'])}</b>"
          f"（prompt {_fmt_tokens(prompt_total)}"
          f" + completion {_fmt_tokens(completion_total)}）"
          f" · LLM 累计耗时 {_fmt_duration(u['llm_time'])}"
          f" · 调用 {len(usage_rows)} 次</p>")
        A("<table><tr><th>模型</th><th>调用</th><th>prompt</th><th>completion</th></tr>")
        for m, d in sorted(u["by_model"].items()):
            A(f"<tr><td class='mono'>{_esc(m)}</td><td>{d['calls']}</td>"
              f"<td>{_fmt_tokens(d['prompt'])}</td><td>{_fmt_tokens(d['completion'])}</td></tr>")
        A("</table>")
        A("<table style='margin-top:10px'><tr><th>用途</th><th>调用</th>"
          "<th>prompt</th><th>completion</th></tr>")
        for p, d in sorted(u["by_purpose"].items()):
            A(f"<tr><td><span class='tag'>{_esc(p)}</span></td><td>{d['calls']}</td>"
              f"<td>{_fmt_tokens(d['prompt'])}</td><td>{_fmt_tokens(d['completion'])}</td></tr>")
        A("</table>")

    doc = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
           f"<title>GranuSci Run { _esc(run_dir.name)}</title>"
           f"<style>{_CSS}</style></head><body><div class='wrap'>"
           + "".join(parts) + "</div></body></html>")
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print("用法: python -m granusci.run_report RUN_DIR [--out path.html]")
        sys.exit(2)
    run_dir = Path(sys.argv[1])
    out = None
    if "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1])
    if not run_dir.is_dir():
        print(f"目录不存在: {run_dir}")
        sys.exit(2)
    path = generate(run_dir, out)
    print(f"[run_report] {path}")


if __name__ == "__main__":
    main()
