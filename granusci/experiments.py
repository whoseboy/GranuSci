# -*- coding: utf-8 -*-
"""⑫ 实验矩阵 runner：消融/横评实验的批量执行与结果收集底盘。

用法：
  1) 写实验矩阵（JSON 文件）：
     {
       "name": "reviewer_ablation",
       "base": {"intent": "研究……", "max_steps": 300},
       "grid": [
         {"id": "review_on"},
         {"id": "review_off", "flags": ["--no-reviewer"]}
       ],
       "seeds": [1, 2, 3]
     }
     grid 单元可覆盖 base 的字段：intent/model/reviewer/max_steps/flags(附加 CLI 开关)。
  2) python -m granusci.experiments matrix.json
  3) 产物在 granusci/experiments/<name>/：
       manifest.jsonl  每 run 的执行日志（journal；跳过判断不看它）
       runs/<tag>/     各 run 目录（--run-dir 固定路径，run.py 原生产物）
       results.csv     逐 run 指标
       summary.md      按配置聚合（均值±标准差）

设计：
- 顺序执行（每 run 是 1-2 小时级的重任务；并行受 LLM 限流与服务器负载约束，
  需要时后续再加 --concurrency）
- 断点续跑以 run 目录 config.json 的 final 为准：自然终态（passed/
  review_failed_twice/max_steps/submitted_no_review）跳过；可恢复终态（配额/
  服务故障）用 --run-dir 续跑而非重开新目录；外层监督器反复调本 runner 即可
  自动吃完配额死亡
- 收集容错：崩溃 run（config.json final="(运行中/中断)"）照收，指标缺什么记什么
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

from . import config

EXP_ROOT = config.ROOT / "granusci" / "experiments"

# 收集的逐 run 指标（列顺序即 CSV 列顺序）
METRIC_FIELDS = [
    "exp", "grid_id", "seed", "run_dir", "final",
    "n_steps", "tool_calls", "ledger_entries", "elapsed_s",
    "review_pass", "review_rounds", "orphan_numbers",
    "llm_calls", "prompt_tokens", "completion_tokens", "total_tokens",
]


# ---------------------------------------------------------------- 矩阵展开
def expand_matrix(spec: dict) -> list[dict]:
    """grid × seeds → run 单元列表。"""
    base = spec.get("base") or {}
    seeds = spec.get("seeds") or [None]
    runs = []
    for cell in spec.get("grid") or []:
        if not cell.get("id"):
            raise ValueError(f"grid 单元缺 id: {cell}")
        for seed in (cell.get("seeds") or seeds):   # 单元可覆盖种子集
            runs.append({
                "exp": spec["name"],
                "grid_id": cell["id"],
                "seed": seed,
                "intent": cell.get("intent", base.get("intent")),
                "model": cell.get("model", base.get("model")),
                "reviewer": cell.get("reviewer", base.get("reviewer")),
                "max_steps": cell.get("max_steps", base.get("max_steps", 300)),
                "flags": list(cell.get("flags", base.get("flags", []))),
            })
    if not runs:
        raise ValueError("矩阵为空（grid 无单元）")
    for r in runs:
        if not r["intent"]:
            raise ValueError(f"intent 缺失（base 与 grid 单元都没给）: {r['grid_id']}")
    return runs


def _run_tag(r: dict) -> str:
    seed_part = f"_s{r['seed']}" if r["seed"] is not None else ""
    return f"exp_{r['exp']}_{r['grid_id']}{seed_part}"


# ---------------------------------------------------------------- 指标收集
def _collect_one(run_dir: Path) -> dict:
    m = {"run_dir": str(run_dir)}
    cfg = {}
    try:
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    m["final"] = cfg.get("final", "(无 config.json)")
    m["n_steps"] = cfg.get("n_steps")
    m["tool_calls"] = cfg.get("tool_calls")
    m["ledger_entries"] = cfg.get("ledger_entries")
    m["elapsed_s"] = cfg.get("elapsed_s")
    # 审稿
    m["review_pass"] = None
    m["review_rounds"] = cfg.get("review_rounds")
    m["orphan_numbers"] = None
    try:
        rv = json.loads((run_dir / "review.md").read_text(encoding="utf-8"))
        m["review_pass"] = bool(rv.get("pass"))
        audit = rv.get("numeric_audit") or {}
        m["orphan_numbers"] = len(audit.get("orphans") or [])
    except Exception:
        pass
    # 用量
    m["llm_calls"] = m["prompt_tokens"] = m["completion_tokens"] = m["total_tokens"] = None
    try:
        u = json.loads((run_dir / "usage.json").read_text(encoding="utf-8"))
        m["llm_calls"] = u.get("total_calls")
        tok = sum(v.get("total_tokens", 0) for v in (u.get("by_model") or {}).values())
        ptok = sum(v.get("prompt_tokens", 0) for v in (u.get("by_model") or {}).values())
        ctok = sum(v.get("completion_tokens", 0) for v in (u.get("by_model") or {}).values())
        m["prompt_tokens"], m["completion_tokens"], m["total_tokens"] = ptok, ctok, tok
    except Exception:
        pass
    return m


def collect(exp_dir: Path, spec: dict) -> list[dict]:
    """扫描 exp_dir/runs/* 收集指标，写 results.csv。"""
    rows = []
    for run_dir in sorted((exp_dir / "runs").iterdir()) if (exp_dir / "runs").is_dir() else []:
        if not run_dir.is_dir():
            continue
        # 从目录名解析 grid_id/seed（exp_<name>_<grid>[_sN]）
        name = run_dir.name
        grid_id, seed = None, None
        if name.startswith("exp_"):
            body = name[len(f"exp_{spec['name']}_"):] if name.startswith(f"exp_{spec['name']}_") else name[4:]
            if body.endswith("]") or "_s" in body:
                i = body.rfind("_s")
                if i > 0 and body[i + 2:].isdigit():
                    grid_id, seed = body[:i], int(body[i + 2:])
            if grid_id is None:
                grid_id = body
        m = _collect_one(run_dir)
        m.update({"exp": spec["name"], "grid_id": grid_id, "seed": seed})
        rows.append({k: m.get(k) for k in METRIC_FIELDS})
    if rows:
        with (exp_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
            w.writeheader()
            w.writerows(rows)
    return rows


def _agg(vals: list) -> str:
    """均值±标准差（样本≥2）；None 全缺记 '-'。"""
    xs = [v for v in vals if isinstance(v, (int, float))]
    if not xs:
        return "-"
    if len(xs) == 1:
        return f"{xs[0]:.1f}"
    return f"{mean(xs):.1f}±{pstdev(xs):.1f}"


def summarize(rows: list[dict], exp_dir: Path) -> Path:
    """按 grid_id 聚合写 summary.md。"""
    by_grid: dict[str, list[dict]] = {}
    for r in rows:
        by_grid.setdefault(r.get("grid_id") or "?", []).append(r)
    lines = [f"# 实验汇总：{rows[0]['exp'] if rows else '?'}",
             f"共 {len(rows)} 个 run\n",
             "| 配置 | n | 终态分布 | 步数 | 工具调用 | 账本条目 | 审稿通过 | 孤儿数值 | total tokens | 时长(s) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for gid, rs in sorted(by_grid.items()):
        finals = {}
        for r in rs:
            finals[r.get("final") or "?"] = finals.get(r.get("final") or "?", 0) + 1
        final_s = ", ".join(f"{k}×{v}" for k, v in sorted(finals.items()))
        passes = [r["review_pass"] for r in rs if r["review_pass"] is not None]
        pass_rate = f"{sum(passes)}/{len(passes)}" if passes else "-"
        lines.append(
            f"| {gid} | {len(rs)} | {final_s} "
            f"| {_agg([r['n_steps'] for r in rs])} "
            f"| {_agg([r['tool_calls'] for r in rs])} "
            f"| {_agg([r['ledger_entries'] for r in rs])} "
            f"| {pass_rate} "
            f"| {_agg([r['orphan_numbers'] for r in rs])} "
            f"| {_agg([r['total_tokens'] for r in rs])} "
            f"| {_agg([r['elapsed_s'] for r in rs])} |")
    path = exp_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- 执行
# 可恢复终态（配额/服务故障/异常退出）：重跑矩阵时用 --run-dir 续跑而非重开
RECOVERABLE_FINALS = {"llm_quota_exhausted", "llm_unavailable", "empty_response",
                      "aborted"}
# 自然完成终态：不再执行
NATURAL_FINALS = {"passed", "review_failed_twice", "max_steps", "submitted_no_review"}


def _dir_final(run_dir: Path) -> str | None:
    """读 run 目录 config.json 的终态字段（无 config 返回 None）。"""
    try:
        return json.loads((run_dir / "config.json").read_text(
            encoding="utf-8")).get("final")
    except Exception:
        return None


def _has_state(run_dir: Path) -> bool:
    """run 目录 checkpoint 有无实际状态（防'config 有但启动即崩'的假续跑）。"""
    ckpt = run_dir / "checkpoint.sqlite"
    if not ckpt.exists():
        return False
    try:
        import sqlite3
        con = sqlite3.connect(str(ckpt))
        try:
            return con.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] > 0
        finally:
            con.close()
    except Exception:
        return False


def _build_cmd(r: dict, run_dir: Path, resume: bool) -> list[str]:
    if resume:
        cmd = [sys.executable, "-m", "granusci.run", "--resume", str(run_dir),
               "--yolo", "--max-steps", str(r["max_steps"])]
        if r["seed"] is not None:
            cmd += ["--seed", str(r["seed"])]
        cmd += r["flags"]
        return cmd
    cmd = [sys.executable, "-m", "granusci.run", r["intent"],
           "--run-dir", str(run_dir), "--yolo",
           "--max-steps", str(r["max_steps"])]
    if r["model"]:
        cmd += ["--model", r["model"]]
    if r["reviewer"]:
        cmd += ["--reviewer", r["reviewer"]]
    if r["seed"] is not None:
        cmd += ["--seed", str(r["seed"])]
    cmd += r["flags"]
    return cmd


def run_matrix(spec_path: Path, only_collect: bool = False,
               log_fn=print) -> Path:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    exp_dir = EXP_ROOT / spec["name"]
    runs_dir = exp_dir / "runs"
    exp_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(exist_ok=True)
    manifest_path = exp_dir / "manifest.jsonl"

    units = expand_matrix(spec)
    # 断点续跑以 run 目录的 config.json 终态为准（manifest 只作日志）
    done = sum(1 for u in units if _dir_final(runs_dir / _run_tag(u)) in NATURAL_FINALS)
    log_fn(f"[exp] {spec['name']}: {len(units)} 个 run（{done} 已自然完成）")

    if not only_collect:
        for u in units:
            tag = _run_tag(u)
            run_dir = runs_dir / tag
            dir_final = _dir_final(run_dir)
            if dir_final in NATURAL_FINALS:
                log_fn(f"[exp] 跳过已完成: {tag} (final={dir_final})")
                continue
            resume = _has_state(run_dir)   # 有 checkpoint 状态 = 真跑过（含可恢复终态）
            cmd = _build_cmd(u, run_dir, resume=resume)
            log_fn(f"[exp] 启动 {tag}{' [续]' if resume else ''}: "
                   f"{' '.join(cmd[2:8])}…")
            t0 = time.time()
            # 逐 run 日志实时落盘（进程留痕 + 监控可见；capture_output 会缓冲到进程结束）
            run_log = exp_dir / "logs" / f"{tag}.log"
            run_log.parent.mkdir(exist_ok=True)
            with open(run_log, "ab") as lf:
                proc = subprocess.run(cmd, cwd=str(config.ROOT),
                                      stdout=lf, stderr=subprocess.STDOUT)
            final = _dir_final(run_dir) or "?"
            if final in NATURAL_FINALS:
                status = "done"
            elif final in RECOVERABLE_FINALS:
                status = "recoverable"
            else:
                status = "failed"
            try:
                tail = run_log.read_text(encoding="utf-8", errors="replace")[-500:]
            except Exception:
                tail = ""
            with manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "tag": tag, "grid_id": u["grid_id"], "seed": u["seed"],
                    "status": status, "rc": proc.returncode, "final": final,
                    "wall_s": round(time.time() - t0, 1),
                    "tail": tail},
                    ensure_ascii=False) + "\n")
            log_fn(f"[exp] {tag} -> {status} (final={final}, "
                   f"{(time.time() - t0) / 60:.1f}min)")
            # run.py 崩溃时产物仍在 run_dir——失败 run 也收集（失败本身是实验数据）

    rows = collect(exp_dir, spec)
    if not rows:
        log_fn("[exp] 无可收集的 run")
        return exp_dir
    summary = summarize(rows, exp_dir)
    log_fn(f"[exp] 收集 {len(rows)} 个 run -> {exp_dir / 'results.csv'}")
    log_fn(f"[exp] 汇总 -> {summary}")
    return exp_dir


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print("用法: python -m granusci.experiments MATRIX.json [--collect-only]")
        sys.exit(2)
    only_collect = "--collect-only" in sys.argv
    run_matrix(Path(sys.argv[1]), only_collect=only_collect,
               log_fn=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
