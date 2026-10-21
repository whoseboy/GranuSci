# -*- coding: utf-8 -*-
"""Beamer 幻灯片工具：叙事结构自检 + XeLaTeX 编译（make_slides 的后端）。

与 latex_tools（论文 pdflatex）平行的汇报通道。MiKTeX 环境已验证 xelatex 可用。
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path


# ---------------------------------------------------------------- 叙事结构自检
def slide_structure_audit(tex: str) -> list[str]:
    """slides-making 技能的机械可检部分（硬约束）。返回问题清单（空=通过）。

    完整的逐帧自检（主张句质量、逻辑递进）由智能体按技能清单自检，这里只拦
    结构性缺失——titlepage/背景/缺口/结论缺失、纯致谢帧、帧数失衡。
    """
    issues: list[str] = []
    # \begin{frame}{标题} 与 \begin{frame}[plain]{标题}；无标题帧单独处理
    body_part = tex.split("\\appendix")[0]
    body_frames = re.findall(
        r"\\begin\{frame\}(?:\[[^\]]*\])?\s*\{(.+?)\}", body_part)
    # 无标题帧内容（Thank-you 式）：\begin{frame} 后不接 {...}
    untitled_blocks = re.findall(
        r"\\begin\{frame\}(?:\[[^\]]*\])?(?!\s*\{)(.*?)\\end\{frame\}",
        body_part, re.S)
    for blk in untitled_blocks:
        if re.search(r"(Thank[s]? ?[Yy]ou|谢谢|致谢)", blk):
            issues.append("存在纯致谢帧——结论帧即结尾，删除或并入结论")
    n = tex.count("\\begin{frame}")
    # 标题帧：\titlepage 或 \maketitle
    has_titlepage = bool(re.search(r"\\(titlepage|maketitle)", tex))
    if not has_titlepage:
        issues.append("缺标题页（第 1 帧应为 \\titlepage：题目+作者+日期，无其他内容）")
    if body_frames:
        # 背景帧：前 3 个正文帧的标题里应有背景性词汇（不含本研究的具体结果）
        head = " ".join(body_frames[:3])
        bg_words = ["背景", "研究意义", "领域", "物理", "现象", "引言", "Introduction",
                    "Background", "Motivation", "意义", "问题", "挑战", "现状"]
        if not any(w in head for w in bg_words):
            issues.append("前 3 帧未见背景性标题——上来就展示工作了；"
                          "应先 1-2 帧讲研究对象与领域意义（不含本研究结果）")
        # 缺口帧
        gap_words = ["缺口", "空白", "未知", "尚未", "没人", "局限", "open", "gap",
                     "Gap", "GAP", "未解决", "留下的问题", "unresolved"]
        if not any(w in " ".join(body_frames) for w in gap_words):
            issues.append("缺缺口帧（\"已知…但…未知\"——为什么要干这个）")
        # 研究问题/目标帧
        q_words = ["研究问题", "本研究", "本文", "目标", "科学问题", "我们研究",
                   "Research question", "This work", "approach", "途径"]
        if not any(w in " ".join(body_frames[:6]) for w in q_words):
            issues.append("缺研究问题帧（缺口之后、方法之前：问题一句话+途径一句话）")
        # 结论帧
        concl = body_frames[-2:] if len(body_frames) >= 2 else body_frames
        c_words = ["结论", "takeaway", "Takeaway", "总结", "Conclusion", "Conclusions"]
        if not any(w in " ".join(concl) for w in c_words):
            issues.append("缺结论帧（3 条带数字的 takeaway，呼应当初缺口）")
        # 纯致谢帧
        for t in body_frames:
            if re.fullmatch(r"\s*(Thank[s]? ?[Yy]ou|谢谢|致谢)\s*", t.strip()):
                issues.append(f"存在纯致谢帧\"{t.strip()}\"——结论帧即结尾，删除或并入结论")
    # 帧数合理性（正文帧 6-30；时长表的下限 8 含 References，硬约束留裕量）
    if body_frames and not (6 <= len(body_frames) <= 30):
        issues.append(f"正文帧数 {len(body_frames)} 失衡（合理区间 6-30，按时长分配表核对）")
    return issues


def compile_slides(run_dir: Path, tex_name: str = "slides.tex") -> dict:
    """XeLaTeX 两轮编译 + 结构化错误返回 + 页数统计。"""
    tex_path = run_dir / tex_name
    if not tex_path.exists():
        return {"error": True, "message": f"{tex_path} 不存在——先 make_slides 写入"}
    errors = []
    for i in range(2):
        r = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_name],
            cwd=str(run_dir), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=180)
        if r.returncode != 0:
            # 提取错误行
            err_lines = [l for l in (r.stdout or "").splitlines()
                         if l.startswith("!")][:5]
            errors = err_lines or [f"exit {r.returncode}"]
            return {"error": True, "errors": errors,
                    "message": "XeLaTeX 编译失败：" + "; ".join(errors)}
    pdf = run_dir / tex_name.replace(".tex", ".pdf")
    if not pdf.exists():
        return {"error": True, "message": "编译完成但 PDF 未生成（检查 documentclass）"}
    # 页数（xaux 无用——直接从 log 拿）
    log = (run_dir / tex_name.replace(".tex", ".log")).read_text(
        encoding="utf-8", errors="replace")
    m = re.search(r"Output written on .*?\((\d+) page", log)
    n_pages = int(m.group(1)) if m else None
    # 未定义引用检查
    undefined = re.findall(r"LaTeX Warning: (Reference|Citation).*undefined", log)
    return {"ok": True, "pdf": str(pdf), "n_pages": n_pages,
            "undefined_refs": len(undefined),
            "message": f"编译成功：{n_pages} 页"
            + (f"，{len(undefined)} 个未定义引用" if undefined else "")}
