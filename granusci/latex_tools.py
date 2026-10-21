# -*- coding: utf-8 -*-
"""LaTeX 手稿工具：write_manuscript / compile_latex 的后端。

编译用本地 MiKTeX pdflatex（REVTeX 4.2 已装，AutoInstall=1）。
编译错误以结构化形式返回，供智能体自主修复迭代。
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

LATEX_TIMEOUT = 240


def compile_latex(work_dir: Path, tex_name: str = "paper.tex", passes: int = 3) -> dict:
    """编译 work_dir/tex_name，跑 passes 遍（交叉引用收敛）。

    编译器自动选择：slides.tex（beamer）用 xelatex，其余 pdflatex
    （beamer 的中文/CJK 与主题需要 xelatex——slides-making 技能硬规则）。
    返回 {ok, pdf, errors: [...], warnings_count, log_tail}
    """
    tex_path = work_dir / tex_name
    if not tex_path.exists():
        return {"ok": False, "error": "tex_not_found", "message": f"找不到 {tex_path}"}

    # beamer 判定：文档类含 beamer 或文件名是 slides
    head = tex_path.read_text(encoding="utf-8", errors="replace")[:800]
    is_beamer = "beamer" in head or Path(tex_name).stem.startswith("slides")
    engine = "xelatex" if is_beamer else "pdflatex"

    env = dict(os.environ)
    env["MIKTEX_AUTOINSTALL"] = "1"          # 无头自动装包（已全局设置，双保险）
    last_log = ""
    for i in range(passes):
        try:
            r = subprocess.run(
                [engine, "-interaction=batchmode", "-halt-on-error", tex_name],
                cwd=str(work_dir), capture_output=True, text=True,
                timeout=LATEX_TIMEOUT, env=env)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "compile_timeout",
                    "message": f"{engine} 超时（>{LATEX_TIMEOUT}s）"}
        last_log = r.stdout or ""

    log_path = work_dir / (Path(tex_name).stem + ".log")
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else last_log

    pdf_path = work_dir / (Path(tex_name).stem + ".pdf")
    if not pdf_path.exists():
        errors = _extract_errors(log_text)
        return {"ok": False, "error": "compile_failed", "errors": errors,
                "message": "编译失败，请按 errors 修复 tex 后重新 compile_latex",
                "log_tail": log_text[-1500:]}

    errors = _extract_errors(log_text)
    undefined_refs = re.findall(r"LaTeX Warning: (Reference .*?|Citation .*?) undefined",
                               log_text)
    n_pages = _count_pages(log_text)
    if not errors and not undefined_refs:
        msg = "编译成功"
    else:
        msg = (f"编译成功但有 {len(errors)} 处错误残留、"
               f"{len(undefined_refs)} 处未定义引用（建议再修一轮后重编译）")
    return {"ok": True, "pdf": str(pdf_path), "n_pages": n_pages,
            "errors": errors, "undefined_references": undefined_refs[:10],
            "message": msg}


def _extract_errors(log: str) -> list[str]:
    errs = []
    for m in re.finditer(r"^! (.*?)(?:\n((?:l\.\d+.*?\n?)?))", log, re.M):
        errs.append((m.group(1) + " " + (m.group(2) or "").replace("\n", " ")).strip()[:300])
    return errs[:15]


def _count_pages(log: str) -> int:
    m = re.search(r"Output written on .*?\((\d+) pages?", log)
    return int(m.group(1)) if m else 0
