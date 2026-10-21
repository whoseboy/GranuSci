# -*- coding: utf-8 -*-
"""slides 叙事结构自检测试：paper13 式"上来就讲工作"的 slide 会被拦截。"""
from __future__ import annotations

from granusci import slides_tools as st

GOOD = r"""
\documentclass[aspectratio=169,10pt]{beamer}
\begin{document}
\title{动能标度在 $a/d$ 方向分裂为两域}
\author{GranuSci}
\date{2026}
\begin{frame}
\titlepage
\end{frame}
\begin{frame}{背景：振动颗粒床的相态现象学}
颗粒床随参数呈现紧致弹跳/过渡/Leidenfrost 三相态。
\end{frame}
\begin{frame}{缺口：势能标度与能量分配未知}
已知动能标度律（文献 A）；已知形态学相图（文献 B）；
但势能从未被标度刻画、能量分配无定量描述。
\end{frame}
\begin{frame}{本研究：516 工况 DEM + DDDA}
一句话问题 + 途径。
\end{frame}
\begin{frame}{系统与协议}
准二维槽，两轮自适应采样。
\end{frame}
\begin{frame}{结果：动能两子域纯幂律}
公式与图。
\end{frame}
\begin{frame}{验证：留出 6 点 MAPE 16.5\%}
表。
\end{frame}
\begin{frame}{结论}
1. 两域标度；2. 连续 crossover；3. 判据。
\end{frame}
\begin{frame}{References}
\begin{thebibliography}{9}
\bibitem{a} A.
\end{thebibliography}
\end{frame}
\end{document}
"""

PAPER13_LIKE = r"""
\documentclass[aspectratio=169,10pt]{beamer}
\begin{document}
\begin{frame}{振动颗粒床的能量标度：动能与势能如何分配？}
张力 + 三个答案。
\end{frame}
\begin{frame}{方法一：批量 DEM + 两轮自适应设计}
方法内容。
\end{frame}
\begin{frame}{结果一：动能两子域纯幂律}
结果内容。
\end{frame}
\begin{frame}
\centering Thank you
\end{frame}
\end{document}
"""


def test_good_structure_passes():
    assert st.slide_structure_audit(GOOD) == []


def test_paper13_like_structure_rejected():
    issues = st.slide_structure_audit(PAPER13_LIKE)
    texts = " ".join(issues)
    # 必须拦住：无标题页、无背景、无缺口、纯致谢帧
    assert any("标题页" in i for i in issues)
    assert any("背景" in i for i in issues)
    assert any("缺口" in i for i in issues)
    assert any("致谢" in i for i in issues)


def test_actual_paper13_slides_pass_after_fix(tmp_path):
    """真实 paper13 slides.tex 已按结构审计意见重构（2026-09-03 重排版，
    决赛报告附录F 选页即基于该修订版），应通过审计；历史版本"上来就讲工作"
    的缺陷形态由 PAPER13_LIKE 合成样例回归覆盖。"""
    src = (r"C:\Users\D0n9\Desktop\2026职工大赛——科研智能平台创新应用"
           r"\granusci\runs\20260902_200025_paper13\slides.tex")
    from pathlib import Path
    p = Path(src)
    if not p.exists():   # run 目录被清理时跳过
        return
    issues = st.slide_structure_audit(p.read_text(encoding="utf-8"))
    assert not issues, f"修订版 paper13 slides 不应再触发结构审计：{issues}"
