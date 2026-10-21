---
name: paper-style
description: 论文英文文风与图表排版规范——AI 词警惕表、句式多样性、图表设计要求、LaTeX 纪律（paper-structure 之后载入）
when_to_use: 论文结构定了、开始写正文时载入
---

# 论文文风与图表规范

## 英文文风（自然科学语域）
**语域**：正式、客观、方法导向；被动语态常见（"was measured"）；术语精确，SI 单位，统计记号规范。

**高频 AI 词警惕表**——遇到这些词自问"这是这里最精确的词，还是默认套话？"：
| 套话 | 更好的选择 |
|---|---|
| delve | examine, investigate, analyze |
| pivotal / crucial | important, central, necessary（按实际分量选）|
| showcase | demonstrate, reveal, present |
| leverage | use, employ, apply |
| landscape / tapestry | field, system, interplay |
| testament to | evidence of, indicates |
| navigate | address, manage |
| plays a crucial role | 用具体作用替代（"dominates the response" 而非 "plays a role"）|

**句式多样性**：长短句交替（复杂思想用短句，简单事实可合并）；相邻句不等长；一段一个论证单元。避免连续同结构句（"X was measured. Y was measured. Z was measured."）。

**精确性**：技术术语首次出现即定义；代词（this/it）须有明确先行词；数据主张带条件与范围（"for Γ ≥ 7"而非裸的"at high Γ"）；不确定的结论用限定语（suggests, is consistent with），确定的用陈述句。

**客观性**：每个 claim 挂证据；承认局限与替代解释；不拔高（"first"须有查新证据，"significant"须有统计支撑）。

## 图表规范
- 每张图：`make_figure` 生成 fig_N.pdf → 论文中
  `\begin{figure}[t] \includegraphics[width=\columnwidth]{fig_1} \caption{...} \label{fig:1} \end{figure}`
  （双栏图用 `figure*` 与 `width=0.85\textwidth`）
- **每张图必须在正文中被引用（Fig.~\ref{fig:1}）并至少有一段实质讨论**——只挂图不讨论是审稿大忌；
- **图的设计层**：一张图回答一个 Results 级科学问题；多面板时各面板承担不同论证角色（现象/机制/边界），不是同一指标的机械重复；若删除某面板论证链不受损，则删；
- 图注自含：不看正文也能懂该图（说明符号、条件、单位）；图号+描述性标题，必要的 Note 说明实验条件；
- **核心定量图的设计要求**：标度律验证图（预测 vs 实测）按主要自变量分色/分符号并画 y=x 参考线，图注注明 R² 与 δ*；参数扫描面板图（同一响应量跨参数组）统一 y 轴范围以便跨面板直接对比；对数跨度的自变量用 log 轴；
- 数据点密集时用透明度或错开符号避免视觉重叠；色板用感知均匀序列（Viridis 系），避免红绿对撞；
- 表格用 `booktabs` 三线表（\toprule\midrule\bottomrule）；
- 数据系列较多时在 caption 里说明符号含义，不要塞满图例。

## LaTeX 语法纪律（避免编译错误）
- 特殊字符转义：& % # → \& \% \#；
- 数学一律 $...$ 或 equation 环境；上下标 $r_s$、$\Gamma$；
- 引用交叉：Fig.~\ref{fig:1}、Table~\ref{tab:1}、Eq.~(\ref{eq:1})；
- 参考文献用 thebibliography 环境手写条目（不用 bibtex），每条来自工具记录中真实检索到的文献；引用处 \cite{key}。

## 数值保真纪律（最高优先）
- 每个数字来自工具返回记录；摘要、正文、图注中的数字须一致；
- 不确定就回查工具记录，不要凭印象。

## 内容要求
- Introduction 的文献综述须基于 lit_literature_search / lit_survey 的真实检索结果，并引用之；
- Methods 给出可复现细节：参数、网格、协议、分析设置（ra 等）；
- Results 里明确区分工具事实与解读推测；局限性如实写（可放 Conclusions 末段）；
- 英文撰写（Physical Review E 为英文期刊），术语准确，句式简洁。
