---
name: paper-structure
description: 论文的论证结构学——摘要证据链、Introduction 漏斗、Results 主张升级链、Discussion 三级解释（Nature 系蒸馏）
when_to_use: 动笔写论文前载入（先于 paper-style）
---

# 论文论证结构学

## 摘要 = 最短证据链
`精确问题 → 研究设计 → 主发现 → 决定性支撑或边界 → 建立了什么 → 为何重要`。
重心必须是 "we found X"，不是 "we performed many simulations" 或 "we propose"。
背景一句带过；方法只讲理解发现所必需的逻辑（对照因素/参数范围），不写流水账；
数字按必要性给（主发现的定量锚点），结尾给概念性收获。单段 100-200 词，不引用文献。

## 文献对话管线（litx 制）

Introduction/Discussion 的每条文献主张走统一管线：`litx_search_external`
（发现）→ `litf_acquire`/`deep_read_paper`（精读）→ `litx_cite_register`
（登记 cite_id）→ 正文 \cite。写作期发现引用不足时的优先动作=回到
lit_survey 子代理补调研（不是稀释现有引用硬凑）。Discussion 的定位段
（"与已有工作的关系"）同样走此管线——每个定位主张一条可溯源引用。

## Introduction = 收敛漏斗，不是领域综述
`重要问题 → 具体现象 → 已有工作确立了什么 → 未解决的张力 → 精确的未知 → 研究问题 → 本文做什么`。
- **Gap 必须写成可填空的未知**："It remains unclear whether/why/under what conditions ___"——
  不是 "existing methods have limitations" 这种空话，更不是"没人用过我们的方法"；
- **文献按论证功能组织**：①建立已知 ②展示已解决 ③暴露边界/矛盾 ④使本研究的问题不可避免——
  每条引用服务一个主张，不做作者目录式罗列；
- 结论性收尾段：`研究问题 → 研究设计 → 决定性评价路线 → 推断范围`，不预告全部结果。

### 文献充分性判据（进入写作前自检）
Introduction 的每一条背景主张都锚定具体文献；研究主线各有代表——①现象发现史、②机制之争、③定量/标度先例、④方法线。总量 **≥20 篇**不足时回到文献工具补检索（gap-mining 的缺口清单是现成素材），不要用稀疏文献硬写。引用与上下文真实相关：每条 \cite 对应一个具体主张。

## 不确定性三元组（每个主结果的强制结构）

Results 中每个 carry 权重的主结果（图表核心/摘要出现）必须以三元组呈现：

1. **结论**：本结果确立了什么（一个可陈述的主张，带定量锚点）；
2. **证据**：支撑它的数据与统计（n、拟合质量、对照、出处=账本/工具记录）；
3. **不确定性**：这个结论的边界——误差量级、适用条件、被排除的替代解释
   的残余风险、样本/预算限制。

第三条不是免责声明堆砌：写"误差 ±X%（n=30 留出验证）"“该边界仅在
Γ≤6 域内确立""替代解释 A 已被对照 Y 排除，但 B 未直接检验"这类**可核查
的边界陈述**。审稿人对"过度声明"的攻击大多落在缺第三条的结果上；
而有无第三条也是"数据报告"与"可投稿论文"的分水岭。

## Results = 主张升级链，不是实验流水账
- 每个 subsection 回答**一个**科学问题、确立**一个**新主张；相邻小节互为因果：
  `观察 → 引出问题 → 定向实验 → 局部解读 → 更强主张 → 下一问题`；
- 论证升级方向：`建立现象 → 压力测试 → 排除替代解释 → 拓宽 → 解读 → 划界`——
  同一主张的重复演示不算新小节，两个小节若推断增益相同则合并；
- **机制问题优先用诊断性证据**（扰动/对照/边界测试），不是"有 X 和没 X 的对比"；
- 每个证据单元以**一个推断收尾**（不超出已评估条件的范围），不要止于数字罗列；
- 局部解读门槛：`suggests/indicates` 只在①回答上一结果引出的问题②证据就在本小节③不确定性与替代解释显式保留时使用。

## Discussion = 综合，不是重放
`压缩的中心发现锚（可含一个关键数字） → 跨结果综合 → 与已有工作的定位关系 → 解释与贡献 → 边界与局限 → 有判别力的下一步问题`。
- **解释分三级，措辞随之校准**：
  ① 确立的贡献（设计直接支撑）→ 陈述句；
  ② 证据支持的解读（机制解释被证据偏好但非唯一）→ "is consistent with / suggests"；
  ③ 开放含义（更广但需新检验）→ "may"——把机制写成定论是最常见的过度声明；
- 文献定位要具体：支持/独立确认/拓展到新机制或范围/限定成立条件/与谁冲突需调和——不是装饰性引用堆。

## 文档类骨架（照抄，勿改选项）
```latex
\documentclass[aps,pre,twocolumn,superscriptaddress,floatfix]{revtex4-2}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\begin{document}
\title{...}
\author{GranuSci Autonomous Research Agent}
\affiliation{Autonomous Research Platform}
\date{\today}
\begin{abstract}
单段约 100--200 词：问题、方法、关键定量结果（含数字）、结论。不引用文献。
\end{abstract}
\maketitle
\section{Introduction}      % 漏斗收敛 → gap → 本文贡献（见上）
\section{System and methods} % 物理模型、参数、协议、分析方法
\section{Results and discussion} % 主张升级链，可拆多个 \subsection
\section{Conclusions}
\begin{thebibliography}{99}
\bibitem{key} 作者, 期刊 \textbf{卷}, 页 (年份).
\end{thebibliography}
\end{document}
```


## 深层参考（load_skill(name="paper-structure", deep=...) 按需展开）
- `evidence-map`：引用-主张证据地图模板与硬门（写 Introduction 前必建——引用不足/断链的根治）
- `modality-calibration`：证据强度-语气校准表（结论措辞分级，防过度声明）
- `claim-escalation`：Results 证据链原型与主张升级模式库
- `discussion-functions`：Discussion 四功能（锚定/定位/解释/贡献）+ 局限=主张边界写法
