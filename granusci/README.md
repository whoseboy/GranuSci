# GranuSci —— 本地自主科研智能体

面向颗粒体系与离散元方向的自主科研智能体：一句研究意图进，一份可审计的研究报告出。
本目录是**平台无关的全新实现**（磐石平台工作流的降级约束全部解除），同时作为后续
论文与消融实验的代码基线。

## 架构

```
用户意图
  │
  ▼
┌─────────────────────────────────────────────────────┐
│ LangGraph 编排                                        │
│   agent ⇄ tools（全量工具自主调用）                    │
│      └─ submit_report ──> reviewer（审稿门）          │
│                           ├─ PASS → 交付              │
│                           └─ FAIL → 反馈回改（≤2轮）  │
├─────────────────────────────────────────────────────┤
│ 工具面（统一注册，智能体一视同仁）                      │
│  MCP·DEM   LIGGGHTS 批量仿真 5 工具                    │
│  MCP·DDDA  量纲分析+π定理+规律发现 7 工具               │
│  MCP·文献  波尔检索/查新 2 工具                        │
│  本地：kb_search（论文知识库 RAG）                     │
│  本地：load_skill（方法论技能，按需载入）              │
│  本地：submit_report（终态动作）                       │
├─────────────────────────────────────────────────────┤
│ 语义层与知识解耦：系统提示词只含行为约束；              │
│ 领域知识全部走 知识库/技能/文献 三条检索线              │
└─────────────────────────────────────────────────────┘
```

关键设计：

- **数值保真双保险**：审稿门内置确定性数值审计（报告中小数必须在工具返回记录中
  找得到出处）+ 异源模型（审稿人与主循环不同厂商）LLM 评审。
- **知识不进提示词**：博士论文全文、DDDA 方法论、工具契约全部在向量库
  （GLM-Embedding-3 + chroma）；操作方法蒸馏为 4 个技能按需加载。
- **全工具自主**：主循环不给任何固定节点顺序，研究路线（文献→π→设计→仿真→
  分析→加密→查新→报告）由智能体依据技能指引与工具证据自主走。

## 使用

```bash
pip install -r requirements.txt

# 构建知识库（首次）
python -m granusci.kb build

# 一句意图跑完整研究
python -m granusci.run "研究振动过程中颗粒床动能与势能的转化规律" --tag e1

# 模型可换（默认 agent=GLM-5.3, reviewer=DeepSeek-V4-Pro）
python -m granusci.run "..." --model DeepSeek-V4-Pro --reviewer GLM-5.1
```

产物在 `runs/<时间戳>_<tag>/`：report.md、review.md、trace.jsonl（全部工具调用）、
messages.json、config.json。

## 目录

```
config.py        端点/模型/路径（.env 在工作区根）
llm.py           paratera chat+embedding 客户端
mcp_bus.py       MCP 工具总线（发现/注册/分发/留痕）
kb.py            知识库构建与检索
skills.py        技能装载
skills/*.md      4 个方法论技能
agent/           state / prompts / graph（LangGraph）
run.py           CLI 入口
runs/            每次运行的全部产物
```

## 依赖

- 三个 MCP 计算服务（granular-dem-mcp / DDDA-MCP / lit-mcp），端点经环境变量配置（.env: DEM_MCP_URL / DDDA_MCP_URL / LITX_MCP_URL）
- paratera LLM API（.env: PARATERA_BASE_URL / PARATERA_API_KEY）
