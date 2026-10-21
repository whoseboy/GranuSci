# -*- coding: utf-8 -*-
"""GranuSci 本地自主科研智能体 —— 全局配置。

设计原则：
- 平台时代的自我设限全部解除：模型、工具、知识库、技能全量可用。
- 语义层（LLM）只带基本行为约束；领域知识一律走知识库/技能/工具，不注入提示词。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 工作区根（granusci/ 的上一级）
ROOT = Path(__file__).resolve().parent.parent
PKG = Path(__file__).resolve().parent

load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------- LLM（paratera + higress 双端点）
PARATERA_BASE_URL = os.environ["PARATERA_BASE_URL"].strip()
PARATERA_API_KEY = os.environ["PARATERA_API_KEY"].strip()

# 国外模型走本地 Higress 网关（New API，OpenAI 兼容，前缀 /v1）
HIGRESS_BASE_URL = os.environ["HIGRESS_URL"].strip().rstrip("/") + "/v1"
HIGRESS_API_KEY = os.environ["HIGRESS_KEY"].strip()

# 经 Higress 路由的模型（消融横评用：国外 SOTA 档）。网关可用模型还包括
# gpt-5.5 / claude-opus-4-5~4-8 / Kimi-K3 / GLM-5.3 / Qwen3.8-Max。
FOREIGN_MODELS = {"gpt-5.6-sol", "gpt-5.5",
                  "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
                  "claude-opus-4-5-20251101"}

EMBED_MODEL = "GLM-Embedding-3"   # 2048 维

# 角色化模型配置。主智能体用 SOTA 模型保证自主推理质量；审稿人用另一家模型保证独立性。
MODEL_PRESETS = {
    "agent":    "GLM-5.3",           # 主循环：意图理解、研究规划、工具编排
    "reviewer": "DeepSeek-V4-Pro-0813",   # 审稿人：报告评审（与主模型异源）
    "vision":   "Qwen3.8-Max",       # 视觉分析：快照/场图/论文图的机制判读
}

# ---------------------------------------------------------------- MCP 服务器（端点经环境变量配置，见 .env.example）
MCP_SERVERS = {
    "dem":  os.environ.get("DEM_MCP_URL",  "http://localhost:8001/mcp"),   # LIGGGHTS 批量离散元仿真
    "ddda": os.environ.get("DDDA_MCP_URL", "http://localhost:8000/mcp"),   # 数据驱动量纲分析 + π 定理推导
    # "lit"(波尔 :8004) 已退役：服务长期不在线，且其挂起会经 anyio cancel
    # scope 连坐污染同 stack 的后续连接（2026-10-10 实测：lit 超时后
    # ddda/litx 全部连坐失败）。文献层统一走 litx(Sciverse 主腿)。
    "litx": os.environ.get("LITX_MCP_URL", "http://localhost:8006/mcp"),   # lit-mcp：统一文献层（外部检索+本地库+引用登记）
}

# DEM run_simulation(wait=true) 同步等待可长达 2400s，MCP 读超时须覆盖
MCP_READ_TIMEOUT = 3000

# granular-dem MCP 鉴权 token（2026-10-12 服务端启用 MCP_AUTH_TOKEN 后，
# tools/call 必须带 x-mcp-token；协议流量免）。从环境/工作区 .env 读取
import os as _os
DEM_MCP_TOKEN = _os.getenv("DEM_MCP_TOKEN", "").strip()

# ---------------------------------------------------------------- 路径
KB_DIR = PKG / "kb_store"        # chroma 持久化目录
SKILLS_DIR = PKG / "skills"
RUNS_DIR = ROOT / "granusci" / "runs"

# KB 源（相对工作区根）。注：王驰博士学位论文全文两份解析文件
# （论文全文解析/全文.md、论文解析/full.md）因版权原因不随仓库发布，
# 如需完整复现知识库请自行获取原文并按上述路径放入。
KB_SOURCES = [
    "granusci/kb_sources/domain-knowledge.md",   # 领域知识综述（多源文献汇编：相态/偏析/DDDA）
    "MCP接口契约.md",           # 工具接口契约
]

# 盲 KB（自主性验证消融）：剔除博士论文全文与综述中论文私有的定量结论（F 指数
# 相态判据、675 工况基准等答案键内容），只保留公共领域知识与方法论文献。
KB_DIR_BLIND = PKG / "kb_store_blind"
KB_SOURCES_BLIND = [
    "granusci/kb_sources_blind/domain-knowledge.md",
    "MCP接口契约.md",           # 工具接口契约
]

# ---------------------------------------------------------------- 智能体参数
MAX_STEPS = 200                 # 主循环 LLM 轮数上限（含计划门/研究/机制观测/论文全链；
                                # 100 已被实测不足——paper8 死在成文前的画图阶段）
TOOL_RESULT_CHAR_CAP = 6000     # 进入 LLM 上下文的单次工具结果截断阈值
FULL_RESULT_CHAR_CAP = 40000    # tool_log 中保留的完整结果上限
