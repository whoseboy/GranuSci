# GranuSci

面向颗粒领域规律自动发现的闭环自主科研智能体（中国科学院力学研究所 KeLiAI 团队）。

给定一句研究意图，智能体自主完成**文献调研 → 实验设计 → 批量离散元仿真 → 规律提炼 → 机制调查 → 论文撰写 → 幻灯片生成**的完整研究链条。系统采用"语义层—计算层"分离架构：大语言模型承担研究决策与学术表达，批量仿真与规律拟合两项核心数值计算由 MCP（模型上下文协议）工具服务执行，产出经数值审计、视觉审计与异源大模型评审三层审稿门把关。

## 目录结构

| 目录 | 内容 |
|---|---|
| `granusci/` | 智能体主程序：LangGraph 状态图主循环、MCP 工具总线、向量知识库、论文/幻灯片工具链 |
| `granular-dem-mcp/` | 离散元批量仿真 MCP 服务（LIGGGHTS-PUBLIC 3.8，Docker 部署） |
| `DDDA-MCP/` | 数据驱动量纲分析（DDDA）MCP 服务（MATLAB 方法的纯 Python 移植） |
| `lit-mcp/` | 文献检索与引用登记 MCP 服务 |
| `报告/` | 项目报告（系统设计、验证实验与消融分析的完整文档） |
| `交付物-基线运行/` | 基线运行中智能体自主产出的 REVTeX 论文与 Beamer 汇报幻灯片 |
| `MCP接口契约.md` | 三个 MCP 服务的工具接口契约 |

## 快速开始

```bash
pip install -r granusci/requirements.txt

# 1. 配置环境变量（复制 .env.example 为 .env 并填写）
#    - LLM 推理服务（OpenAI 兼容）的 base_url 与 api_key
#    - 三个 MCP 服务的端点（DEM / DDDA / 文献）

# 2. 启动计算服务（分别见各服务目录内的部署说明）
#    granular-dem-mcp/（Docker）/ DDDA-MCP/（run.sh）/ lit-mcp/

# 3. 构建知识库（首次）
python -m granusci.kb build

# 4. 一句意图跑完整研究
python -m granusci.run "研究振动颗粒床的能量标度规律" --tag demo
```

运行产物落盘于 `granusci/runs/<时间戳>_<tag>/`：论文与幻灯片源文件及编译产物、消息史、工具调用轨迹、账本、审稿记录与用量明细。详细用法（无人值守模式、消融配置、断点续跑）见 `granusci/README.md` 与项目报告附录A。

## 说明

- **LIGGGHTS 源码**：`granular-dem-mcp/Dockerfile` 按路径 `liggghts-src/` 编译 [LIGGGHTS-PUBLIC 3.8](https://github.com/CFDEMproject/LIGGGHTS-PUBLIC)，该第三方源码树不含在本仓库，构建镜像前请自行获取并置于该路径。
- **验证文献全文**：知识库源中的博士学位论文全文因版权原因不随仓库发布（见 `granusci/config.py` 中 KB_SOURCES 注释）。
- **安全**：仓库不含任何密钥；全部端点与凭据经环境变量配置，模板见 `.env.example`。
