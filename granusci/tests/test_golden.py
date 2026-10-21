# -*- coding: utf-8 -*-
"""⑪ Golden test 基线：确定性组件的固定输入→固定输出契约。

守护对象（改动这些组件必须 consciously 更新此处基线）：
- numeric_audit：数字级溯源审计引擎（论文支柱②的核心，容差规则不许悄悄变）
- compress_view：上下文压缩视图（结构契约：配对完整、角色保留、边界行为）
- 图拓扑：节点集合（拓扑变更须显式）
- 系统提示词契约：关键节存在（防 format 破坏/意外丢失关键约束）
"""
from __future__ import annotations

import json

from granusci.agent import graph as g
from granusci.agent import prompts
from granusci.skills import load_skills


# ================================================================ numeric_audit
def test_numeric_audit_tolerance_matrix():
    """容差矩阵 golden：舍入/百分数/绝对值/字面子串四种格式差异放过，孤儿抓住。"""
    tool_log = [
        {"tool": "t1", "args": {}, "ok": True,
         "result": {"values": [0.630234, 0.1311, -0.37257745, 0.500, -1.86]}},
    ]
    # 报告数字：0.6302=舍入；13.11=百分数；0.3726=绝对值舍入；0.500=字面子串；1.86=负数取幅
    report = "系数 0.6302 与 13.11 与 0.3726 与 0.500 与 1.86，另有一个 99.99。"
    audit = g.numeric_audit(report, tool_log)
    assert audit["report_decimal_count"] == 6
    assert audit["orphans"] == ["99.99"]     # 唯一孤儿；五种格式差异全部放过


def test_numeric_audit_no_false_pass():
    """近似但不同的数字不许通过（容差不许放水）。"""
    tool_log = [{"tool": "t", "args": {}, "ok": True, "result": {"v": 0.1234}}]
    audit = g.numeric_audit("结果为 0.1235。", tool_log)
    assert audit["orphans"] == ["0.1235"]    # 末位不同=无出处


def test_numeric_audit_empty_report():
    audit = g.numeric_audit("全文无小数。", [])
    assert audit == {"report_decimal_count": 0, "orphans": []}


# ================================================================ compress_view
def _mk_tool_msg(idx: int, content: str) -> dict:
    return {"role": "tool", "tool_call_id": f"call_{idx}", "content": content}


def _mk_assistant_with_call(idx: int) -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": f"call_{idx}", "type": "function",
                            "function": {"name": "t", "arguments": "{}"}}]}


def test_compress_view_golden_structure():
    """压缩 golden（两区制）：冻结区压缩、追加区原文、user 永不压缩。"""
    msgs = [{"role": "user", "content": "意图"}]
    for i in range(12):
        msgs.append(_mk_assistant_with_call(i))
        long = f"数据{i} " + "x" * 1000
        msgs.append(_mk_tool_msg(i, long))
    msgs.append({"role": "assistant", "content": "短文本"})
    # 冻结边界在第 13 条（前 6 对 assistant+tool 压缩，其后原文）
    cut = 13
    out = g.compress_view(msgs, compact_cut=cut)
    # 长度与结构
    assert len(out) == len(msgs)
    assert [m["role"] for m in out] == [m["role"] for m in msgs]
    # user 全保（含冻结区内的第 0 条）
    assert out[0]["content"] == "意图"
    # 冻结区内长 tool 压缩、追加区全保
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert len(tool_msgs) == 12
    frozen_tools = tool_msgs[:6]
    live_tools = tool_msgs[6:]
    assert all("已压缩" in m["content"] for m in frozen_tools)
    assert all("已压缩" not in m["content"] for m in live_tools)
    # 压缩保头 250 字符 + 标记；tool_call_id 全保留（配对完整）
    assert frozen_tools[0]["content"].startswith("数据0 " + "x" * 245)
    assert all(m["tool_call_id"] for m in tool_msgs)
    # assistant tool_calls 未被破坏
    assistants = [m for m in out if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(assistants) == 12


def test_compress_view_prefix_stability():
    """缓存友好核心性质：追加消息后，旧消息的视图渲染逐字节不变。

    （服务端前缀缓存按最长公共前缀命中——视图稳定 = 每轮只按新增部分计费）
    """
    msgs = [{"role": "user", "content": "意图"}]
    for i in range(8):
        msgs.append(_mk_assistant_with_call(i))
        msgs.append(_mk_tool_msg(i, f"数据{i}" + "x" * 800))
    cut = 5
    v1 = g.compress_view(msgs, compact_cut=cut)
    # 追加新一轮（assistant + tool + 新 user 指令），cut 不动（未超阈值）
    msgs2 = msgs + [_mk_assistant_with_call(99),
                    _mk_tool_msg(99, "新结果" + "n" * 900),
                    {"role": "user", "content": "继续"}]
    v2 = g.compress_view(msgs2, compact_cut=cut)
    # 旧部分逐字节一致（前缀缓存可命中到上轮请求末尾）
    assert v2[:len(v1)] == v1
    # 新消息原文进入视图
    assert v2[-1]["content"] == "继续"


def test_compress_view_boundaries():
    """边界 golden：400 字符为严格阈值（冻结区内）；追加区永不压缩。"""
    msgs = [_mk_assistant_with_call(0), _mk_tool_msg(0, "y" * 400),
            _mk_assistant_with_call(1), _mk_tool_msg(1, "y" * 401)]
    # cut=0：全部在追加区，都不压缩
    out = g.compress_view(msgs, compact_cut=0)
    tools = [m for m in out if m.get("role") == "tool"]
    assert "已压缩" not in tools[0]["content"]
    assert "已压缩" not in tools[1]["content"]
    # cut=4（全部冻结）：401 压缩、400 不压（严格大于）
    out2 = g.compress_view(msgs, compact_cut=4)
    tools2 = [m for m in out2 if m.get("role") == "tool"]
    assert "已压缩" not in tools2[0]["content"]   # 恰 400 不压
    assert "已压缩" in tools2[1]["content"]       # 401 → 压


def test_compress_view_long_assistant_head_tail():
    msgs = [{"role": "assistant", "content": "A" * 3001}]
    out = g.compress_view(msgs, compact_cut=1)
    c = out[0]["content"]
    assert "中段压缩" in c and c.startswith("A" * 1500) and c.endswith("A" * 1000)


def test_compress_view_deterministic():
    """同一输入两次压缩输出完全一致（golden 前提）。"""
    msgs = []
    for i in range(15):
        msgs.append(_mk_assistant_with_call(i))
        msgs.append(_mk_tool_msg(i, f"内容{i}" + "z" * 500))
    assert g.compress_view(msgs, compact_cut=10) == g.compress_view(msgs, compact_cut=10)


def test_advance_compact_cut():
    """阈值触发推进：未超不动；超了推进且保留尾部工作集；单调不回头。"""
    msgs = [{"role": "user", "content": "意图"}]
    for i in range(30):
        msgs.append(_mk_assistant_with_call(i))
        msgs.append(_mk_tool_msg(i, "d" * 1000))    # 每对 ~1000 字符
    # 远低于默认阈值（100k）：不动
    assert g._advance_compact_cut(msgs, 0) == 0
    # 超阈值：从 cut 起的追加区太大 → 推进到保留 ~50k 的位置
    big = [{"role": "user", "content": "意图"}]
    for i in range(120):
        big.append(_mk_assistant_with_call(i))
        big.append(_mk_tool_msg(i, "d" * 1000))    # ~120k 字符追加区
    cut2 = g._advance_compact_cut(big, 0)
    assert cut2 > 0
    tail = sum(len(str(m.get("content") or "")) for m in big[cut2:])
    assert tail < g.COMPACT_KEEP_CHARS + 2000      # 追加区收敛到工作集规模
    # 再追加少量消息：未再超（tail < trigger）→ 边界不动（前缀稳定）
    big2 = big + [_mk_assistant_with_call(999), _mk_tool_msg(999, "e" * 500)]
    assert g._advance_compact_cut(big2, cut2) == cut2
    # 单调：新 cut 不会小于旧 cut
    assert g._advance_compact_cut(big, cut2) >= cut2


# ================================================================ 图拓扑
def test_graph_topology_golden():
    graph, _, _ = g.build_graph("GLM-5.3", "DeepSeek-V4-Pro", 50, log_fn=lambda m: None)
    nodes = set(graph.get_graph().nodes.keys())
    assert nodes == {"__start__", "__end__", "agent", "tools",
                     "plan_gate", "plan_gate_exit", "inbox_gate", "reviewer",
                     "no_review_gate"}


def test_local_tools_contract_golden():
    """本地工具面契约：核心工具在册且 schema 关键字段齐全。"""
    tools = {t.name: t for t in g._make_local_tools({}, run_dir=None)}
    expected = {"kb_search", "load_skill", "record_finding", "submit_research_plan",
                "lit_search_openalex", "litf_acquire", "litf_read", "deep_read_paper",
                "make_figure", "vision_analyze", "write_manuscript", "compile_latex",
                "submit_paper", "submit_report"}
    assert expected <= set(tools), f"缺工具: {expected - set(tools)}"
    assert "kind" in tools["record_finding"].schema["properties"]
    assert "required" in tools["submit_research_plan"].schema


# ================================================================ 提示词契约
def test_system_prompt_contract():
    sp = prompts.build_system_prompt(load_skills(), experience="")
    for key in ["研究启动协议", "学术交付三件套", "数值保真", "研究状态账本",
                "过往运行经验", "机制调查", "用户意图如下"]:
        assert key in sp, f"系统提示词丢失关键节: {key}"
    # 行为准则 8 条
    assert sp.count("\n8.") >= 1 or "8. **研究状态账本**" in sp


def test_reviewer_prompt_contract():
    rp = prompts.REVIEWER_SYSTEM_PROMPT
    for key in ["数值一致性", "比较性结论", "原创性", "诚实性", "pass",
                "失败模式", "图文断层", "过度声明", "二轮"]:
        assert key in rp, f"审稿人提示词丢失: {key}"


def test_prompt_format_survives():
    """format 占位符健全性：两个命名槽都能填充不抛 KeyError。"""
    sp = prompts.build_system_prompt(load_skills(), experience="测试经验")
    assert "测试经验" in sp and "{experience}" not in sp and "{skill_catalog}" not in sp
