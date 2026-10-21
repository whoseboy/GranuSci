# -*- coding: utf-8 -*-
"""技能（Skills）：按需加载的操作性方法论指引——三层渐进加载。

与知识库的分工：KB 是原始文献（论文全文、契约），技能是从中蒸馏的
"怎么干"——只在智能体判断需要时经 load_skill 载入，不预占上下文。

三层结构（对齐 Claude Code / 前沿 agent 技能形态）：
  ① 目录层：catalog_text() 一句话索引进系统提示词（常驻）
  ② 主体层：load_skill() 载入 SKILL.md 正文（紧凑方法论，≤~6KB）
  ③ 参考层：load_skill(name, deep="<文件名>") 载入 references/ 深层文档
     （成熟技能搬来的完整框架——审稿 rubric、写作范式等，按需展开）

目录形态：skills/<name>/SKILL.md + skills/<name>/references/*.md
单文件形态（兼容）：skills/<name>.md
"""
from __future__ import annotations

import re
from pathlib import Path

from . import config


class Skill:
    def __init__(self, name: str, description: str, when_to_use: str,
                 body: str, refs: dict[str, str] | None = None):
        self.name = name
        self.description = description
        self.when_to_use = when_to_use
        self.body = body
        self.refs = refs or {}       # {参考文件名（不含.md）: 全文}

    @property
    def ref_list(self) -> list[str]:
        """参考层文件清单（进主体层末尾的导航，提示 agent 可深入）。"""
        return sorted(self.refs.keys())


def _parse_skill_md(text: str, fallback_name: str) -> tuple[str, str, str, str] | None:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        return None
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    name = meta.get("name", fallback_name)
    return name, meta.get("description", ""), meta.get("when_to_use", ""), m.group(2).strip()


def load_skills() -> dict[str, Skill]:
    """扫描技能目录：单文件与目录形态都认。"""
    root = Path(config.SKILLS_DIR)
    skills: dict[str, Skill] = {}
    # 目录形态（主体 + 参考层）
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        skill_md = d / "SKILL.md"
        if not skill_md.exists():
            continue
        parsed = _parse_skill_md(skill_md.read_text(encoding="utf-8"), d.name)
        if not parsed:
            continue
        name, desc, when, body = parsed
        refs: dict[str, str] = {}
        ref_dir = d / "references"
        if ref_dir.is_dir():
            for rf in sorted(ref_dir.glob("*.md")):
                refs[rf.stem] = rf.read_text(encoding="utf-8").strip()
        skills[name] = Skill(name, desc, when, body, refs)
    # 单文件形态（兼容旧技能）
    for path in sorted(root.glob("*.md")):
        parsed = _parse_skill_md(path.read_text(encoding="utf-8"), path.stem)
        if not parsed:
            continue
        name, desc, when, body = parsed
        skills[name] = Skill(name, desc, when, body)
    return skills


def catalog_text(skills: dict[str, Skill]) -> str:
    """给系统提示词用的技能目录（只有一句话索引，不含正文）。"""
    lines = []
    for s in sorted(skills.values(), key=lambda x: x.name):
        lines.append(f"- {s.name}: {s.description}（适用：{s.when_to_use}）")
    return "\n".join(lines)
