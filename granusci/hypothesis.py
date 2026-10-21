# -*- coding: utf-8 -*-
"""假设层模块（A4 研究记忆 + B1 假设对象 + B6 战役编排 的存储与状态机）。

设计输入（五道门+漏项扫描的落地）：
- 形式化假设对象（门 3 §1）：分区变量/区数/数值边界/机制叙事/来源（cite_id
  或 reasoning）/置信/可证伪预测——LLM 自然语言假设的结构化容器；
- 假设生命周期（B1）：draft → registered → testing → supported/refuted/
  revised（状态迁移记录全部留痕，死亡位置=新科学问题候选）；
- 研究记忆（A4）：跨 run 持久化——假设档案 + 失败模式库 + 领域知识自增长
  （战役发现的规律按文献库同格式入库，成为下一次战役先验）；
- 战役编排（B6）：批次状态机（phase/experiment/budget）+ 终止决策记录。

存储：JSON 文件（granusci/memory/research/），run 内经工具读写，跨 run 追加。
不引入新服务——研究记忆是 granusci 的本地资产，随工作区走。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

RESEARCH_DIR = Path(__file__).parent / "memory" / "research"

# 假设状态机
STATES = ("draft", "registered", "testing", "supported", "refuted", "revised", "retired")
TRANSITIONS = {
    "draft": {"registered", "retired"},
    "registered": {"testing", "retired"},
    "testing": {"supported", "refuted", "revised", "retired"},
    "revised": {"testing", "retired"},
    "supported": {"revised", "retired"},   # 支持不等于永真——新证据可修订
    "refuted": {"revised", "retired"},
    "retired": set(),
}

KINDS = ("partition", "mechanism", "scaling", "boundary", "sign", "other")


class HypothesisStore:
    """假设档案：register / update_state / query / lifecycle 记录。"""

    def __init__(self, root: Path | None = None):
        self.root = root or RESEARCH_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self._hyp_path = self.root / "hypotheses.jsonl"
        self._phase_path = self.root / "campaign.json"

    # ---------------- 假设对象 ----------------
    def register(self, hyp: dict, run_tag: str = "") -> dict:
        """登记一条形式化假设。必填：statement（可证伪表述）；强烈建议：
        prediction（可检验预测）、kind、source（{type:'cite',cite_id} 或
        {type:'reasoning'}）。数值边界（boundaries）供仪器对账。"""
        missing = [k for k in ("statement",) if not hyp.get(k)]
        if missing:
            return {"ok": False, "message": f"假设缺必填字段: {missing}"}
        kind = hyp.get("kind", "other")
        if kind not in KINDS:
            return {"ok": False, "message": f"kind 须为 {KINDS}"}
        src = hyp.get("source") or {"type": "reasoning"}
        if src.get("type") == "cite" and not src.get("cite_id"):
            return {"ok": False, "message": "文献型假设须带 cite_id（litx_cite_register）"}
        hid = f"H{int(time.time() * 1000) % 10**10:010d}"
        # 同毫秒双注册撞 ID（v3hr 实况：两假设共用 H0228441555，对账建议
        # 指代不明）——撞则在档重生成
        _existing = {r["hid"] for r in self._all()}
        while hid in _existing:
            hid = f"H{(int(time.time() * 1000) + 1) % 10**10:010d}"
            time.sleep(0.001)
        rec = {
            "hid": hid, "kind": kind, "state": "registered",
            "statement": hyp["statement"],
            "prediction": hyp.get("prediction", ""),
            "partition_vars": hyp.get("partition_vars", []),
            "boundaries": hyp.get("boundaries", []),
            "mechanism": hyp.get("mechanism", ""),
            "source": src, "confidence": hyp.get("confidence", "low"),
            "run_tag": run_tag, "created": time.time(),
            "history": [{"ts": time.time(), "event": "registered"}],
        }
        self._append(rec)
        return {"ok": True, "hid": hid, "hypothesis": rec}

    def update_state(self, hid: str, new_state: str,
                     evidence: str = "", note: str = "") -> dict:
        rec = self._find(hid)
        if not rec:
            return {"ok": False, "message": f"未登记的假设: {hid}"}
        if new_state not in STATES:
            return {"ok": False, "message": f"未知状态 {new_state}; 合法: {STATES}"}
        if new_state not in TRANSITIONS.get(rec["state"], set()):
            return {"ok": False,
                    "message": f"非法迁移 {rec['state']}→{new_state}; "
                               f"从 {rec['state']} 可去 {sorted(TRANSITIONS[rec['state']])}"}
        # 裁决类迁移强制 evidence（审计 A-14：工具描述宣称"refuted/support
        # 必填"但代码零校验——一句话无证据裁决会进生死档案污染论文素材）
        if new_state in ("supported", "refuted") and not evidence.strip():
            return {"ok": False,
                    "message": f"{new_state} 是裁决性迁移，evidence 必填："
                               "须引用具体工具记录（如 plot_script 输出、"
                               "dem_analyze_frame 结果、工况 job_id），"
                               "无证据的裁决不被接受"}
        rec["state"] = new_state
        rec["history"].append({
            "ts": time.time(), "event": new_state,
            "evidence": evidence[:400], "note": note[:200]})
        self._rewrite(hid, rec)
        # refuted 的死亡位置提示（B1：死亡=新科学问题候选）
        extra = ""
        if new_state == "refuted":
            extra = ("（该假设已被证据杀死——死亡位置与新数据一起构成新科学问题候选，"
                     "写报告时在 limitation/next-step 引用本 hid）")
        return {"ok": True, "hid": hid, "state": new_state, "note": extra}

    def revise(self, hid: str, new_statement: str, why: str,
               new_prediction: str = "") -> dict:
        """修订=新版本（旧记录保留于 history，链式 parent）。"""
        rec = self._find(hid)
        if not rec:
            return {"ok": False, "message": f"未登记的假设: {hid}"}
        old = {"statement": rec["statement"], "prediction": rec["prediction"],
               "version_ts": time.time()}
        rec["statement"] = new_statement
        if new_prediction:
            rec["prediction"] = new_prediction
        rec["state"] = "revised"
        rec.setdefault("versions", []).append(old)
        rec["history"].append({"ts": time.time(), "event": "revised",
                               "why": why[:300]})
        self._rewrite(hid, rec)
        return {"ok": True, "hid": hid, "state": "revised"}

    def query(self, state: str = "", kind: str = "") -> dict:
        recs = [r for r in self._all()
                if (not state or r["state"] == state)
                and (not kind or r["kind"] == kind)]
        return {"ok": True, "n": len(recs), "hypotheses": recs}

    def lifecycle(self, hid: str) -> dict:
        rec = self._find(hid)
        return {"ok": bool(rec), "hypothesis": rec}

    # ---------------- 战役编排（B6）----------------
    def campaign_phase(self, phase: str, note: str = "",
                       budget_used: dict | None = None) -> dict:
        """记录/推进战役阶段：screening / mapping / targeted / confirmatory /
        writing。每次迁移记录预算消耗与决策理由（终止决策也走这里）。"""
        rec = {"ts": time.time(), "phase": phase, "note": note[:300],
               "budget_used": budget_used or {}}
        with open(self._phase_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"ok": True, "phase": phase, "note": "战役阶段已记录（含预算与理由）"}

    def campaign_log(self) -> dict:
        if not self._phase_path.exists():
            return {"ok": True, "phases": []}
        phases = [json.loads(l) for l in
                  self._phase_path.read_text(encoding="utf-8").splitlines() if l]
        return {"ok": True, "n": len(phases), "phases": phases}

    # ---------------- 存储 ----------------
    def _append(self, rec: dict):
        with open(self._hyp_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _all(self) -> list[dict]:
        if not self._hyp_path.exists():
            return []
        return [json.loads(l) for l in
                self._hyp_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def _find(self, hid: str) -> dict | None:
        for r in self._all():
            if r["hid"] == hid:
                return r
        return None

    def _rewrite(self, hid: str, rec: dict):
        recs = self._all()
        out = [rec if r["hid"] == hid else r for r in recs]
        with open(self._hyp_path, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


class KnowledgeLibrary:
    """A4 领域知识自增长库：战役发现的规律按文献格式入库=下一次战役先验。

    条目=战役产出的可复用知识（标度律/相边界/机制判据），格式与 lit-mcp
    的书目条目同构（source 指向战役 run 而非文献）——检索时与文献平权。
    存储：memory/research/knowledge.jsonl
    """

    def __init__(self, root: Path | None = None):
        self.root = root or RESEARCH_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self._path = self.root / "knowledge.jsonl"

    def deposit(self, entry: dict, run_tag: str = "") -> dict:
        """入库一条战役知识。必填：statement（规律陈述）、kind、evidence
        （工具记录/账本引用）、run_tag（来源战役）。"""
        missing = [k for k in ("statement", "kind", "evidence") if not entry.get(k)]
        if missing:
            return {"ok": False, "message": f"知识条目缺必填: {missing}"}
        rec = {
            "kid": f"K{int(time.time() * 1000) % 10**10:010d}",
            "kind": entry["kind"],          # law / boundary / mechanism / pitfall
            "statement": entry["statement"],
            "quantitative": entry.get("quantitative", ""),
            "evidence": entry["evidence"],
            "confidence": entry.get("confidence", "medium"),
            "source_run": run_tag,
            "created": time.time(),
        }
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"ok": True, "kid": rec["kid"],
                "note": "已入库——后续战役的研究计划阶段检索本库（与文献平权）"}

    def query(self, keyword: str = "", kind: str = "") -> dict:
        if not self._path.exists():
            return {"ok": True, "n": 0, "entries": []}
        recs = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if keyword and keyword not in r["statement"] + r.get("quantitative", ""):
                continue
            if kind and r["kind"] != kind:
                continue
            recs.append(r)
        return {"ok": True, "n": len(recs), "entries": recs}


class FailureModeLibrary:
    """A4 失败模式库：跨战役的行为陷阱沉淀（与 cross_memory 的 pitfall 分工：
    那里是单 run 教训的自动蒸馏，这里是战役级的系统性失败模式——
    '假设编码瓶颈''均匀加密零增益'这类结构性教训，供研究计划阶段规避）。
    存储：memory/research/failure_modes.jsonl
    """

    def __init__(self, root: Path | None = None):
        self.root = root or RESEARCH_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self._path = self.root / "failure_modes.jsonl"

    def record(self, mode: dict, run_tag: str = "") -> dict:
        missing = [k for k in ("title", "description", "countermeasure")
                   if not mode.get(k)]
        if missing:
            return {"ok": False,
                    "message": f"失败模式缺必填: {missing}（title/description/countermeasure）"}
        rec = {"fid": f"F{int(time.time() * 1000) % 10**10:010d}",
               "title": mode["title"],
               "description": mode["description"],
               "countermeasure": mode["countermeasure"],
               "source_run": run_tag, "created": time.time()}
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"ok": True, "fid": rec["fid"]}

    def query(self, keyword: str = "") -> dict:
        if not self._path.exists():
            return {"ok": True, "n": 0, "modes": []}
        recs = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if keyword and keyword not in r["title"] + r["description"]:
                continue
            recs.append(r)
        return {"ok": True, "n": len(recs), "modes": recs}
