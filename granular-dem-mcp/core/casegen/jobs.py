# -*- coding: utf-8 -*-
"""casegen 作业执行器：单个结构化 case 的运行管理（独立于 legacy 的批量 runner）。

与 legacy.runner（振动槽批量工况）不同，这里是"一个 case 描述 = 一个作业"：
  build(case dict) → 校验+渲染+落盘 → run（单进程 nice）→ 状态/结果读取。
安全护栏（服务器侧最终防线，与 renderer.validate 双保险）：
  粒子数 ≤5000 / 总步数 ≤2e6 / 墙钟超时 600s / nice -n 19 单核。
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import threading
import time

from .renderer import render_case

CASEGEN_ROOT = "casegen_jobs"
RUN_TIMEOUT_S = 600
MAX_PARTICLES = 5000
MAX_STEPS = 2_000_000


class CaseJob:
    def __init__(self, job_id: str, case_dir: str, meta: dict):
        self.job_id = job_id
        self.case_dir = case_dir
        self.meta = meta


def _write_json_atomic(path: str, obj: dict):
    tmp = f"{path}.tmp.{threading.get_ident()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


class CaseJobManager:
    """casegen 作业管理（线程池=1：单核串行，护栏内）。"""

    def __init__(self, work_dir: str, liggghts_cmd: str):
        self.work_dir = work_dir
        self.liggghts_cmd = shlex.split(liggghts_cmd)
        self.root = os.path.join(work_dir, CASEGEN_ROOT)
        os.makedirs(self.root, exist_ok=True)
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}

    # ---------- 构建 ----------
    def build(self, case: dict) -> dict:
        """校验+渲染+落盘（不运行）。返回 job_id 或错误列表。"""
        # 服务器侧二次护栏（双保险）
        try:
            n_par = sum(c.get("count", 0)
                        for c in case["particles"]["insertion"].get("composition", []))
            n_steps = sum(p.get("steps", 0) for p in case.get("phases", []))
        except (KeyError, TypeError):
            return {"ok": False, "error": "bad_schema",
                    "message": "case 结构不完整，无法统计粒子数/步数"}
        if n_par > MAX_PARTICLES:
            return {"ok": False, "error": "limit",
                    "message": f"粒子总数 {n_par} 超服务器上限 {MAX_PARTICLES}"}
        if n_steps > MAX_STEPS:
            return {"ok": False, "error": "limit",
                    "message": f"总步数 {n_steps} 超服务器上限 {MAX_STEPS}"}

        job_id = f"cg_{time.strftime('%Y%m%d_%H%M%S')}_{id(case) % 10000:04d}"
        jdir = os.path.join(self.root, job_id)
        os.makedirs(jdir, exist_ok=True)
        script, errs = render_case(case, out_dir=jdir)
        if errs:
            import shutil
            shutil.rmtree(jdir, ignore_errors=True)
            return {"ok": False, "error": "validation",
                    "errors": errs,
                    "message": "case 校验失败（按 errors 逐条修复后重新提交）"}
        with open(os.path.join(jdir, "case.json"), "w", encoding="utf-8") as f:
            json.dump(case, f, ensure_ascii=False, indent=1)
        with open(os.path.join(jdir, "case.in"), "w", encoding="utf-8") as f:
            f.write(script)
        self._write_status(job_id, "built", {"n_particles": n_par, "n_steps": n_steps})
        # 回显关键参数摘要——数值审计的可溯源语料（v2_hopper 教训：正文引用
        # 泊松比/重力等输入参数时，工具结果里必须出现过这些数字，否则审稿门
        # 判无出处，而 agent 无法自证）
        _m = case.get("material", {}) or {}
        _g = (case.get("fields", {}) or {}).get("gravity")
        _ex = case.get("excitation") or {}
        return {"ok": True, "job_id": job_id, "n_particles": n_par,
                "n_steps": n_steps, "script_lines": len(script.splitlines()),
                "case_dir": jdir,
                "params_echo": {
                    "gravity": _g,
                    "youngs_modulus": _m.get("youngs_modulus"),
                    "poissons_ratio": _m.get("poissons_ratio"),
                    "friction": _m.get("friction"),
                    "restitution": _m.get("restitution"),
                    "timestep": (case.get("numerics", {}) or {}).get("timestep"),
                    "excitation": {k: _ex.get(k) for k in
                                   ("type", "motion", "amplitude",
                                    "frequency_hz", "shear_velocity") if _ex.get(k) is not None},
                }}

    # ---------- 运行 ----------
    def modify_and_rebuild(self, job_id: str, patch: dict) -> dict:
        """仿真门处置动作：对已 build 的 case 打补丁（浅合并顶层/指定段）后
        重建为新作业。patch 例：{'phases':[{'name':'settle','steps':600000,...}]}
        或 {'particles':{'insertion':{'region':[...], ...}}}（段级深合并）。
        原 job 保留（对照/审计），新 job 继承 meta+加 parent 字段。"""
        import copy
        case_path = os.path.join(self.root, job_id, "case.json")
        if not os.path.exists(case_path):
            return {"ok": False, "error": "no_job", "message": f"未知 job_id: {job_id}"}
        with open(case_path, encoding="utf-8") as f:
            case = json.load(f)

        def deep_merge(dst, src):
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    deep_merge(dst[k], v)
                else:
                    dst[k] = v

        deep_merge(case, patch)
        case.setdefault("meta", {})["parent_job"] = job_id
        case["meta"]["purpose"] = str(case["meta"].get("purpose", "")) + f"（仿真门处置，源 {job_id}）"
        return self.build(case)

    def run(self, job_id: str, wait: bool = False, timeout: int = RUN_TIMEOUT_S) -> dict:
        st = self.status(job_id)
        if not st.get("ok"):
            return st
        if st["status"] == "running":
            # v3hr 教训：已在运行的作业 wait=true 也应阻塞——"已在运行"立即
            # 返回使 agent 只能高频轮询烧 LLM 步数。有本进程线程则 join；
            # 无（跨进程启动）则轮询状态文件直到终态或超时。
            if not wait:
                return {"ok": True, "job_id": job_id, "status": "running",
                        "message": "已在运行"}
            t = self._threads.get(job_id)
            if t is not None:
                t.join(timeout + 30)
                return self.status(job_id)
            deadline = time.time() + timeout + 30
            while time.time() < deadline:
                time.sleep(5)
                st = self.status(job_id)
                if st.get("status") not in ("running", "queued"):
                    return st
            st = self.status(job_id)
            st["waited"] = False
            st["message"] = f"等待超时(>{timeout + 30}s)，作业仍在运行，可继续查询"
            return st
        t = threading.Thread(target=self._worker, args=(job_id, timeout), daemon=True)
        with self._lock:
            self._threads[job_id] = t
        t.start()
        if not wait:
            return {"ok": True, "job_id": job_id, "status": "running"}
        t.join(timeout + 30)
        return self.status(job_id)

    def _worker(self, job_id: str, timeout: int):
        jdir = os.path.join(self.root, job_id)
        self._write_status(job_id, "running", {"started": time.time()})
        cmd = self.liggghts_cmd + ["-in", "case.in"]
        try:
            r = subprocess.run(
                ["nice", "-n", "19"] + cmd, cwd=jdir,
                stdout=open(os.path.join(jdir, "run.log"), "w"),
                stderr=subprocess.STDOUT, timeout=timeout)
            code = r.returncode
        except subprocess.TimeoutExpired:
            self._write_status(job_id, "failed",
                               {"reason": f"超时(>{timeout}s)", "exit_code": -9})
            return
        except Exception as e:  # noqa: BLE001
            self._write_status(job_id, "failed", {"reason": str(e)[:200]})
            return
        dumps = sorted(f for f in os.listdir(jdir) if f.startswith("out_") and f.endswith(".txt"))
        # 末帧按解析的 step 数值取 max（字符串排序坑：
        # v1_dryrun 实测 out_settle_90000 > out_main_1400000）
        def _step_of(fn):
            m = re.search(r"_(\d+)\.txt$", fn)
            return int(m.group(1)) if m else -1
        last_dump = max(dumps, key=_step_of) if dumps else None
        if code == 0 and dumps:
            self._write_status(job_id, "done",
                               {"exit_code": code, "n_dumps": len(dumps),
                                "finished": time.time()})
        elif code == 0:
            self._write_status(job_id, "done_warn",
                               {"exit_code": code, "n_dumps": 0,
                                "note": "运行完成但无 dump 产出，检查 output 段"})
        else:
            tail = self._log_tail(jdir)
            self._write_status(job_id, "failed",
                               {"exit_code": code, "log_tail": tail})

    # ---------- 状态/结果 ----------
    def status(self, job_id: str) -> dict:
        sp = os.path.join(self.root, job_id, "job.json")
        if not os.path.exists(sp):
            return {"ok": False, "error": "no_job",
                    "message": f"未知 job_id: {job_id}"}
        with open(sp, encoding="utf-8") as f:
            st = json.load(f)
        st["ok"] = True
        return st

    def diagnose(self, job_id: str) -> dict:
        """全套诊断（诊断箱 v0）：守恒/弛豫/速度 sanity/静置判据。"""
        st = self.status(job_id)
        if not st.get("ok"):
            return st
        from .diagnostics import diagnose
        import json as _json
        case_path = os.path.join(self.root, job_id, "case.json")
        n_exp = st.get("n_particles")
        h = None
        try:
            with open(case_path, encoding="utf-8") as f:
                c = _json.load(f)
            b = c["domain"]["box"]
            h = max(b[3] - b[2], b[5] - b[4])
        except Exception:  # noqa: BLE001
            pass
        return diagnose(os.path.join(self.root, job_id),
                        n_expected=n_exp, domain_h=h)

    def result(self, job_id: str) -> dict:
        """读取运行结果摘要：状态 + dump 文件列表 + 末帧统计。"""
        st = self.status(job_id)
        if not st.get("ok"):
            return st
        jdir = os.path.join(self.root, job_id)
        dumps = sorted(f for f in os.listdir(jdir) if f.startswith("out_") and f.endswith(".txt"))
        # 末帧按解析的 step 数值取 max（字符串排序坑：
        # v1_dryrun 实测 out_settle_90000 > out_main_1400000）
        def _step_of(fn):
            m = re.search(r"_(\d+)\.txt$", fn)
            return int(m.group(1)) if m else -1
        last_dump = max(dumps, key=_step_of) if dumps else None
        out = dict(st)
        out["dumps"] = dumps
        if st["status"] in ("done", "done_warn") and dumps:
            try:
                out["last_frame"] = self._frame_summary(os.path.join(jdir, last_dump))
            except Exception as e:  # noqa: BLE001
                out["last_frame_error"] = str(e)[:100]
            # 诊断箱 v0 摘要（详细诊断走 diagnose）
            try:
                out["diagnostics"] = self.diagnose(job_id)["diagnostics"]
            except Exception as e:  # noqa: BLE001
                out["diagnostics_error"] = str(e)[:80]
            # 粒子守恒对账（e2e 实测失败模式：坠出域被 lost ignore 静默删除）
            n_expect = st.get("n_particles")
            n_last = (out.get("last_frame") or {}).get("n")
            if n_expect and n_last is not None and n_last < 0.8 * n_expect:
                out["status"] = "done_leak"
                out["leak_note"] = (f"末帧 {n_last} 颗 vs 插入 {n_expect} 颗——"
                                     f"超过 20% 颗粒丢失（坠出域/被删除），检查壁面封闭性")
        return out

    def _frame_summary(self, path: str) -> dict:
        """末帧轻量摘要：粒子数/type 分布/质心/包围盒（诊断箱 v0）。"""
        n = 0
        types: dict[int, int] = {}
        cx = cy = cz = 0.0
        xmin = ymin = zmin = float("inf")
        xmax = ymax = zmax = float("-inf")
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) < 5 or not p[0].isdigit():
                    continue
                n += 1
                t = int(p[1])
                types[t] = types.get(t, 0) + 1
                x, y, z = float(p[2]), float(p[3]), float(p[4])
                cx += x; cy += y; cz += z
                xmin = min(xmin, x); xmax = max(xmax, x)
                ymin = min(ymin, y); ymax = max(ymax, y)
                zmin = min(zmin, z); zmax = max(zmax, z)
        if n == 0:
            return {"n": 0}
        return {"n": n, "types": types,
                "centroid": [round(cx / n, 5), round(cy / n, 5), round(cz / n, 5)],
                "bbox": [[round(xmin, 4), round(xmax, 4)],
                         [round(ymin, 4), round(ymax, 4)],
                         [round(zmin, 4), round(zmax, 4)]],
                "file": os.path.basename(path)}

    @staticmethod
    def _log_tail(jdir: str, n: int = 8) -> str:
        try:
            with open(os.path.join(jdir, "run.log"), encoding="utf-8",
                      errors="replace") as f:
                return "".join(f.readlines()[-n:])
        except OSError:
            return "(no log)"

    def _write_status(self, job_id: str, status: str, extra: dict | None = None):
        sp = os.path.join(self.root, job_id, "job.json")
        st = {"job_id": job_id, "status": status, "updated": time.time()}
        st.update(extra or {})
        _write_json_atomic(sp, st)
