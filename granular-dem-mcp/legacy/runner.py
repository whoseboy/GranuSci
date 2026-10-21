# -*- coding: utf-8 -*-
"""批量 LIGGGHTS 作业管理。

流程（每工况）：
  1) 确保该 (d, F) 的自然堆积 restart 存在（不存在则先跑初始化脚本），restart 缓存复用
  2) 由模板生成 in.run，用 -var 传参（不做字符串替换）
  3) 后台线程池调度执行，实时更新 job.json

异步模式：submit() 立即返回 job_id；worker 线程逐工况执行并写状态文件。
"""
import json
import os
import shlex
import shutil
import subprocess
import threading
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE, "legacy", "templates")
MESH_FILE = os.path.join(BASE, "legacy", "meshes", "fangcao.stl")
G = 9.81
# restart 缓存版本：init 模板的域/材料等实质性变更时 +1，旧缓存自动失效
RESTART_VERSION = 2


class JobManager:
    def __init__(self, work_dir, liggghts_cmd, host_work_dir):
        self.work_dir = work_dir
        self.host_work_dir = host_work_dir
        self.liggghts_cmd = shlex.split(liggghts_cmd)
        self._lock = threading.Lock()
        self._sem = None  # 每作业独立

    # ---------- 路径 ----------
    def _jobs_dir(self):
        return os.path.join(self.work_dir, "jobs")

    def _job_dir(self, job_id):
        return os.path.join(self._jobs_dir(), job_id)

    def _restarts_dir(self):
        return os.path.join(self.work_dir, "restarts")

    def host_path(self, p):
        """容器路径 -> 宿主机路径（返回给调用方，供同机 DDDA-MCP 读取）。"""
        if (self.host_work_dir and p
                and p.startswith(self.work_dir)
                and self.host_work_dir.startswith("/")):
            return self.host_work_dir + p[len(self.work_dir):]
        return p

    # ---------- 状态 ----------
    def _read_job(self, job_id):
        with open(os.path.join(self._job_dir(job_id), "job.json"), encoding="utf-8") as f:
            return json.load(f)

    def _write_job(self, job_id, job):
        """原子写状态文件。tmp 文件名带线程ID，防止并发写互相覆盖。
        注意：调用方若涉及 读-改-写 须持有 self._lock。"""
        job["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        p = os.path.join(self._job_dir(job_id), "job.json")
        tmp = "%s.tmp.%s" % (p, threading.get_ident())
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False, indent=1)
        os.replace(tmp, p)

    def _update_job(self, job_id, mutate):
        """线程安全的 读-改-写：持锁读取、应用变更函数、原子落盘。"""
        with self._lock:
            job = self._read_job(job_id)
            mutate(job)
            self._write_job(job_id, job)
            return job

    # ---------- 提交 ----------
    def submit(self, design_id, max_parallel=4):
        if not design_id:
            return {"ok": False, "error": "missing_design_id",
                    "message": "需要 design_id（先运行 generate_design）"}
        dpath = os.path.join(self.work_dir, "designs", design_id + ".json")
        if not os.path.exists(dpath):
            return {"ok": False, "error": "design_not_found",
                    "message": "找不到设计 %s" % design_id}
        with open(dpath, encoding="utf-8") as f:
            design = json.load(f)
        cases = design["cases"]
        if len(cases) > 1000:
            return {"ok": False, "error": "too_many_cases",
                    "message": "单次作业最多 1000 工况"}

        job_id = "job_%s" % time.strftime("%Y%m%d_%H%M%S")
        jd = os.path.join(self._jobs_dir(), job_id)
        os.makedirs(jd, exist_ok=True)
        os.makedirs(self._restarts_dir(), exist_ok=True)
        job = {"job_id": job_id, "design_id": design_id, "status": "running",
               "total": len(cases), "done": 0, "failed": [], "current": None,
               "cases": cases, "max_parallel": max(1, min(int(max_parallel), 64)),
               "created": time.strftime("%Y-%m-%d %H:%M:%S"), "updated": ""}
        self._write_job(job_id, job)

        t = threading.Thread(target=self._driver, args=(job_id,), daemon=True)
        t.start()
        return {"ok": True, "job_id": job_id, "n_cases": len(cases),
                "max_parallel": job["max_parallel"], "mode": "async",
                "hint": "用 job_status 查询进度；完成后 postprocess 提取响应量"}

    def status(self, job_id):
        try:
            job = self._read_job(job_id)
        except FileNotFoundError:
            return {"ok": False, "error": "job_not_found",
                    "message": "找不到作业 %s" % job_id}
        log_tail = ""
        logp = os.path.join(self._job_dir(job_id), "driver.log")
        if os.path.exists(logp):
            with open(logp, encoding="utf-8", errors="replace") as f:
                log_tail = f.read()[-400:]
        return {"ok": True, "job_id": job_id, "status": job["status"],
                "progress": "%s/%s" % (job["done"], job["total"]),
                "failed": job["failed"], "current": job["current"],
                "log_tail": log_tail}

    def find_job_by_design(self, design_id):
        """按 design_id 找最近的作业（轮询续接/防重复提交用）。找不到返回 None。"""
        if not design_id:
            return None
        jd = self._jobs_dir()
        if not os.path.isdir(jd):
            return None
        for name in sorted(os.listdir(jd), reverse=True):  # job_时间戳，倒序=最新优先
            try:
                with open(os.path.join(jd, name, "job.json"), encoding="utf-8") as f:
                    if json.load(f).get("design_id") == design_id:
                        return name
            except Exception:
                continue
        return None

    # ---------- 执行 ----------
    def _driver(self, job_id):
        job = self._read_job(job_id)
        sem = threading.Semaphore(job["max_parallel"])
        threads = []

        def run_case(case):
            with sem:
                if self._read_job(job_id).get("status") != "running":
                    return
                cid = case["case_id"]
                self._update_job(job_id, lambda j: j.__setitem__("current", cid))
                cdir = os.path.join(self._job_dir(job_id), "case_%s" % cid)
                os.makedirs(cdir, exist_ok=True)
                try:
                    ok, msg, stage = self._run_single(job_id, cdir, case)
                    def _fin(j):
                        if not ok:
                            j["failed"].append({"case_id": cid, "stage": stage,
                                                "message": (msg or "")[:200]})
                        j["done"] += 1
                        if j.get("current") == cid:
                            j["current"] = None
                    self._update_job(job_id, _fin)
                except Exception as e:
                    def _err(j):
                        j["failed"].append({"case_id": cid, "stage": "exception",
                                            "message": str(e)[:200]})
                        j["done"] += 1
                    self._update_job(job_id, _err)

        for case in job["cases"]:
            t = threading.Thread(target=run_case, args=(case,), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        def _close(j):
            j["current"] = None
            j["status"] = "failed" if j["failed"] else "done"
        self._update_job(job_id, _close)
        final = self._read_job(job_id)
        with open(os.path.join(self._job_dir(job_id), "driver.log"), "a",
                  encoding="utf-8") as f:
            f.write("[driver] 作业结束: status=%s done=%s failed=%s\n" %
                    (final["status"], final["done"], len(final["failed"])))

    def _run_single(self, job_id, case_dir, case):
        """执行单工况；返回 (ok, msg, stage)。"""
        logf = os.path.join(case_dir, "run.log")
        with open(logf, "a", encoding="utf-8") as lf:
            # 1) restart 缓存（二元床层按全部二元参数缓存）
            if case.get("study") == "segregation":
                rkey = "packbin_ds%s_F%s_rs%s_rr%s_x%s_v1.restart" % (
                    case["d"], case["F"], case["r_s"], case["r_rho"], case["x_large"])
            else:
                rkey = "pack_d%s_F%s_v%d.restart" % (case["d"], case["F"], RESTART_VERSION)
            rpath = os.path.join(self._restarts_dir(), rkey)
            if not os.path.exists(rpath):
                lf.write("[case %s] 生成堆积 restart F=%s ...\n" % (case["case_id"], case["F"]))
                ok, msg = self._run_init(case_dir, case, rpath, lf)
                if not ok:
                    return False, msg, "init"

            # 2) 振动工况
            lf.write("[case %s] 运行振动工况 ...\n" % case["case_id"])
            ok, msg = self._run_case(case_dir, case, rpath, lf)
            if not ok:
                return False, msg, "run"
        return True, "", "run"

    def _run_init(self, case_dir, case, restart_target, lf):
        script = self._gen_init(case_dir, case)
        cmd = self.liggghts_cmd + ["-in", os.path.basename(script)]
        try:
            r = subprocess.run(cmd, cwd=case_dir, capture_output=True, text=True,
                               timeout=7200)
        except subprocess.TimeoutExpired:
            return False, "init 超时(>2h)"
        produced = os.path.join(case_dir, "pack.restart")
        if r.returncode == 0 and os.path.exists(produced):
            shutil.move(produced, restart_target)
            return True, ""
        tail = ((r.stdout or "") + (r.stderr or ""))[-500:]
        lf.write("init 失败 exit=%s:\n%s\n" % (r.returncode, tail))
        return False, "init 失败 exit=%s；log尾: %s" % (r.returncode, tail[-200:])

    def _run_case(self, case_dir, case, restart_path, lf):
        script = self._gen_run(case_dir, case)
        cmd = self.liggghts_cmd + [
            "-in", os.path.basename(script),
            "-var", "restart", restart_path,
            "-var", "meshfile", MESH_FILE,
            "-var", "amp", "%.6f" % case["amplitude"],
            "-var", "period", "%.6f" % case["period"],
            "-var", "save", str(case["save"]),
            "-var", "run_dev", str(case["run_dev"]),
            "-var", "run_stat", str(case["run_stat"]),
        ]
        try:
            r = subprocess.run(cmd, cwd=case_dir, capture_output=True, text=True,
                               timeout=86400)
        except subprocess.TimeoutExpired:
            return False, "run 超时(>24h)"
        dumps = [f for f in os.listdir(case_dir)
                 if f.startswith("dmp_") and f.endswith(".txt")]
        if r.returncode == 0 and dumps:
            return True, ""
        tail = ((r.stdout or "") + (r.stderr or ""))[-500:]
        lf.write("run 失败 exit=%s dumps=%s:\n%s\n" % (r.returncode, len(dumps), tail))
        return False, "run 失败 exit=%s, dumps=%s；log尾: %s" % (
            r.returncode, len(dumps), tail[-200:])

    # ---------- 脚本生成 ----------
    def _gen_init(self, case_dir, case):
        import math
        if case.get("study") == "segregation":
            return self._gen_init_binary(case_dir, case)
        with open(os.path.join(TEMPLATE_DIR, "in.init.liggghts"), encoding="utf-8") as f:
            tpl = f.read()
        d = case["d"]
        n = case["n_particles"]
        vp = math.pi / 6.0 * d ** 3
        mass_in = n * vp * 2500.0
        h_ins = n * vp / (0.5 * 0.094 * 0.0045)  # 目标颗数在松装体积分数~0.5下的高度
        script = (tpl
                  .replace("${meshfile}", MESH_FILE)
                  .replace("${radius}", "%.6f" % (d / 2))
                  .replace("${restart_out}", os.path.join(case_dir, "pack.restart"))
                  .replace("${ins_ymax}", "%.4f" % (0.005 + h_ins))
                  .replace("${mass_in}", "%.6e" % mass_in)
                  .replace("${ninsert}", str(max(50, n // 2))))
        path = os.path.join(case_dir, "in.init")
        with open(path, "w", encoding="utf-8") as f:
            f.write(script)
        return path

    def _gen_init_binary(self, case_dir, case):
        """二元混合床层初始化脚本（振动偏析研究）。"""
        with open(os.path.join(TEMPLATE_DIR, "in.init.binary.liggghts"), encoding="utf-8") as f:
            tpl = f.read()
        import math
        d_s = case["d"]
        n = case["n_particles"]          # 以小颗粒计的层数×列数
        vp = math.pi / 6.0 * d_s ** 3
        # 总插入质量按混合体系平均密度折算（质量分数加权）
        rho_mix = case["w1"] * case["rho_s"] + case["w2"] * case["rho_l"]
        mass_in = n * vp * rho_mix
        # 大颗粒使床层更高，松装体积分数取保守 0.45
        h_ins = n * vp / (0.45 * 0.094 * 0.0045)
        script = (tpl
                  .replace("${meshfile}", MESH_FILE)
                  .replace("${radius_s}", "%.6f" % case["radius_s"])
                  .replace("${radius_l}", "%.6f" % case["radius_l"])
                  .replace("${rho_s}", "%.1f" % case["rho_s"])
                  .replace("${rho_l}", "%.1f" % case["rho_l"])
                  .replace("${w1}", "%.6f" % case["w1"])
                  .replace("${w2}", "%.6f" % case["w2"])
                  .replace("${restart_out}", os.path.join(case_dir, "pack.restart"))
                  .replace("${ins_ymax}", "%.4f" % (0.005 + h_ins))
                  .replace("${mass_in}", "%.6e" % mass_in))
        path = os.path.join(case_dir, "in.init")
        with open(path, "w", encoding="utf-8") as f:
            f.write(script)
        return path

    def _gen_run(self, case_dir, case):
        name = "in.run.binary.liggghts" if case.get("study") == "segregation" else "in.run.liggghts"
        with open(os.path.join(TEMPLATE_DIR, name), encoding="utf-8") as f:
            tpl = f.read()
        script = tpl.replace("${outdir}", case_dir.replace("\\", "/"))
        path = os.path.join(case_dir, "in.run")
        with open(path, "w", encoding="utf-8") as f:
            f.write(script)
        return path
