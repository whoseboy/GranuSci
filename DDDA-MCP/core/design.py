"""设计仪器（design instrument）：区制可辨识性驱动的实验设计建议。

在已完成的 DDDA 分析状态（聚类+拟合产物目录）上，对候选工况计算复合期望
信息增益并输出批次建议：

    score(x) = [A_bnd(x) + A_law(x) + w_e·A_exp(x)] / c(x)

  A_bnd  归属信息：I(z;y) = H(π) − E_y[H(π(·|y))]（κ 校准票 → 归属后验，
         两区预测分布混合信道，1D 求积；未拟合簇=中性成分+质量因子）
  A_law  定律信息：Σ_k π_k·½·log(1+h̃_k)（log-log 岭重拟合的归一化杠杆）
  A_exp  探索项：最近已仿真点距离（模型外声明启发项）
  c(x)   成本：v0 为 1（cost_mode="none"）；墙钟回归版待作业数据接入

κ 校准：票比 → 实测硬标签正确率（留一票，isotonic 单调化）——只用当前
分析状态内部数据，不依赖任何评估集。方法学与假设清单见论文算法设计文档
（A1-A7/D1-D5），本模块为其 v0 实现的服务端移植。

只读：本模块不向 clustering_dir 写任何文件。
"""

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

LAM0 = 1e-2          # log-log 重拟合弱岭（预注册值）
ALPHA_K = 0.5        # κ_u 平滑（诊断模式用）
NQUAD = 161          # A_bnd 求积节点
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def _find_deriv_file(clustering_dir, explicit=None):
    if explicit:
        return explicit
    cands = sorted(glob.glob(os.path.join(clustering_dir, "*偏导数*.xlsx")))
    cands += sorted(glob.glob(os.path.join(clustering_dir, "deriv*.xlsx")))
    if not cands:
        raise FileNotFoundError(f"未找到偏导数文件: {clustering_dir}")
    return cands[0]


def _find_fitting_json(clustering_dir, explicit=None):
    if explicit:
        return explicit
    for p in (os.path.join(clustering_dir, "fitting_results.json"),
              os.path.join(os.path.dirname(clustering_dir), "fitting_results.json")):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"未找到 fitting_results.json: {clustering_dir}")


def _find_cluster_coords(clustering_dir, k, x_cols):
    for p in (os.path.join(clustering_dir, f"cluster_{k}_coordinates.csv"),
              os.path.join(clustering_dir, "cluster_coordinates", f"cluster_{k}_coordinates.csv")):
        if os.path.exists(p):
            return pd.read_csv(p)[x_cols].values
    return None


class DesignInstrument:
    """一次 DDDA 分析状态上的设计仪器。"""

    def __init__(self, clustering_dir, fitting_json=None, deriv_file=None,
                 k_votes=7, y_col=None):
        from core.fitting import _robust_fit
        self.dir = clustering_dir
        fj = _find_fitting_json(clustering_dir, fitting_json)
        fit = json.loads(open(fj, encoding="utf-8").read())
        self.x_cols = fit["x_cols"]
        self.y_col = y_col or fit.get("y_col")
        self.fit_entries = {e["cluster_id"]: e for e in fit["fitting_results"] if "a" in e}

        # 标签与坐标（log10 空间）
        lab_path = os.path.join(clustering_dir, "clustering_components", "cluster_labels.npy")
        if not os.path.exists(lab_path):
            lab_path = os.path.join(clustering_dir, "cluster_labels.npy")
        self.labels = np.load(lab_path)

        df = pd.read_excel(_find_deriv_file(clustering_dir, deriv_file))
        # x 用 float32——与 clustering.load_data_chunked 的 analysis.x 完全同值，
        # 保证 KNN 平局裁决与 predict/聚类几何一致（规则网格等距点对此敏感）；
        # y 用 float64——与 fitting.py 的读取口径一致
        self.x = df[self.x_cols].values.astype(np.float32).astype(np.float64)  # log10
        self.y = df[self.y_col].values.astype(np.float64)           # log10
        if len(self.x) != len(self.labels):
            raise ValueError(f"坐标 {len(self.x)} 与标签 {len(self.labels)} 数量不一致")

        # u 矩阵（新版落盘产物；旧目录缺省时跳过 κ_u 诊断）
        u_path = os.path.join(clustering_dir, "clustering_components", "u_matrix.npy")
        self.u = np.load(u_path) if os.path.exists(u_path) else None

        self.K = int(self.labels.max()) + 1
        self.unfitted = [k for k in range(self.K) if k not in self.fit_entries]

        lab = self.labels >= 0
        self.cloud_x, self.cloud_lab = self.x[lab], self.labels[lab]
        self.tree = KDTree(self.cloud_x)

        # 清洗掩码（复现 _robust_fit 规则，c_mode 跟随原始拟合）+ log-log 贝叶斯重拟合
        c_mode = fit.get("c_mode", "fixed")
        self.clean_mask = np.zeros(len(self.labels), dtype=bool)
        self.D, self.beta, self.Minv, self.sigma2 = {}, {}, {}, {}
        for k, e in self.fit_entries.items():
            idx = np.where(self.labels == k)[0]
            Di = np.array(e["Di"])
            psi_all = self.x[idx] @ Di
            C_all, A_all = 10.0 ** psi_all, 10.0 ** self.y[idx]
            res, C_cl, A_cl, _, _ = _robust_fit(C_all, A_all, c_mode=c_mode,
                                                fixed_c=float(e.get("c", 0.0) or 0.0))
            if "error" in res:
                continue
            key_clean = set(zip(np.round(C_cl, 12), np.round(A_cl, 12)))
            in_clean = np.array([(c_, a_) in key_clean for c_, a_ in
                                 zip(np.round(C_all, 12), np.round(A_all, 12))])
            self.clean_mask[idx[in_clean]] = True
            psi, eta = psi_all[in_clean], self.y[idx[in_clean]]
            Psi = np.column_stack([np.ones(in_clean.sum()), psi])
            M = Psi.T @ Psi + LAM0 * np.eye(2)
            beta = np.linalg.solve(M, Psi.T @ eta)
            r = eta - Psi @ beta
            self.D[k] = Di
            self.beta[k] = beta
            self.Minv[k] = np.linalg.inv(M)
            self.sigma2[k] = float(r @ r / max(len(eta) - 2, 1))

        self.kappa_anchors = self._fit_kappa_label()

    # ---------- κ ----------
    def _loo_votes(self):
        n = len(self.labels)
        out = np.zeros((n, self.K))
        d, j = self.tree.query(self.x, k=min(8, len(self.cloud_x)))
        j = np.atleast_2d(j)
        for i in range(n):
            # 留一：剔除自身（坐标精确匹配；网格数据坐标唯一）
            keep = [t for t in range(j.shape[1])
                    if not np.array_equal(self.cloud_x[j[i][t]], self.x[i])][:7]
            for t in keep:
                out[i, self.cloud_lab[j[i][t]]] += 1
        return out

    def _fit_kappa_label(self):
        from sklearn.isotonic import IsotonicRegression
        votes = self._loo_votes()
        lab = self.labels >= 0
        if lab.sum() < 10:
            return (np.array([1.0]), np.array([1.0]))
        maj = votes[lab].argmax(axis=1)
        r = votes[lab].max(axis=1) / np.maximum(votes[lab].sum(axis=1), 1)
        correct = (maj == self.labels[lab]).astype(float)
        rs, idx = np.unique(r, return_inverse=True)
        cnt = np.bincount(idx).astype(float)
        p = np.bincount(idx, weights=correct) / cnt
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip",
                                 y_min=1.0 / max(self.K, 1), y_max=1.0)
        return (rs, iso.fit_transform(rs, p, sample_weight=cnt))

    def _apply_kappa(self, votes):
        rs, ps = self.kappa_anchors
        tot = np.maximum(votes.sum(axis=1), 1)
        r = votes.max(axis=1) / tot
        p_maj = np.interp(r, rs, ps, left=ps[0], right=ps[-1])
        maj = votes.argmax(axis=1)
        pi = np.zeros(votes.shape)
        rest = (1.0 - p_maj) / max(self.K - 1, 1)
        pi[:, :] = rest[:, None] if rest.ndim else rest
        pi[np.arange(len(votes)), maj] = p_maj
        return pi

    # ---------- 批量查询 ----------
    def votes_and_dnn(self, Xq):
        Xq = np.atleast_2d(Xq)
        kq = min(7, len(self.cloud_x))
        d, j = self.tree.query(Xq, k=kq)
        d = np.atleast_2d(d); j = np.atleast_2d(j)
        votes = np.zeros((len(Xq), self.K))
        np.add.at(votes, (np.repeat(np.arange(len(Xq)), kq), self.cloud_lab[j.ravel()]), 1)
        return votes, d[:, 0]

    def pi(self, Xq):
        votes, _ = self.votes_and_dnn(Xq)
        return self._apply_kappa(votes)

    def law_predictive(self, Xq):
        Xq = np.atleast_2d(Xq)
        nq = len(Xq)
        mu = np.zeros((nq, self.K)); s2 = np.zeros((nq, self.K)); ht = np.zeros((nq, self.K))
        y_bar = float(np.mean(self.y))
        for k in sorted(self.fit_entries):
            if k not in self.D:
                continue
            psi = np.column_stack([np.ones(nq), Xq @ self.D[k]])
            h = np.einsum("ij,jk,ik->i", psi, self.Minv[k], psi)
            mu[:, k] = psi @ self.beta[k]
            s2[:, k] = self.sigma2[k] * (1.0 + h)
            ht[:, k] = h
        for k in self.unfitted:
            mu[:, k] = y_bar; s2[:, k] = 1e4
        return mu, s2, ht

    # ---------- 获取函数 ----------
    def A_bnd(self, Xq, pi=None, mu=None, s2=None):
        if pi is None: pi = self.pi(Xq)
        if mu is None: mu, s2, _ = self.law_predictive(Xq)
        fitted = s2 < 1e3
        w = np.where(fitted, pi, 0.0)
        mass = w.sum(axis=1, keepdims=True)
        out = np.zeros(len(Xq))
        have = mass[:, 0] > 1e-12
        if not have.any():
            return out
        pi_n = np.where(have[:, None], w / np.maximum(mass, 1e-12), 0.0)
        mu_n = np.where(fitted, mu, 0.0)
        s2_n = np.where(fitted, s2, 1.0)
        s = np.sqrt(s2_n[have])
        mu_h, pi_h = mu_n[have], pi_n[have]
        lo = (mu_h - 5 * s).min(axis=1); hi = (mu_h + 5 * s).max(axis=1)
        H_pri = -(pi_h * np.log(np.clip(pi_h, 1e-12, 1))).sum(axis=1)
        E_H = np.zeros(have.sum())
        for i in range(have.sum()):
            yv = np.linspace(lo[i], hi[i], NQUAD)
            comp = pi_h[i] * np.exp(-0.5 * ((yv[:, None] - mu_h[i]) / s[i]) ** 2) \
                   / (s[i] * np.sqrt(2 * np.pi))
            m = comp.sum(axis=1)
            q = comp / np.clip(m[:, None], 1e-300, None)
            E_H[i] = TRAPZ(m * (-(q * np.log(np.clip(q, 1e-12, 1))).sum(axis=1)), yv)
        out[have] = (H_pri - E_H) * mass[have, 0]
        return out

    def A_law(self, Xq, pi=None, ht=None):
        if pi is None: pi = self.pi(Xq)
        if ht is None: _, _, ht = self.law_predictive(Xq)
        return (pi * 0.5 * np.log1p(ht)).sum(axis=1)

    def A_exp(self, Xq):
        _, dnn = self.votes_and_dnn(Xq)
        return dnn

    # ---------- 批次建议 ----------
    def candidate_grid(self, grid_n=20):
        pr_path = os.path.join(self.dir, "clustering_components", "parameter_ranges.json")
        if os.path.exists(pr_path):
            pr = json.loads(open(pr_path, encoding="utf-8").read())
            lo, hi = np.array(pr["min_values"]), np.array(pr["max_values"])
        else:
            lo, hi = self.x.min(axis=0), self.x.max(axis=0)
        axs = [np.linspace(lo[j], hi[j], grid_n) for j in range(len(lo))]
        G = np.meshgrid(*axs, indexing="ij")
        return np.column_stack([g.ravel() for g in G])

    def advise(self, budget=30, w_e=0.3, candidates=None, grid_n=20,
               exclude=None, repel_radius=None, cost=None):
        """candidates/exclude 均为**物理空间**坐标（与 predict 工具同口径），
        内部转 log10；candidates=None 时用 parameter_ranges 上的 log 网格。"""
        if candidates is None:
            Xc = self.candidate_grid(grid_n)
        else:
            Xc = np.log10(np.atleast_2d(np.asarray(candidates, dtype=float)))
        if exclude is not None and len(exclude):
            ex = np.log10(np.atleast_2d(np.asarray(exclude, dtype=float)))
            keep = np.ones(len(Xc), dtype=bool)
            for e in ex:
                keep &= np.linalg.norm(Xc - e, axis=1) > 1e-6
            Xc = Xc[keep]
        pi = self.pi(Xc)
        mu, s2, ht = self.law_predictive(Xc)
        ab = self.A_bnd(Xc, pi, mu, s2)
        al = self.A_law(Xc, pi, ht)
        ae = self.A_exp(Xc)
        c = np.ones(len(Xc)) if cost is None else np.asarray(cost, dtype=float)
        score = (ab + al + w_e * ae) / c
        if repel_radius is None:
            d, _ = self.tree.query(self.cloud_x, k=2)
            repel_radius = float(np.median(d[:, 1])) * 1.5
        s = score.copy()
        sel = []
        for _ in range(min(budget, len(Xc))):
            i = int(np.argmax(s))
            if not np.isfinite(s[i]):
                break
            sel.append(i)
            s[np.linalg.norm(Xc - Xc[i], axis=1) < repel_radius] = -np.inf
        batch = []
        for rank, i in enumerate(sel):
            votes, dnn = self.votes_and_dnn(Xc[i][None, :])
            dom = int(np.argmax([ab[i], al[i], w_e * ae[i]]))
            reason = ["边界辨识：归属票 {v}，π={p}，两区预测可分性驱动".format(
                          v=votes[0].astype(int).tolist(), p=np.round(pi[i], 3).tolist()),
                      "定律收紧：杠杆率 {h}，归属加权后定律信息主导".format(
                          h=np.round(ht[i], 3).tolist()),
                      "探索：最近已仿真点距离 {d:.3f}（log10），覆盖驱动".format(d=dnn[0])][dom]
            batch.append({
                "rank": rank + 1,
                "params_log10": dict(zip(self.x_cols, np.round(Xc[i], 4).tolist())),
                "params_physical": dict(zip(self.x_cols, np.round(10.0 ** Xc[i], 6).tolist())),
                "A_bnd": round(float(ab[i]), 4), "A_law": round(float(al[i]), 4),
                "A_exp": round(float(ae[i]), 4), "score": round(float(score[i]), 4),
                "pi": np.round(pi[i], 3).tolist(),
                "pred_mu": np.round(mu[i], 3).tolist(), "pred_sd": np.round(np.sqrt(s2[i]), 3).tolist(),
                "reason": reason,
            })
        return {
            "budget": budget, "w_e": w_e, "repel_radius_log10": round(repel_radius, 4),
            "K": self.K, "unfitted_clusters": self.unfitted,
            "n_fuzzy": int((self.labels < 0).sum()),
            "kappa_anchors": {"vote_ratio": np.round(self.kappa_anchors[0], 3).tolist(),
                              "p_correct": np.round(self.kappa_anchors[1], 3).tolist()},
            "laws": {str(k): {"Di": np.round(self.D[k], 4).tolist(),
                               "b": round(float(self.beta[k][1]), 4),
                               "log10a": round(float(self.beta[k][0]), 4),
                               "sigma_log": round(float(np.sqrt(self.sigma2[k])), 4)}
                      for k in self.D},
            "cost_mode": "none (v0)",
            "batch": batch,
            "note": "建议为咨询性质：agent 可采纳/修改/否决，决策请入账。"
                    "A_bnd/A_law 为复合 EIG 的链式分解（nat），A_exp 为声明的模型外启发项。",
        }
