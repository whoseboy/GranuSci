# -*- coding: utf-8 -*-
"""core/design.py（设计仪器）部署门槛测试。

覆盖：①合成双区数据全流程→仪器构建→advise 结构性正确 ②确定性（同输入同输出）
③κ 锚点单调 ④π 归一 ⑤与本地回放仪器（replay/instrument.py）在真实 405 状态上
的成分级等价（skipif 缺目录）。
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.derivatives import compute_derivatives          # noqa: E402
from core.clustering import AdvancedClusteringAnalysis    # noqa: E402
from core.fitting import fit_power_function               # noqa: E402
from core.design import DesignInstrument                  # noqa: E402

REPLAY_405 = os.path.join(
    os.path.dirname(ROOT), "论文ICML2027", "replay", "work", "state_405")


def _synthetic_state(tmp_path):
    """双区合成：分离角≈49°（远高于梯度噪声地板），抖动网格 324 点。"""
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(5)
    D0 = np.array([0.40, 0.80, -0.45]); D0 /= np.linalg.norm(D0)
    D1 = np.array([0.95, 0.20, -0.25]); D1 /= np.linalg.norm(D1)
    n = np.array([0.60, 0.65, -0.34]); n /= np.linalg.norm(n)
    box = np.array([[-0.6, 0.6], [0.1, 2.2], [-0.3, 1.2]])
    axs = [np.linspace(box[j, 0], box[j, 1], m) for j, m in ((0, 9), (1, 12), (2, 3))]
    G = np.meshgrid(*axs, indexing="ij")
    xi = np.column_stack([g.ravel() for g in G])
    cell = (box[:, 1] - box[:, 0]) / np.array([8, 11, 2]) * 0.25
    xi = xi + rng.uniform(-cell, cell, xi.shape)
    t = np.median(xi @ n)
    s = 1.0 / (1.0 + np.exp(-(xi @ n - t) / 0.12))
    eta = (1 - s) * (0.5 + (-1.4) * (xi @ D0)) + s * (0.3 + 1.4 * (xi @ D1))
    import zlib
    noise = np.array([np.random.default_rng(zlib.crc32(np.round(r, 9).tobytes())).normal(0, 0.05)
                      for r in xi])
    df = pd.DataFrame(np.column_stack([10 ** xi, 10 ** (eta + noise)]),
                      columns=["ad", "gamma", "F", "y"])
    csv = tmp_path / "in.csv"
    df.to_csv(csv, index=False)
    dx = tmp_path / "d.xlsx"
    compute_derivatives(input_file=str(csv), output_file=str(dx),
                        space="physical", k_neighbors=40)
    an = AdvancedClusteringAnalysis(data_file=str(dx), n_clusters=0,
                                    subclust_radius_factor=0.3, y_col="y")
    an.run_analysis(output_dir=str(tmp_path))
    fit_power_function(derivative_file=str(dx), output_dir=str(tmp_path),
                       clustering_analysis=an, c_mode="fixed", fixed_c=0.0,
                       y_col="y")
    # run_analysis 的偏导数文件名即 dx；DesignInstrument 按 *偏导数* / deriv* 通配定位
    os.rename(dx, tmp_path / "偏导数.xlsx")
    return str(tmp_path)


def test_advise_structure_and_determinism(tmp_path):
    d = _synthetic_state(tmp_path)
    inst = DesignInstrument(d)
    assert inst.K >= 1
    rs, ps = inst.kappa_anchors
    assert np.all(np.diff(ps) >= -1e-12), "κ 锚点须单调"
    out1 = inst.advise(budget=8, w_e=0.3, grid_n=8)
    out2 = inst.advise(budget=8, w_e=0.3, grid_n=8)
    assert out1["batch"] and len(out1["batch"]) <= 8
    assert json.dumps(out1["batch"], sort_keys=True) == json.dumps(out2["batch"], sort_keys=True)
    for p in out1["batch"]:
        assert p["A_bnd"] >= 0 and p["score"] == pytest.approx(
            p["A_bnd"] + p["A_law"] + 0.3 * p["A_exp"], abs=2e-4)
        assert sum(p["pi"]) == pytest.approx(1.0, abs=1e-6)
    # 候选/exclude 物理空间口径
    cand = [[10 ** v for v in row] for row in
            np.column_stack([np.linspace(-0.5, 0.5, 5), np.linspace(0.3, 2.0, 5),
                             np.full(5, 0.9)])]
    out3 = inst.advise(budget=3, candidates=cand, exclude=[cand[0]])
    assert len(out3["batch"]) <= 3
    for p in out3["batch"]:
        assert p["params_log10"]["ad"] != pytest.approx(np.log10(cand[0][0]), abs=1e-5)


def test_equivalence_with_local_replay_instrument():
    if not os.path.exists(os.path.join(REPLAY_405, "state.npz")):
        pytest.skip("本地回放状态不存在")
    replay_dir = os.path.join(os.path.dirname(ROOT), "论文ICML2027", "replay")
    sys.path.insert(0, replay_dir)
    from instrument import InstrumentState
    from pathlib import Path
    loc = InstrumentState(Path(REPLAY_405), y_col="e_k_star")
    srv = DesignInstrument(REPLAY_405)
    rng = np.random.default_rng(3)
    Xq = loc.x[rng.choice(len(loc.x), 25, replace=False)]
    pi_l, pi_s = loc.pi(Xq), srv.pi(Xq)
    assert np.abs(pi_l - pi_s).max() < 1e-9
    mu_l, s2_l, ht_l = loc.law_predictive(Xq)
    mu_s, s2_s, ht_s = srv.law_predictive(Xq)
    assert np.abs(mu_l - mu_s).max() < 1e-9
    assert np.abs(ht_l - ht_s).max() < 1e-9
    ab_l, ab_s = loc.A_bnd(Xq, pi_l, mu_l, s2_l), srv.A_bnd(Xq, pi_s, mu_s, s2_s)
    assert np.abs(ab_l - ab_s).max() < 1e-8
