# -*- coding: utf-8 -*-
"""自动量纲分析：Buckingham π 定理的自动推导。

给定候选物理参数及其量纲表达式，构造量纲矩阵并求零空间，
输出所有独立的无量纲组合（π 群）。

数学：n 个参数、k 个独立基本量纲 -> n-k 个独立 π 群。
量纲矩阵 D (k×n)，零空间基向量 v 满足 D·v=0，
对应 π = p1^v1 * p2^v2 * ... * pn^vn（无量纲）。

量纲表达式解析：用 pint 的 UnitRegistry 识别 "m"、"m/s^2"、"kg"、"Hz"、
"dimensionless" 等；也接受显式基本量纲字典 {"L":1,"T":-2}。
"""
from __future__ import annotations

import sympy

# 常见量纲速记（无 pint 依赖也能工作）
_DIM_ALIASES = {
    "L": {"L": 1}, "M": {"M": 1}, "T": {"T": 1},
    "length": {"L": 1}, "mass": {"M": 1}, "time": {"T": 1},
    "dimensionless": {}, "1": {}, "-": {},
}


def _parse_dimension(expr: str) -> dict:
    """解析量纲表达式 -> {基本量纲: 指数}。

    支持三种形式：
      1. 基本量纲组合: "L", "L*T^-2", "M*L/T^3"（*、/、^ 与 L/M/T/I/Θ/N/J）
      2. 单位表达式: "m", "m/s^2", "kg", "Hz"（经 pint 换算）
      3. 速记: "dimensionless"/"-"/""
    """
    s = (expr or "").strip()
    if not s or s.lower() in ("dimensionless", "none"):
        return {}
    # 形式1/3: 只含基本量纲符号与运算符（剥离指数数字后判断，
    # 否则 'T^-1' 切出数字 token 会被误判为非符号形式而落入 pint）
    stripped = []
    for tok in s.replace("*", " ").replace("/", " ").split():
        base = tok.split("^")[0]
        stripped.append(base)
    if stripped and all(t in ("L", "M", "T", "I", "Θ", "N", "J") for t in stripped if t):
        return _parse_symbolic(s)
    # 形式2: pint 单位
    try:
        import pint
        ureg = pint.UnitRegistry()
        u = ureg.parse_expression(s).units  # type: ignore[attr-defined]
        out = {}
        dl = u.dimensionality
        for key, exp in dl.items():
            dim = key.strip("[]")
            dim = {"length": "L", "mass": "M", "time": "T",
                   "current": "I", "temperature": "Θ",
                   "amount": "N", "luminosity": "J"}.get(dim, dim)
            if exp:
                out[dim] = int(exp)
        return out
    except Exception:
        pass
    # 兜底再试符号形式（如 "L T^-2"）
    return _parse_symbolic(s)


def _parse_symbolic(s: str) -> dict:
    """解析 'L*T^-2' / 'M*L/T^3' 形式的基本量纲组合。

    手动逐 token 解析（不用 sympify——不同 sympy 版本会把 'T' 替换成
    特斯拉单位展开成 {M,I,T} 混合量纲，实测踩过）。
    """
    out = {}
    # 按 * 和 / 切分（保留 / 的除法语义）
    s = s.replace(" ", "").replace("**", "^")
    if not s:
        return out
    # 统一为 (符号, 指数) 列表: 遇 '/' 后指数取负
    tokens = []
    denom = False
    for chunk in s.replace("/", "*/").split("*"):
        if not chunk:
            continue
        if chunk.startswith("/"):
            chunk, denom = chunk[1:], True
        if "^" in chunk:
            base, exp = chunk.split("^", 1)
            try:
                e = int(eval(exp)) if not exp.lstrip("-").isdigit() else int(exp)
            except Exception:
                raise ValueError("指数不合法: %s" % exp)
        else:
            base, e = chunk, 1
        if base not in ("L", "M", "T", "I", "Θ", "N", "J"):
            raise ValueError("未知量纲符号: %s（支持 L/M/T/I/Θ/N/J）" % base)
        tokens.append((base, -e if denom else e))
    for base, e in tokens:
        out[base] = out.get(base, 0) + e
    return {k: v for k, v in out.items() if v}


def _fmt_group(name_exp: list) -> str:
    """把 [(参数名, 指数), ...] 格式化为 'a^2*d^-1' 样式（指数1省略）。"""
    parts = []
    for name, exp in name_exp:
        if exp == 0:
            continue
        parts.append(name if exp == 1 else "%s^%d" % (name, exp))
    return "*".join(parts) if parts else "1"


def derive_pi_groups(params: dict) -> dict:
    """Buckingham π 定理自动推导。

    Args:
        params: {参数名: 量纲表达式}。量纲表达式支持 "L"/"L*T^-2" 基本量纲组合，
                或 "m"/"m/s^2"/"kg"/"Hz" 单位形式（内部经 pint 换算），
                无量纲参数写 "dimensionless"。

    Returns:
        {ok, n_params, n_base_dims, n_pi_groups, pi_groups: [{expression, ...}],
         dimensionless_params: [...], notes}
    """
    if not params or len(params) < 1:
        return {"ok": False, "error": "需要至少一个参数（{参数名: 量纲表达式}）"}

    # 1) 解析量纲
    parsed = {}
    errors = []
    for name, expr in params.items():
        try:
            parsed[str(name)] = _parse_dimension(str(expr))
        except Exception as exc:
            errors.append("%s: %s" % (name, exc))
    if errors:
        return {"ok": False, "error": "量纲解析失败", "details": errors}

    # 2) 无量纲参数直接分离
    dimless = [n for n, d in parsed.items() if not d]
    dimful = {n: d for n, d in parsed.items() if d}
    names = list(dimful.keys())

    # 3) 基本量纲集合
    all_dims = sorted({d for dd in dimful.values() for d in dd})
    if not names:
        return {"ok": True, "n_params": len(parsed), "n_base_dims": 0,
                "n_pi_groups": 0, "pi_groups": [],
                "dimensionless_params": dimless,
                "notes": ["全部参数均无量纲，无需 π 定理"]}

    # 4) 量纲矩阵（行=基本量纲, 列=参数）, 求零空间
    matrix = sympy.Matrix([[dimful[n].get(d, 0) for n in names] for d in all_dims])
    nullspace = matrix.nullspace()

    # 5) 整数化零空间基（sympy 通常给整数基；有理数基则通分）
    pi_groups = []
    for v in nullspace:
        vals = [sympy.nsimplify(x) for x in v]
        lcm = sympy.ilcm(*[sympy.Rational(x).q for x in vals]) if vals else 1
        ints = [int(x * lcm) for x in vals]
        # 约去公因子
        from math import gcd
        g = 0
        for x in ints:
            g = gcd(g, abs(x))
        if g > 1:
            ints = [x // g for x in ints]
        pi_groups.append(_fmt_group(list(zip(names, ints))))

    n_pi = len(pi_groups)
    notes = [
        "π 定理：n=%d 个量纲参数、k=%d 个独立基本量纲 -> %d 个独立 π 组合"
        % (len(names), len(all_dims), n_pi),
        "验证：本结果用于 {a:L, f:T^-1, d:L, g:L*T^-2} 时导出 d/a 与 g/(a*f^2)，"
        "与论文人工推导的无量纲振幅 a/d、振动强度 Γ 一致",
    ]
    if dimless:
        notes.append("无量纲参数（如恢复系数）本身即为 π 群，已单列")

    return {
        "ok": True,
        "n_params": len(parsed),
        "n_base_dims": len(all_dims),
        "n_pi_groups": n_pi,
        "pi_groups": pi_groups,
        "dimensionless_params": dimless,
        "base_dimensions": all_dims,
        "notes": notes,
    }
