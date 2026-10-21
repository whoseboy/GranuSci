# -*- coding: utf-8 -*-
"""参数化 STL 几何生成器（casegen v2：mesh 壁面支持）。

基本几何（ASCII STL，单位 mm→case schema 用 SI；生成器内统一 SI）：
  funnel    漏斗：圆柱上段+圆锥收口+出料管
  hopper_wedge  楔形料仓（2D 挤出）
  incline   斜面板（角度+长度）
  cylinder  开口圆柱筒（可做提壁堆积角）
  box_tray  矩形槽（四面壁+底，开口向上——legacy 方槽的通用版）
设计约束：法向朝内（颗粒侧）；三角度制网格（粗网格对 DEM 足够——颗粒
接触走 LIGGGHTS 三角面片，网格密度影响接触精度与性能，默认 5-10mm 边长）。
"""
from __future__ import annotations

import math
from pathlib import Path


def _facet(v1, v2, v3):
    """一个 STL 三角形（自动法向）。"""
    ux, uy, uz = v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[0]*0+v1[2]*0+ (v2[2]-v1[2])
    # 上面写法易错，直接写清楚：
    ux, uy, uz = (v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2])
    wx, wy, wz = (v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2])
    nx, ny, nz = (uy*wz-uz*wy, uz*wx-ux*wz, ux*wy-uy*wx)
    n = math.sqrt(nx*nx+ny*ny+nz*nz) or 1.0
    nx, ny, nz = nx/n, ny/n, nz/n
    return (f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}\n"
            f"    outer loop\n"
            f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n"
            f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n"
            f"      vertex {v3[0]:.6e} {v3[1]:.6e} {v3[2]:.6e}\n"
            f"    endloop\n"
            f"  endfacet\n")


def _stl(name: str, facets: str) -> str:
    return f"solid {name}\n{facets}endsolid {name}\n"


def _ring(r, y, n, cx=0.0, cz=0.0):
    """xz 平面圆环采样（y 高度）。"""
    pts = []
    for i in range(n):
        a = 2*math.pi*i/n
        pts.append((cx + r*math.cos(a), y, cz + r*math.sin(a)))
    return pts


def _side_facets(upper, lower, facets):
    """两环之间侧面（法向朝外由顶点序决定——上环顺时针连下环）。"""
    n = len(upper)
    for i in range(n):
        j = (i+1) % n
        facets.append(_facet(lower[i], lower[j], upper[j]))
        facets.append(_facet(lower[i], upper[j], upper[i]))
    return facets


def funnel(r_top=0.03, r_throat=0.008, h_cone=0.04, h_tube=0.03,
           h_cyl=0.02, n=32) -> str:
    """漏斗（竖直）：圆柱段(h_cyl) + 锥段(h_cone, r_top→r_throat) + 直管(h_tube)。
    原点在锥喉中心；颗粒从上方进。只生成侧壁（无顶底盖）。"""
    fs = []
    y_cyl_bottom = h_cone/2 + h_cyl
    y_cyl_top = h_cone/2 + h_cyl + h_cyl  # 冗余行防笔误——实际只用一段
    # 三段环
    y0 = -h_cone/2 - h_tube        # 管底
    y1 = -h_cone/2                  # 管顶=锥底（喉）
    y2 = h_cone/2                   # 锥顶
    y3 = h_cone/2 + h_cyl           # 圆柱顶
    r_tube = r_throat
    ring_bottom = _ring(r_tube, y0, n)
    ring_throat = _ring(r_throat, y1, n)
    ring_topcone = _ring(r_top, y2, n)
    ring_cyltop = _ring(r_top, y3, n)
    _side_facets(ring_throat, ring_bottom, fs)   # 管段
    _side_facets(ring_topcone, ring_throat, fs)  # 锥段
    _side_facets(ring_cyltop, ring_topcone, fs)  # 柱段
    return _stl("funnel", "".join(fs))


def hopper_wedge(width=0.1, depth=0.02, h_up=0.06, h_down=0.04,
                 slot_half=0.005, n_seg=16) -> str:
    """楔形料仓（y 竖直，x 宽度向，z 挤出向）：上部矩形段 h_up，下部收缩到
    中心缝（缝半宽 slot_half）高 h_down；两端 z 面封板。开口：顶进料、底缝出料。"""
    fs = []
    # 侧面轮廓（x-y）：右壁
    y0 = 0.0
    y1 = h_down
    y2 = h_down + h_up
    w_top = width/2
    # 右壁内表面（x>0）：从 (slot_half,0) 到 (slot_half,h_down)?? 不——
    # 楔形：缝在底部，右壁从 (slot_half, 0) 斜到 (w_top, h_down)，再直上 (w_top, y2)
    # z 面位置
    z0, z1 = -depth/2, depth/2
    def wall(x0, y_a, x1, y_b, zn, zt):
        """一面墙（x-y 线段沿 z 挤出），法向朝内容区。"""
        fs.append(_facet((x0, y_a, zn), (x1, y_b, zn), (x1, y_b, zt)))
        fs.append(_facet((x0, y_a, zn), (x1, y_b, zt), (x0, y_a, zt)))
    # 右壁（法向朝 -x 侧）：顶点序决定
    wall(slot_half, 0.0, w_top, h_down, z0, z1)      # 斜段
    wall(w_top, h_down, w_top, y2, z0, z1)           # 直段
    # 左壁镜像
    wall(-slot_half, 0.0, -w_top, h_down, z0, z1)
    wall(-w_top, h_down, -w_top, y2, z0, z1)
    # 前后 z 封板（法向朝内）：多边形分两个三角形/梯形
    for zn, sign in ((z0, 1), (z1, -1)):
        # 右半剖面多边形：(slot_half,0)(w_top,h_down)(w_top,y2)(0,y2)(0,0)
        poly = [(slot_half, 0.0), (w_top, h_down), (w_top, y2), (0.0, y2), (0.0, 0.0)]
        for i in range(1, len(poly)-1):
            a, b, c = poly[0], poly[i], poly[i+1]
            if sign > 0:
                fs.append(_facet((a[0], a[1], zn), (b[0], b[1], zn), (c[0], c[1], zn)))
            else:
                fs.append(_facet((a[0], a[1], zn), (c[0], c[1], zn), (b[0], b[1], zn)))
        # 左半
        poly = [(-slot_half, 0.0), (-w_top, h_down), (-w_top, y2), (0.0, y2), (0.0, 0.0)]
        for i in range(1, len(poly)-1):
            a, b, c = poly[0], poly[i], poly[i+1]
            if sign > 0:
                fs.append(_facet((a[0], a[1], zn), (c[0], c[1], zn), (b[0], b[1], zn)))
            else:
                fs.append(_facet((a[0], a[1], zn), (b[0], b[1], zn), (c[0], c[1], zn)))
    return _stl("hopper_wedge", "".join(fs))


def incline(length=0.3, width=0.1, angle_deg=30, thickness=0.005,
            n_seg=8) -> str:
    """斜面板（x 下坡向，y 竖直，z 宽度）：顶面为滑移面。原点在板顶面
    上端中点。"""
    fs = []
    a = math.radians(angle_deg)
    dx = length*math.cos(a)
    dy = -length*math.sin(a)
    z0, z1 = -width/2, width/2
    # 顶面四角（法向朝上）：P0(0,0,z0) P1(0,0,z1) P2(dx,dy,z1) P3(dx,dy,z0)
    p0, p1, p2, p3 = (0,0,z0), (0,0,z1), (dx,dy,z1), (dx,dy,z0)
    fs.append(_facet(p0, p1, p2)); fs.append(_facet(p0, p2, p3))
    # 底面+侧+端封板（厚度 thickness 沿法向下移）
    nx, ny = math.sin(a), math.cos(a)   # 顶面法向
    off = (nx*thickness, ny*thickness, 0)
    q0 = tuple(p - o for p, o in zip(p0, off))
    q1 = tuple(p - o for p, o in zip(p1, off))
    q2 = tuple(p - o for p, o in zip(p2, off))
    q3 = tuple(p - o for p, o in zip(p3, off))
    fs.append(_facet(q0, q2, q1)); fs.append(_facet(q0, q3, q2))
    fs.append(_facet(p0, q0, q1)); fs.append(_facet(p0, q1, p1))
    fs.append(_facet(p3, q3, q2)); fs.append(_facet(p3, q2, p2))
    fs.append(_facet(p0, p3, q3)); fs.append(_facet(p0, q3, q0))
    fs.append(_facet(p1, q1, q2)); fs.append(_facet(p1, q2, p2))
    return _stl("incline", "".join(fs))


def cylinder_open(r=0.03, h=0.08, n=32) -> str:
    """开口圆柱筒（无底无盖——配 zplane 底壁用，做提壁堆积角）。"""
    fs = []
    lower = _ring(r, 0.0, n)
    upper = _ring(r, h, n)
    _side_facets(upper, lower, fs)
    return _stl("cylinder_open", "".join(fs))


def box_tray(lx=0.1, lz=0.0045, h=0.15, wall_t=0.004, n_seg=4) -> str:
    """矩形槽：四壁+底（开口向上）。y=0 为内底面。壁厚 wall_t 仅供封闭
    几何（接触只看内表面）。"""
    fs = []
    x0, x1 = -lx/2, lx/2
    z0, z1 = -lz/2, lz/2
    X0, X1 = x0-wall_t, x1+wall_t
    Z0, Z1 = z0-wall_t, z1+wall_t
    # 底（法向 +y）
    fs.append(_facet((x0, 0, z0), (x0, 0, z1), (x1, 0, z1)))
    fs.append(_facet((x0, 0, z0), (x1, 0, z1), (x1, 0, z0)))
    # 四壁（内表面法向朝内）
    # x=x0 壁（法向 +x）
    fs.append(_facet((x0, 0, z0), (x0, h, z1), (x0, h, z0)))
    fs.append(_facet((x0, 0, z0), (x0, 0, z1), (x0, h, z1)))
    # x=x1 壁（法向 -x）
    fs.append(_facet((x1, 0, z1), (x1, 0, z0), (x1, h, z0)))
    fs.append(_facet((x1, 0, z1), (x1, h, z0), (x1, h, z1)))
    # z=z0 壁（法向 +z）
    fs.append(_facet((x1, 0, z0), (x0, 0, z0), (x0, h, z0)))
    fs.append(_facet((x1, 0, z0), (x0, h, z0), (x1, h, z0)))
    # z=z1 壁（法向 -z）
    fs.append(_facet((x0, 0, z1), (x1, 0, z1), (x1, h, z1)))
    fs.append(_facet((x0, 0, z1), (x1, h, z1), (x0, h, z1)))
    return _stl("box_tray", "".join(fs))


GENERATORS = {
    "funnel": funnel, "hopper_wedge": hopper_wedge, "incline": incline,
    "cylinder_open": cylinder_open, "box_tray": box_tray,
}


def _to_zup(stl: str) -> str:
    """y-up → z-up 旋转变换（绕 x 轴 +90°：T(x,y,z)=(x,-z,y)）。

    生成器内部全部按 y 轴朝上构建（elevation/开口沿 +y），而 casegen 世界
    约定 z 轴朝上（重力默认 [0,0,-9.81]，诊断按 z 轴分层）。v1_dryrun 实测：
    不变换时 incline 平板对 -z 下落的颗粒呈侧立姿态，颗粒从旁边掉落，
    表现为"mesh 穿透"假象。变换是正旋转（det=+1），三角绕向保持不变。
    vertex 与 facet normal 行同步变换。"""
    out_lines = []
    for line in stl.splitlines():
        s = line.strip()
        if s.startswith("vertex") or s.startswith("facet normal"):
            parts = s.split()
            if s.startswith("facet"):
                # "facet normal x y z" → 坐标在 parts[2:5]
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                out_lines.append(f"  facet normal {x:.6e} {-z:.6e} {y:.6e}")
            else:
                # "vertex x y z" → 坐标在 parts[1:4]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                out_lines.append(f"      vertex {x:.6e} {-z:.6e} {y:.6e}")
        else:
            out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def generate(kind: str, params: dict, out_path: str) -> dict:
    """主入口：生成参数化 STL（z-up 世界约定）。返回几何信息（包围盒等，供 case 校验）。"""
    if kind not in GENERATORS:
        return {"ok": False, "message": f"未知几何 {kind}; 可用: {list(GENERATORS)}"}
    stl = _to_zup(GENERATORS[kind](**params))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(stl, encoding="utf-8")
    # 包围盒粗算（解析顶点）
    xs, ys, zs = [], [], []
    for line in stl.splitlines():
        if line.strip().startswith("vertex"):
            _, x, y, z = line.split()
            xs.append(float(x)); ys.append(float(y)); zs.append(float(z))
    return {"ok": True, "kind": kind, "file": str(out_path),
            "bbox": [[min(xs), max(xs)], [min(ys), max(ys)], [min(zs), max(zs)]],
            "n_facets": stl.count("facet normal")}
