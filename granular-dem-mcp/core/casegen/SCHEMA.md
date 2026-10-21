# Case Description Schema v1（casegen 契约层）

结构化 case 描述 = agent（LLM）产出的**唯一**建模接口；渲染器把它变成
LIGGGHTS-PUBLIC 3.8 脚本。设计原则（C1 判决的直接执行）：

1. 语法零自由度——schema 里没有的字段不存在，渲染器是唯一语法出口
2. 物理护栏内建——渲染前校验（SI 单位下限/时间步上限/域-几何包含）
3. 每个字段对应 C1 剥错实验中出过错的层——组件参数化封死
4. 向后兼容 legacy 战役——振动槽 case 可被本 schema 精确表达

## Schema（JSON）

```jsonc
{
  "meta": {
    "name": "str（case 名，文件安全字符）",
    "purpose": "str（研究目的，账本用）"
  },

  // ---- 几何与域 ----
  "domain": {
    "box": [xmin, xmax, ymin, ymax, zmin, zmax],   // SI 米；须含 geometry 全部+运动余量
    "walls": [                                       // 壁面列表（v2：原语+mesh）
      {"kind": "xplane", "pos": 0.1},               // 原语：xplane/yplane/zplane/xcylinder...
      {"kind": "zplane", "pos": 0.0},
      // mesh 壁（v2）：参数化生成或预置 STL
      // 几何约定（v2.1 起）：所有 gen 均为 z-up（开口/法向朝 +z，与默认重力 [0,0,-9.81] 对齐）。
      // incline：length 沿 x（下坡向）、高差沿 z、width 沿 y；funnel/box_tray：开口朝 +z；
      // cylinder_open：轴沿 z。例：angle=30 的 incline 顶面从 (0,0,0) 降到 (L·cos30, 0, -L·sin30)。
      // 常见坑：mesh 与相邻壁之间留缝 > 粒径时颗粒会从缝中漏出（v1_dryrun 实测 3mm 缝漏 2mm 粒）。
      {"kind": "mesh", "stl": {
          "gen": "funnel",                          // 参数化生成：funnel/hopper_wedge/incline/cylinder_open/box_tray
          "params": {"r_top": 0.03, "r_throat": 0.008},   // 各几何默认参数见 stl_gen.py
          "offset": [0, 0, 0]                       // 平移（STL 原点→目标位置）
      }}
      // 或预置：{"kind":"mesh","stl":{"file":"/work/meshes/my.stl"}}
    ],
    "margin": 0.005                                  // 壁面运动余量建议值（校验提示用）
  },

  // ---- 颗粒 ----
  "particles": {
    "types": [                                       // 1-3 种颗粒
      {"id": 1, "diameter": 0.002, "density": 2500}
      // v3 增 "shape": "sphere"|"superquadric"
    ],
    "insertion": {
      "method": "pack",                              // pack（自然堆积式插入）| rain
      "region": [xmin,xmax,ymin,ymax,zmin,zmax],     // 插入区（须在 domain 内）
      "composition": [                               // 各 type 的目标
        {"type": 1, "count": 500}
        // 或 {"type":1,"mass_frac":0.7} + total_mass
      ],
      "seed": 32452843
    }
    // v1 不做 create_atoms 单颗粒直插——单球任务用 pack+count=1
  },

  // ---- 材料与接触（C1 错误层 2/4/7 的封死）----
  "material": {
    "youngs_modulus": 5.0e7,          // ≥5e6（SI 硬下限，校验）
    "poissons_ratio": 0.45,
    "restitution": 0.95,              // 颗粒-颗粒 & 颗粒-壁（v1 单值）
    "friction": 0.3,
    "rolling_friction": 0.03          // 可 0
  },

  // ---- 物理场 ----
  "fields": {
    "gravity": [gx, gy, gz],          // m/s²（倾斜=倾斜重力矢量，如斜面任务）
    "boundary": "fff"                  // f/p 组合，v1 支持 fff 与 p f f（剪切）
  },

  // ---- 激励（可选；无则静置）----
  // LIGGGHTS 3.8 查证：primitive 壁仅 shear；wiggle/linear 在 mesh 壁
  // （fix move/mesh）。v2 新增 mesh 壁运动型：
  "excitation": {
    "type": "vibrate_domain" | "wall_shear" | "mesh_motion" | null,
    // mesh_motion（v2）：指定 mesh 壁运动（振动槽=mesh wiggle，传送带=mesh linear）
    "mesh_wall_index": 0,             // walls 里 mesh 壁的下标
    "motion": "wiggle" | "linear",
    "amplitude": [ax, ay, az],        // wiggle：振幅矢量（米）
    "frequency_hz": 10.0,             // wiggle 周期
    "velocity": [vx, vy, vz],         // linear：恒速（m/s，传送带用）
    // v1 两型字段保留：
    "shear_wall_index": 0, "shear_dim": "x", "shear_velocity": 0.2,
    // vibrate_domain：振动激励（legacy 战役型）——整域振动：所有壁+颗粒
    //   由 fix move all wiggle 实现——⚠ v1_dryrun 实测：所有颗粒被运动学
    //   prescribed 振动锁定（输运/流动类问题物理失效，仅适用于"颗粒群整体
    //   振动、关注相对分布"的问题）。输运/流动问题请用 mesh_motion。
    //   幅值矢量任一向非零
    "amplitude": [ax, ay, az],        // 米（vibrate_domain）
    "frequency_hz": 10.0,
    // wall_shear：单壁恒速剪切（primitive shear）——传送带/库埃特
    //   ⚠ v1_dryrun 曾被误判"不生效"：床形指标（床高/φ）对整体平移不敏感——
    //   剪切有效性判据须看质心漂移与底层颗粒速度，不能只看床形。
    "shear_wall_index": 0,            // domain.walls 的下标
    "shear_dim": "x",                 // x|y|z
    "shear_velocity": 0.2             // m/s
  },

  // ---- 数值（C1 错误层 1/6 的封死：渲染器写死正确语法）----
  "numerics": {
    "timestep": 1.0e-5,               // 上限校验：dt ≤ 0.2*sqrt(rho d³/(E*...))，保守查表
    "neighbor_skin": 0.002
  },

  // ---- 运行分段 ----
  "phases": [                          // 顺序执行；每段一个 run
    {"name": "settle", "steps": 400000, "thermo_every": 10000},
    {"name": "main",   "steps": 200000, "thermo_every": 10000}
  ],

  // ---- 输出 ----
  "output": {
    "dump_columns": ["id","type","x","y","z","vx","vy","vz","radius"],
    "dump_every": 100,                 // 步
    "final_state": true                // 是否写 restart（供续跑）
  }
}
```

## 校验规则（渲染前强制，全部来自 C1 七层错误与已知物理）

| 规则 | 错误消息（给 agent 的修复字典） |
|---|---|
| youngs_modulus ≥ 5e6 | "SI 单位下杨氏模量须 ≥5e6（LIGGGHTS 硬检查）" |
| timestep ≤ Rayleigh 时间的 20%（球，按最小粒径） | "时间步超 Rayleigh 上限，建议 ≤{x}" |
| insertion.region ⊂ domain.box | "插入区超出模拟域" |
| excitation 振幅+margin ⊂ domain | "壁面振动幅超出域边界，扩域或减幅" |
| 粒子总数 ≤ 5000（v1 硬顶） | "超过 v1 粒子数上限" |
| 总步数 ≤ 2e6 | "总步数超限" |
| composition 计数×单颗粒体积 ≤ 插入区体积×0.55 | "插入区过小（堆积分数上限 0.55）" |

## 渲染映射（schema → in.liggghts 的固定骨架）

- header：echo/log/atom_style granular/atom_modify map array/boundary（从
  fields.boundary 渲染）/newton off/**communicate single vel yes**（写死）
- units si
- create_box：atom types 数=len(particles.types)
- **property/global 四件套**：peratomtype / peratomtype 1 / peratomtypepair
  N / peratomtypepair N（按 type 数渲染，写死格式）
- pair_style gran model hertz tangential history rolling_friction epsd2
  [torsionTorque on 当 rolling>0]（写死）
- wall/gran：model hertz tangential history **primitive** type 1
  {kind} {args}（逐 wall 渲染；v1 无 mesh）
- gravity / nve/sphere / timestep / neighbor（写死结构，参数来自 schema）
- insertion：insert/pack（写死安全语法：insert_every once overlapcheck yes
  all_in yes；count 按 composition 反算 mass_in 或用 maxattempt）
- excitation 渲染（2026-09-23 查证后定案）：
  - vibrate_domain → `fix movewall all move wiggle 0 {ay} 0 period T`
    （颗粒群整域正弦振动，重力场下壁面相对振动；legacy 等效机制）
  - wall_shear → 对应 wall/gran 行追加 `shear {dim} {v}`
  （primitive 壁 wiggle 不存在，源码实证：primitive_wall.h 无运动逻辑）
- phases：逐段 run + thermo_style custom step atoms ke c_rke（写死，
  compute erotate/sphere 预定义）
- dump：custom {every} ./out_{phase}_*.txt {columns}（写死路径格式）

## 能力边界（v2）

- 壁面=box 原语（平面/圆柱）+ 参数化 mesh 几何（funnel/hopper_wedge/
  incline/cylinder_open/box_tray，参数化生成）+ 预置 STL 文件
- 激励=整域振动 / 单壁剪切 / **mesh 壁 wiggle 振动（真实振动槽）/
  mesh 壁 linear 恒速（传送带）**
- 颗粒=球形 1-3 type；超二次/多球 v3
- 插入=pack（自然堆积式）；流场/连续注入 v3
