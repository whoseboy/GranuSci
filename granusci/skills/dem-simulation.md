---
name: dem-simulation
description: 离散元仿真的工具用法、协议选择与工程约束——含自定义 case 构建（casegen）与振动槽批量管线（legacy）
when_to_use: 需要构建自定义仿真 case、设计仿真工况、提交仿真作业、查询进度或后处理时载入
---

# DEM 仿真（LIGGGHTS）

## 两条仿真路径（按研究需要选）

**路径 A：自定义 case 构建（casegen v1，2026-09 起）**——自由建模：任意 box 域 +
平面/圆柱壁 + 1-3 种球形颗粒 + 整域振动/单壁剪切激励。适合非振动槽问题
（堆积角/剪切/输运/碰撞/新几何研究）。工具链：

1. `dem_build_case(case)` → job_id：提交**结构化 case JSON**（完整契约在工具
   description 里——域/壁面/颗粒/材料/激励/数值/分段/输出八段）。**不要写
   LIGGGHTS 脚本**——渲染器是唯一语法出口，脚本会被校验拒绝。
   - 物理护栏会强制校验（杨氏模量≥5e6/时间步≤0.3 Rayleigh/堆积分数≤0.55/
     插入区每维≥2×最大粒径/重力方向必须有底壁/粒子≤5000/步数≤2e6）；
   - 校验错误是字典化的（每条错误消息直接告诉你怎么改）——按提示修 JSON 重交；
   - 设计纪律：先想物理（域要围住颗粒+运动余量、壁面要封闭、颗粒数×体积
     不超插入区一半），再填数字。
2. `dem_run_case(job_id, wait, timeout)` → 运行（单核，默认超时 600s 墙钟顶）。
   wait=true 同步等待；长仿真用 wait=false + 轮询。
2b. `dem_modify_case(job_id, patch)` → 仿真门处置：补丁深合并重建为新 case
   （原作业保留），配合下方仿真门处置表使用。
3. `dem_case_result(job_id)` → 状态 + dump 列表 + **末帧摘要与诊断箱**
   （D1 粒子守恒 / D2 动能弛豫 / D4 速度 sanity；overall=PASS/WARN/FAIL）。
   **仿真门纪律：status=done 且 diagnostics.overall≠FAIL 才算有效结果**；
   done_leak（颗粒坠出域）= 无效，须修壁面后重建 case。
   - 动能弛豫 verdict=RUNNING_IN 表示沉降/发展未完成——统计段之前应加长
     发展段（phases 加步数）重建。
   - dump 文件在宿主机 work/casegen_jobs/<job_id>/，可传 DDDA 分析。

**路径 B：振动槽批量管线（legacy，energy/segregation 战役用）**——固定准二维
方槽 + 无量纲参数网格批量扫参。见下文。两条路径的作业体系独立
（job_id 前缀区分：cg_=casegen，job_=legacy）。

## 仿真门（casegen 路径强制纪律）

**有效结果的必要条件：status=done ∧ D1 守恒（PASS/WARN）∧ 与物理目标匹配的
诊断通过。** 静置类目标额外要求 D6 准静态 PASS；振动/剪切类目标 D2=STEADY
可接受。诊断不满足时按处置表自主行动：

| 诊断信号 | 处置（用 dem_modify_case 打补丁重建后重跑） | patch 要点 |
|---|---|---|
| D2=RUNNING_IN / D6=FAIL | 延长运行：步数加倍起改 | `{'phases':[{'name':..., 'steps':2×原, ...}]}` |
| D1=FAIL / done_leak | 修壁面封闭性（补底壁/封开面） | `{'domain':{'walls':[...全量重写...]}}` |
| 颗粒雨（插入区堆积分数过低→床不形成） | 缩插入区让颗粒集中落下 | `{'particles':{'insertion':{'region':[...]}}}` |
| D4=FAIL（速度爆炸） | timestep 减半重跑；仍爆→上报 | `{'numerics':{'timestep':...}}` |

**处置上限：同一 case 最多 2 轮自主处置。两轮后仍不达标 → 停止重试，向
用户报告完整诊断链（各轮诊断结果+已试 patch），等人工裁决。** 无限重试烧
机时和无头绪乱试都禁止。每次处置须在账本记录：源 job_id→新 job_id、诊断
信号、patch 内容、结果。

**激励有效性判别（v1_dryrun 教训）**：判断激励（wall_shear/mesh_motion/
振动）是否生效，**不能只看床形指标**（床高/堆积分数/footprint 对床层整体
平移不敏感——床被推向端墙堆积后床形近似不变，造成"激励不生效"假象）。
正确判据：①质心位置的时间序列（各帧对比）②底层颗粒平均速度（贴底 2.5mm
内颗粒的 vx/vy/vz）③极端对照（无激励 vs 10×激励应有可测差异）。任一判据
显示运动即激励生效，即使床形不变。

## 路径 B：振动槽批量管线（legacy）
容器：准二维方槽（槽长 0.1 m × 槽高 0.2 m × 槽宽 4.5 mm，竖直正弦振动）。
- 单分散研究（study="energy"，默认）：一种颗粒（直径 d、密度 2500 kg/m³）；
- 二元混合研究（study="segregation"）：小颗粒（type 1，直径 d）+ 大/重颗粒（type 2），
  粒径比 r_s=d_l/d_s、密度比 r_rho=rho_l/rho_s、大颗粒体积分数 x_large。
  槽宽 4.5 mm 限制 r_s ≤ 2（大颗粒须小于槽宽）；a/d 与 F 以小颗粒直径计。

## 工具链（固定顺序）
1. `dem_generate_design(ad_list, gamma_list, F_list, d, dev_periods, stat_periods[, study, r_s_list, r_rho_list, x_large_list])` → design_id
   - 三个无量纲参数的水平列表做全排列（segregation 再乘二元参数列表）；**全部必填**。
   - ad=a/d 无量纲振幅（范围约 [0.05,10]）；gamma=Γ=a(2πf)²/g 振动强度（约 [0.5,500]）；
     F=层数（1–64，按槽长/直径的整列数折算颗数）。
2. `dem_run_simulation(design_id, max_parallel, wait, wait_timeout)` → job_id
   - `wait=true`：同步等待至完成（推荐；每工况约 20–60s（快速协议），大设计内部轮询，wait_timeout 默认足够 60+ 工况）；
   - `wait=false`：立即返回 job_id。**异步纪律（paper11 教训）**：提交后当场检查返回值含 job_id；
     每拿到一个 job_id 立即用 record_finding(kind=artifact) 记账（design_id→job_id 映射+responses 路径格式
     /work/results/<job_id>/responses.csv）——消息历史会被压缩，不记账就会丢失收割线索；
     提交多个设计时逐个确认落成，全部提交完再统一轮询收割，不要提交完就不管。
   - **max_parallel 建议 32**（服务器 128 核，单工况单核进程级并行——32 路并行仍有充足余量；批量大设计传 32，少量验证工况可 4-8）。
3. `dem_postprocess(job_id)` → responses.csv：每工况无量纲响应量（物理空间）：
   e_k*（平均动能）、t_y*（纵向颗粒温度）、h_m*（质心高度）、ep*（平均离底高度）；
   segregation 另有 seg_dh_star（物种高度差，正=大颗粒在上）、mix_index（Lacey 混合指数，0=完全偏析，1=完全混合）。
   **幂等**：对同一 job 重复调用返回相同结果——收割过的路径记入账本后不要再刷。
4. `dem_merge_responses(job_ids)` → 多轮数据合并（同参数自动去重，最新优先）。
   合并输入的 job 清单从账本的 artifact 条目取（那里有完整 design→job→路径映射），不要凭记忆。

## 机制观测工具（单工况深挖，用于机制调查）
- `dem_get_snapshot(job_id, case_id, frame)`：单帧颗粒快照（x/y/vx/vy/speed/type/radius 平行数组）。
  返回数据大——配 make_figure 的 from_call 引用直接渲染 particles 图（colorby=type 看分层/speed 看流动），
  再用 vision_analyze 视觉判读结构（分层/气隙/对流/团簇）。
- `dem_get_profiles(job_id, case_id)`：时间平均剖面与场——
  large_fraction(y) 偏析剖面 | solid_fraction 场（气隙/流化诊断）| vx/vy 场（对流环诊断）|
  逐帧 time_series（收敛检查）。fields 为 [ix][iy]。同样用 from_call 引用画 heatmap/line。

## 仿真协议（发展段/统计段周期数）
| 协议 | dev_periods | stat_periods | 适用 |
|---|---|---|---|
| 快速（默认 20/8） | 20 | 8 | 能量类响应的规律结构与符号快速发现 |
| 加长 | 100–200 | 30–50 | 结构类响应（偏析、分层——收敛慢）或需要更好统计时 |

协议时长与工况周期数成正比；偏析时标通常远慢于能量时标，偏析研究发展段建议 ≥100 周期（可先试探性跑 1–2 个工况确认收敛）。

## 工程约束（违反会浪费算力）
- 每个（粒径, 层数[, 二元参数]）组合首次运行会先生成自然堆积床层并缓存，之后复用——相同床层参数的批量工况放一个 design 里比拆多个 design 高效。
- 一次全排列工况数 = 各列表长度连乘，提交前自己先乘一遍，确认在预算内。
- 后处理得到的 responses.csv 路径可直接作为 DDDA 分析输入（space="physical"）。

## 失败处置
- `dem_job_status` 返回 failed 时看失败详情与日志尾部；个别工况失败可剔除该工况重新生成小设计补跑，不必全部重跑。
