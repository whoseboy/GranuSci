# DDDA MCP 接口契约（v1.3 —— 与实现完全对齐，2026-09-09；v1.2=2026-08-25）

> 已按实际实现更新。DDDA-MCP 是标准 FastMCP 服务（stdio/SSE/streamable-http 三传输），
> 5 个工具。LIGGGHTS MCP（granular-dem）由用户自行实现，见文末待定部分。

## DDDA-MCP（已实现，代码在 `DDDA-MCP/`）

部署：本地/服务器 `python server.py`（stdio），或 `MCP_TRANSPORT=streamable-http MCP_HOST=0.0.0.0 MCP_PORT=8000 python server.py`（HTTP，端点 `/mcp`）。

**数据格式约定**：输入 Excel（.xlsx），前 N-1 列自变量 + 末列因变量，**所有数据须为 log10 对数空间**；偏导数列以 `dY_d` 前缀自动识别。

**数据流总设计**（MCP 协议无文件传输原语，工具参数只能传 JSON 值）：
- **主线（科研自动化）**：granular-dem MCP（LIGGGHTS，同一台服务器）生成新仿真数据 → 返回路径 → 智能体转述给 DDDA MCP 分析。数据不跨机器，路径即通道。
- **平台侧数据入口**（三种，均已实测）：
  1. `load_data_from_url` 工具：传 URL，服务器自取（平台数据集"挂载地址"、共享存储等 URL 可寻址数据源）
  2. `data={"columns":[...], "rows":[[...],...]}` 行内参数：几千行内的表格数据直接塞参数
  3. `input_file="builtin:4mm_675"`：内置基准数据集（复刻论文/演示保险丝）

**智能体接力已实测**（2026-08-25，Claude Code 无头模式）：LLM 自主完成 compute_derivatives → cluster_analysis → fit_power_function → predict 四步接力，路径全部从上一步返回值转述，无人工干预，全链路成功。

| 工具 | 输入要点 | 返回要点 |
|---|---|---|
| `compute_derivatives` | input_file（必填）、x_cols/y_col、k_neighbors/k_bandwidth | ok、output_file、各偏导数统计 |
| `cluster_analysis` | input_file（偏导数文件）、**n_clusters=0 自动定簇数**、fcm_m、subclust_radius_factor（MATLAB ra 语义，默认0.10）、membership_threshold（默认0.6，低于剔除为模糊点） | ok、n_clusters、n_fuzzy_points、cluster_summary（各子域点数/占比）、files |
| `fit_power_function` | derivative_file、clustering_dir、c_mode（fixed/free）、max_clean_ratio | ok、**delta_star（论文指标δ\*）**、fitting_results[]（每子域：a/b/c、r²、**delta、effective_exponents、effective_formula**） |
| `run_full_pipeline` | input_file + 上述参数并集 | 三段结果合并，~5KB |
| `predict` | clustering_dir、params（**物理空间** dict，如 {"ad":1.0,"gamma":10.0,"F":8.0}）、fitting_json（默认自动查找）、k=7 | ok、**subdomain_id（KNN投票）**、predicted_value、formula、effective_exponents |
| `load_data_from_url` | url（http/https）、filename（可选） | ok、local_path（可直接作后续工具的 input_file）、size_bytes |

**智能体验证结论**（2026-08-25，Claude Code 作为标准 MCP 客户端实测）：
- 5 工具全部可列出、可调用；工具链文件路径传递正常
- 返回严格合法 JSON（无 NaN/Inf），分步 <3KB / 全流程 ~5KB，字段齐全
- 错误路径结构化（ok=False + 中文 message）
- 675 工况基准：自动 3 子域，ARI=0.90，δ\*=8.26%（论文 13.86%）
- **磐石平台托管"工具验证"失败为平台侧问题**（同代码在标准客户端下全部通过）

## v1.3 增量（2026-09-09）：设计仪器工具（ICML 2027 论文 E3 前置）

新增 `core/design.py`（DesignInstrument）+ 两个工具（服务器 8000 已部署，
备份 `backup_20260909_114652`，本地 pytest 2/2 + 服务器端老战役状态逐值验证通过）：

| 工具 | 输入要点 | 返回要点 |
|---|---|---|
| `batch_query` | clustering_dir、params_list（**物理空间** [{参数名:值},...]）、fitting_json（可选） | ok、K、results[]（votes、**pi=κ 校准归属概率**、nearest_distance_log10） |
| `design_advice` | clustering_dir、budget（默认30）、w_e（默认0.3 预注册值）、candidates/exclude（物理空间坐标列表，可选）、grid_n（默认20）、repel_radius（默认=中位近邻距×1.5） | ok、batch[]（rank、params_physical、**A_bnd/A_law/A_exp/score 逐点分解**、pi、pred_mu/pred_sd、reason 人话解释）、kappa_anchors、laws、K/unfitted/n_fuzzy 诊断 |

score=(A_bnd+A_law+w_e·A_exp)/c；A_bnd=归属信息（校准票→π，混合信道容量，nat）、
A_law=定律信息（贝叶斯杠杆 ½log(1+h̃)）、A_exp=探索（最近点距离，声明的模型外项）、
c=成本（v0 恒 1，cost_mode 字段披露）。**建议为咨询性质：agent 可采纳/修改/否决，
决策应入账**（ledger）。只读工具，不向 clustering_dir 写任何文件。旧分析目录
（无 u_matrix.npy）自动降级可用（κ 校准只依赖 labels）。

同版聚类落盘增量（加法）：`clustering_components/u_matrix.npy`（FCM 隶属度矩阵）、
`local_eigenvalues.npy`（逐点 |g|²）——供下游诊断，缺失不影响任何既有工具。

## granular-dem MCP（LIGGGHTS，已实现：granular-dem-mcp:1.4 Docker 镜像）

FastMCP 服务（与 DDDA 同风格），4 工具：
- `generate_design(ad_list, gamma_list, F_list, d, dev_periods, stat_periods)` -> design_id
- `run_simulation(design_id, max_parallel)` -> job_id（异步；每(d,F)自动生成堆积restart并缓存，restart带版本号）
- `job_status(job_id)` -> status/progress/failed
- `postprocess(job_id)` -> **responses.csv（物理空间：case_id, ad, gamma, F, n_particles, e_k_star, t_y_star, h_m_star）**

响应量按论文第2章定义：E_k\*=⟨½\|U\|²⟩/(gd)、T_y\*=T̄_y/(dg)（2d网格胞元）、h_m\*=h̄_m/d。

**服务器部署**：
- DDDA-MCP: :8000（venv 部署）
- granular-dem: :8001（docker run -v <工作目录>:/work -e HOST_WORK_DIR=<工作目录>）
- 共享数据通道：~/mcp/work（designs/jobs/restarts/results）

**双MCP联调已实测**（2026-08-25，27工况全闭环）：设计->仿真(27/27成功)->后处理->responses.csv
物理空间直传 DDDA run_full_pipeline(space="physical")。注意：27点规模下 DDDA 自动分区会
因数据稀疏而退化（正确返回 ok=False）——质量分析需 675 级别数据（builtin:4mm_675 或
大规模仿真网格）。27工况偏导数符号与基准一致（+0.83/+0.81/-0.72 vs 基准+0.85/+0.63/-0.81）。


## granular-dem MCP v1.4（2026-09-23）：casegen v1 三工具上线

架构（C1 判决）：LLM 只产结构化 case JSON（契约=core/casegen/SCHEMA.md，
全文内嵌于 build_case 的 description），渲染器为唯一语法出口，物理护栏
（杨氏模量≥5e6/时间步≤0.3 Rayleigh/堆积分数≤0.55/插入区每维≥2×最大粒径/
粒子≤5000/步数≤2e6/重力方向必须有底壁）在 build 时强制校验，错误消息
字典化（按提示修复 JSON 重交即可）。

### tool 8: build_case(case) → job_id | 校验错误列表
### tool 9: run_case(job_id, wait?, timeout=600) → 运行（单核 nice，墙钟顶 600s）
### tool 10: case_result(job_id) → 状态+dump 列表+末帧摘要
   末帧摘要含粒子数/type 分布/质心/包围盒；粒子丢失>20% 时状态转 done_leak
   并给出泄漏警示（壁面封闭性检查的运行时兜底）。

v1 能力边界：box 原语壁（平面/圆柱）+球形 1-3 type+pack 插入+整域振动/
单壁剪切激励。v2=mesh 几何，v3=单壁独立运动/多球型/连续注入。
C1 二测（结构化路径 10 任务）10/10 vs 裸写 0/10。

## granular-dem MCP v1.5（2026-09-23）：诊断箱 v1 标准算子

### tool 11: analyze_frame(job_id, op, n_layers, dump_file)
- op=bed_geometry：床层高度/床层包围体积/堆积分数/孔隙率（重力轴向，
  含粒径修正；带 sanity 提示——随机密堆典型 φ≈0.55-0.64）
- op=density_profile：沿重力轴分层局部堆积分数（分层/流化/表面结构诊断）
- 设计纪律：agent 不要手工从 bbox 推导这些量——工具数字自带口径可过审计
- 诊断箱新增 D6_quasi_static（特征速度 <0.1×粒径/s 才算静置达成）；
  D2 修正：RUNNING_IN 只要求末段持续下降（去掉与中值比较的过严条件）

C1 三测复盘修正：agent 报的堆积分数 0.0023 不是算错——是仿真本身未沉降
（300 球颗粒雨 2s 后特征速度仍 0.45 m/s，床未形成）。教训进 D6：
静置 case 的有效性=跑完∧守恒∧准静态，三条件缺一不可。

## lit-mcp v1.0（2026-09-23，端口 8006）：统一文献层上线（A3）

granusci 侧前缀 litx_。架构=外部检索（主通道，OpenAlex 主/Crossref 兜底，
服务器白名单出网）+ 本地文献库（代理 sci-evo-extract :8005 的 papers/evidence）
+ 项目源文档登记（wangchi2024 学位论文等）+ 引用登记（cite_register）。

- litx_search_external(query, limit)：书目元数据候选（DOI/标题/作者/年份/venue）
- litx_library_search / litx_library_evidence：本地已处理文献与证据单元
- litx_get_source_doc：源文档书目条目（王驰论文 cite_id=wangchi2024）
- litx_cite_register(bib)：正式引用登记→cite_id（引用门核心）

**引用门 v2（cite_id 制）**：每条正式 \cite 必须对应 litx 书目元数据记录或
cite_id 登记；kb_search 文本块不再构成可溯源出处（b101 引用幻觉的结构性修复，
审稿 prompt 已升级）。domain-knowledge.md 已改造为机制图式卡（物理知识保留
供 RAG，文献标注仅作检索线索）。王驰论文全文块保留（RAG），出处走 litx。
运行：~/mcp/lit-mcp/（DDDA-MCP venv 拷贝，nohup，日志 server.log）。

## granular-dem MCP v2.0（2026-09-23）：casegen v2 mesh 几何上线

- mesh 壁：wall {kind:mesh, stl:{gen|file, params, offset}}——五种参数化几何
  （funnel/hopper_wedge/incline/cylinder_open/box_tray）或预置 STL；
  渲染器自动生成 STL 落盘 job 目录
- mesh 运动激励：excitation {type:mesh_motion, mesh_wall_index, motion:
  wiggle|linear}——fix move/mesh（振动槽 wiggle=legacy 战役同款；linear=传送带）
- 新校验：mesh STL bbox+offset 须在 simbox 内（越界=面元被切→运行时
  mesh elements lost，服务器实测教训转护栏）
- LIGGGHTS 3.8 语法查证补充：mesh/surface 平移关键词是 move（非 trans）
线上 e2e：200 球 box_tray+wiggle 振动槽 build→run→result 全通（粒子守恒
200/200，11 dump，D2=RUNNING_IN 正确）。本地测试 20/20。
