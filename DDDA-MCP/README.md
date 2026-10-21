# DDDA-MCP: 数据驱动维度分析 MCP 服务

将原脚本 3（光滑微分求导）、4（FCM 聚类）、5（幂函数拟合）打包为标准 MCP 服务，
通过 stdio 传输，供支持 MCP 的 AI 客户端调用。

## 数据格式要求

- **文件格式**：Excel (`.xlsx`)，第一行为表头。
- **列约定**：前 N-1 列为自变量，最后一列为因变量（列数、列名任意，自动识别）。
- **对数空间**：所有输入数据**必须固定在 log10 对数空间**（即已对原始物理量取以 10 为底的对数）。
  工具内部在拟合阶段会通过 `B=10^X`、`A=10^Y` 转换到物理空间，用户无需也不能更改此行为。
- **偏导数列**：`compute_derivatives` 输出的偏导数列为 `dY_d{原列名}`，聚类和拟合工具通过 `dY_d` 前缀自动识别。

示例（3 列，2 个自变量）：

| X1  | X2  | Y   |
|-----|-----|-----|
| 0.5 | 1.2 | 2.1 |
| ... | ... | ... |

## 安装

```bash
pip install -r requirements.txt
```

依赖：`mcp`、`numpy`、`pandas`、`scipy`、`scikit-learn`、`scikit-fuzzy`、
`matplotlib`、`openpyxl`、`joblib`、`tqdm`。

## 启动服务

### 方式一：stdio（本地 MCP 客户端）

```bash
python server.py
```

默认通过 stdio 与 MCP 客户端通信。在 MCP 客户端中配置：

```json
{
  "mcpServers": {
    "ddda-mcp": {
      "command": "python",
      "args": ["server.py"],
      "cwd": "/path/to/DDDA-MCP"
    }
  }
}
```

### 方式二：Streamable HTTP（填写服务地址 URL）

```bash
set MCP_TRANSPORT=streamable-http
set MCP_HOST=127.0.0.1
set MCP_PORT=8000
python server.py
```

PowerShell 可使用：

```powershell
$env:MCP_TRANSPORT = "streamable-http"
$env:MCP_HOST = "127.0.0.1"
$env:MCP_PORT = "8000"
python server.py
```

连接配置：

```json
{
  "mcpServers": {
    "ddda-mcp": {
      "url": "http://127.0.0.1:8000/mcp",
      "type": "streamable_http"
    }
  }
}
```

如果客户端选择 SSE，则启动时设置 `MCP_TRANSPORT=sse`，服务地址使用 `http://127.0.0.1:8000/sse`。

## 提供的工具

### 1. `compute_derivatives` — 光滑偏导数计算

使用 KDE 自适应高斯加权局部线性拟合，计算因变量对各自变量的偏导数。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `input_file` | string | 是 | 输入 Excel 路径（log10 空间） |
| `output_file` | string | 否 | 输出路径，默认在输入同目录自动生成 |
| `n_indep` | integer | 否 | 自变量列数，默认总列数-1 |
| `x_cols` | string[] | 否 | 显式自变量列名（覆盖 n_indep） |
| `y_col` | string | 否 | 因变量列名，默认末列 |
| `k_neighbors` | integer | 否 | 局部拟合邻居数，默认随维度自适应 |
| `k_bandwidth` | integer | 否 | KDE 带宽近邻数，默认 30 |

输出：原数据 + `dY_d{列名}` 偏导数列的 Excel，返回每列导数统计。

### 2. `cluster_analysis` — subclust+FCM 复合聚类

对偏导数数据执行减法聚类确定初始中心，再用 FCM 精细聚类，输出每个子空间的
特征值/特征向量（含主导 Di 值）。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `input_file` | string | 必填 | 含 `dY_d` 偏导数列的 Excel |
| `output_dir` | string | 输入同目录 | 结果输出目录 |
| `n_clusters` | integer | 1 | 目标聚类数；**0 = subclust 自动确定**（论文"自动分区"） |
| `n_indep` / `x_cols` / `y_col` | — | — | 列识别参数 |
| `fcm_m` | float | 2.0 | FCM 模糊指数 |
| `fcm_error` | float | 0.0003 | 收敛阈值 |
| `fcm_maxiter` | integer | 2000 | 最大迭代次数 |
| `max_samples` | integer | 200000 | 聚类抽样上限 |
| `subclust_radius_factor` | float | 0.10 | 减法聚类半径因子（MATLAB ra 语义，[0,1]归一化特征空间） |
| `membership_threshold` | float | 0.6 | FCM 最大隶属度阈值，低于则标记为模糊点(-1)不参与拟合；0 关闭 |
| `subclust_suppression_ratio` | float | 1.25 | 减法聚类抑制比（rb=sq·ra，MATLAB 默认） |

输出：`clustering_eigen_analysis.xlsx`、`clustering_components/`（标签/中心/参数）、
聚类坐标 CSV。

### 3. `fit_power_function` — 物理空间幂函数拟合

对每个聚类子空间拟合 `A = a·C^b + c`，其中组合变量
`C = B1^D1·B2^D2·...·Bn^Dn`（Di 为主导特征向量，|Di|<0.05 置零）。
含残差离群点清洗。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `derivative_file` | string | 必填 | 含偏导数的 Excel |
| `output_dir` | string | 输入同目录 | 结果目录 |
| `clustering_dir` | string | 输入同目录 | 聚类结果目录 |
| `c_mode` | string | `"fixed"` | `"fixed"`（固定 c）或 `"free"`（自由拟合） |
| `fixed_c` | float | 1.0 | 固定 c 值（仅 fixed 模式） |
| `max_clean_ratio` | float | 0.1 | 离群点清洗比例 |
| `di_threshold` | float | 0.05 | Di 置零阈值 |
| `min_fit_points` | integer | 10 | 拟合最少数据点 |
| `curve_fit_maxfev` | integer | 20000 | curve_fit 最大迭代 |

输出：拟合曲线图（总览 + 各子空间）、`fitting_data.xlsx`、文本报告。

### 5. `predict` — 新参数预测（规律->应用闭环）

给定物理空间的新参数点，用已拟合的分区幂律模型预测响应量，并以 KNN 投票判断
该点所属子域。需先运行 run_full_pipeline（或分步的 cluster+fit）。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `clustering_dir` | string | 是 | cluster_analysis 输出目录（含 cluster_*_coordinates.csv） |
| `params` | object | 是 | 各自变量物理空间值，如 `{"ad": 1.0, "gamma": 10.0, "F": 8.0}` |
| `fitting_json` | string | 否 | fitting_results.json 路径，默认在 clustering_dir 及上级查找 |
| `k` | integer | 否 | KNN 投票近邻数，默认 7 |

返回：`{subdomain_id, predicted_value, formula, effective_exponents, knn_votes, ...}`

### 4. `run_full_pipeline` — 完整流水线

依次执行 偏导数 → 聚类 → 拟合。聚类与拟合在内存中直接传递中间结果，
所有最终结果保存到 `output_dir`。参数为上述三个工具参数的并集。

## 项目结构

```
DDDA-MCP/
├── server.py              # MCP 服务入口（注册 4 个工具，stdio 传输）
├── config.py              # 全局默认参数配置
├── requirements.txt
├── mcp.json               # MCP 客户端配置示例
├── core/
│   ├── derivatives.py     # 脚本3：KDE 光滑偏导数
│   ├── clustering.py      # 脚本4：subclust+FCM 聚类
│   └── fitting.py         # 脚本5：幂函数拟合（带清洗）
└── utils/
    ├── data_io.py         # Excel 读写、列推断、数据校验
    └── visualization.py   # matplotlib 字体/后端配置（微软雅黑）
```

## 默认参数调整

所有默认参数集中在 `config.py` 的 `Config` 类中，可直接修改或通过工具参数覆盖。
字体默认为微软雅黑（`Microsoft YaHei`），自动回退到 SimHei、DejaVu Sans。

## 算法说明

1. **偏导数**：Silverman 带宽 → KDE 局部密度 → 自适应权重（与密度成反比）→
   中位数 KNN 距离确定核宽 h → 加权局部线性最小二乘 → 导数缩放回原始量纲。
2. **聚类**：逐点 `∇f·∇fᵀ` 矩阵的主导特征向量作为局部特征 → RobustScaler 标准化 →
   大数据集 KMeans 预抽样 → 减法聚类确定初始中心 → skfuzzy FCM 精细聚类 →
   子空间全局 `∑∇f·∇fᵀ` 特征分解得到主导 Di。
3. **拟合**：log10→物理空间 → 按 Di 计算组合变量 C → 初步 `curve_fit` →
   剔除残差最大的 `max_clean_ratio` 比例点 → 二次拟合 → 计算 R²。
