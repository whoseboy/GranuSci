---
name: plot-script
description: plot_script 自由绘图的代码手册——环境约定、常用模板、沙箱纪律、常见坑
when_to_use: 用 plot_script 写定制绘图代码时载入
---

# plot_script 代码手册

## 环境约定（已配置，直接用）
- `plt`（rcParams 已按 APS：8.5pt 字号/单栏 3.4in/双栏 7.0in/600dpi/Viridis 默认色序+线型冗余循环/mathtext dejavusans/开放轴/中文 CJK 字体链）
- `np`（numpy）
- `data(N)`：取第 N 次工具调用的完整返回 dict（大数据旁路，不占上下文）——快照/剖面/时间序列从这拿；
- `SINGLE_COL`=3.4、`DOUBLE_COL`=7.0（英寸宽度）
- 图保存自动（PDF+PNG），不要自己 savefig
- 标签文本裸写记号会自动数学化（r_s→$r_s$ 下标排版），也可手写 LaTeX

## 常用模板
```python
# 双栏多面板组图
fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 5.0))
for ax, (grp, d) in zip(axes.flat, groups.items()):
    ax.plot(d["gamma"], d["dh"], "o-")
    ax.set_xlabel(r"$\Gamma$"); ax.set_ylabel(r"$\Delta h^*$")
fig.tight_layout(pad=0.5)

# 面板标号（带白色背景框，防压数据线）
ax.text(0.03, 0.95, "(a)", transform=ax.transAxes, fontsize=10,
        fontweight="bold", va="top",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

# 双 y 轴（轴刻度与曲线同色联动，读者不用回查图例）
ax2 = ax.twinx()
ax2.spines["right"].set_color("#5EC962")
ax2.tick_params(axis="y", colors="#5EC962")
ax2.set_ylabel("M", color="#5EC962")

# inset（局部放大）——规则：只放真正需要放大的区段（如低Γ区），置于数据空白角，
# 用 ax.indicate_inset_zoom 标注框线，x 范围收窄到放大段
axins = ax.inset_axes([0.60, 0.12, 0.35, 0.32])
axins.plot(gam_low, dh_low, "o-", color="#3B528B")
axins.set_xlim(3, 10)                     # 真放大，不是重复全域
ax.indicate_inset_zoom(axins, edgecolor="0.6")

# 大数据旁路
snap = data(171)          # 第 171 次调用（如 get_snapshot）的完整返回
ax.scatter(snap["x"], snap["y"], c=snap["type"], s=2)

# 多系列手动分配线型冗余（make_figure 自动，手写时自己保证）
MARKERS = ["o", "s", "^", "D"]; LINESTYLES = ["-", "--", "-.", ":"]
for i, (name, d) in enumerate(series.items()):
    ax.plot(d["x"], d["y"], MARKERS[i % 4] + LINESTYLES[i % 4], label=name)
```

## 沙箱纪律
- import 白名单：matplotlib / numpy / math / mpl_toolkits——其他被拒；
- 禁 os / subprocess / open( / exec( / eval( / shutil / pathlib——文件与系统调用被拦；
- 数据只有两条路：`data(N)` 引用工具记录（可审计）或照抄工具返回内联——**禁止凭记忆补数据点**。

## 常见坑
- matplotlib 3.11 删了 `basex/basey`——log 轴用 `ax.set_xscale("log", base=10)`；
- **log 轴跨 ≤2 个数量级时刻度标签叠印**（"3×10⁰"撞"10⁰"）——稀疏数据点少时干脆用
  `ax.set_xscale("linear")` + 手动 `ax.set_xticks([3,7,15,30])`（参数值就是刻度，物理量离散扫描本就不该有插值刻度）；
- mathtext 不支持部分 LaTeX 命令——失败时剥 `$` 用纯文本（平台自动降级）。
