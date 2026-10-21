"""可视化工具（matplotlib Agg 后端）。

默认字体为微软雅黑（Windows 首选），提供 SimHei / DejaVu Sans 回退链。
"""

import matplotlib

matplotlib.use("Agg")  # 无 GUI 后端，服务端必需

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from config import get_config


def setup_plot_style():
    """设置 matplotlib 全局样式。"""
    cfg = get_config()
    plt.rcParams["font.sans-serif"] = cfg.font_fallback
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "stixsans"
    matplotlib.rcParams["font.family"] = cfg.font_family
    matplotlib.rcParams["font.size"] = 12


# 导入时自动设置样式
setup_plot_style()
