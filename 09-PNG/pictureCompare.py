import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


def load_dataframe(input_path: Path) -> pd.DataFrame:
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)
    return pd.read_csv(input_path)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    default="植株数据.xlsx - Sheet1.csv",
    help="Input CSV/XLSX file path",
)
parser.add_argument(
    "--output_dir",
    default=".",
    help="Directory to write generated figures",
)
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# 1. 加载数据
df = load_dataframe(Path(args.input))

# 2. 提取并映射宏观与微观表型数据
# 株高
height_gt = df['株高真值'].values
height_pred = df['株高虚拟植'].values

# 冠幅
width_gt = df['冠幅真值'].values
width_pred = df['冠幅虚拟植'].values

# 叶长 - 矩阵展平
leaf_len_gt = np.concatenate([df['叶长真值1'].values, df['叶长真值2'].values, df['叶长真值3'].values])
leaf_len_pred = np.concatenate([df['叶长虚拟植1'].values, df['叶长虚拟植2'].values, df['叶长虚拟植3'].values])

# 叶宽 - 矩阵展平
leaf_wid_gt = np.concatenate([df['叶宽真值1'].values, df['叶宽真值2'].values, df['叶宽真值3'].values])
leaf_wid_pred = np.concatenate([df['叶宽虚拟植1'].values, df['叶宽虚拟植2'].values, df['叶宽虚拟植3'].values])

# 3. 全局学术排版与字体设定
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['figure.dpi'] = 300

# 4. 独立制图与自动图例导出函数
def plot_and_save_regression(x_manual, y_virtual, title, unit, filename):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 强制绝对等距坐标系
    ax.set_aspect('equal', adjustable='box')
    
    # 绘制带有 95% 置信区间的回归分布图
    sns.regplot(x=x_manual, y=y_virtual, ax=ax, 
                scatter_kws={'alpha': 0.6, 'color': '#1f77b4', 'edgecolor': 'w'},
                line_kws={'color': '#08306b', 'linewidth': 2})
    
    # 建立底层对角理想基准线 (y=x)
    min_val = min(np.min(x_manual), np.min(y_virtual))
    max_val = max(np.max(x_manual), np.max(y_virtual))
    buffer = (max_val - min_val) * 0.05
    ax.plot([min_val - buffer, max_val + buffer], 
            [min_val - buffer, max_val + buffer], 
            color='black', linestyle='--', linewidth=1.5, zorder=0)
    
    # 计算统计学量化指标
    slope, intercept = np.polyfit(x_manual, y_virtual, 1)
    r_squared = np.corrcoef(x_manual, y_virtual)[0, 1]**2
    rmse = np.sqrt(mean_squared_error(x_manual, y_virtual))
    n = len(x_manual)
    
    # 注入数据铭牌至左上角空白区
    stats_text = (f"$R^2 = {r_squared:.3f}$\n"
                  f"RMSE = {rmse:.2f} {unit}\n"
                  f"$y = {slope:.2f}x {intercept:+.2f}$\n"
                  f"$n = {n}$")
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # 构建显式代理图例 (Proxy Artists) 并添加 95% CI
    scatter_proxy = mlines.Line2D([], [], color='none', marker='o', markerfacecolor='#1f77b4', markeredgecolor='w', alpha=0.6, markersize=8, label='Data Points')
    ideal_line_proxy = mlines.Line2D([], [], color='black', linestyle='--', linewidth=1.5, label='Ideal Fit ($y=x$)')
    reg_line_proxy = mlines.Line2D([], [], color='#08306b', linewidth=2, label='Linear Regression')
    ci_proxy = mpatches.Patch(color='#1f77b4', alpha=0.2, label='95% Confidence Interval')
    
    # 部署图例至右下角并强制边缘渲染
    ax.legend(handles=[scatter_proxy, ideal_line_proxy, reg_line_proxy, ci_proxy], 
              loc='lower right', fontsize=10, framealpha=0.9, edgecolor='gray')
    
    # 轴向物理绑定
    ax.set_xlabel(f'Manual Measurement ({unit})', fontweight='bold', fontsize=12)
    ax.set_ylabel(f'Virtual Extraction ({unit})', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14, pad=15)
    
    # 独立渲染与物理文件导出
    plt.tight_layout()
    plt.savefig(output_dir / f'{filename}.png', format='png', bbox_inches='tight')
    plt.savefig(output_dir / f'{filename}.pdf', format='pdf', bbox_inches='tight')
    plt.close()

# 5. 执行串行批处理导出（解耦子图架构）
plot_and_save_regression(height_gt, height_pred, "Plant Height Regression", "cm", "Plant_Height")
plot_and_save_regression(width_gt, width_pred, "Canopy Width Regression", "cm", "Canopy_Width")
plot_and_save_regression(leaf_len_gt, leaf_len_pred, "Leaf Length Regression", "cm", "Leaf_Length")
plot_and_save_regression(leaf_wid_gt, leaf_wid_pred, "Leaf Width Regression", "cm", "Leaf_Width")
