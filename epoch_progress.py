"""
Line-plot of Florence-2 fine-tuning progress (Epoch 0–8)
========================================================
• 数据来源：表 1  
• 任务     ：DocVQA，500 × validation  
• 绘图规范：科研投稿 (Times New Roman, 300 dpi, PDF & PNG)

必要依赖：
pip install matplotlib seaborn pandas
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------------------------------------------
# 1) 数据录入
# ------------------------------------------------------------------
df = pd.DataFrame(
    {
        "Epoch":          list(range(9)),
        "Accuracy (%)":   [15.20, 36.20, 38.00, 40.20, 41.80, 43.20, 44.40, 45.80, 45.60],
        "Similarity":     [0.2437, 0.5014, 0.5164, 0.5264, 0.5449, 0.5632, 0.5670, 0.5791, 0.5816],
        "EM (%)":         [None, 36.32, 38.83, 40.12, 40.74, 41.50, 42.14, 42.29, 42.40],
        "F1 (%)":         [None, 44.60, 47.37, 48.87, 49.50, 50.32, 50.99, 51.06, 51.13],
        "Val Loss":      [None, 0.6237, 0.5966, 0.5811, 0.5791, 0.5736, 0.5711, 0.5725, 0.5725],
    }
)

# 将 None 替换为 NaN，方便绘制连续折线
df = df.astype("float64")

# ------------------------------------------------------------------
# 2) 全局风格
# ------------------------------------------------------------------
sns.set_theme(
    style="whitegrid",
    font="Times New Roman",
    rc={"axes.titlesize": 13, "axes.labelsize": 12, "figure.dpi": 120, "savefig.dpi": 300},
)

# ------------------------------------------------------------------
# 3) 绘图
# ------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(7.5, 5.0))

# ——— 主 Y 轴：四个评价指标 ———
metrics = {
    "Accuracy (%)":  ("o", "#1f77b4"),
    "EM (%)":        ("s", "#ff7f0e"),
    "F1 (%)":        ("^", "#2ca02c"),
    "Similarity":    ("d", "#9467bd"),
}

for m, (marker, color) in metrics.items():
    ax1.plot(
        df["Epoch"],
        df[m] if m != "Similarity" else df[m] * 100,  # 将 Similarity × 100 转为 %
        marker=marker,
        linewidth=2,
        markersize=5,
        label=m,
        color=color,
    )

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Score (%)")
ax1.set_ylim(0, 105)
ax1.set_xticks(df["Epoch"])
ax1.legend(loc="upper left", frameon=False)
ax1.grid(alpha=0.3)

# ——— 副 Y 轴：验证损失 ———
ax2 = ax1.twinx()
ax2.plot(
    df["Epoch"],
    df["Val Loss"],
    marker="x",
    linewidth=2,
    markersize=6,
    color="#d62728",
    label="Val Loss",
)
ax2.set_ylabel("Val Loss")
ax2.set_ylim(min(df["Val Loss"].dropna()) - 0.02, max(df["Val Loss"].dropna()) + 0.02)

# 合并图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper center", ncol=3, frameon=False)

# ——— 标题 & 布局 ———
fig.suptitle("Florence-2 Fine-tuning on DocVQA (500 Val Samples)")
fig.tight_layout(rect=[0, 0, 1, 0.96])

# ------------------------------------------------------------------
# 4) 导出
# ------------------------------------------------------------------
out = Path("figures"); out.mkdir(exist_ok=True)
fig.savefig(out / "epoch_progress.pdf",  bbox_inches="tight")
fig.savefig(out / "epoch_progress.png",  bbox_inches="tight", dpi=300)

plt.show()
