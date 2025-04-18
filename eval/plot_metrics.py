import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--metrics_file', type=str, required=True, help='Path to txt or csv with metrics')
parser.add_argument('--output', type=str, default='metrics_bar.png', help='Output image file')
args = parser.parse_args()

# 读取指标
with open(args.metrics_file) as f:
    lines = [l.strip() for l in f if l.strip()]
metric_dict = {}
for l in lines:
    if ':' in l:
        k, v = l.split(':')
        metric_dict[k.strip().upper()] = float(v.strip())

models = ['Improved CycleGAN']
FID = [metric_dict.get('FID', 0)]
SSIM = [metric_dict.get('SSIM', 0)]
LPIPS = [metric_dict.get('LPIPS', 0)]

x = np.arange(len(models))
width = 0.25

fig, ax1 = plt.subplots(figsize=(6, 4))

# FID
rects1 = ax1.bar(x - width, FID, width, label='FID', color='#4C72B0')
# SSIM
rects2 = ax1.bar(x, SSIM, width, label='SSIM', color='#55A868')
# LPIPS
rects3 = ax1.bar(x + width, LPIPS, width, label='LPIPS', color='#C44E52')

ax1.set_ylabel('Metric Value')
ax1.set_title('Quantitative Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=15)
ax1.legend()

# 数值标注
for rect in rects1 + rects2 + rects3:
    height = rect.get_height()
    ax1.annotate(f'{height:.3f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(args.output, dpi=200)
plt.show()
