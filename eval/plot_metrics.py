import matplotlib.pyplot as plt
import numpy as np

# 手动填入各模型的评测结果
models = ['Improved CycleGAN', 'Original CycleGAN', 'pix2pix']
FID = [1390.01, 1247.35, 1384.27]
SSIM = [0.4401, 0.4199, 0.3437]
LPIPS = [0.4803, 0.4836, 0.5309]

# FID/LPIPS 越低越好，SSIM 越高越好
x = np.arange(len(models))
width = 0.25

fig, ax1 = plt.subplots(figsize=(8, 5))

# FID
rects1 = ax1.bar(x - width, FID, width, label='FID', color='#4C72B0')
# SSIM
rects2 = ax1.bar(x, SSIM, width, label='SSIM', color='#55A868')
# LPIPS
rects3 = ax1.bar(x + width, LPIPS, width, label='LPIPS', color='#C44E52')

ax1.set_ylabel('Metric Value')
ax1.set_title('Quantitative Comparison of Models')
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
plt.savefig('metrics_bar.png', dpi=200)
plt.show()
