import matplotlib.pyplot as plt
import seaborn as sns

# 数据
models = ['Improved CycleGAN', 'Original CycleGAN', 'pix2pix']
FID = [1390.01, 1247.35, 1384.27]
SSIM = [0.4401, 0.4199, 0.3437]
LPIPS = [0.4803, 0.4836, 0.5309]

sns.set(style="whitegrid", font_scale=1.2)

# 1. FID
plt.figure(figsize=(5,4))
bars = plt.bar(models, FID, color="#4C72B0", width=0.6)
plt.title('FID Comparison (Lower is Better)')
plt.ylabel('FID')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('fid_comparison.png', dpi=200)
plt.show()

# 2. SSIM
plt.figure(figsize=(5,4))
bars = plt.bar(models, SSIM, color="#55A868", width=0.6)
plt.title('SSIM Comparison (Higher is Better)')
plt.ylabel('SSIM')
plt.ylim(0, 0.6)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.3f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('ssim_comparison.png', dpi=200)
plt.show()

# 3. LPIPS
plt.figure(figsize=(5,4))
bars = plt.bar(models, LPIPS, color="#C44E52", width=0.6)
plt.title('LPIPS Comparison (Lower is Better)')
plt.ylabel('LPIPS')
plt.ylim(0, 0.6)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.3f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('lpips_comparison.png', dpi=200)
plt.show()
