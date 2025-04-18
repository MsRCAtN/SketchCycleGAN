import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--metrics_file', type=str, required=True, help='Path to txt or csv with metrics')
parser.add_argument('--output_prefix', type=str, default='', help='Prefix for output images')
args = parser.parse_args()

with open(args.metrics_file) as f:
    lines = [l.strip() for l in f if l.strip()]
metric_dict = {}
for l in lines:
    if ':' in l:
        k, v = l.split(':')
        metric_dict[k.strip().upper()] = float(v.strip())

models = ['Improved CycleGAN', 'Original CycleGAN', 'pix2pix']
FID = [metric_dict.get('FID', 0)]
SSIM = [metric_dict.get('SSIM', 0)]
LPIPS = [metric_dict.get('LPIPS', 0)]

sns.set(style="whitegrid", font_scale=1.2)

# 1. FID
plt.figure(figsize=(5,4))
bars = plt.bar(models[:1], FID, color="#4C72B0", width=0.6)
plt.title('FID Comparison (Lower is Better)')
plt.ylabel('FID')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{args.output_prefix}fid_comparison.png', dpi=200)
plt.close()

# 2. SSIM
plt.figure(figsize=(5,4))
bars = plt.bar(models[:1], SSIM, color="#55A868", width=0.6)
plt.title('SSIM Comparison (Higher is Better)')
plt.ylabel('SSIM')
plt.ylim(0, 0.6)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.3f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{args.output_prefix}ssim_comparison.png', dpi=200)
plt.close()

# 3. LPIPS
plt.figure(figsize=(5,4))
bars = plt.bar(models[:1], LPIPS, color="#C44E52", width=0.6)
plt.title('LPIPS Comparison (Lower is Better)')
plt.ylabel('LPIPS')
plt.ylim(0, 0.6)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.3f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{args.output_prefix}lpips_comparison.png', dpi=200)
plt.close()
