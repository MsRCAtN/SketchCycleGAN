import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--metrics_file', type=str, required=True, help='Path to csv with metrics')
parser.add_argument('--output_prefix', type=str, default='', help='Prefix for output images')
args = parser.parse_args()

if args.metrics_file.endswith('.csv'):
    df = pd.read_csv(args.metrics_file)
    # 
    model_map = {
        'cyclegan_mod_STN': 'CycleGAN (STN+LSGAN)',
        'cyclegan_mod_TPS': 'CycleGAN (TPS+WGAN-GP)',
        'cyclegan_orig': 'CycleGAN (Original)',
        'pix2pix': 'pix2pix'
    }
    models = [model_map.get(m, m) for m in df['model'].tolist()]
    metrics = [col for col in ['FID', 'SSIM', 'LPIPS'] if col in df.columns]
else:
    raise ValueError('Only csv is supported for multi-model comparison!')

sns.set(style="whitegrid", font_scale=1.3)

palette = sns.color_palette("Set2", n_colors=len(models))

for metric in metrics:
    plt.figure(figsize=(6,5))
    values = df[metric].tolist()
    bars = plt.bar(models, values, color=palette)
    # Adjust y-axis limits and add descriptive text based on the metric
    if metric == 'LPIPS':
        plt.ylim(0, max(values)*1.15)
        extra_text = 'Lower is better'
        title_str = f'{metric} Comparison'
    elif metric == 'FID':
        plt.ylim(0, max(values)*1.15)
        extra_text = 'Lower is better'
        title_str = f'{metric} Comparison'
    elif metric == 'SSIM':
        plt.ylim(0, max(0.3, max(values)*1.2))
        extra_text = 'Higher is better'
        title_str = f'{metric} Comparison'
    # Remove grid for a cleaner look
    plt.grid(False)
    # Add text labels on top of each bar
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.3f}' if metric != 'FID' else f'{bar.get_height():.2f}',
                 ha='center', va='bottom', fontsize=13, fontweight='bold')
    # Set plot title and labels
    plt.title(title_str, fontsize=17, fontweight='bold', pad=14, loc='center')
    plt.ylabel(metric, fontsize=15)
    plt.xticks(fontsize=10, rotation=15)
    plt.yticks(fontsize=13)
    # Add 'Higher/Lower is better' text annotation
    plt.text(0.98, 0.98, extra_text, fontsize=13, color='#555', ha='right', va='top', 
             transform=plt.gca().transAxes, fontweight='bold', 
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    plt.tight_layout()
    out_path = f'{args.output_prefix}{metric.lower()}_comparison.png'
    plt.savefig(out_path, dpi=220)
    print(f'Saved: {out_path}')
    plt.close()
