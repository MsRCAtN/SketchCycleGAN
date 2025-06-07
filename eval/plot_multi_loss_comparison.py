import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.ndimage import gaussian_filter1d

# Log files and model names
log_files = [
    ("output/cyclegan/2025-04-19_18-54-02_STN+LSGAN/train_loss_log.txt", "cyclegan_mod_STN"),
    ("output/cyclegan/2025-04-26_15-49-40_TPS+WGAN-GP/train_loss_log.txt", "cyclegan_mod_TPS"),
    ("output/cyclegan_orig/2025-04-27_13-31-23_LSGAN/train_loss_log.txt", "cyclegan_orig"),
    ("output/pix2pix/2025-04-27_13-29-01/train_log.txt", "pix2pix"),
]

regex = re.compile(r"Epoch (\d+).*?Batch (\d+)/(\d+).*?Loss_G: ([\d\.eE+-]+)\s*Loss_D: ([\d\.eE+-]+)")

losses = {}
for log_path, model_name in log_files:
    if not os.path.exists(log_path):
        print(f"[Warning] File not found: {log_path}")
        continue
    losses[model_name] = {'epoch': [], 'batch': [], 'loss_g': [], 'loss_d': []}
    with open(log_path, 'r') as f:
        for line in f:
            m = regex.search(line)
            if m:
                epoch = int(m.group(1))
                batch = int(m.group(2))
                total_batch = int(m.group(3))
                loss_g = float(m.group(4))
                loss_d = float(m.group(5))
                # If batch==1, use frac_epoch=epoch (not epoch+batch/total_batch)
                if batch == 1:
                    frac_epoch = epoch
                else:
                    frac_epoch = epoch + (batch - 1) / total_batch
                losses[model_name]['epoch'].append(frac_epoch)
                losses[model_name]['batch'].append(batch)
                losses[model_name]['loss_g'].append(loss_g)
                losses[model_name]['loss_d'].append(loss_d)
    if len(losses[model_name]['loss_g']) == 0:
        print(f"[Warning] No loss parsed: {log_path}")

# Plot settings
plt.figure(figsize=(10, 5))
for model_name, data in losses.items():
    if len(data['epoch']) == 0:
        continue
    # Remove outliers (clip to 99th percentile for each loss)
    g = np.array(data['loss_g'])
    d = np.array(data['loss_d'])
    g_clip = np.clip(g, None, np.percentile(g, 99))
    d_clip = np.clip(d, None, np.percentile(d, 99))
    # Smooth
    g_smooth = gaussian_filter1d(g_clip, sigma=5)
    d_smooth = gaussian_filter1d(d_clip, sigma=5)
    plt.plot(data['epoch'], g_smooth, label=f"{model_name}")
plt.xlabel("Epoch")
plt.ylabel("Generator Loss (G)")
plt.title("Generator Loss Curve Comparison")
plt.legend()
plt.tight_layout()
plt.xlim(left=1)
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
os.makedirs("output/plots", exist_ok=True)
plt.savefig("output/plots/loss_curve_G.png", dpi=200)
plt.show()

plt.figure(figsize=(10, 5))
for model_name, data in losses.items():
    if len(data['epoch']) == 0:
        continue
    d = np.array(data['loss_d'])
    d_clip = np.clip(d, None, np.percentile(d, 99))
    d_smooth = gaussian_filter1d(d_clip, sigma=5)
    plt.plot(data['epoch'], d_smooth, label=f"{model_name}")
plt.xlabel("Epoch")
plt.ylabel("Discriminator Loss (D)")
plt.title("Discriminator Loss Curve Comparison")
plt.legend()
plt.tight_layout()
plt.xlim(left=1)
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
os.makedirs("output/plots", exist_ok=True)
plt.savefig("output/plots/loss_curve_D.png", dpi=200)
plt.show()
print("Saved: output/plots/loss_curve_G.png, output/plots/loss_curve_D.png")
