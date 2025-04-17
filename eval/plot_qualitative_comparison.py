import matplotlib.pyplot as plt
from PIL import Image
import os

# 随机挑选几组样例（这里选 batch1, batch1001, batch2001）
samples = ['epoch1_batch1', 'epoch1_batch1001', 'epoch1_batch2001']
columns = ['realA', 'fakeB_improved', 'fakeB_orig', 'fakeB_pix2pix', 'realB']
titles = ['Input (A)', 'Improved CycleGAN', 'Original CycleGAN', 'pix2pix', 'Target (B)']

# 路径模板
def get_path(sample, col):
    if col == 'realA':
        return f'output/2025-04-17_10-45-44/{sample}_realA.png'
    elif col == 'fakeB_improved':
        return f'output/2025-04-17_10-45-44/{sample}_fakeB.png'
    elif col == 'fakeB_orig':
        return f'output/cyclegan_orig/2025-04-17_11-45-35/{sample}_fakeB.png'
    elif col == 'fakeB_pix2pix':
        return f'output/pix2pix/2025-04-17_12-26-34/{sample}_fakeB.png'
    elif col == 'realB':
        return f'output/2025-04-17_10-45-44/{sample}_realB.png'
    else:
        return None

plt.figure(figsize=(15, 9))
for i, sample in enumerate(samples):
    for j, col in enumerate(columns):
        img_path = get_path(sample, col)
        plt.subplot(len(samples), len(columns), i*len(columns)+j+1)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            plt.imshow(img)
        else:
            plt.text(0.5, 0.5, 'Missing', fontsize=12, ha='center', va='center')
        plt.axis('off')
        if i == 0:
            plt.title(titles[j], fontsize=13)
plt.tight_layout()
plt.savefig('qualitative_comparison.png', dpi=200)
plt.show()
