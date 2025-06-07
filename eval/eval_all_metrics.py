import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import torch
import lpips
from pytorch_fid import fid_score
import re

# ========== CONFIG ==========
MODELS = {
    'cyclegan_mod_STN': 'cyclegan_mod_STN/fakeB',
    'cyclegan_mod_TPS': 'cyclegan_mod_TPS/fakeB',
    'cyclegan_orig': 'cyclegan_orig/fakeB',
    'pix2pix': 'pix2pix/fakeB',
}
REAL_DIR = 'photo'  # realB
ROOT = os.path.dirname(os.path.dirname(__file__)) + '/sampled_pairs'

# ========== HELPERS ==========
def get_base_stem(filename):
    stem = os.path.splitext(filename)[0]
    base = re.sub(r'-\d+$', '', stem)
    return base

def get_all_pairs(fake_dir, real_dir):
    # collect all real images (base_stem -> fullpath)
    real_map = {}
    for cat in os.listdir(real_dir):
        real_cat_dir = os.path.join(real_dir, cat)
        if not os.path.isdir(real_cat_dir):
            continue
        for fname in os.listdir(real_cat_dir):
            base = get_base_stem(fname)
            real_map[(cat, base)] = os.path.join(real_cat_dir, fname)
    pairs = []
    for cat in os.listdir(fake_dir):
        fake_cat_dir = os.path.join(fake_dir, cat)
        if not os.path.isdir(fake_cat_dir):
            continue
        for fname in os.listdir(fake_cat_dir):
            base = get_base_stem(fname)
            key = (cat, base)
            if key in real_map:
                fake_path = os.path.join(fake_cat_dir, fname)
                real_path = real_map[key]
                pairs.append((fake_path, real_path))
    return pairs

def compute_ssim(fake_paths, real_paths):
    scores = []
    for f, r in zip(fake_paths, real_paths):
        img1 = np.array(Image.open(f).convert('RGB'))
        img2 = np.array(Image.open(r).convert('RGB'))
        score = ssim(img1, img2, channel_axis=-1, data_range=255)
        scores.append(score)
    return float(np.mean(scores))

def compute_lpips(fake_paths, real_paths, net='alex', device='cuda' if torch.cuda.is_available() else 'cpu'):
    loss_fn = lpips.LPIPS(net=net).to(device)
    scores = []
    for f, r in zip(fake_paths, real_paths):
        img1 = Image.open(f).convert('RGB').resize((256,256))
        img2 = Image.open(r).convert('RGB').resize((256,256))
        t1 = torch.from_numpy(np.array(img1)).permute(2,0,1).unsqueeze(0).float()/127.5 - 1
        t2 = torch.from_numpy(np.array(img2)).permute(2,0,1).unsqueeze(0).float()/127.5 - 1
        t1 = t1.to(device)
        t2 = t2.to(device)
        with torch.no_grad():
            d = loss_fn(t1, t2).item()
        scores.append(d)
    return float(np.mean(scores))

def compute_fid_from_pairs(fake_paths, real_paths, device='cuda' if torch.cuda.is_available() else 'cpu'):
    import shutil, tempfile
    tmp_root = tempfile.mkdtemp(prefix='eval_metrics_tmp_')
    fake_dir = os.path.join(tmp_root, 'fake')
    real_dir = os.path.join(tmp_root, 'real')
    os.makedirs(fake_dir)
    os.makedirs(real_dir)
    # tmp
    for i, (f, r) in enumerate(zip(fake_paths, real_paths)):
        fake_ext = os.path.splitext(f)[1]
        real_ext = os.path.splitext(r)[1]
        fake_target = os.path.join(fake_dir, f'{i:04d}{fake_ext}')
        real_target = os.path.join(real_dir, f'{i:04d}{real_ext}')
        shutil.copy2(f, fake_target)
        shutil.copy2(r, real_target)
    # pytorch-fid
    fid = fid_score.calculate_fid_given_paths([fake_dir, real_dir], batch_size=32, device=device, dims=2048)
    # 
    shutil.rmtree(tmp_root)
    return float(fid)

# ========== MAIN ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='metrics.csv', help='Output CSV file')
    args = parser.parse_args()

    results = []
    for model, rel_fake_dir in MODELS.items():
        fake_dir = os.path.join(ROOT, rel_fake_dir)
        real_dir = os.path.join(ROOT, REAL_DIR)
        pairs = get_all_pairs(fake_dir, real_dir)
        print(f'[{model}] Found {len(pairs)} pairs (fake_dir={fake_dir}, real_dir={real_dir})')
        # DEBUG: 10
        for i, (f, r) in enumerate(pairs[:10]):
            print(f'  Pair {i+1}: fake={f} | real={r}')
        if len(pairs) == 0:
            print(f'[{model}] WARNING: No valid pairs found! Skipping this model.')
            continue
        fake_paths, real_paths = zip(*pairs)
        # DEBUG: 
        from collections import Counter
        cat_counter = Counter([os.path.basename(os.path.dirname(fp)) for fp in fake_paths])
        print(f'[{model}] Category image counts: {dict(cat_counter)}')
        # FID
        if len(fake_paths) == 0 or len(real_paths) == 0:
            print(f'[{model}] WARNING: No images for FID! Skipping FID.')
            fid = float('nan')
        else:
            fid = compute_fid_from_pairs(fake_paths, real_paths)
        ssim_score = compute_ssim(fake_paths, real_paths)
        lpips_score = compute_lpips(fake_paths, real_paths)
        results.append({'model': model, 'FID': fid, 'SSIM': ssim_score, 'LPIPS': lpips_score})
        print(f'[{model}] FID={fid:.2f}, SSIM={ssim_score:.4f}, LPIPS={lpips_score:.4f}')
    if results:
        pd.DataFrame(results).to_csv(args.output, index=False)
        print(f'Saved: {args.output}')
    else:
        print('No valid results, nothing saved.')

if __name__ == '__main__':
    main()
