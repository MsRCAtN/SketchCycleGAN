import os
import shutil
import random
from glob import glob

# 
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Dataset'))
MODES = ['sketch', 'photo']
SPLITS = ['train', 'test']
SPLIT_RATIO = 0.8  # 8:2
SEED = 99
random.seed(SEED)

for mode in MODES:
    # （train/test）
    mode_root = os.path.join(DATA_ROOT, mode)
    all_sets = [d for d in os.listdir(mode_root) if os.path.isdir(os.path.join(mode_root, d)) and d not in ['train', 'test']]
    for subset in all_sets:
        src_root = os.path.join(mode_root, subset)
        class_names = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
        for cls in class_names:
            src_cls_dir = os.path.join(src_root, cls)
            imgs = [f for f in os.listdir(src_cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            random.shuffle(imgs)
            n_train = int(len(imgs) * SPLIT_RATIO)
            split_imgs = {'train': imgs[:n_train], 'test': imgs[n_train:]}
            for split in SPLITS:
                tgt_cls_dir = os.path.join(DATA_ROOT, mode, split, cls)
                os.makedirs(tgt_cls_dir, exist_ok=True)
                for img in split_imgs[split]:
                    src_path = os.path.join(src_cls_dir, img)
                    tgt_path = os.path.join(tgt_cls_dir, f'{subset}_{img}')
                    if not os.path.exists(tgt_path):
                        shutil.copy2(src_path, tgt_path)
print('Dataset splitting complete!')
