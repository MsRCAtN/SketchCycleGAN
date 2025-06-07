import os
import random
import shutil
from collections import defaultdict

# 
sketch_root = os.path.join(os.path.dirname(__file__), '../Dataset/sketch/tx_000000000000')
photo_root = os.path.join(os.path.dirname(__file__), '../Dataset/photo/tx_000000000000')
output_root = os.path.join(os.path.dirname(__file__), '../sampled_pairs')

# 
IMG_EXTS = ['.png', '.jpg', '.jpeg', '.bmp']

# 
paired_by_class = defaultdict(list)

for class_name in os.listdir(sketch_root):
    class_sketch_dir = os.path.join(sketch_root, class_name)
    class_photo_dir = os.path.join(photo_root, class_name)
    if not os.path.isdir(class_sketch_dir) or not os.path.isdir(class_photo_dir):
        continue
    # photosketchphoto
    used_photos = set()
    for fname in os.listdir(class_sketch_dir):
        if not any(fname.lower().endswith(ext) for ext in IMG_EXTS):
            continue
        base = fname.split('-')[0]
        photo_path = None
        for ext in IMG_EXTS:
            candidate = os.path.join(class_photo_dir, base + ext)
            if os.path.exists(candidate):
                photo_path = candidate
                break
        if photo_path and photo_path not in used_photos:
            paired_by_class[class_name].append((os.path.join(class_sketch_dir, fname), photo_path, fname))
            used_photos.add(photo_path)

# 
for sub in ['sketch', 'photo']:
    subdir = os.path.join(output_root, sub)
    os.makedirs(subdir, exist_ok=True)

# ，3
count = 0
for class_name, pairs in paired_by_class.items():
    if len(pairs) < 3:
        continue
    sampled = random.sample(pairs, 3)
    for idx, (sketch_path, photo_path, sketch_fname) in enumerate(sampled):
        # 
        sketch_out_dir = os.path.join(output_root, 'sketch', class_name)
        photo_out_dir = os.path.join(output_root, 'photo', class_name)
        os.makedirs(sketch_out_dir, exist_ok=True)
        os.makedirs(photo_out_dir, exist_ok=True)
        # 
        sketch_out_path = os.path.join(sketch_out_dir, sketch_fname)
        # sketch，.jpg
        photo_fname = os.path.splitext(sketch_fname)[0] + '.jpg'
        photo_out_path = os.path.join(photo_out_dir, photo_fname)
        shutil.copy2(sketch_path, sketch_out_path)
        shutil.copy2(photo_path, photo_out_path)
        count += 1

print(f"， {count} 。：{output_root}")
