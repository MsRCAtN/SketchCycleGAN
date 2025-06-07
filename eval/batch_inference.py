import os
import argparse
import subprocess
import torch

# 
SKETCH_ROOT = '../sampled_pairs/sketch'
FAKEB_ROOT_TEMPLATE = '../sampled_pairs/{model_name}/fakeB'

parser = argparse.ArgumentParser(description='Batch inference to generate fake B images using a trained model.')
parser.add_argument('--sketch_root', type=str, default=SKETCH_ROOT, help='Root directory of input sketches.')
parser.add_argument('--out_root', type=str, default=None, help='Root directory for saving generated fake B images. Defaults to ../sampled_pairs/{model_name}/fakeB')
parser.add_argument('--model_name', type=str, required=True, help='Name of the model, used for naming the output directory.')
parser.add_argument('--weights', type=str, required=True, help='Path to the trained model weights (.pth file).')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing fake B images if they exist.')
args = parser.parse_args()

sketch_root = args.sketch_root
model_name = args.model_name
weights = args.weights
out_root = args.out_root or FAKEB_ROOT_TEMPLATE.format(model_name=model_name)
overwrite = args.overwrite

# 
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch, 'has_mps') and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

os.makedirs(out_root, exist_ok=True)

total, done = 0, 0
for category in os.listdir(sketch_root):
    sketch_dir = os.path.join(sketch_root, category)
    fakeB_dir = os.path.join(out_root, category)
    os.makedirs(fakeB_dir, exist_ok=True)
    for fname in os.listdir(sketch_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        sketch_path = os.path.join(sketch_dir, fname)
        fakeB_path = os.path.join(fakeB_dir, fname)
        total += 1
        if os.path.exists(fakeB_path) and not overwrite:
            done += 1
            continue
        # 
        result = subprocess.run([
            'python', os.path.join(os.path.dirname(__file__), '../inference_cyclegan.py'),
            '--weights', weights,
            '--input', sketch_path,
            '--output', fakeB_path,
            '--device', device
        ])
        if result.returncode == 0:
            done += 1
        else:
            print(f"Error processing: {sketch_path}")
print(f"Processed: {done}/{total} images. Output saved to: {out_root}")
