import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms, models
import argparse
import csv

# Example paths: ../sampled_pairs/{model_name}/fakeB/{category}/*.png, ../sampled_pairs/photo/{category}/*.jpg
PHOTO_ROOT = '../sampled_pairs/photo'

def extract_vgg16_fc2(model, img_tensor, device):
    x = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(x)
    return features.cpu().numpy().squeeze()

def preprocess(img, size=224):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return tf(img)

def eval_dir(fakeB_dir, photo_dir, category, device, size=224, model_name='model'):
    fakeB_imgs = sorted([f for f in os.listdir(fakeB_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    results = []
    vgg16 = models.vgg16(pretrained=True).to(device)
    vgg16.eval()
    vgg16_fc2 = torch.nn.Sequential(*list(vgg16.children())[:-1], torch.nn.Flatten(), *list(vgg16.classifier.children())[:5])
    for fakeB_name in tqdm(fakeB_imgs, desc=f"{model_name}/{category}"):
        fakeB_img_path = os.path.join(fakeB_dir, fakeB_name)
        base = os.path.splitext(fakeB_name)[0]
        real_img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            candidate = os.path.join(photo_dir, base + ext)
            if os.path.exists(candidate):
                real_img_path = candidate
                break
        if not real_img_path:
            print(f"Warning: real photo not found: {photo_dir}/{base}.*")
            continue
        fakeB_img = Image.open(fakeB_img_path).convert('RGB')
        real_img = Image.open(real_img_path).convert('RGB')
        fakeB_tensor = preprocess(fakeB_img, size)
        real_tensor = preprocess(real_img, size)
        vec1 = extract_vgg16_fc2(vgg16_fc2, fakeB_tensor, device)
        vec2 = extract_vgg16_fc2(vgg16_fc2, real_tensor, device)
        sim = 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        results.append({'model': model_name, 'category': category, 'fakeB': fakeB_name, 'real': os.path.basename(real_img_path), 'similarity': sim})
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fakeB_root', type=str, required=True, help='Root directory of generated fakeB images (e.g., ../sampled_pairs/model_name/fakeB).')
    parser.add_argument('--photo_root', type=str, default=PHOTO_ROOT, help='Root directory of real photo images (e.g., ../sampled_pairs/photo).')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model being evaluated.')
    parser.add_argument('--csv', type=str, default='similarity_results.csv', help='Output CSV file path for detailed similarity scores.')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--device', type=str, default=None, help='Device to use: cpu, cuda, mps. Auto-detects if None.')
    args = parser.parse_args()
    fakeB_root = args.fakeB_root
    photo_root = args.photo_root
    model_name = args.model_name
    csv_path = args.csv 
    size = args.size
    # 
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = 'cuda'
        elif getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    # 
    all_categories = [d for d in os.listdir(fakeB_root) if os.path.isdir(os.path.join(fakeB_root, d))]
    all_stats = []
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'category', 'fakeB', 'real', 'similarity'])
        for category in sorted(all_categories):
            fakeB_dir = os.path.join(fakeB_root, category)
            photo_dir = os.path.join(photo_root, category)
            if not os.path.isdir(fakeB_dir) or not os.path.isdir(photo_dir):
                continue
            results = eval_dir(fakeB_dir, photo_dir, category, device, size, model_name)
            for r in results:
                writer.writerow([r['model'], r['category'], r['fakeB'], r['real'], r['similarity']])
            # 
            if results:
                sims = [r['similarity'] for r in results]
                mean = np.mean(sims)
                std = np.std(sims)
                all_stats.append({'model': model_name, 'category': category, 'mean': mean, 'std': std, 'n': len(sims)})
    # 
    print("\n=== Similarity Summary ===")
    for stat in all_stats:
        print(f"{stat['model']}/{stat['category']}: mean={stat['mean']:.4f}, std={stat['std']:.4f}, n={stat['n']}")
    # Save markdown summary relative to the CSV output path
    summary_md_path = os.path.join(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', 'similarity_summary.md')
    with open(summary_md_path, 'w') as f:
        f.write('| Model | Category | Mean Similarity | Std | N |\n')
        f.write('|-------|----------|-----------------|-----|---|\n')
        for stat in all_stats:
            f.write(f"| {stat['model']} | {stat['category']} | {stat['mean']:.4f} | {stat['std']:.4f} | {stat['n']} |\n")
    print(f"Results saved to: {csv_path} and {summary_md_path}")

if __name__ == '__main__':
    main()
