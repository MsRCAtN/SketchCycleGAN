import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import torch
import lpips
from torchvision import transforms
from scipy import linalg

def get_image_paths(output_dir, fake_type, real_type):
    # 匹配所有 fake 和 real 图片（按 batch index 配对）
    fake_paths = sorted([f for f in os.listdir(output_dir) if fake_type in f and f.endswith('.png')])
    real_paths = sorted([f for f in os.listdir(output_dir) if real_type in f and f.endswith('.png')])
    # 按 batch index 匹配
    pairs = []
    for f in fake_paths:
        idx = f.split('_batch')[-1].split('_')[0]
        real_match = [r for r in real_paths if f'_batch{idx}_' in r]
        if real_match:
            pairs.append((os.path.join(output_dir, f), os.path.join(output_dir, real_match[0])))
    return pairs

def compute_fid(fake_imgs, real_imgs):
    # 计算 FID，使用 InceptionV3
    from torchvision.models import inception_v3
    import torch.nn.functional as F
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()
    resize = transforms.Resize((299, 299))
    def get_activations(imgs):
        acts = []
        for img in imgs:
            x = resize(img).unsqueeze(0).to(device)
            x = x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x
            with torch.no_grad():
                pred = model(x)
            acts.append(pred.cpu().squeeze(0).numpy())
        return np.array(acts)
    act1 = get_activations(fake_imgs)
    act2 = get_activations(real_imgs)
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def compute_ssim_lpips(pairs, device):
    ssim_scores = []
    lpips_scores = []
    loss_fn = lpips.LPIPS(net='alex').to(device)
    tf = transforms.ToTensor()
    for fake_path, real_path in tqdm(pairs, desc='Evaluating SSIM/LPIPS'):
        fake = Image.open(fake_path).convert('RGB')
        real = Image.open(real_path).convert('RGB')
        fake_np = np.array(fake)
        real_np = np.array(real)
        ssim_score = ssim(fake_np, real_np, channel_axis=2, data_range=255)
        ssim_scores.append(ssim_score)
        fake_t = tf(fake).unsqueeze(0).to(device)
        real_t = tf(real).unsqueeze(0).to(device)
        lpips_score = loss_fn(fake_t, real_t).item()
        lpips_scores.append(lpips_score)
    return np.mean(ssim_scores), np.mean(lpips_scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help='output dir (with generated images)')
    parser.add_argument('--fake_type', type=str, default='fakeB', help='fake image type (fakeB or fakeA)')
    parser.add_argument('--real_type', type=str, default='realB', help='real image type (realB or realA)')
    args = parser.parse_args()
    output_dir = args.output_dir
    pairs = get_image_paths(output_dir, args.fake_type, args.real_type)
    if len(pairs) == 0:
        print('No image pairs found!')
        return
    print(f'Found {len(pairs)} image pairs for evaluation.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载图片到内存
    fake_imgs = [transforms.ToTensor()(Image.open(f).convert('RGB')) for f, _ in pairs]
    real_imgs = [transforms.ToTensor()(Image.open(r).convert('RGB')) for _, r in pairs]
    # FID
    print('Computing FID...')
    fid = compute_fid(fake_imgs, real_imgs)
    # SSIM & LPIPS
    ssim_score, lpips_score = compute_ssim_lpips(pairs, device)
    # 输出
    result_str = f"FID: {fid:.4f}\nSSIM: {ssim_score:.4f}\nLPIPS: {lpips_score:.4f}\n"
    print(result_str)
    with open(os.path.join(output_dir, 'eval_results.txt'), 'w') as f:
        f.write(result_str)

if __name__ == '__main__':
    main()
