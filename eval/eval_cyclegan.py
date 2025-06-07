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

def split_concat_grid(img, nrow=4, ncol=4):
    # img: PIL.Image, a concatenated grid image
    w, h = img.size
    cell_w, cell_h = w // nrow, h // ncol
    imgs = []
    for r in range(ncol):
        for c in range(nrow):
            crop = img.crop((c*cell_w, r*cell_h, (c+1)*cell_w, (r+1)*cell_h))
            imgs.append(crop)
    return imgs  # List of 16 individual images

def get_concat_pairs(output_dir):
    # Find files like pair_epoch*_batch*.png
    concat_paths = sorted([f for f in os.listdir(output_dir) if f.startswith('pair_epoch') and f.endswith('.png')])
    pairs = []
    for f in concat_paths:
        img = Image.open(os.path.join(output_dir, f)).convert('RGB')
        imgs = split_concat_grid(img, nrow=4, ncol=4)  # 16 images
        # 0-3: real_A, 4-7: fake_B, 8-11: real_B, 12-15: fake_A
        for i in range(4):
            pairs.append((imgs[4+i], imgs[8+i]))  # (fake_B, real_B)
    return pairs

def compute_fid(fake_imgs, real_imgs):
    from torchvision.models import inception_v3
    import torch.nn.functional as F
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()
    resize = transforms.Resize((299, 299))
    def get_activations(imgs):
        acts = []
        for img in imgs:
            x = resize(transforms.ToTensor()(img)).unsqueeze(0).to(device)
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
    for fake_img, real_img in tqdm(pairs, desc='Evaluating SSIM/LPIPS'):
        fake_np = np.array(fake_img)
        real_np = np.array(real_img)
        ssim_score = ssim(fake_np, real_np, channel_axis=2, data_range=255)
        ssim_scores.append(ssim_score)
        fake_t = tf(fake_img).unsqueeze(0).to(device)
        real_t = tf(real_img).unsqueeze(0).to(device)
        lpips_score = loss_fn(fake_t, real_t).item()
        lpips_scores.append(lpips_score)
    return np.mean(ssim_scores), np.mean(lpips_scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help='output dir (with generated images)')
    args = parser.parse_args()
    output_dir = args.output_dir
    pairs = get_concat_pairs(output_dir)
    if len(pairs) == 0:
        print('No image pairs found!')
        return
    print(f'Found {len(pairs)} image pairs for evaluation.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fake_imgs = [f for f, _ in pairs]
    real_imgs = [r for _, r in pairs]
    print('Computing FID...')
    fid = compute_fid(fake_imgs, real_imgs)
    ssim_score, lpips_score = compute_ssim_lpips(pairs, device)
    result_str = f"FID: {fid:.4f}\nSSIM: {ssim_score:.4f}\nLPIPS: {lpips_score:.4f}\n"
    print(result_str)
    with open(os.path.join(output_dir, 'eval_results.txt'), 'w') as f:
        f.write(result_str)

if __name__ == '__main__':
    main()
