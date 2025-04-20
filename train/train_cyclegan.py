import sys
import os
import datetime
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils as vutils
from models.cyclegan import CycleGAN
from datasets.sketch_photo_dataset import SketchPhotoDataset
from tqdm import tqdm
import time
import random

# 配置
BATCH_SIZE = 2
EPOCHS = 20
# 设备选择兼容新版 PyTorch，无警告
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_built() and torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
if DEVICE == 'cuda':
    print(f"[INFO] Using device: cuda ({torch.cuda.get_device_name(torch.cuda.current_device())})")
elif DEVICE == 'mps':
    print("[INFO] Using device: mps (Apple Silicon GPU)")
else:
    print("[INFO] Using device: cpu")
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Dataset')

# 简单 transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def save_image(tensor, path, nrow=2):
    vutils.save_image(tensor, path, nrow=nrow, normalize=True, value_range=(-1, 1))

# 匹配通道数工具
def match_channels(input, target):
    if input.shape[1] != target.shape[1]:
        if input.shape[1] == 1 and target.shape[1] == 3:
            input = input.repeat(1, 3, 1, 1)
        elif input.shape[1] == 3 and target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
    return input, target

class ImagePool:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []
    def query(self, images):
        out = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(img.clone().detach())
                out.append(img)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    tmp = self.images[idx].clone()
                    self.images[idx] = img.clone().detach()
                    out.append(tmp)
                else:
                    out.append(img)
        return torch.cat(out, dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='output dir to resume from')
    args = parser.parse_args()

    # resume 逻辑
    if args.resume:
        OUTPUT_DIR = args.resume
        if not os.path.isdir(OUTPUT_DIR):
            print(f"Resume directory {OUTPUT_DIR} does not exist.")
            sys.exit(1)
        LOG_FILE = os.path.join(OUTPUT_DIR, 'train_log.txt')
        CHECKPOINT = os.path.join(OUTPUT_DIR, 'cyclegan_minimal.pth')
        if not os.path.exists(CHECKPOINT):
            print(f"Checkpoint {CHECKPOINT} does not exist in resume directory.")
            sys.exit(1)
        RESUME_MODE = True
    else:
        RUN_TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'cyclegan', RUN_TIMESTAMP)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        LOG_FILE = os.path.join(OUTPUT_DIR, 'train_log.txt')
        CHECKPOINT = os.path.join(OUTPUT_DIR, 'cyclegan_minimal.pth')
        RESUME_MODE = False

    # 数据集
    dataset = SketchPhotoDataset(DATA_ROOT, paired=True, transform=transform, sketch_set='tx_000000000000', photo_set='tx_000000000000')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 模型
    model = CycleGAN(input_nc=1, output_nc=3).to(DEVICE)
    optimizer_G = torch.optim.Adam(list(model.G_AB.parameters()) + list(model.G_BA.parameters()), lr=0.0001, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(list(model.D_A.parameters()) + list(model.D_B.parameters()), lr=0.0001, betas=(0.5, 0.999))
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    fake_A_pool = ImagePool(50)
    fake_B_pool = ImagePool(50)

    # 检查 checkpoint
    start_epoch = 0
    if RESUME_MODE:
        checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(checkpoint['model'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint at epoch {start_epoch} in {OUTPUT_DIR}")

    logf = open(LOG_FILE, 'a')
    loss_log_path = os.path.join(OUTPUT_DIR, 'train_loss_log.txt')
    monitor_log_path = os.path.join(OUTPUT_DIR, 'train_monitor_log.txt')
    loss_logf = open(loss_log_path, 'a')
    monitor_logf = open(monitor_log_path, 'a')
    if not RESUME_MODE:
        logf.write(f"# Training run at {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\n")
    try:
        for epoch in range(start_epoch, EPOCHS):
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{EPOCHS}")
            epoch_start = time.time()
            for i, batch in pbar:
                real_A = batch['sketch'].to(DEVICE)
                real_B = batch['photo'].to(DEVICE)
                # 生成器前向
                fake_B, rec_A, fake_A, rec_B = model(real_A, real_B)
                # 生成器 loss
                valid = torch.ones_like(model.D_B(fake_B), device=DEVICE)
                fake = torch.zeros_like(model.D_B(fake_B), device=DEVICE)
                loss_GAN_AB = criterion_GAN(model.D_B(fake_B), valid)
                loss_GAN_BA = criterion_GAN(model.D_A(fake_A), valid)
                loss_cycle_A = criterion_cycle(rec_A, real_A)
                loss_cycle_B = criterion_cycle(rec_B, real_B)
                # Identity loss
                identity_A = model.G_BA(real_B)
                identity_B = model.G_AB(real_A)
                identity_A, real_B_matched = match_channels(identity_A, real_B)
                identity_B, real_A_matched = match_channels(identity_B, real_A)
                loss_identity_A = criterion_identity(identity_A, real_B_matched)
                loss_identity_B = criterion_identity(identity_B, real_A_matched)
                lambda_id = 0.5
                loss_G = loss_GAN_AB + loss_GAN_BA + 20 * (loss_cycle_A + loss_cycle_B) + lambda_id * (loss_identity_A + loss_identity_B)
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

                # 判别器多步训练
                n_D_steps = 3
                for _ in range(n_D_steps):
                    # 判别器输入加噪声（0.05）
                    def add_noise(x, std=0.05):
                        return x + torch.randn_like(x) * std

                    # 判别器label smoothing
                    real_label = torch.full((real_A.size(0), 1, 30, 30), 0.9, device=DEVICE)
                    fake_label = torch.zeros_like(real_label)

                    # 判别器A
                    pred_real_A = model.D_A(add_noise(real_A))
                    loss_D_A_real = criterion_GAN(pred_real_A, real_label)
                    fake_A_buffer = fake_A_pool.query(fake_A.detach())
                    pred_fake_A = model.D_A(add_noise(fake_A_buffer))
                    loss_D_A_fake = criterion_GAN(pred_fake_A, fake_label)
                    loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

                    # 判别器B
                    pred_real_B = model.D_B(add_noise(real_B))
                    loss_D_B_real = criterion_GAN(pred_real_B, real_label)
                    fake_B_buffer = fake_B_pool.query(fake_B.detach())
                    pred_fake_B = model.D_B(add_noise(fake_B_buffer))
                    loss_D_B_fake = criterion_GAN(pred_fake_B, fake_label)
                    loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

                    loss_D = loss_D_A + loss_D_B
                    optimizer_D.zero_grad()
                    loss_D.backward()
                    optimizer_D.step()

                # tqdm显示
                pbar.set_postfix({
                    'Loss_G': f"{loss_G.item():.4f}",
                    'Loss_D': f"{loss_D.item():.4f}"
                })
                # 自动化模式崩溃检测
                with torch.no_grad():
                    mean_fakeB = fake_B.mean().item()
                    std_fakeB = fake_B.std().item()
                    mean_fakeA = fake_A.mean().item()
                    std_fakeA = fake_A.std().item()
                    if std_fakeB < 0.05 or std_fakeA < 0.05:
                        print(f"[Warning] Possible mode collapse detected! std_fakeB: {std_fakeB:.4f}, std_fakeA: {std_fakeA:.4f}")
                        logf.write(f"[Warning][Epoch {epoch+1}][Batch {i+1}] std_fakeB: {std_fakeB:.4f}, std_fakeA: {std_fakeA:.4f}\n")
                    monitor_msg = f"[Epoch {epoch+1}][Batch {i+1}] mean_fakeB: {mean_fakeB:.4f}, std_fakeB: {std_fakeB:.4f}, mean_fakeA: {mean_fakeA:.4f}, std_fakeA: {std_fakeA:.4f}\n"
                    monitor_logf.write(monitor_msg)
                    monitor_logf.flush()
                # 日志
                if i % 10 == 0:
                    msg = f"[Epoch {epoch+1}] [Batch {i+1}/{len(dataloader)}] Loss_G: {loss_G.item():.4f} Loss_D: {loss_D.item():.4f}\n"
                    loss_logf.write(msg)
                    loss_logf.flush()
                def to3c(x):
                    return x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)
                # 保存成对图片，方便对比
                if i % 200 == 0:
                    grid = torch.cat([
                        to3c(real_A[:4]), to3c(fake_B[:4]), to3c(real_B[:4]), to3c(fake_A[:4])
                    ], dim=0)
                    save_image(grid, os.path.join(OUTPUT_DIR, f'pair_epoch{epoch}_batch{i}.png'), nrow=4)
            # 保存 checkpoint
            torch.save({
                'model': model.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'epoch': epoch
            }, CHECKPOINT)
            print(f"Checkpoint saved to {CHECKPOINT}")
            logf.write(f"Checkpoint saved to {CHECKPOINT}\n")
            logf.flush()
            print(f"Epoch time: {time.time()-epoch_start:.1f}s")
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Saving checkpoint before exit...")
        torch.save({
            'model': model.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
            'epoch': epoch if 'epoch' in locals() else start_epoch
        }, CHECKPOINT)
        print(f"[Interrupted] Checkpoint saved to {CHECKPOINT}")
        logf.write(f"[Interrupted] Checkpoint saved to {CHECKPOINT}\n")
        logf.flush()
        raise
    except Exception as e:
        print(f"Exception: {e}")
        logf.write(f"Exception: {e}\n")
        logf.flush()
        raise
    finally:
        logf.close()
        loss_logf.close()
        monitor_logf.close()

if __name__ == '__main__':
    main()
