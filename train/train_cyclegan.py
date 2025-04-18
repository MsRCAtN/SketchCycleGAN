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

# 配置
BATCH_SIZE = 2
EPOCHS = 100
DEVICE = 'mps' if hasattr(torch, 'has_mps') and torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Dataset')

# 简单 transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def save_image(tensor, path, nrow=2):
    vutils.save_image(tensor, path, nrow=nrow, normalize=True, value_range=(-1, 1))

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
    optimizer_G = torch.optim.Adam(list(model.G_AB.parameters()) + list(model.G_BA.parameters()), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(list(model.D_A.parameters()) + list(model.D_B.parameters()), lr=2e-4, betas=(0.5, 0.999))
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

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
                # G_BA(B) ≈ B, G_AB(A) ≈ A
                identity_A = model.G_BA(real_B)
                identity_B = model.G_AB(real_A)
                loss_identity_A = criterion_identity(identity_A, real_B)
                loss_identity_B = criterion_identity(identity_B, real_A)
                lambda_id = 0.5
                loss_G = loss_GAN_AB + loss_GAN_BA + 10 * (loss_cycle_A + loss_cycle_B) + lambda_id * (loss_identity_A + loss_identity_B)
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()
                # 判别器 loss
                loss_D_A = (criterion_GAN(model.D_A(real_A), valid) + criterion_GAN(model.D_A(fake_A.detach()), fake)) * 0.5
                loss_D_B = (criterion_GAN(model.D_B(real_B), valid) + criterion_GAN(model.D_B(fake_B.detach()), fake)) * 0.5
                loss_D = loss_D_A + loss_D_B
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
                # tqdm显示
                pbar.set_postfix({
                    'Loss_G': f"{loss_G.item():.4f}",
                    'Loss_D': f"{loss_D.item():.4f}"
                })
                # 日志
                if i % 10 == 0:
                    msg = f"[Epoch {epoch+1}] [Batch {i+1}/{len(dataloader)}] Loss_G: {loss_G.item():.4f} Loss_D: {loss_D.item():.4f}\n"
                    logf.write(msg)
                    logf.flush()
                # 保存生成图片
                if i % 200 == 0:
                    save_image(fake_B[:4], os.path.join(OUTPUT_DIR, f'epoch{epoch+1}_batch{i+1}_fakeB.png'))
                    save_image(fake_A[:4], os.path.join(OUTPUT_DIR, f'epoch{epoch+1}_batch{i+1}_fakeA.png'))
                    save_image(real_A[:4], os.path.join(OUTPUT_DIR, f'epoch{epoch+1}_batch{i+1}_realA.png'))
                    save_image(real_B[:4], os.path.join(OUTPUT_DIR, f'epoch{epoch+1}_batch{i+1}_realB.png'))
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

if __name__ == '__main__':
    main()
