import sys
import os
import datetime
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils as vutils
from models.cyclegan_mod import CycleGANMod
from datasets.sketch_photo_dataset import SketchPhotoDataset
from tqdm import tqdm
import time
import random
import mlflow
import mlflow.pytorch

# 
BATCH_SIZE = 2
EPOCHS = 20
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

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def save_image(tensor, path, nrow=2):
    vutils.save_image(tensor, path, nrow=nrow, normalize=True, value_range=(-1, 1))

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
    mlflow.set_experiment("CycleGAN-LSGAN")
    with mlflow.start_run():
        parser = argparse.ArgumentParser()
        parser.add_argument('--resume', type=str, default=None, help='output dir to resume from')
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--epochs', type=int, default=20)
        opt = parser.parse_args()
        BATCH_SIZE = opt.batch_size
        NUM_WORKERS = opt.num_workers
        EPOCHS = opt.epochs
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("num_workers", NUM_WORKERS)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("model", "CycleGAN-LSGAN")
        # resume 
        if opt.resume:
            OUTPUT_DIR = opt.resume
            if not os.path.isdir(OUTPUT_DIR):
                print(f"Resume directory {OUTPUT_DIR} does not exist.")
                sys.exit(1)
            RESUME_MODE = True
            start_epoch = 0
        else:
            OUTPUT_DIR = os.path.join('output', f'cyclegan_lsgan_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            RESUME_MODE = False
            start_epoch = 0
        LOG_FILE = os.path.join(OUTPUT_DIR, 'train_log.txt')
        CHECKPOINT = os.path.join(OUTPUT_DIR, 'cyclegan_lsgan.pth')
        # 
        dataset = SketchPhotoDataset(DATA_ROOT, paired=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        # 
        model = CycleGANMod(input_nc=1, output_nc=3).to(DEVICE)
        optimizer_G = torch.optim.Adam(list(model.G_AB.parameters()) + list(model.G_BA.parameters()), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(list(model.D_A.parameters()) + list(model.D_B.parameters()), lr=0.0002, betas=(0.5, 0.999))
        criterion_GAN = nn.MSELoss()
        criterion_cycle = nn.L1Loss()
        criterion_identity = nn.L1Loss()
        fake_A_pool = ImagePool()
        fake_B_pool = ImagePool()
        # 
        if opt.resume and os.path.exists(CHECKPOINT):
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
                    fake_B, rec_A, fake_A, rec_B = model(real_A, real_B)
                    # ========== 1.  loss（LSGAN） ==========
                    valid = torch.ones_like(model.D_B(fake_B), device=DEVICE)
                    fake = torch.zeros_like(model.D_B(fake_B), device=DEVICE)
                    loss_GAN_AB = criterion_GAN(model.D_B(fake_B), valid)
                    loss_GAN_BA = criterion_GAN(model.D_A(fake_A), valid)
                    loss_cycle_A = criterion_cycle(rec_A, real_A)
                    loss_cycle_B = criterion_cycle(rec_B, real_B)
                    # Identity loss
                    identity_A = model.G_BA(real_B)
                    identity_B = model.G_AB(real_A)
                    loss_identity_A = criterion_identity(identity_A, real_A)
                    loss_identity_B = criterion_identity(identity_B, real_B)
                    # 
                    loss_G = loss_GAN_AB + loss_GAN_BA + 10 * (loss_cycle_A + loss_cycle_B) + 5 * (loss_identity_A + loss_identity_B)
                    optimizer_G.zero_grad()
                    loss_G.backward()
                    optimizer_G.step()
                    # ========== 2.  loss（LSGAN） ==========
                    # A
                    fake_A_buffer = fake_A_pool.query(fake_A.detach())
                    loss_D_A_real = criterion_GAN(model.D_A(real_A), valid)
                    loss_D_A_fake = criterion_GAN(model.D_A(fake_A_buffer), fake)
                    loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
                    # B
                    fake_B_buffer = fake_B_pool.query(fake_B.detach())
                    loss_D_B_real = criterion_GAN(model.D_B(real_B), valid)
                    loss_D_B_fake = criterion_GAN(model.D_B(fake_B_buffer), fake)
                    loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
                    loss_D = loss_D_A + loss_D_B
                    optimizer_D.zero_grad()
                    loss_D.backward()
                    optimizer_D.step()
                    # tqdm
                    pbar.set_postfix({
                        'loss_G': loss_G.item(),
                        'loss_D': loss_D.item()
                    })
                    # 
                    if (i+1) % 100 == 0:
                        mlflow.log_metric("loss_G", loss_G.item(), step=epoch*len(dataloader)+i)
                        mlflow.log_metric("loss_D", loss_D.item(), step=epoch*len(dataloader)+i)
                    if i % 10 == 0:
                        msg = f"[Epoch {epoch+1}] [Batch {i+1}/{len(dataloader)}] Loss_G: {loss_G.item():.4f} Loss_D: {loss_D.item():.4f}\n"
                        loss_logf.write(msg)
                        loss_logf.flush()
                    def to3c(x):
                        return x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)
                    if i % 200 == 0:
                        grid = torch.cat([
                            to3c(real_A[:4]), to3c(fake_B[:4]), to3c(real_B[:4]), to3c(fake_A[:4])
                        ], dim=0)
                        save_image(grid, os.path.join(OUTPUT_DIR, f'pair_epoch{epoch}_batch{i}.png'), nrow=4)
                sample_path = os.path.join(OUTPUT_DIR, f"sample_epoch{epoch+1}.png")
                if os.path.exists(sample_path):
                    mlflow.log_artifact(sample_path, artifact_path="samples")
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
                if (epoch+1) % 5 == 0 or (epoch+1)==EPOCHS:
                    mlflow.pytorch.log_model(model.G_AB, f"generator_AB_epoch{epoch+1}")
                    mlflow.pytorch.log_model(model.D_A, f"discriminator_A_epoch{epoch+1}")
                    mlflow.pytorch.log_model(model.G_BA, f"generator_BA_epoch{epoch+1}")
                    mlflow.pytorch.log_model(model.D_B, f"discriminator_B_epoch{epoch+1}")
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
