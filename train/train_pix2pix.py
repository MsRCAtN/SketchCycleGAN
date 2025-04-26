import sys
import os
import datetime
import argparse
import mlflow
import mlflow.pytorch
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils as vutils
from models.pix2pix import Pix2Pix
from datasets.sketch_photo_dataset import SketchPhotoDataset
from tqdm import tqdm
import time

def save_image(tensor, path, nrow=2):
    vutils.save_image(tensor, path, nrow=nrow, normalize=True, value_range=(-1, 1))

def main():
    mlflow.set_experiment("pix2pix")
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
        mlflow.log_param("model", "pix2pix")

        DEVICE = 'cpu'
        if torch.cuda.is_available():
            DEVICE = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_built() and torch.backends.mps.is_available():
            DEVICE = 'mps'
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

        model_tag = 'pix2pix'
        if opt.resume:
            OUTPUT_DIR = opt.resume
            if not os.path.isdir(OUTPUT_DIR):
                print(f"Resume directory {OUTPUT_DIR} does not exist.")
                sys.exit(1)
            LOG_FILE = os.path.join(OUTPUT_DIR, 'train_log.txt')
            CHECKPOINT = os.path.join(OUTPUT_DIR, 'pix2pix.pth')
            if not os.path.exists(CHECKPOINT):
                print(f"Checkpoint {CHECKPOINT} does not exist in resume directory.")
                sys.exit(1)
            RESUME_MODE = True
        else:
            RUN_TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', model_tag, RUN_TIMESTAMP)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            LOG_FILE = os.path.join(OUTPUT_DIR, 'train_log.txt')
            CHECKPOINT = os.path.join(OUTPUT_DIR, 'pix2pix.pth')
            RESUME_MODE = False

        sketch_sets = [
            'tx_000000000000',
            'tx_000000000010',
            'tx_000000000110',
            'tx_000000001010',
            'tx_000000001110',
            'tx_000100000000',
        ]
        photo_sets = [
            'tx_000000000000',
            'tx_000100000000',
        ]
        dataset = SketchPhotoDataset(
            root_dir='Dataset',
            paired=True,
            transform=transform,
            sketch_sets=sketch_sets,
            photo_sets=photo_sets
        )
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        model = Pix2Pix(in_channels=1, out_channels=3).to(DEVICE)
        optimizer_G = torch.optim.Adam(model.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        criterion_GAN = torch.nn.BCEWithLogitsLoss()
        criterion_L1 = torch.nn.L1Loss()

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
                    real_A = batch['sketch'].to(DEVICE)  # input (sketch)
                    real_B = batch['photo'].to(DEVICE)   # target (photo)
                    # 生成器前向
                    fake_B = model.generator(real_A)
                    # 判别器前向
                    pred_real = model.discriminator(real_A, real_B)
                    pred_fake = model.discriminator(real_A, fake_B.detach())
                    valid = torch.ones_like(pred_real, device=DEVICE)
                    fake = torch.zeros_like(pred_fake, device=DEVICE)
                    # 判别器 loss
                    loss_D_real = criterion_GAN(pred_real, valid)
                    loss_D_fake = criterion_GAN(pred_fake, fake)
                    loss_D = (loss_D_real + loss_D_fake) * 0.5
                    optimizer_D.zero_grad()
                    loss_D.backward()
                    optimizer_D.step()
                    # 生成器 loss
                    pred_fake_for_G = model.discriminator(real_A, fake_B)
                    loss_G_GAN = criterion_GAN(pred_fake_for_G, valid)
                    loss_G_L1 = criterion_L1(fake_B, real_B)
                    loss_G = loss_G_GAN + 100 * loss_G_L1
                    optimizer_G.zero_grad()
                    loss_G.backward()
                    optimizer_G.step()
                    pbar.set_postfix({
                        'Loss_G': f"{loss_G.item():.4f}",
                        'Loss_D': f"{loss_D.item():.4f}"
                    })
                    if (i+1) % 100 == 0:
                        mlflow.log_metric("loss_G", loss_G.item(), step=epoch*len(dataloader)+i)
                        mlflow.log_metric("loss_D", loss_D.item(), step=epoch*len(dataloader)+i)
                    if i % 10 == 0:
                        msg = f"[Epoch {epoch+1}] [Batch {i+1}/{len(dataloader)}] Loss_G: {loss_G.item():.4f} Loss_D: {loss_D.item():.4f}\n"
                        logf.write(msg)
                        logf.flush()
                    if i % 200 == 0:
                        save_image(fake_B[:4], os.path.join(OUTPUT_DIR, f'epoch{epoch+1}_batch{i+1}_fakeB.png'))
                        save_image(real_A[:4], os.path.join(OUTPUT_DIR, f'epoch{epoch+1}_batch{i+1}_realA.png'))
                        save_image(real_B[:4], os.path.join(OUTPUT_DIR, f'epoch{epoch+1}_batch{i+1}_realB.png'))
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
                # 保存样例图片
                sample_path = os.path.join(OUTPUT_DIR, f"sample_epoch{epoch+1}.png")
                if os.path.exists(sample_path):
                    mlflow.log_artifact(sample_path, artifact_path="samples")
                # 保存模型
                if (epoch+1) % 5 == 0 or (epoch+1)==EPOCHS:
                    mlflow.pytorch.log_model(model.G, f"generator_epoch{epoch+1}")
                    mlflow.pytorch.log_model(model.D, f"discriminator_epoch{epoch+1}")
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
