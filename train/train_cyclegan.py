import sys
import os
import datetime
import argparse
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader
from torchvision import transforms, utils as vutils
from models.cyclegan import CycleGAN
from datasets.sketch_photo_dataset import SketchPhotoDataset
from tqdm import tqdm
import time
import random

# Training Hyperparameters
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

# Image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def save_image(tensor, path, nrow=2):
    vutils.save_image(tensor, path, nrow=nrow, normalize=True, value_range=(-1, 1))

# Function to match image channels (e.g., 1-channel to 3-channel)
def match_channels(input, target):
    if input.shape[1] != target.shape[1]:
        if input.shape[1] == 1 and target.shape[1] == 3:
            input = input.repeat(1, 3, 1, 1)
        elif input.shape[1] == 3 and target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
    return input, target

class ImagePool:
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """
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

# --- WGAN-GP Gradient Penalty ---
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones_like(d_interpolates, device=device, requires_grad=False)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Output directory to resume training from (e.g., output/cyclegan/YYYY-MM-DD_HH-MM-SS)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs to train for')
    opt = parser.parse_args()

    # Update BATCH_SIZE and EPOCHS from command line arguments if provided
    BATCH_SIZE_EFFECTIVE = opt.batch_size
    EPOCHS_EFFECTIVE = opt.epochs
    NUM_WORKERS_EFFECTIVE = opt.num_workers

    # Resume training or setup new run
    if opt.resume:
        OUTPUT_DIR = opt.resume
        if not os.path.isdir(OUTPUT_DIR):
            print(f"Error: Resume directory {OUTPUT_DIR} does not exist.")
            sys.exit(1)
        LOG_FILE = os.path.join(OUTPUT_DIR, 'train_log.txt')
        CHECKPOINT = os.path.join(OUTPUT_DIR, 'cyclegan.pth')
        if not os.path.exists(CHECKPOINT):
            print(f"Error: Checkpoint {CHECKPOINT} does not exist in resume directory.")
            sys.exit(1)
        RESUME_MODE = True
        print(f"Resuming training from {OUTPUT_DIR}")
    else:
        RUN_TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'cyclegan', RUN_TIMESTAMP)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        LOG_FILE = os.path.join(OUTPUT_DIR, 'train_log.txt')
        CHECKPOINT = os.path.join(OUTPUT_DIR, 'cyclegan.pth')
        RESUME_MODE = False
        print(f"Starting new training run. Output will be saved to {OUTPUT_DIR}")

    # sketch/photo set
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
    # Create dataset and dataloader
    dataset = SketchPhotoDataset(DATA_ROOT, transform=transform, mode='train')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_EFFECTIVE, shuffle=True, num_workers=NUM_WORKERS_EFFECTIVE)

    # Initialize model, optimizers, and loss functions
    model = CycleGAN(input_nc=1, output_nc=3).to(DEVICE)
    optimizer_G = torch.optim.Adam(list(model.G_AB.parameters()) + list(model.G_BA.parameters()), lr=0.0001, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(list(model.D_A.parameters()) + list(model.D_B.parameters()), lr=0.0001, betas=(0.5, 0.999))
    # Cycle consistency loss
    criterion_cycle = nn.L1Loss()
    # Identity loss
    criterion_identity = nn.L1Loss()

    fake_A_pool = ImagePool(50)
    fake_B_pool = ImagePool(50)

    # Load checkpoint if resuming
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
        # Helper function to ensure 3 channels for visualization, defined inside main for proper scoping if needed by loop
        def to3c(x):
            return x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)

        for epoch in range(start_epoch, EPOCHS_EFFECTIVE):
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{EPOCHS_EFFECTIVE}")
            epoch_start = time.time()
            for i, batch in pbar:
                real_A = batch['sketch'].to(DEVICE)
                real_B = batch['photo'].to(DEVICE)
                fake_B, rec_A, fake_A, rec_B = model(real_A, real_B)

                # ========== 1. Generator loss (WGAN-GP) ==========
                loss_GAN_AB = -model.D_B(fake_B).mean()
                loss_GAN_BA = -model.D_A(fake_A).mean()
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

                # ========== 2. Discriminator loss (WGAN-GP) ==========
                n_D_steps = 3
                lambda_gp = 10
                for _ in range(n_D_steps):
                    # A
                    real_validity_A = model.D_A(real_A)
                    fake_A_buffer = fake_A_pool.query(fake_A.detach())
                    fake_validity_A = model.D_A(fake_A_buffer)
                    gp_A = compute_gradient_penalty(model.D_A, real_A, fake_A_buffer, DEVICE)
                    loss_D_A = fake_validity_A.mean() - real_validity_A.mean() + lambda_gp * gp_A
                    optimizer_D.zero_grad()
                    loss_D_A.backward(retain_graph=True)
                    optimizer_D.step()
                    # B
                    real_validity_B = model.D_B(real_B)
                    fake_B_buffer = fake_B_pool.query(fake_B.detach())
                    fake_validity_B = model.D_B(fake_B_buffer)
                    gp_B = compute_gradient_penalty(model.D_B, real_B, fake_B_buffer, DEVICE)
                    loss_D_B = fake_validity_B.mean() - real_validity_B.mean() + lambda_gp * gp_B
                    optimizer_D.zero_grad()
                    loss_D_B.backward(retain_graph=True)
                    optimizer_D.step()
                loss_D = loss_D_A + loss_D_B

                # tqdm progress bar update
                pbar.set_postfix({
                    'Loss_G': f"{loss_G.item():.4f}",
                    'Loss_D': f"{loss_D.item():.4f}"
                })
                # Monitor for potential mode collapse by checking std of generated images
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

                # Log losses periodically
                if i % 10 == 0:
                    msg = f"[Epoch {epoch+1}] [Batch {i+1}/{len(dataloader)}] Loss_G: {loss_G.item():.4f} Loss_D: {loss_D.item():.4f}\n"
                    loss_logf.write(msg)
                    loss_logf.flush()
                
                # Save a grid of samples periodically
                if i % 200 == 0:
                    grid = torch.cat([
                        to3c(real_A[:4]), to3c(fake_B[:4]), to3c(real_B[:4]), to3c(fake_A[:4])
                    ], dim=0)
                    save_image(grid, os.path.join(OUTPUT_DIR, f'pair_epoch{epoch}_batch{i}.png'), nrow=4)
            
            # Save checkpoint at the end of each epoch
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

    except KeyboardInterrupt: # Handle Ctrl+C, save checkpoint
        print("KeyboardInterrupt: Saving checkpoint before exit...")
        if 'model' in locals() and 'optimizer_G' in locals() and 'optimizer_D' in locals(): # Ensure variables are defined
            torch.save({
                'model': model.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'epoch': epoch if 'epoch' in locals() else start_epoch # Use current epoch or start_epoch if loop didn't start
            }, CHECKPOINT)
            print(f"[Interrupted] Checkpoint saved to {CHECKPOINT}")
            if 'logf' in locals() and not logf.closed: logf.write(f"[Interrupted] Checkpoint saved to {CHECKPOINT}\n")
        else:
            print("[Interrupted] Model not initialized, cannot save checkpoint.")
        if 'logf' in locals() and not logf.closed: logf.flush()
        raise
    except Exception as e:
        print(f"Exception: {e}")
        if 'logf' in locals() and not logf.closed: 
            logf.write(f"Exception: {e}\n")
            logf.flush()
        raise
    finally:
        if 'logf' in locals() and not logf.closed: logf.close()
        if 'loss_logf' in locals() and not loss_logf.closed: loss_logf.close()
        if 'monitor_logf' in locals() and not monitor_logf.closed: monitor_logf.close()

if __name__ == '__main__':
    main()
