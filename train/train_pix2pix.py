import sys
import os
import datetime
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Use abspath for robustness
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils as vutils
from models.pix2pix_model import Pix2PixModel as Pix2Pix # Assuming Pix2PixModel is the class name in pix2pix_model.py
from datasets.aligned_dataset import AlignedDataset as SketchPhotoDataset # Assuming AlignedDataset is used for pix2pix
from tqdm import tqdm
import time

# Utility function to save image tensors
def save_image(tensor, path, nrow=2, normalize=True, value_range=(-1, 1)):
    """Saves a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            will save the tensor as a grid of images.
        path (str): Path to save image.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default: 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
            Default: True.
        value_range (tuple, optional): Tuple (min, max) where min and max are numbers,
            then the image is normalized to this range. Default: (-1, 1).
    """
    vutils.save_image(tensor, path, nrow=nrow, normalize=normalize, value_range=value_range)

def main():
    parser = argparse.ArgumentParser(description='Pix2Pix Training Script')
    parser.add_argument('--resume', type=str, default=None, help='Path to the output directory to resume training from (e.g., output/pix2pix/timestamp). Checkpoint and logs will be loaded from here.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading.')
    parser.add_argument('--epochs', type=int, default=20, help='Total number of epochs to train.')
    opt = parser.parse_args()

    BATCH_SIZE = opt.batch_size
    NUM_WORKERS = opt.num_workers
    EPOCHS = opt.epochs

    # --- Device Selection ---
    DEVICE = 'cpu' # Default to CPU
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        print(f"[INFO] Using CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built(): # Check for Apple MPS
        DEVICE = 'mps'
        print("[INFO] Using MPS device (Apple Silicon GPU)")
    else:
        print("[INFO] Using CPU device")

    # --- Path Definitions ---
    # Root directory of the project
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Root directory of the dataset
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'Dataset')

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
        try:
            if not RESUME_MODE:
                logf.write(f"# Training run at {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\n")

            for epoch in range(start_epoch, EPOCHS):
                pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{EPOCHS}")
                epoch_start = time.time()
                for i, batch in pbar:
                    # Get input and target images
                    real_A = batch['sketch'].to(DEVICE)  # input (sketch)
                    real_B = batch['photo'].to(DEVICE)   # target (photo)
                    
                    # --- Train Discriminator ---
                    fake_B = model.generator(real_A)
                    pred_real = model.discriminator(real_A, real_B)
                    pred_fake = model.discriminator(real_A, fake_B.detach()) # Detach fake_B for discriminator training
                    valid = torch.ones_like(pred_real, device=DEVICE)
                    fake = torch.zeros_like(pred_fake, device=DEVICE)
                    # Discriminator loss
                    loss_D_real = criterion_GAN(pred_real, valid)
                    loss_D_fake = criterion_GAN(pred_fake, fake)
                    loss_D = (loss_D_real + loss_D_fake) * 0.5
                    optimizer_D.zero_grad()
                    loss_D.backward()
                    optimizer_D.step()
                    
                    # --- Train Generator ---
                    # Re-evaluate fake_B (not detached) with updated discriminator for generator loss calculation
                    pred_fake_for_G = model.discriminator(real_A, fake_B) 
                    loss_G_GAN = criterion_GAN(pred_fake_for_G, valid) # Adversarial loss
                    loss_G_L1 = criterion_L1(fake_B, real_B) # L1 loss (pixel-wise)
                    loss_G = loss_G_GAN + 100 * loss_G_L1 # Total generator loss
                    optimizer_G.zero_grad()
                    loss_G.backward()
                    optimizer_G.step()

                    pbar.set_postfix({
                        'Loss_G': f"{loss_G.item():.4f}",
                        'Loss_D': f"{loss_D.item():.4f}"
                    })

                    # Log losses periodically
                    if (i+1) % 10 == 0: 
                        msg = f"[Epoch {epoch+1}] [Batch {i+1}/{len(dataloader)}] Loss_G: {loss_G.item():.4f} Loss_D: {loss_D.item():.4f}\n"
                        logf.write(msg)
                        logf.flush()

                    # Save sample images periodically
                    if (i+1) % 200 == 0:
                        save_image(fake_B[:4], os.path.join(OUTPUT_DIR, f'epoch{epoch+1}_batch{i+1}_fakeB.png'))
                        save_image(real_A[:4], os.path.join(OUTPUT_DIR, f'epoch{epoch+1}_batch{i+1}_realA.png'))
                        save_image(real_B[:4], os.path.join(OUTPUT_DIR, f'epoch{epoch+1}_batch{i+1}_realB.png'))
                
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
            if 'model' in locals() and 'optimizer_G' in locals() and 'optimizer_D' in locals() and 'CHECKPOINT' in locals():
                current_epoch_to_save = epoch if 'epoch' in locals() else start_epoch
                torch.save({
                    'model': model.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'optimizer_D': optimizer_D.state_dict(),
                    'epoch': current_epoch_to_save
                }, CHECKPOINT)
                print(f"[Interrupted] Checkpoint saved to {CHECKPOINT} at epoch {current_epoch_to_save}")
                if 'logf' in locals() and not logf.closed:
                    logf.write(f"[Interrupted] Checkpoint saved to {CHECKPOINT} at epoch {current_epoch_to_save}\n")
                    logf.flush()
            else:
                print("[Interrupted] Critical variables (model, optimizers, checkpoint path) not defined. Cannot save checkpoint.")
            raise 
        except Exception as e:
            print(f"Exception during training: {e}")
            if 'logf' in locals() and not logf.closed:
                logf.write(f"Exception during training: {e}\n")
                logf.flush()
            # Optionally, save a checkpoint here too if possible and desired
            if 'model' in locals() and 'optimizer_G' in locals() and 'optimizer_D' in locals() and 'CHECKPOINT' in locals():
                current_epoch_to_save = epoch if 'epoch' in locals() else start_epoch
                torch.save({
                    'model': model.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'optimizer_D': optimizer_D.state_dict(),
                    'epoch': current_epoch_to_save
                }, CHECKPOINT)
                print(f"[Exception] Checkpoint saved to {CHECKPOINT} at epoch {current_epoch_to_save} due to exception.")
                if 'logf' in locals() and not logf.closed:
                    logf.write(f"[Exception] Checkpoint saved to {CHECKPOINT} at epoch {current_epoch_to_save} due to exception: {e}\n")
                    logf.flush()
            raise 
        finally:
            if 'logf' in locals() and not logf.closed:
                logf.close()
            print("Training finished or terminated.")

if __name__ == '__main__':
    main()
