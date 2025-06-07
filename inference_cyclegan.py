import argparse
import torch
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from models.cyclegan import CycleGAN  # or cyclegan_mod if needed

def find_corresponding_photo(sketch_path):
    """
    Finds the corresponding photo for a given sketch.
    Assumes photo path like: Dataset/photo/train/tx_000000000000/{filename}.jpg
    """
    sketch_filename = os.path.basename(sketch_path)
    # Get base filename without extension (e.g., .png/.jpg)
    base, _ = os.path.splitext(sketch_filename)
    photo_dir = os.path.join(os.path.dirname(__file__), 'Dataset', 'photo', 'train', 'tx_000000000000')
    # Try common image extensions
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        candidate = os.path.join(photo_dir, base + ext)
        if os.path.exists(candidate):
            return candidate
    return None

def main():
    parser = argparse.ArgumentParser(description="CycleGAN Inference Script")
    parser.add_argument('--weights', type=str, default='output/cyclegan/latest.pth', help='Path to model weights (.pth file)')
    parser.add_argument('--input', type=str, default='input.png', help='Path to input image')
    parser.add_argument('--output', type=str, default='output.png', help='Path to save output image')
    parser.add_argument('--direction', type=str, default='A2B', choices=['A2B', 'B2A'], help='Translation direction')
    parser.add_argument('--size', type=int, default=256, help='Image size (default: 256)')
    parser.add_argument('--device', type=str, default=None, help='Device to use: cpu, cuda, mps, or auto for auto-detection.')
    parser.add_argument('--find_photo', action='store_true', help='If set, tries to find the corresponding photo for the input sketch and prints its path instead of performing inference.')
    args = parser.parse_args()

    # Determine device (cuda, mps, cpu)
    if args.device is not None:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    print(f"Using device: {device}")

    # If --find_photo is used, locate corresponding photo and exit
    if args.find_photo:
        photo_path = find_corresponding_photo(args.input)
        if photo_path:
            print(f"Found corresponding photo: {photo_path}")
        else:
            print("No corresponding photo found.")
        return

    # Load model
    model = CycleGAN()
    # Load model weights (state_dict)
    state = torch.load(args.weights, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        state_dict = state['model']
    else:
        state_dict = state
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load and preprocess input image (ensure grayscale)
    img = Image.open(args.input).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] as CycleGAN expects
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        if args.direction == 'A2B':
            output = model.G_AB(input_tensor)
        else:
            output = model.G_BA(input_tensor)
        # If output is in [-1,1], map to [0,1]
        output_img = (output.squeeze(0).detach().cpu() + 1) / 2
        save_image(output_img, args.output)
        print(f"Saved output image to {args.output}")

if __name__ == '__main__':
    main()
