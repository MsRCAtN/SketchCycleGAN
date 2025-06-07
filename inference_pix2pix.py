import argparse
import torch
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from models.pix2pix import Pix2Pix

def main():
    parser = argparse.ArgumentParser(description="Pix2Pix Inference Script")
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pth file)')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, required=True, help='Path to save output image')
    parser.add_argument('--size', type=int, default=256, help='Image size (default: 256)')
    parser.add_argument('--device', type=str, default=None, help='Device to use: cpu, cuda, mps, or auto for auto-detection.')
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

    # Load model
    model = Pix2Pix()
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
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] as Pix2Pix expects
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model.generator(input_tensor)
        # If output is in [-1,1], map to [0,1]
        output_img = (output.squeeze(0).detach().cpu() + 1) / 2
        save_image(output_img, args.output)
        print(f"Saved output image to {args.output}")

if __name__ == '__main__':
    main()
