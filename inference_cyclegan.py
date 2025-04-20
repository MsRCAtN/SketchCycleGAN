import argparse
import torch
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from models.cyclegan import CycleGAN  # or cyclegan_mod if needed

def main():
    parser = argparse.ArgumentParser(description="CycleGAN Inference Script")
    parser.add_argument('--weights', type=str, default='output/cyclegan/latest.pth', help='Path to model weights (.pth file)')
    parser.add_argument('--input', type=str, default='input.png', help='Path to input image')
    parser.add_argument('--output', type=str, default='output.png', help='Path to save output image')
    parser.add_argument('--direction', type=str, default='A2B', choices=['A2B', 'B2A'], help='Translation direction')
    parser.add_argument('--size', type=int, default=256, help='Image size (default: 256)')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # Load model
    model = CycleGAN()
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Preprocess input
    transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        # Uncomment if training used normalization
        # transforms.Normalize((0.5,), (0.5,))
    ])
    img = Image.open(args.input).convert('RGB')
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
