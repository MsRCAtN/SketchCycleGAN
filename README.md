# SketchCycleGAN: Sketch-to-Image Translation in PyTorch

This project provides PyTorch implementations of CycleGAN and Pix2Pix models tailored for sketch-to-image translation tasks. It includes support for an original CycleGAN, an improved CycleGAN (potentially incorporating geometric consistency or attention mechanisms), and a standard Pix2Pix model. The framework is designed for training, evaluation, and comparative analysis of these image translation techniques.

This work was developed as part of a dissertation project focusing on generative adversarial networks for image synthesis from sketches.

## Features

- **Multiple Models**: Implements Original CycleGAN, Improved CycleGAN, and Pix2Pix.
- **Unified Training Scripts**: Clear and consistent training scripts for each model (`train/train_cyclegan.py`, `train/train_cyclegan_orig.py`, `train/train_pix2pix.py`).
- **Organized Outputs**: Training outputs (checkpoints, logs, samples) are saved to `output/{model_tag}/{timestamp}/`.
- **Resume Training**: Easily resume training from saved checkpoints using the `--resume` flag.
- **Automatic Checkpoint Saving**: Checkpoints are saved automatically during training and upon interruption (e.g., Ctrl+C).
- **Comprehensive Evaluation**: Scripts to compute common image quality metrics like FID, SSIM, and LPIPS (e.g., `eval/eval_cyclegan.py`).
- **Cross-Platform GPU Support**: Supports NVIDIA GPUs (CUDA), Apple Silicon GPUs (MPS), and CPU fallback for training and inference.
- **Modular Code**: Designed for easy understanding, modification, and extension for further research and experimentation.

## Setup

### 1. Prerequisites
- Python 3.12+ recommended.
- NVIDIA GPU with CUDA support (for CUDA acceleration) or Apple Silicon Mac (for MPS acceleration).

### 2. Clone the Repository
```bash
git clone https://github.com/MsRCAtN/SketchCycleGAN.git
cd SketchCycleGAN
```

### 3. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

## Dataset Preparation

This project is designed to work with datasets like the "Sketchy Database". The dataset is expected to be placed in the `Dataset/` directory, which is not included in this repository due to its size. You will need to obtain and structure your dataset according to the following format, which the scripts are currently configured to use:

```
Dataset/
├── photo/
│   ├── tx_000000000000/  
│   │   ├── image1.jpg
│   │   └── ...
│   └── tx_000100000000/  
│       └── ...
└── sketch/
    ├── tx_000000000000/  
    │   ├── sketch1.png
    │   └── ...
    └── tx_000100000000/  
        └── ...
```

The training scripts (`train_cyclegan.py`, `train_pix2pix.py`) currently use hardcoded lists of these `tx_*` subdirectories (found within `Dataset/photo/` and `Dataset/sketch/`) for training and testing. If you use a different dataset or wish to use different subsets, you may need to modify these lists within the respective training scripts or adapt the `SketchPhotoDataset` class in `datasets/sketch_photo_dataset.py`.

It is recommended to split your data into training and validation/testing sets, for example, using a 60:40 ratio. The `datasets/split_dataset.py` script can be used as a starting point or adapted for this purpose, or you can manually organize your files into appropriate training and testing subdirectories within the `tx_*` folders if your chosen dataset structure supports this directly.

## Training

Training scripts are located in the `train/` directory.

### Common Training Arguments:
- `--batch_size <int>`: Batch size for training (default: 1 or 2, depending on script).
- `--epochs <int>`: Total number of epochs to train (default: 20 or 200).
- `--num_workers <int>`: Number of worker threads for data loading (default: 4).
- `--resume <path_to_output_dir>`: Path to a previous output directory (e.g., `output/cyclegan/2024-06-07_10-30-00`) to resume training. The script will load the checkpoint and logs from this directory.
- The scripts currently use a hardcoded dataset structure based on subdirectories within `Dataset/photo/` and `Dataset/sketch/`. There is no `--dataset_name` argument for specifying a top-level dataset subfolder.
- `--lr <float>`: Learning rate (specific to some scripts, check argparse defaults).

### CycleGAN (Improved Version)
```bash
python train/train_cyclegan.py --epochs 200 --batch_size 1
```

### Original CycleGAN
```bash
python train/train_cyclegan_orig.py --epochs 200 --batch_size 1
```

### Pix2Pix
```bash
python train/train_pix2pix.py --epochs 200 --batch_size 2
```

Note: The specific sketch and photo subsets used for training (e.g., `tx_000000000000`) are currently hardcoded within the training scripts. Modify these lists in the scripts if you need to use different data subsets.

- Training progress, logs, and model checkpoints (`.pth` files) will be saved in a new timestamped subdirectory under `output/{model_tag}/` (e.g., `output/cyclegan/YYYY-MM-DD_HH-MM-SS/`).

## Evaluation

Evaluation scripts are in the `eval/` directory. They typically require paths to the dataset and the trained model's output directory or specific checkpoint file.

### Example: Evaluating CycleGAN
```bash
python eval/eval_cyclegan.py \
    --model_path output/cyclegan/YYYY-MM-DD_HH-MM-SS/cyclegan.pth \
    --dataroot_A Dataset/sketch/tx_your_test_sketch_set \
    --dataroot_B Dataset/photo/tx_your_test_photo_set \
    --domain A2B # or B2A, depending on which generator you want to test
    # Ensure --dataroot_A and --dataroot_B point to the correct test set subdirectories.
    # Add other necessary arguments like --output_dir_eval if the script saves generated images.
```

### Example: Evaluating Pix2Pix
(Assuming an `eval_pix2pix.py` script exists or `eval_cyclegan.py` is adapted)
```bash
python eval/eval_pix2pix.py \
    --model_path output/pix2pix/YYYY-MM-DD_HH-MM-SS/pix2pix.pth \
    --dataroot_A Dataset/sketch/tx_your_test_sketch_set \
    --dataroot_B Dataset/photo/tx_your_test_photo_set \
    # Ensure --dataroot_A and --dataroot_B point to the correct paired test set subdirectories.
    # Add other necessary arguments.
```
- Consult the specific evaluation script for its required arguments (e.g., `--fake_type`, `--real_type` if comparing generated vs. real images for metrics).
- Evaluation results (metrics, generated images) are typically saved in the model's output directory or a specified evaluation output directory.

## Inference

Use the inference scripts to generate images using a trained model.

### CycleGAN Inference
```bash
python inference_cyclegan.py \
    --model_path path/to/your/cyclegan_checkpoint.pth \
    --input_dir path/to/input_sketches_or_photos \
    --output_dir path/to/save_generated_images \
    --domain A2B # or B2A
```

### Pix2Pix Inference
```bash
python inference_pix2pix.py \
    --model_path path/to/your/pix2pix_checkpoint.pth \
    --input_dir path/to/input_sketches \
    --output_dir path/to/save_generated_images
```

## Directory Structure Overview

- **`/` (Root Directory)**:
  - `README.md`: This document.
  - `requirements.txt`: Python dependencies.
  - `.gitignore`: Files and directories ignored by Git.
  - `inference_cyclegan.py`, `inference_pix2pix.py`: Scripts for model inference.
  - `plot_multi_loss_comparison.py`: Utility to plot comparative loss curves.
- **`Dataset/`**: (User-provided, Git-ignored) Stores raw image datasets.
- **`datasets/`**: Data loading and preprocessing scripts (e.g., `aligned_dataset.py`, `unaligned_dataset.py`, `split_dataset.py`).
- **`eval/`**: Model evaluation scripts (e.g., `eval_cyclegan.py`, `eval_all_metrics.py`) and visualization tools.
- **`models/`**: Model architecture definitions (e.g., `cyclegan.py`, `pix2pix_model.py`, `stn.py`, `tps.py`).
- **`output/`**: (Git-ignored by default) Stores all outputs from training and evaluation (checkpoints, logs, sample images, metrics).
  - `cyclegan/`, `cyclegan_orig/`, `pix2pix/`: Model-specific output subdirectories.
  - `plots/`: Generated charts (loss curves, metric comparisons).
- **`train/`**: Training scripts for different models (e.g., `train_cyclegan.py`, `train_pix2pix.py`).

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for bugs, feature requests, or suggestions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This project is inspired by and builds upon the foundational work of:
  - [CycleGAN](https://junyanz.github.io/CycleGAN/)
  - [pix2pix](https://phillipi.github.io/pix2pix/)
- Gratitude to the PyTorch team and the open-source community for their invaluable tools and libraries.

---

**For questions or issues, please open an issue on GitHub.**
