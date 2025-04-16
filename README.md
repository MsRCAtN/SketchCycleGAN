# SketchCycleGAN: Sketch to Image Translation

This project implements three models for sketch-to-image translation:
- **Original CycleGAN**
- **pix2pix**
- **Improved CycleGAN** (with geometric deformation/keypoint alignment)

All models support training, evaluation, and fair comparison experiments.

## Features
- Unified training scripts for each model: `train/train_cyclegan.py`, `train/train_cyclegan_orig.py`, `train/train_pix2pix.py`
- Output directories are organized as `output/{model_name}/{timestamp}/`
- Resume training and automatic checkpoint saving on interruption
- Evaluation script computes FID, SSIM, LPIPS (see `eval/eval_cyclegan.py`)
- Cross-platform GPU support: Apple M1/M2 (mps), NVIDIA RTX (cuda), or CPU fallback
- Modular code structure for easy extension and experiment management

## Directory Structure
- `models/`: Model definitions
- `datasets/`: Dataset and preprocessing
- `train/`: Training scripts (one per model)
- `eval/`: Evaluation and comparison scripts
- `output/`: Training results, checkpoints, generated images
- `utils/`: Logging, visualization, checkpoint helpers

## Quick Start
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train a model:**
   ```bash
   python train/train_cyclegan.py           # Improved CycleGAN
   python train/train_cyclegan_orig.py      # Original CycleGAN
   python train/train_pix2pix.py            # pix2pix
   ```
   - Each script will create a new output folder automatically.
   - To resume training, use `--resume output/{model_name}/{timestamp}`

3. **Evaluate results:**
   ```bash
   python eval/eval_cyclegan.py --output_dir output/{model_name}/{timestamp} --fake_type fakeB --real_type realB
   ```
   - Results are saved to `eval_results.txt` in the output directory.

## Notes
- See comments in each script for details and customization.
- For ablation or batch experiments, copy and modify the training scripts as needed.
- For more advanced features (multi-GPU, mixed precision, etc.), please extend the scripts or contact the author.

---

**For questions or issues, please open an issue on GitHub or contact the project maintainer.**
