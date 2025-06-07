import re
import matplotlib.pyplot as plt
import argparse

# 
parser = argparse.ArgumentParser()
parser.add_argument('--log_file', type=str, required=True, help='Path to train_log.txt')
parser.add_argument('--title', type=str, default='Training Loss Curve', help='Plot title')
parser.add_argument('--output', type=str, default='loss_curve.png', help='Output image file')
args = parser.parse_args()

# Get log file path from arguments
log_path = args.log_file

# Parse log file to extract losses
steps, G_loss, D_loss = [], [], []
with open(log_path, 'r') as f:
    for line in f:
        if line.startswith('['):
            m = re.match(r'\[Epoch (\d+)\] \[Batch (\d+)', line)
            if m:
                epoch = int(m.group(1))
                batch = int(m.group(2))
                # Extract G and D losses
                g = float(line.split('Loss_G:')[1].split('Loss_D:')[0].strip())
                d = float(line.split('Loss_D:')[1].strip())
                steps.append((epoch, batch))
                G_loss.append(g)
                D_loss.append(d)

# Calculate fractional step index (epoch + batch/total_batches_per_epoch, assuming 6250 for now)
step_idx = [e + b/6250 for (e, b) in steps]

plt.figure(figsize=(8,5))
plt.plot(step_idx, G_loss, label='Generator Loss', color='#4C72B0')
plt.plot(step_idx, D_loss, label='Discriminator Loss', color='#C44E52')
plt.xlabel('Epoch (fractional)')
plt.ylabel('Loss')
plt.title(args.title)
plt.legend()
plt.tight_layout()
plt.savefig(args.output, dpi=200)
plt.show()
