import re
import matplotlib.pyplot as plt

# 日志文件路径
log_path = 'output/2025-04-17_10-45-44/train_log.txt'

# 解析日志
steps, G_loss, D_loss = [], [], []
with open(log_path, 'r') as f:
    for line in f:
        m = re.match(r'\[Epoch (\d+)\] \[Batch (\d+)', line)
        if m:
            epoch = int(m.group(1))
            batch = int(m.group(2))
            # 获取loss
            g = float(line.split('Loss_G:')[1].split('Loss_D:')[0].strip())
            d = float(line.split('Loss_D:')[1].strip())
            steps.append((epoch, batch))
            G_loss.append(g)
            D_loss.append(d)

# 横轴为 step（可选：epoch+batch/6250）
step_idx = [e + b/6250 for (e, b) in steps]

plt.figure(figsize=(8,5))
plt.plot(step_idx, G_loss, label='Generator Loss', color='#4C72B0')
plt.plot(step_idx, D_loss, label='Discriminator Loss', color='#C44E52')
plt.xlabel('Epoch (fractional)')
plt.ylabel('Loss')
plt.title('Training Loss Curve (Improved CycleGAN)')
plt.legend()
plt.tight_layout()
plt.savefig('loss_curve_cyclegan_improved.png', dpi=200)
plt.show()
