import re
import matplotlib.pyplot as plt
import argparse

# 
# LOG_PATH = '../output/cyclegan/2025-04-26_15-49-40_TPS+WGAN-GP/train_monitor_log.txt'

def parse_log(log_path):
    batches = []
    mean_fakeB = []
    std_fakeB = []
    mean_fakeA = []
    std_fakeA = []
    # ，
    pattern = re.compile(r'\[Epoch (\d+)\]\[Batch (\d+)\] mean_fakeB: ([\d.eE\-]+), std_fakeB: ([\d.eE\-]+), mean_fakeA: ([\d.eE\-]+), std_fakeA: ([\d.eE\-]+)')
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epoch = int(m.group(1))
                batch = int(m.group(2))
                batches.append((epoch, batch))
                mean_fakeB.append(float(m.group(3)))
                std_fakeB.append(float(m.group(4)))
                mean_fakeA.append(float(m.group(5)))
                std_fakeA.append(float(m.group(6)))
    return batches, mean_fakeB, std_fakeB, mean_fakeA, std_fakeA

def plot_mode_collapse_points(log_path):
    batches, mean_fakeB, std_fakeB, mean_fakeA, std_fakeA = parse_log(log_path)
    if not batches:
        print("No valid log lines found. Check the log format or path.")
        return
    x = list(range(len(batches)))
    # std_fakeB<0.05std_fakeA<0.05
    collapse_idx_fakeB = [i for i, v in enumerate(std_fakeB) if v < 0.05]
    collapse_idx_fakeA = [i for i, v in enumerate(std_fakeA) if v < 0.05]
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    # fakeB
    if collapse_idx_fakeB:
        axes[0].scatter([x[i] for i in collapse_idx_fakeB], [std_fakeB[i] for i in collapse_idx_fakeB], color='r', label='std_fakeB < 0.05', s=1)
    axes[0].set_ylim(0, 0.05)  # 
    axes[0].set_xlim(0, 250000)
    axes[0].set_ylabel('Std (fakeB)')
    axes[0].legend()
    axes[0].set_title('Potential Mode Collapse Points for fakeB (std < 0.05)')
    # fakeA
    if collapse_idx_fakeA:
        axes[1].scatter([x[i] for i in collapse_idx_fakeA], [std_fakeA[i] for i in collapse_idx_fakeA], color='b', label='std_fakeA < 0.05', s=1)
    axes[1].set_ylim(0, 0.05)  # 
    axes[1].set_xlim(0, 250000)
    axes[1].set_xlabel('Batch')
    axes[1].set_ylabel('Std (fakeA)')
    axes[1].legend()
    axes[1].set_title('Potential Mode Collapse Points for fakeA (std < 0.05)')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot mode collapse points from train_monitor_log.txt')
    parser.add_argument('log_path', type=str, help='Path to train_monitor_log.txt')
    args = parser.parse_args()
    plot_mode_collapse_points(args.log_path)
