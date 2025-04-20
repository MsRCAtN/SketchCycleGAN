import subprocess
import sys
import os
import time

def run_script(script_path):
    print(f"\n==== Running: {script_path} ====")
    start = time.time()
    result = subprocess.run([sys.executable, script_path], stdout=sys.stdout, stderr=sys.stderr)
    end = time.time()
    elapsed = end - start
    print(f"==== Finished: {script_path} | Time: {elapsed/60:.2f} min ====")
    return elapsed, result.returncode

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scripts = [
        os.path.join(base_dir, 'train_cyclegan.py'),
        os.path.join(base_dir, 'train_cyclegan_orig.py'),
        os.path.join(base_dir, 'train_pix2pix.py'),
    ]
    total_time = 0
    for script in scripts:
        elapsed, code = run_script(script)
        total_time += elapsed
        if code != 0:
            print(f"Error: {script} exited with code {code}")
            sys.exit(code)
    print(f"\n==== All training finished! Total time: {total_time/60:.2f} min ====")
