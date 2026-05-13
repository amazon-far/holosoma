import subprocess
import os

env = os.environ.copy()
env['VK_ICD_FILENAMES'] = '/usr/share/vulkan/icd.d/nvidia_icd.json'

cmd = "source /home/ubuntu22/.holosoma_deps/miniconda3/bin/activate hssim && python src/holosoma/holosoma/train_cvae.py --checkpoint_dir ./logs/WholeBodyTracking/20260329_023617-t1_23dof_wbt_fast_sac_manager-locomotion/model_0400000.pt --episodes 2 --epochs 1 --latent_dim 16 --batch_size 256 --save_dir ./logs/cvae_models/"
print(f"Running: {cmd}")
proc = subprocess.Popen(cmd, shell=True, executable='/bin/bash', env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

try:
    for line in iter(proc.stdout.readline, ''):
        print(line, end='')
except KeyboardInterrupt:
    proc.kill()
proc.wait()
