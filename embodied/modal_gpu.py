"""Run the self-measurement irreducibility test on a Modal GPU (T4) — a genuinely different substrate.

GPU temperature (nvidia-smi) perturbed by GPU compute; the same hysteresis/floor/extrapolation
analysis. Robust long holds. Returns stdout + the result JSON.

    /tmp/modalenv/bin/modal run modal_gpu.py
"""
import sys

import modal

app = modal.App("selfmeasure-gpu")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch")
         .add_local_file("selfmeasure.py", remote_path="/root/selfmeasure.py"))


@app.function(image=image, gpu="T4", timeout=1800)
def run():
    import os
    import subprocess
    env = dict(os.environ, SELFMEASURE_HOLD="45", SELFMEASURE_SETTLE="10")
    os.makedirs("/root/results", exist_ok=True)
    import torch
    head = f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} " \
           f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}\n"
    r = subprocess.run([sys.executable, "/root/selfmeasure.py"], capture_output=True, text=True,
                       cwd="/root", env=env)
    try:
        res = open("/root/results/selfmeasure-linux-gpu_compute.json").read()
    except OSError:
        res = "(no result file written)"
    return head + r.stdout + "\n--STDERR--\n" + r.stderr[-2000:] + "\n--RESULT--\n" + res


@app.local_entrypoint()
def main():
    sys.stdout.write(run.remote() + "\n")
