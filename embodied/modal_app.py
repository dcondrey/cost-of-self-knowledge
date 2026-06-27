"""Run the self-introspection benchmark on a Modal cloud container (Linux substrate).

Usage:
    pip install modal && modal token new        # one-time auth, if not already set up
    modal run modal_app.py

Prints the machine's JSON; save it as embodied/results/<name>.json and run aggregate.py.
"""
import sys

import modal

app = modal.App("introspection-benchmark")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy")
         .add_local_file("benchmark.py", remote_path="/root/benchmark.py"))


@app.function(image=image, cpu=2.0)
def run():
    import subprocess
    r = subprocess.run(["python", "/root/benchmark.py"], capture_output=True, text=True, cwd="/root")
    return r.stdout or r.stderr


@app.local_entrypoint()
def main():
    sys.stdout.write(run.remote() + "\n")
