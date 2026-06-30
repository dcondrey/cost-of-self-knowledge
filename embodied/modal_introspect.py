"""Run the introspective-calibration scale sweep on Modal (A100-80GB).

Why Modal over Colab for the large run:
  - 72B in 4-bit needs ~40GB VRAM + activations; a free Colab A100 is 40GB and OOMs. A100-80GB fits it.
  - persistent HF cache volume: each model downloads once, reused across runs.
  - no session timeout: the full 0.5B->72B curve runs unattended.

    pip install modal && modal setup        # one time
    modal run modal_introspect.py            # full curve, default models
    modal run modal_introspect.py --models "Qwen/Qwen2.5-32B-Instruct,Qwen/Qwen2.5-72B-Instruct"
"""

import modal

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("torch", "transformers", "accelerate", "bitsandbytes", "numpy")
)
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
results = modal.Volume.from_name("introspect-results", create_if_missing=True)
app = modal.App("introspect-calibration", image=image)


@app.function(
    gpu="A100-80GB",
    timeout=14400,
    volumes={"/root/.cache/huggingface": cache, "/out": results},
)
def run(models: str, script: str):
    import subprocess

    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/dcondrey/cost-of-self-knowledge",
            "/tmp/repo",
        ],
        check=True,
    )
    subprocess.run(
        ["python", script, "--models", models],
        cwd="/tmp/repo/embodied",
        check=True,
    )
    subprocess.run(
        "cp /tmp/repo/embodied/results/*.json /out/",
        shell=True,
        check=False,
    )
    cache.commit()
    results.commit()


@app.local_entrypoint()
def main(
    models: str = (
        "Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct,"
        "Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-14B-Instruct,Qwen/Qwen2.5-32B-Instruct,"
        "Qwen/Qwen2.5-72B-Instruct"
    ),
    script: str = "introspect_calibration.py",
):
    run.remote(models, script)
