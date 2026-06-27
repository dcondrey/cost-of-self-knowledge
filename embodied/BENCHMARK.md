# Cross-substrate self-introspection benchmark

Measures, on each machine in its own CPU-seconds, the cost of self-introspection at three
modalities (shallow in-process / rich in-process / deep subprocess telemetry) versus the cost of a
unit of work, giving `look_frac` per modality. Run it on as many real substrates as possible; each
emits `benchmark_result.json`. Collect them into `results/` and run `aggregate.py` for the
cross-machine table + figure.

## Machines

**1. This Mac (done)** — `python benchmark.py` → saved as `results/01-mac-m4.json`.

**2. Modal (Linux cloud)**
```
pip install modal && modal token new      # one-time auth
modal run modal_app.py                     # prints JSON
```
Save the JSON to `results/02-modal-linux.json`.

**3. Google Colab** — paste this one cell, run it, copy the JSON it prints into
`results/03-colab.json` (set `REPO` to your pushed repo URL first):
```python
!git clone -q REPO repo && pip -q install numpy
import subprocess; print(subprocess.run(["python","repo/embodied/benchmark.py"],
                                         capture_output=True, text=True, cwd="repo/embodied").stdout)
```
(If no repo yet, paste the contents of `benchmark.py` into a cell and run it directly.)

**4. GitHub CI (ubuntu + macOS + windows, automatic)** — pushing this repo triggers
`.github/workflows/benchmark.yml`, which runs the benchmark on all three OS runners and uploads
each `benchmark_result.json` as an artifact (Actions tab → run → Artifacts). Download them into
`results/` (e.g. `04-ci-ubuntu.json`, `05-ci-macos.json`, `06-ci-windows.json`).

**5. Your other computer** — after pulling the repo:
```
cd embodied && pip install numpy && python benchmark.py
```
Save to `results/07-<name>.json`.

## Aggregate
```
python aggregate.py        # -> benchmark_cross_machine.csv, fig3_cross_machine.png
```

Notes: `numpy` is the only dependency (used for the work-unit; falls back to a pure-Python loop if
absent). `proc_inproc` (rich in-process telemetry) is Linux-only — it fills in the middle modality
on the cloud/CI Linux runners. Deep-telemetry cost on shared CI runners may differ from bare metal;
that variation across substrates is itself part of the result.
