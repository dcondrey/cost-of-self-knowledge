"""Quick probe: does a Modal Linux container expose /sys thermal that RESPONDS to CPU load?

Reads thermal zones, burns 4 cores for 10s, reads again. If after > before, the cloud substrate's
thermal is real and responsive (usable for the self-measurement test); if zones are empty or static,
container thermal is masked and the irreducibility test needs bare metal.
"""
import sys

import modal

app = modal.App("thermal-probe")
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, cpu=4.0)
def probe():
    import glob
    import multiprocessing as mp
    import time

    def temps():
        out = []
        for f in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
            try:
                out.append(int(open(f).read().strip()) / 1000.0)
            except (OSError, ValueError):
                pass
        return out

    def busy(end):
        x = 1.0
        while time.time() < end:
            for _ in range(20000):
                x = (x * 1.0000001 + 1.0) % 1e6

    before = temps()
    end = time.time() + 10
    ps = [mp.Process(target=busy, args=(end,)) for _ in range(4)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    return {"zones_before": before, "zones_after": temps()}


@app.local_entrypoint()
def main():
    sys.stdout.write(str(probe.remote()) + "\n")
