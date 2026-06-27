"""The body: real interoceptive senses of the machine this agent runs on.

Samples genuine hardware telemetry (no simulation) into a structured BodyReading the agent
can feel. Two tiers, like a real organism:
  - FAST senses (load, memory, the agent's own CPU/RAM footprint): cheap, sampled every tick.
  - SLOW senses (drive temperatures + wear via smartctl, network latency): expensive, cached
    and refreshed on a slow cadence (these physically change slowly anyway).

Each signal is tagged recoverable (heat/effort: comes back) or irreversible (wear/age: never
does) - the cost-of-thinking vs cost-of-remembering distinction. The agent's OWN footprint is
read separately from the ambient system, so "my heat" can be told from "the room's heat".

A sense that cannot be read is marked unavailable; it is never faked.

Requires macOS tools: sysctl, vm_stat, ps, ping (all base), and smartctl (brew smartmontools)
for drive temps/wear. CPU/GPU chip temperature and watts need `sudo powermetrics` and are left
out of this no-sudo reader (add later behind an explicit grant).
"""
import logging
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("body")

SLOW_REFRESH_S = 30.0          # drive temp / wear change slowly; refresh cadence
NET_REFRESH_S = 5.0
PING_HOST = "1.1.1.1"


def _run(cmd, timeout=8.0):
    """Run a command, return stdout or None on any failure (recorded, not swallowed)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 or r.stdout else None
    except (subprocess.SubprocessError, OSError) as e:
        log.debug(f"cmd failed {cmd}: {e}")
        return None


def _sysctl(name):
    out = _run(["sysctl", "-n", name], timeout=2.0)
    return out.strip() if out else None


@dataclass
class BodyReading:
    t: float
    temps_c: dict = field(default_factory=dict)        # organ -> Celsius (recoverable-ish)
    load1: float = None
    load_norm: float = None                            # load1 / ncpu  (recoverable effort)
    self_cpu_pct: float = None                         # the agent's OWN cpu (self vs ambient)
    self_rss_mb: float = None
    mem_used_frac: float = None                        # strain: how cramped (recoverable)
    mem_compressed_frac: float = None
    aging: dict = field(default_factory=dict)          # drive -> wear dict (IRREVERSIBLE)
    world_latency_ms: float = None                     # sense of the outside / lifeline
    unavailable: list = field(default_factory=list)

    def summary(self):
        temps = " ".join(f"{k}={v}C" for k, v in self.temps_c.items()) or "n/a"
        ag = "; ".join(
            f"{k}:{v.get('power_on_hours','?')}h"
            + (f",used{v['percent_used']}%" if v.get('percent_used') is not None else "")
            + (f",realloc{v['reallocated']}" if v.get('reallocated') is not None else "")
            for k, v in self.aging.items()) or "n/a"
        return (
            f"temps[{temps}]  load={self.load1}(x{self.load_norm:.2f}/cpu)  "
            f"self_cpu={self.self_cpu_pct}% self_rss={self.self_rss_mb}MB  "
            f"mem_used={_pct(self.mem_used_frac)} compressed={_pct(self.mem_compressed_frac)}  "
            f"world={self.world_latency_ms}ms\n"
            f"  aging(irreversible): {ag}"
            + (f"\n  unavailable: {','.join(self.unavailable)}" if self.unavailable else ""))


def _pct(x):
    return f"{x*100:.0f}%" if x is not None else "n/a"


class Body:
    def __init__(self, pid=None):
        import os
        self.pid = pid or os.getpid()
        self.ncpu = int(_sysctl("hw.ncpu") or 1)
        self.memsize = int(_sysctl("hw.memsize") or 0)
        self.page_size = 16384
        self.drives = self._discover_drives()
        self._slow_cache = {"temps": {}, "aging": {}, "t": 0.0}
        self._net_cache = {"latency": None, "t": 0.0}

    # ---- drive discovery (run once) ----
    def _discover_drives(self):
        drives = {}
        if not shutil.which("smartctl"):
            return drives
        out = _run(["diskutil", "list"], timeout=5.0) or ""
        candidates = re.findall(r"^(/dev/disk\d+) \((?:internal|external), physical\)", out, re.M)
        for dev in candidates:
            for flag in ([], ["-d", "sat"]):
                txt = _run(["smartctl", "-a", *flag, dev], timeout=10.0)
                if txt and ("Temperature" in txt):
                    name = self._drive_name(txt, dev)
                    drives[name] = {"dev": dev, "flag": flag}
                    break
        return drives

    @staticmethod
    def _drive_name(txt, dev):
        m = re.search(r"(?:Device Model|Model Number):\s*(.+)", txt)
        short = dev.replace("/dev/", "")
        if m:
            return f"{short}:{m.group(1).strip().split()[0][:10]}"
        return short

    # ---- fast senses (every tick) ----
    def _fast(self, r):
        la = _sysctl("vm.loadavg")               # "{ 2.73 2.90 3.19 }"
        if la:
            nums = re.findall(r"[\d.]+", la)
            if nums:
                r.load1 = float(nums[0])
                r.load_norm = r.load1 / self.ncpu
        else:
            r.unavailable.append("load")

        ps = _run(["ps", "-o", "%cpu=,rss=", "-p", str(self.pid)], timeout=2.0)
        if ps and ps.strip():
            parts = ps.split()
            try:
                r.self_cpu_pct = float(parts[0])
                r.self_rss_mb = round(float(parts[1]) / 1024.0, 1)
            except (ValueError, IndexError):
                r.unavailable.append("self")
        else:
            r.unavailable.append("self")

        vm = _run(["vm_stat"], timeout=2.0)
        if vm and self.memsize:
            pg = {k: int(v) for k, v in re.findall(r"Pages ([\w\s]+?):\s+(\d+)\.", vm)}
            total = self.memsize / self.page_size
            free = pg.get("free", 0) + pg.get("speculative", 0)
            comp = pg.get("occupied by compressor", 0)
            r.mem_used_frac = round(1.0 - free / total, 3)
            r.mem_compressed_frac = round(comp / total, 3)
        else:
            r.unavailable.append("memory")

    # ---- slow senses (cached) ----
    def _slow(self, r, now):
        if now - self._slow_cache["t"] > SLOW_REFRESH_S or not self._slow_cache["temps"]:
            temps, aging = {}, {}
            for name, d in self.drives.items():
                txt = _run(["smartctl", "-a", *d["flag"], d["dev"]], timeout=10.0)
                if not txt:
                    continue
                tc = self._parse_temp(txt)
                if tc is not None:
                    temps[name] = tc
                aging[name] = self._parse_wear(txt)
            if temps or aging:
                self._slow_cache = {"temps": temps, "aging": aging, "t": now}
        r.temps_c = dict(self._slow_cache["temps"])
        r.aging = dict(self._slow_cache["aging"])
        if not r.temps_c:
            r.unavailable.append("temps")

        if now - self._net_cache["t"] > NET_REFRESH_S:
            out = _run(["ping", "-c", "1", "-t", "2", PING_HOST], timeout=4.0)
            lat = None
            if out:
                m = re.search(r"=\s*[\d.]+/([\d.]+)/", out)
                if m:
                    lat = round(float(m.group(1)), 1)
            self._net_cache = {"latency": lat, "t": now}
        r.world_latency_ms = self._net_cache["latency"]
        if r.world_latency_ms is None:
            r.unavailable.append("world")

    @staticmethod
    def _parse_temp(txt):
        m = re.search(r"Temperature:\s+(\d+)\s+Celsius", txt)          # NVMe
        if m:
            return int(m.group(1))
        m = re.search(r"Temperature_Celsius.*?-\s+(\d+)", txt)          # ATA SMART table
        if m:
            return int(m.group(1))
        m = re.search(r"Current Drive Temperature:\s+(\d+)", txt)
        return int(m.group(1)) if m else None

    @staticmethod
    def _parse_wear(txt):
        w = {}
        m = re.search(r"Percentage Used:\s+(\d+)%", txt)
        w["percent_used"] = int(m.group(1)) if m else None
        m = re.search(r"Power[_ ]On[_ ]Hours.*?-\s+(\d+)", txt) or re.search(r"Power On Hours:\s+([\d,]+)", txt)
        w["power_on_hours"] = int(m.group(1).replace(",", "")) if m else None
        m = re.search(r"Reallocated_Sector_Ct.*?-\s+(\d+)", txt)
        w["reallocated"] = int(m.group(1)) if m else None
        return w

    def read(self):
        now = time.time()
        r = BodyReading(t=now)
        self._fast(r)
        self._slow(r, now)
        return r


def main():
    body = Body()
    log.info(f"discovered organs: {', '.join(body.drives) or 'none (smartctl?)'}  "
             f"cpu={body.ncpu} ram={body.memsize // (1024**3)}GB")
    log.info("=" * 70)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for i in range(n):
        r = body.read()
        log.info(f"[{i}] " + r.summary())
        if i < n - 1:
            time.sleep(2.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
