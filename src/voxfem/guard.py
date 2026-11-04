"""Run-time guards for GPU measurements on a Windows (WDDM) workstation.

Three failure modes have contaminated measurements in this project:
  1. a second GPU job running concurrently (inflates wall times),
  2. VRAM oversubscription: since NVIDIA driver 536.40 the Windows driver
     does not fail an oversized CUDA allocation but backs it with system RAM
     ("shared GPU memory"), silently slowing kernels by 5-1000x,
  3. other processes (the desktop compositor, remote desktop) holding VRAM
     that cudaMemGetInfo does not account for.

The guards below detect all three.  Shared-memory spill is read from the
Windows "GPU Process Memory" performance counters for this process; any
growth beyond a small threshold marks the measurement invalid.
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import cupy as cp

SPILL_LIMIT_BYTES = 256 * 2**20


def _ps(cmd: str, timeout=60) -> str:
    return subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd],
                                   text=True, timeout=timeout).strip()


def gpu_process_memory():
    """{pid: {"dedicated": bytes, "shared": bytes}} from WDDM counters."""
    cmd = ("$d=(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage').CounterSamples;"
           "$s=(Get-Counter '\\GPU Process Memory(*)\\Shared Usage').CounterSamples;"
           "$o=@();foreach($x in $d){$o+=@{n=$x.InstanceName;k='d';v=$x.CookedValue}};"
           "foreach($x in $s){$o+=@{n=$x.InstanceName;k='s';v=$x.CookedValue}};"
           "$o|ConvertTo-Json -Compress")
    try:
        rows = json.loads(_ps(cmd))
    except Exception:
        return {}
    out = {}
    for r in rows if isinstance(rows, list) else [rows]:
        name = r["n"]
        try:
            pid = int(name.split("_")[1])
        except Exception:
            continue
        d = out.setdefault(pid, {"dedicated": 0, "shared": 0})
        d["dedicated" if r["k"] == "d" else "shared"] += int(r["v"])
    return out


def python_gpu_processes(min_bytes=200 * 2**20):
    """Other python.exe processes holding GPU memory."""
    try:
        txt = _ps("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                  "Select-Object -ExpandProperty ProcessId")
        pids = {int(x) for x in txt.split()}
    except Exception:
        pids = set()
    pids.discard(os.getpid())
    mem = gpu_process_memory()
    return {p: mem[p] for p in pids if p in mem and mem[p]["dedicated"] > min_bytes}


def device_state():
    q = "name,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu,pstate"
    try:
        s = subprocess.check_output(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                                    text=True, timeout=20).strip()
        keys = q.split(",")
        return dict(zip(keys, [v.strip() for v in s.split(",")]))
    except Exception as ex:
        return {"error": repr(ex)}


class SpillWatch:
    """Records this process's WDDM dedicated/shared usage at start and on
    every check; `spilled` is True once shared usage grew past the limit."""

    def __init__(self, limit=SPILL_LIMIT_BYTES):
        self.pid = os.getpid()
        self.limit = limit
        self.samples = []
        self.base = self._sample("start")

    def _sample(self, tag):
        m = gpu_process_memory()
        mine = m.get(self.pid, {"dedicated": 0, "shared": 0})
        others = sum(v["dedicated"] for p, v in m.items() if p != self.pid)
        rec = {"tag": tag, "t": time.time(), "dedicated": mine["dedicated"],
               "shared": mine["shared"], "others_dedicated": others}
        self.samples.append(rec)
        return rec

    def check(self, tag="check"):
        rec = self._sample(tag)
        return rec

    @property
    def max_shared_growth(self):
        return max(s["shared"] for s in self.samples) - self.base["shared"]

    @property
    def spilled(self):
        return self.max_shared_growth > self.limit

    def report(self):
        return {"pid": self.pid, "limit_bytes": self.limit,
                "max_shared_growth_bytes": int(self.max_shared_growth),
                "spilled": bool(self.spilled), "samples": self.samples}


def effective_free_bytes():
    """Conservative free VRAM: min(cudaMemGetInfo free, total minus every
    other process's WDDM dedicated usage minus this process's pool)."""
    free, total = cp.cuda.runtime.memGetInfo()
    m = gpu_process_memory()
    pid = os.getpid()
    others = sum(v["dedicated"] for p, v in m.items() if p != pid)
    pool = cp.get_default_memory_pool().total_bytes()
    wddm_free = total - others - max(pool, m.get(pid, {}).get("dedicated", 0))
    return int(min(free, wddm_free)), {"cuda_free": int(free), "total": int(total),
                                       "others_dedicated": int(others), "wddm_free": int(wddm_free)}


def require_budget(need_bytes, margin_bytes=1536 * 2**20):
    eff, info = effective_free_bytes()
    ok = need_bytes + margin_bytes <= eff
    info.update({"need": int(need_bytes), "margin": int(margin_bytes), "effective_free": eff, "ok": ok})
    return ok, info


def require_exclusive():
    others = python_gpu_processes()
    if others:
        raise RuntimeError(f"another python process holds GPU memory: {others}")


class ClockPoller:
    """Background nvidia-smi sampling (every `ms` ms) of SM/memory clocks,
    utilization, power and temperature while a measurement runs.  Samples
    taken between kernels show idle clocks, so summaries use only samples
    with utilization >= busy_pct."""

    FIELDS = ["clocks.sm", "clocks.mem", "utilization.gpu", "power.draw", "temperature.gpu"]

    def __init__(self, ms=100):
        import threading
        self.samples = []
        self._p = subprocess.Popen(["nvidia-smi", f"--query-gpu={','.join(self.FIELDS)}",
                                    "--format=csv,noheader,nounits", f"-lms", str(ms)],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._t = threading.Thread(target=self._read, daemon=True)
        self._t.start()

    def _read(self):
        for line in self._p.stdout:
            try:
                v = [float(x) for x in line.strip().split(",")]
                self.samples.append(dict(zip(self.FIELDS, v)))
            except Exception:
                pass

    def stop(self):
        self._p.terminate()
        try:
            self._p.wait(timeout=5)
        except Exception:
            self._p.kill()

    def summary(self, busy_pct=50):
        import numpy as np
        busy = [s for s in self.samples if s["utilization.gpu"] >= busy_pct]
        out = {"n_samples": len(self.samples), "n_busy": len(busy)}
        for f in self.FIELDS:
            vals = [s[f] for s in busy]
            if vals:
                out[f + "_median"] = float(np.median(vals))
                out[f + "_min"] = float(np.min(vals))
                out[f + "_max"] = float(np.max(vals))
        return out


class PeakMemory:
    """Peak pool usage sampled at explicit points, plus device high-water."""

    def __init__(self):
        self.pool = cp.get_default_memory_pool()
        self.peak_used = 0
        self.peak_total = 0
        self.peak_device = 0
        free, total = cp.cuda.runtime.memGetInfo()
        self.base_device = total - free

    def sample(self):
        self.peak_used = max(self.peak_used, self.pool.used_bytes())
        self.peak_total = max(self.peak_total, self.pool.total_bytes())
        free, total = cp.cuda.runtime.memGetInfo()
        self.peak_device = max(self.peak_device, total - free)

    def report(self):
        return {"peak_pool_used_bytes": int(self.peak_used),
                "peak_pool_reserved_bytes": int(self.peak_total),
                "peak_device_used_bytes": int(self.peak_device),
                "baseline_device_used_bytes": int(self.base_device)}
