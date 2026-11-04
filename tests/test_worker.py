"""GPU worker: events, frames, cancel, supersede, warm start across rungs, invalid specs."""
import time

import numpy as np
import pytest

from agentto import library
from agentto.worker import GPUWorker, ladder


@pytest.fixture(scope="module")
def worker():
    w = GPUWorker()
    ev = w.get(timeout=120)
    assert ev and ev["type"] == "ready"
    yield w
    w.close()


def collect(w, job_id, until=("done", "error"), timeout=300):
    out, t0 = [], time.time()
    while time.time() - t0 < timeout:
        e = w.get(timeout=1)
        if e is None or e.get("job") != job_id:
            continue
        out.append(e)
        if e["type"] in until:
            return out
    raise TimeoutError(job_id)


def test_ladder_doubles():
    r = ladder(library.cantilever(40))
    assert len(r) >= 4
    for a, b in zip(r, r[1:]):
        assert all(y == 2 * x for x, y in zip(a["grid"], b["grid"]))


def test_job_streams_progress_and_frames(worker):
    jid = worker.submit({"spec": library.cantilever(12).model_dump(), "frame_every_s": 0.0, "frame_max_dim": 16})
    ev = collect(worker, jid)
    kinds = [e["type"] for e in ev]
    done = ev[-1]
    assert kinds[0] == "started" and done["type"] == "done" and done["status"] == "converged"
    assert kinds.count("progress") == done["iterations"]
    fr = [e for e in ev if e["type"] == "frame"]
    assert fr and fr[-1].get("final")
    shape = fr[-1]["shape"]
    assert len(fr[-1]["data"]) == int(np.prod(shape)) and max(shape) <= 16
    assert done["compliance_phys"] > 0


def test_cancel_and_supersede(worker):
    a = worker.submit({"spec": library.cantilever(24).model_dump()})
    t0 = time.time()
    while time.time() - t0 < 120:                 # wait until it is running
        e = worker.get(timeout=1)
        if e and e.get("job") == a and e["type"] == "progress" and e["it"] >= 2:
            break
    b = worker.submit({"spec": library.cantilever(8).model_dump(), "supersede": True})
    c = worker.submit({"spec": library.cantilever(8).model_dump(), "supersede": True})
    ea = collect(worker, a)
    assert ea[-1]["status"] == "cancelled"
    eb = collect(worker, b, until=("done", "error", "skipped"))
    assert eb[-1]["type"] == "skipped"
    ec = collect(worker, c)
    assert ec[-1]["status"] == "converged"


def test_warm_start_chain(worker):
    spec = library.cantilever(10).model_dump()
    j1 = worker.submit({"spec": spec})
    d1 = collect(worker, j1)[-1]
    j2 = worker.submit({"spec": spec, "h": 1.0 / 20, "warm_from": j1})
    ev2 = collect(worker, j2)
    assert ev2[0]["warm"] is True
    assert ev2[-1]["status"] == "converged" and ev2[-1]["iterations"] < d1["iterations"]


def test_invalid_spec_rejected_before_gpu(worker):
    d = library.bridge(10).model_dump()
    d["supports"] = [x for x in d["supports"] if x["id"] == "roller"]
    jid = worker.submit({"spec": d})
    ev = collect(worker, jid)
    assert ev[-1]["type"] == "error" and ev[-1]["kind"] == "invalid_spec"
