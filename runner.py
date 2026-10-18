"""Resumable parallel runner.
Usage: python runner.py HOST NPROC [--reverse] [--any]
  HOST: mac | cloud | all ; results go to results/<name>.npz (existing files are skipped).
  --any runs other hosts' unfinished jobs too (for load balancing; order reversed).
Native macOS, all cores:  python runner.py all 10
"""
import os, sys, time, pickle
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
import experiments as ex
import shelfmelt as sm

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def path(j): return os.path.join(RES, j["name"] + ".npz")

def work(j):
    if os.path.exists(path(j)): return j["name"], 0.0, "skip"
    t = time.time()
    try:
        out = sm.run(j["params"], j["regime"])
    except Exception as e:
        return j["name"], time.time() - t, f"FAIL {e!r}"
    meta = {k: j[k] for k in ("name", "group", "label", "variant", "regime", "seed")}
    meta["params"] = {k: v for k, v in j["params"].items() if not isinstance(v, np.ndarray)}
    meta["C"] = ex.slope_contrast(j["params"], j["regime"])
    tmp = path(j) + ".tmp.npz"
    np.savez_compressed(tmp, meta=np.frombuffer(pickle.dumps(meta), dtype=np.uint8),
                        **{k: np.asarray(v) for k, v in out.items()})
    os.replace(tmp, path(j))
    return j["name"], time.time() - t, "ok"

if __name__ == "__main__":
    host, nproc = sys.argv[1], int(sys.argv[2])
    os.makedirs(RES, exist_ok=True)
    jobs = [j for j in ex.JOBS if host in ("all", j["host"])]
    if "--any" in sys.argv:
        jobs = jobs + [j for j in ex.JOBS if j["host"] != host][::-1]
    jobs = [j for j in jobs if not os.path.exists(path(j))]
    if "--any" not in sys.argv:
        jobs.sort(key=ex.cost, reverse=True)          # longest first
    print(f"{len(jobs)} jobs on {nproc} procs", flush=True)
    t0 = time.time()
    with Pool(nproc, maxtasksperchild=4) as pool:
        for i, (name, dt, st) in enumerate(pool.imap_unordered(work, jobs), 1):
            print(f"[{i}/{len(jobs)} {(time.time()-t0)/60:6.1f} min] {name} {dt/60:5.1f} min {st}", flush=True)
