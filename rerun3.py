"""Remaining DNS-law reruns: resolution, slope-scale at 5 m, GL slope with fixed ends (2 km wide)."""
import os, sys, shutil
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import experiments as ex, runner
jobs = []
for j in ex.JOBS:
    if not j["variant"].startswith("dns") or j["host"] in ("zc", "zm"): continue
    if j["group"] not in ("resolution", "GL slope (fixed ends)"): continue
    j = dict(j)
    if j["group"] == "GL slope (fixed ends)": j["params"] = dict(j["params"], Ly=2e3)
    jobs.append(j)
for j in jobs:
    for d in ("results", "mac/results"):
        f = os.path.join(d, j["name"] + ".npz")
        if os.path.exists(f):
            dst = os.path.join("results_offset", d.replace("/", "_") + "__" + j["name"] + ".npz")
            if not os.path.exists(dst): shutil.move(f, dst)
jobs.sort(key=ex.cost, reverse=True)
if __name__ == "__main__":
    print(len(jobs), "jobs", flush=True)
    with Pool(int(sys.argv[1]), maxtasksperchild=2) as P:
        for i, (name, dt, st) in enumerate(P.imap_unordered(runner.work, jobs), 1):
            print(f"[{i}/{len(jobs)}] {name} {dt/60:.1f} min {st}", flush=True)
