"""Mac-share (zm) jobs of the revised DNS law, run here with a 2 km wide domain to save time."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import experiments as ex, runner
jobs = []
for j in ex.JOBS:
    if j["host"] != "zm": continue
    j = dict(j); j["params"] = dict(j["params"], Ly=2e3); jobs.append(j)
pri = ["crevasseheight(steepwalls)", "ens__base__dnsA", "ens__base__dnsP", "channels"]
jobs.sort(key=lambda j: min([i for i, p in enumerate(pri) if j["name"].startswith(p)] or [99]))
if __name__ == "__main__":
    jobs = [j for j in jobs if not os.path.exists(runner.path(j))]
    print(len(jobs), "jobs", flush=True)
    with Pool(int(sys.argv[1]), maxtasksperchild=2) as P:
        for i, (name, dt, st) in enumerate(P.imap_unordered(runner.work, jobs), 1):
            print(f"[{i}/{len(jobs)}] {name} {dt/60:.1f} min {st}", flush=True)
