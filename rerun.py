"""Run the revised-DNS-law jobs (hosts zc then zm) in priority order; old offset-law files are moved aside."""
import os, sys, shutil
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import experiments as ex, runner
nproc = int(sys.argv[1])
jobs = [j for j in ex.JOBS if j["host"] == "zc"] + [j for j in ex.JOBS if j["host"] == "zm"]
os.makedirs("results_offset", exist_ok=True)
for j in jobs:
    for d in ("results", "mac/results"):
        f = os.path.join(d, j["name"] + ".npz")
        if os.path.exists(f) and not os.path.exists(f + ".new"):
            dst = os.path.join("results_offset", d.replace("/", "_") + "__" + j["name"] + ".npz")
            if not os.path.exists(dst): shutil.move(f, dst)
first = [j for j in jobs if j["name"] in ("ens__base__dns__s1", "ens__base__dnsQ__s1", "ens__base__dnsF__s1")]
show = [j for j in jobs if j["group"] == "showcase"]
rest = [j for j in jobs if j not in first and j not in show]
order = first + show[:1] + rest[:6] + show[1:] + rest[6:]
if __name__ == "__main__":
    print(len(order), "jobs", flush=True)
    with Pool(nproc, maxtasksperchild=2) as P:
        for i, (name, dt, st) in enumerate(P.imap(runner.work, order), 1):
            print(f"[{i}/{len(order)}] {name} {dt/60:.1f} min {st}", flush=True)
