from multiprocessing import Pool
import experiments as ex, runner, os
v3 = [j for j in ex.JOBS if j["host"] == "v3"]
def prio(j):
    if j["group"] == "GL slope (fixed ends)": return (0 if j["seed"] == 1 else 5, j["name"])
    if j["group"] == "smoothing-5m": return (1, j["name"])
    if j["group"] == "ens": return (2, j["name"])
    return (3 if j["seed"] == 1 else 6, j["name"])
jobs = [j for j in sorted(v3, key=prio) if not os.path.exists(runner.path(j))]
print(len(jobs), "jobs; first:", jobs[0]["name"], flush=True)
with Pool(2, maxtasksperchild=4) as p:
    for r in p.imap_unordered(runner.work, jobs): print(*r, flush=True)
