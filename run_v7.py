from multiprocessing import Pool
import experiments as ex, runner, os
jobs = [j for j in ex.JOBS if j["host"] in ("v7",) and not os.path.exists(runner.path(j))]
jobs.sort(key=lambda j: j["group"] != "stair")
print(len(jobs), [j["group"] for j in jobs if j["group"]!="stair"], flush=True)
with Pool(2, maxtasksperchild=4) as p:
    for r in p.imap_unordered(runner.work, jobs): print(*r, flush=True)
