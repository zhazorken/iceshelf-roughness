from multiprocessing import Pool
import experiments as ex, runner, os
jobs = sorted([j for j in ex.JOBS if j["host"] == "v3"], key=ex.cost)   # cheapest first (Mac does costliest first)
jobs = [j for j in jobs if not os.path.exists(runner.path(j))]
with Pool(2, maxtasksperchild=4) as p:
    for r in p.imap_unordered(runner.work, jobs): print(*r, flush=True)
