from multiprocessing import Pool
import experiments as ex, runner, os
jobs = [j for j in ex.JOBS if j["host"] == "v4" and not os.path.exists(runner.path(j))]
jobs.sort(key=lambda j: j["group"] != "layers-main")
with Pool(2, maxtasksperchild=4) as p:
    for r in p.imap_unordered(runner.work, jobs): print(*r, flush=True)
