from multiprocessing import Pool
import experiments as ex, runner
jobs = [j for j in ex.JOBS if j["group"].startswith("resolution") and j["label"] == "40"]
with Pool(2) as p:
    for r in p.imap_unordered(runner.work, jobs): print(*r, flush=True)
