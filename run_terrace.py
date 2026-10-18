import os, sys, time
from multiprocessing import Pool
import experiments as ex, runner
jobs = [j for j in ex.JOBS if j["group"].startswith("terrace")]
with Pool(2) as p:
    for name, dt, st in p.imap_unordered(runner.work, jobs):
        print(name, round(dt, 1), st, flush=True)
