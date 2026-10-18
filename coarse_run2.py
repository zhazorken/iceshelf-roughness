from coarse import job, work
from multiprocessing import Pool
jobs = []
for tc in (4.0, 6.0, 8.0, 12.0):
    for b in (0.0, 0.5):
        jobs.append(job(b, 1.0, "dnsF", extra=dict(strat_theta_c=tc), tag=f"_tc{tc:g}"))
for tc in (6.0, 12.0):
    jobs.append(job(0.0, 1.0, "plumeF", extra=dict(strat_theta_c=tc), tag=f"_tc{tc:g}"))
if __name__ == "__main__":
    with Pool(2) as P:
        for r in P.imap_unordered(work, jobs): print(*r, flush=True)
