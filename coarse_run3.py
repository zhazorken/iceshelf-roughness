from coarse import job, work
from multiprocessing import Pool
jobs = []
for tc in (1.0, 2.0):
    for M in (1.0, 0.9):
        for b in (0.0, 0.33):
            jobs.append(job(b, M, "dnsF", Lx=8e3, extra=dict(strat_theta_c=tc, strat_n=8.0), tag=f"_tc{tc:g}n8"))
if __name__ == "__main__":
    with Pool(2) as P:
        for r in P.imap_unordered(work, jobs): print(*r, flush=True)
