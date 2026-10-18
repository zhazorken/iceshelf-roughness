from coarse import job, work
from multiprocessing import Pool
jobs = []
for v in ("dnsF", "convF"):
    for tc in (2.0, 3.0):
        for na in (2.0, 6.0):
            jobs.append(job(0.0, 1.0, v, Lx=8e3, extra=dict(strat_theta_c=tc, strat_n=8.0, noise_amp=na), tag=f"_tc{tc:g}n8_na{na:g}"))
if __name__ == "__main__":
    with Pool(2) as P:
        for r in P.imap_unordered(work, jobs): print(*r, flush=True)
