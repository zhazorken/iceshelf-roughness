"""Step coarsening test: riser melt scaled by (riser height / h0)^beta on a uniform 4 deg base."""
import sys, os, time, pickle, numpy as np
from multiprocessing import Pool
import shelfmelt as sm, experiments as ex

def job(beta, M, v, Lx=10e3, years=None, extra=None, tag=""):
    regime, q = ex.variant(dict(ex.TER, Lx=Lx, Ly=500.0, seed=1, keep_field=True), v)
    if v.endswith("F"): q["strat_theta_c"] = 2.0
    q["pool_tau"] = 0.1
    q.update(face_beta=beta)
    if extra: q.update(extra)
    q = ex.with_M(q, regime, M)
    if years: q["years"] = years
    name = f"b{beta:g}_M{M:g}_{v}_L{Lx/1e3:g}{tag}"
    return name, q, regime

def work(a):
    name, q, regime = a
    f = f"results_coarse/{name}.pkl"
    if os.path.exists(f): return name, "skip"
    t = time.time(); out = sm.run(q, regime)
    out["params"] = q
    pickle.dump(out, open(f, "wb")); return name, f"{time.time()-t:.0f}s"

if __name__ == "__main__":
    jobs = [job(b, M, v) for v in ("dnsF",) for M in (1.0, 0.7) for b in (-0.25, 0.0, 0.33, 0.5)]
    with Pool(2) as P:
        for r in P.imap_unordered(work, jobs): print(*r, flush=True)
