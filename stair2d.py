"""2D (plan-view) staircase runs: offset-linear convection with flat-ice fraction a, sharp meltwater cutoff on flats."""
import os, time, pickle
from multiprocessing import Pool
import shelfmelt as sm, experiments as ex
def job(a, M=1.0, Lx=8e3, Ly=1e3, adapt=2000.0, tag="", beta=0.0, extra=None):
    regime, q = ex.variant(dict(ex.TER, Lx=Lx, Ly=Ly, seed=1, keep_field=True, snap_years=2.0), "dnsF")
    q.update(conv_a=a, strat_theta_c=2.0, strat_n=8.0, pool_tau=0.1, adapt_ref=adapt, face_beta=beta)
    if extra: q.update(extra)
    q = ex.with_M(q, regime, M); q["years"] = 2.0 * Lx / q["U_ice"]
    return f"a{a:g}_M{M:g}_ad{adapt:g}_b{beta:g}{tag}", q, regime
def work(j):
    name, q, regime = j; f = f"results_coarse/2d_{name}.pkl"
    if os.path.exists(f): return name, "skip"
    t = time.time(); out = sm.run(q, regime); out["params"] = q
    pickle.dump(out, open(f, "wb")); return name, f"{time.time()-t:.0f}s"
if __name__ == "__main__":
    jobs = [job(0.02, tag="_wide", extra=dict(crev_len=(1000.0, 1000.0))), job(0.05, beta=0.33, tag="_wide", extra=dict(crev_len=(1000.0, 1000.0))),
            job(0.2, tag="_wide", extra=dict(crev_len=(1000.0, 1000.0))), job(0.76, tag="_wide", extra=dict(crev_len=(1000.0, 1000.0)))]
    with Pool(2) as P:
        for r in P.imap_unordered(work, jobs): print(*r, flush=True)
