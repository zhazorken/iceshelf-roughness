"""2D (plan view) terrace runs with the Monin-Obukhov closure and closed-basin pooling."""
import os, time, pickle, numpy as np
from multiprocessing import Pool
import shelfmelt as sm, experiments as ex
def job(ua, pool_h, Lx=8e3, Ly=1e3, U=200.0, M=0.95, seed=1, tag="", extra=None):
    regime, q = ex.variant(dict(ex.TER, Lx=Lx, Ly=Ly, seed=seed, keep_field=True, U_ice=U), "dns")
    q.update(conv_a=0.76, strat_theta_c=0.0, mo=True, mo_ua=ua, mo_gD=0.01, mo_smin=0.01, mo_delta=0.2,
             pool_h=pool_h, pool_tau=0.1, slope_smooth=10.0, adapt_ref=2000.0, noise_amp=1.0)
    if extra: q.update(extra)
    target = M * U * np.tan(np.deg2rad(4.0)) / sm.SPY
    lo, hi = 1e-3, 1e4
    for _ in range(50):
        mid = np.sqrt(lo * hi); q["melt_mult"] = mid
        lo, hi = (mid, hi) if sm.smooth_mean_melt(q, regime) < target else (lo, mid)
    q["melt_mult"] = np.sqrt(lo * hi); q["years"] = 1.5 * Lx / U
    return f"ua{ua*1e3:g}_ph{pool_h:g}_s{seed}{tag}", q, regime
def work(j):
    name, q, regime = j; f = f"results_coarse/terr2d_{name}.pkl"
    if os.path.exists(f): return name, "skip"
    t = time.time(); out = sm.run(q, regime); out["params"] = q
    pickle.dump(out, open(f, "wb")); return name, f"{time.time()-t:.0f}s"
if __name__ == "__main__":
    jobs = [job(1e-4, 1.0, tag="_ss5", extra=dict(slope_smooth=5.0)), job(1e-4, 1.0, tag="_noad", extra=dict(adapt_ref=0.0)),
            job(1e-4, 2.0), job(1e-4, 1.0, tag="_ss5_noad_M1", M=1.0, extra=dict(slope_smooth=5.0, adapt_ref=0.0))]
    with Pool(2) as P:
        for r in P.imap_unordered(work, jobs): print(*r, flush=True)
