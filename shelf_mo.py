"""Main 40 km shelf with the DNS-fit law + Monin-Obukhov closure + closed-basin pooling (terrace regime)."""
import os, sys, time, pickle, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
import shelfmelt as sm, experiments as ex
def params(dx=20.0, Ly=1e3, ua=1e-4, ph=1.0, seed=1, keep=True):
    base = dict(ex.BASE, dx=dx, Ly=Ly, seed=seed, keep_field=keep, ustar_gl=ua, ustar_out=ua)   # quiet water: u* = ambient
    regime, q = ex.variant(base, "dns")
    q.update(mo=True, mo_ua=ua, mo_gD=0.01, mo_smin=0.01, mo_delta=0.2, strat_theta_c=0.0,
             pool_h=ph, pool_tau=0.1)
    lo, hi = 1e-2, 1e4
    for _ in range(40):
        mid = np.sqrt(lo*hi); q["dns_scale"] = mid
        lo, hi = (mid, hi) if sm.smooth_mean_melt(q, regime) < ex.M_CONV0 else (lo, mid)
    q["dns_scale"] = np.sqrt(lo*hi)
    return regime, q
if __name__ == "__main__":
    dx, Ly, ua, ph = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    regime, q = params(dx, Ly, ua, ph)
    name = f"shelfmoZ_dx{dx:g}_Ly{Ly:g}_ua{ua*1e3:g}_ph{ph:g}"
    print(name, "dns_scale", q["dns_scale"], flush=True)
    t = time.time(); out = sm.run(q, regime); out["params"] = q
    out["C"] = ex.slope_contrast(q, regime)
    pickle.dump(out, open(f"results_coarse/{name}.pkl", "wb"))
    print(name, f"{time.time()-t:.0f}s", "G", out.get("G"), "inc", out.get("inc999"), flush=True)
