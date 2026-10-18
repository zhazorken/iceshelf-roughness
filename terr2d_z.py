"""Plan-view terrace run with the revised DNS law in quiet water (u* = u_a), MO closure and pooling."""
import pickle, time, numpy as np
import shelfmelt as sm, experiments as ex
def job(ua=3e-4, ph=1.0, M=0.5, Lx=8e3, Ly=1e3, U=200.0, seed=1):
    base = dict(ex.TER, Lx=Lx, Ly=Ly, seed=seed, keep_field=True, U_ice=U, ustar_gl=ua, ustar_out=ua)
    regime, q = ex.variant(base, "dns")
    q.update(mo=True, mo_ua=ua, mo_gD=0.01, mo_smin=0.01, mo_delta=0.2, pool_h=ph, pool_tau=0.1,
             slope_smooth=10.0, adapt_ref=2000.0, noise_amp=1.0)
    target = M * U * np.tan(np.deg2rad(4.0)) / sm.SPY; lo, hi = 1e-3, 1e4
    for _ in range(50):
        mid = np.sqrt(lo * hi); q["melt_mult"] = mid
        lo, hi = (mid, hi) if sm.smooth_mean_melt(q, regime) < target else (lo, mid)
    q["melt_mult"] = np.sqrt(lo * hi); q["years"] = 1.5 * Lx / U
    return q, regime
if __name__ == "__main__":
    q, regime = job(); t = time.time(); out = sm.run(q, regime); out["params"] = q
    pickle.dump(out, open("results_coarse/terr2dZ_ua0.3_ph1_M0.5.pkl", "wb")); print("done", time.time() - t)
