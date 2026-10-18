"""Build-up of melt-mechanism complexity on the 40 km shelf, toward Pine Island-like roughness."""
import os, sys, pickle, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import shelfmelt as sm, experiments as ex
TALL = dict(crev_Hmax=250.0, crev_wall=70.0)
def case(k):
    seed = int(k.split("s")[1]) if "s" in k else 1; k = k.split("s")[0]
    b = dict(ex.BASE, Ly=4e3, dx=20.0, seed=seed, keep_field=True, snap_years=0.5, snap_dx=40.0)
    if k == "b1": return ex.variant(b, "shear")
    if k == "b2": return ex.variant(dict(b, **TALL), "shear")
    if k == "b3": return ex.variant(ex._q(dict(b, **TALL), ex.QUIET), "dns")
    if k == "b4": return ex.variant(ex._q(dict(b, **TALL), ex.QUIET), "dnsF")
    if k == "b5":
        r, q = ex.variant(ex._q(dict(b, **TALL), ex.QUIET), "dnsF"); q.update(pool_mode="accumulate", pool_prod=0.03); return r, q
    if k == "b6":
        r, q = ex.variant(ex._q(dict(b, **TALL), ex.QUIET), "dns"); ua = q["ustar_gl"]
        q.update(mo=True, mo_ua=3e-4, mo_gD=0.01, mo_smin=0.01, mo_delta=0.2, pool_mode="accumulate", pool_prod=0.03); return r, q
    if k == "b7":
        r, q = ex.variant(ex._q(dict(b, **TALL), ex.QUIET), "dnsF"); q.update(pool_mode="accumulate", pool_prod=0.01); return r, q
    if k == "b8":
        r, q = ex.variant(ex._q(dict(b, crev_Hmax=300.0, crev_wall=70.0), ex.QUIET), "dnsF"); q.update(pool_mode="accumulate", pool_prod=0.01); return r, q
def work(k):
    f = f"results_coarse/build_{k}.pkl"
    if os.path.exists(f): return k, 0, "skip"
    t = time.time(); regime, q = case(k); out = sm.run(q, regime); out["params"] = q
    pickle.dump(out, open(f, "wb")); return k, round(time.time() - t), "G %.2f inc %.0f" % (out["G"], out["inc999"])
if __name__ == "__main__":
    ks = sys.argv[2:] or ["b2", "b3", "b4", "b5", "b6", "b1"]
    with Pool(int(sys.argv[1])) as P:
        for r in P.imap_unordered(work, ks): print(r, flush=True)
