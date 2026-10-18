"""Runs with relief and melt snapshots for Movie S1 (shear, sin^(2/3), terrace case)."""
import sys, pickle, time, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, shelfmelt as sm, experiments as ex, terr2d_z as T
case = sys.argv[1]
if case in ("shear", "conv"):
    regime, q = ex.variant(dict(ex.BASE, Ly=6e3, dx=20.0, seed=1, snap_years=0.5, snap_dx=40.0), case)
else:
    kw = {"terr": {}, "saw": dict(ph=0.0), "smo": dict(ua=1e-3)}[case]
    q, regime = T.job(Ly=3e3, **kw); q.update(snap_years=0.5, snap_dx=20.0)
t = time.time(); out = sm.run(q, regime); out["params"] = q
pickle.dump({k: out[k] for k in ("x", "y", "zbar", "snap_t", "snaps", "snap_melt", "params", "G")},
            open(f"results_coarse/movie_{case}.pkl", "wb"))
print(case, "done", round(time.time() - t), "s", "G", round(out["G"], 2))
