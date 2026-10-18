import sys, pickle, time, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, shelfmelt as sm, experiments as ex
prod = float(sys.argv[1]); v = sys.argv[2] if len(sys.argv) > 2 else "conv"
Ly = float(sys.argv[3]) if len(sys.argv) > 3 else 4e3; dx = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0
regime, q = ex.variant(dict(ex.BASE, Ly=Ly, dx=dx, seed=1, keep_field=True), v + "A")
q.update(pool_prod=prod)
if len(sys.argv) > 5: q.update(snap_years=0.5, snap_dx=40.0)
t = time.time(); out = sm.run(q, regime); out["params"] = q
tag = f"{v}A_prod{prod:g}_Ly{Ly:.0f}_dx{dx:g}"
pickle.dump(out, open(f"results_coarse/prod_{tag}.pkl", "wb"))
print(tag, round(time.time() - t), "s G %.2f inc999 %.1f mbar %.2f" % (out["G"], out["inc999"], out["mbar"]))
