import sys, pickle, time, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, shelfmelt as sm, experiments as ex
tag = sys.argv[1]
show = tag.endswith("show"); tag0 = tag[:-4] if show else tag
base = dict(ex.BASE, Ly=6e3 if show else 4e3, dx=20.0, seed=1, keep_field=True, crev_Hmax=150.0)
if show: base.update(snap_years=0.5, snap_dx=40.0)
if tag0 == "convP2":   regime, q = ex.variant(base, "conv"); q.update(pool_h=2.0)
elif tag0 == "convP5": regime, q = ex.variant(base, "conv"); q.update(pool_h=5.0)
elif tag0 == "convA03": regime, q = ex.variant(base, "convA"); q.update(pool_prod=0.03)
elif tag0 == "dnsQP2": regime, q = ex.variant(ex._q(dict(base, crev_wall=70.0), ex.QUIET), "dns"); q.update(pool_h=2.0)
t = time.time(); out = sm.run(q, regime); out["params"] = q
pickle.dump(out, open(f"results_coarse/mix_{tag}.pkl", "wb"))
print(tag, round(time.time() - t), "s G %.2f inc999 %.1f" % (out["G"], out["inc999"]), flush=True)
