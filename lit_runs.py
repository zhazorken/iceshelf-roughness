import os, sys, pickle, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import shelfmelt as sm, experiments as ex
s = int(sys.argv[1])
regime, q = ex.variant(dict(ex.BASE, seed=s, keep_field=True, dns_lit=True), "dns")
print("C10", ex.slope_contrast(q, regime) if False else "", flush=True)
t = time.time(); out = sm.run(q, regime); out["params"] = q
pickle.dump(out, open(f"results_coarse/dnslit_s{s}.pkl", "wb")); print("dnslit", s, round(time.time() - t), "G %.2f" % out["G"], flush=True)
