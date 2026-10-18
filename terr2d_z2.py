import pickle, sys, time
import shelfmelt as sm
from terr2d_z import job
for ua, ph in ((3e-4, 0.0), (1e-3, 1.0)):
    q, regime = job(ua=ua, ph=ph); t = time.time(); out = sm.run(q, regime); out["params"] = q
    pickle.dump(out, open(f"results_coarse/terr2dZ_ua{ua*1e3:g}_ph{ph:g}_M0.5.pkl", "wb")); print(ua, ph, time.time() - t, out["G"], flush=True)
