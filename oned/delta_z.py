"""Sensitivity of the 1D terrace regime to the MO stability depth delta and to the noise seed."""
import pickle
from regime import run, classify, BASE
B = dict(BASE, law="dnsmax", M=0.5)
G = [dict(mo_ua=ua, pool_h=1.0, mo_delta=d, rng=0) for d in (0.05, 0.1, 0.4, 0.8) for ua in (2e-4, 3e-4, 5e-4)]
G += [dict(mo_ua=ua, pool_h=ph, rng=r) for r in (1, 2) for ua in (2e-4, 3e-4, 5e-4) for ph in (0.0, 1.0)]
res = []
for kw in G:
    o = run(**dict(B, **kw)); m = classify(o)
    res.append((kw, {k: m[k] for k in ("regime", "rms", "lvl") if k in m}))
    print(kw, m["regime"], round(m["rms"], 1), round(m.get("lvl", float("nan")), 2), flush=True)
    pickle.dump(res, open("delta_z.pkl", "wb"))
