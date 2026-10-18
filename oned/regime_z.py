"""Regime maps with the revised DNS law (max of theta-convection and shear) + MO closure + pooling."""
import pickle, sys
from multiprocessing import Pool
from regime import run, classify, BASE
B = dict(BASE, law="dnsmax", M=0.5)
def one(kw):
    o = run(**dict(B, **{k: v for k, v in kw.items() if k != "tag"})); m = classify(o); m["zb"] = o["zb"].astype("float32"); m["zbar"] = o["zbar"]; m["x"] = o["x"]
    return kw, m
if __name__ == "__main__":
    g1 = [dict(mo_ua=ua, pool_h=ph, rng=0) for ua in (1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3) for ph in (0.0, 0.5, 1.0, 2.0)]
    g2 = [dict(mo_ua=ua, pool_h=1.0, M=M, rng=0, tag="M") for M in (0.2, 0.3, 0.7, 0.95) for ua in (1e-4, 3e-4, 1e-3)]
    res = []
    with Pool(int(sys.argv[1])) as P:
        for r in P.imap_unordered(one, g1 + g2):
            res.append(r); print(len(res), {k: v for k, v in r[0].items()}, r[1]["regime"], round(r[1]["rms"], 1), round(r[1]["lvl"], 2), flush=True)
            pickle.dump(res, open("regime_z.pkl", "wb"))
