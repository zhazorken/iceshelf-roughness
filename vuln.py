"""Rank melt mechanisms by the deepest basal incision relative to local ice thickness."""
import numpy as np, pickle, figures as F, shelfmelt as sm
def incH(rec, x0=5e3, x1=38e3):
    p = rec["params"]; z = rec["field40"].astype(float); x = rec["x"]; zb = rec["zbar"]
    s = max(1, int(round(40.0 / p["dx"]))); xs = x[::s][: z.shape[1]]; H = sm.thickness(zb, p)[::s][: z.shape[1]]
    hp = sm.highpass(z, 40.0); m = (xs > x0) & (xs < x1)
    q = hp[:, m] / H[None, m]
    tot = z[:, m] / H[None, m]
    return float(q.max()), float(np.percentile(q, 99.9)), float(tot.max())
if __name__ == "__main__":
    R = F.load(["results", "mac/results"]); out = {}
    for r in R:
        key = None
        if r["group"] == "ens": key = ("40", r["variant"])
        elif r["group"] == "crevasse height (steep walls)": key = (r["label"] + "s", r["variant"])
        elif r["group"] == "showcase": key = ("show", r["variant"])
        if key is None: continue
        out.setdefault(key, []).append(incH(r))
    for f, k in (("results_coarse/mix_convA03show.pkl", ("40-150", "convA03")),):
        d = pickle.load(open(f, "rb")); d["params"] = dict(sm.P0, **d["params"]); out.setdefault(k, []).append(incH(d))
    pickle.dump(out, open("results_coarse/vuln.pkl", "wb"))
    for k in sorted(out, key=lambda k: (k[0], -np.mean([v[0] for v in out[k]]))):
        a = np.array(out[k]); print(k, len(a), "max %.3f  p999 %.3f  totmax %.3f" % tuple(a.mean(0)))
