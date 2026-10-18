import numpy as np, figures as F, shelfmelt as sm, collections, pickle, glob
def aniso(z, dx, x, x0=10e3, x1=38e3):
    hp = sm.highpass(z.astype(float), dx); m = (x > x0) & (x < x1); hp = hp[:, m]
    gx = np.diff(hp, axis=1) / dx; gy = np.diff(hp, axis=0) / dx
    return float(np.var(gx) / np.var(gy))
R = F.load(["results", "mac/results"]); d = collections.defaultdict(list)
for r in R:
    if r["group"] != "ens": continue
    s = max(1, int(round(40.0 / r["params"]["dx"]))); x = r["x"][::s][: r["field40"].shape[1]]
    d[r["variant"]].append(aniso(r["field40"], 40.0, x))
for k in sorted(d, key=lambda k: np.mean(d[k])): print(f"{k:8s} {np.mean(d[k]):6.2f}")
for f in ["results_coarse/terrW_terr_s2.pkl", "results_coarse/terrW_saw_s2.pkl", "results_coarse/terrW_smo_s2.pkl", "results_coarse/mix_convA03show.pkl"]:
    r = pickle.load(open(f, "rb")); dx = float(r["params"]["dx"]); z = r["field"]
    if "terrW" in f:
        Fz = z.astype(float) + r["zbar"][None, :]; z = Fz - np.polyval(np.polyfit(r["x"], Fz.mean(0), 1), r["x"])[None, :]
        print(f, round(aniso(z, dx, r["x"], 1e3, 7.5e3), 2))
    else: print(f, round(aniso(z, dx, r["x"]), 2))
