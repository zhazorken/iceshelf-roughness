import numpy as np, glob, pickle, shelfmelt as sm, experiments as ex
def Cang(p, regime, ang):
    p = dict(sm.P0, **p); x, y, zbar, u = sm.setup(p); dT = sm.thermal_driving(zbar, p)
    sth = np.sin(np.arctan(np.abs(np.gradient(zbar, p["dx"]))))
    f = sm.REGIMES[regime]
    return float(f(dT, np.full_like(sth, np.sin(np.deg2rad(ang))), u, p).mean() / f(dT, sth, u, p).mean())
rows = {}
for f in sorted(glob.glob("results/ens__base__*.npz") + glob.glob("mac/results/ens__base__*.npz") + glob.glob("results/dnsu*")
                + glob.glob("mac/results/contrast__*.npz")):
    try: z = np.load(f, allow_pickle=False); m = pickle.loads(z["meta"].tobytes())
    except Exception: continue
    k = (m["group"], m["label"], m["variant"])
    rows.setdefault(k, []).append((float(z["G"]), m["C"], Cang(m["params"], m["regime"], 5), Cang(m["params"], m["regime"], 15)))
for k, v in sorted(rows.items()):
    v = np.array(v); print(k, "G %.2f  C90 %.1f  C10 %.2f  C20 %.2f  n=%d" % (v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean(), v[:, 3].mean(), len(v)))
