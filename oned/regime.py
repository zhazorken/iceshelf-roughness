"""Regime map: ambient friction velocity (tides) x pooling depth scale, MO closure + DNS offset-linear law."""
import numpy as np, pickle, sys
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter1d
from stair1d import run
BASE = dict(law="dns", thc=0.0, mo=True, U=200.0, years=60.0, noise=1.0, ds=5.0, mo_smin=0.01, mo_gD=0.01, adapt=2000.0, M=0.95)
def classify(o):
    x, zb, zbar = o["x"], o["zb"], o["zbar"]; dx = x[1] - x[0]
    sel = (x > 3000) & (x < 7500)
    ang = np.degrees(np.arctan(np.gradient(gaussian_filter1d(zb, 1), dx)))[sel]
    zp = (zb - zbar)[sel]; hp = zp - gaussian_filter1d(zp, 2000 / dx / 2.355, mode="nearest")
    rms = float(hp.std()); riser = float((ang > 20).mean()); flatcells = np.abs(ang) < 15
    lvl = float((np.abs(ang[flatcells]) < 0.5).mean()) if flatcells.any() else 0.0
    tilt = float(np.median(ang[flatcells])) if flatcells.any() else np.nan
    back = float((ang < -2).mean())
    regime = "smooth" if rms < 2.5 else ("terraces" if lvl > 0.3 else "sawtooth")
    return dict(rms=rms, riser=riser, lvl=lvl, tilt=tilt, back=back, regime=regime)
def one(kw):
    o = run(**dict(BASE, **kw)); m = classify(o); m["zb"] = o["zb"].astype(np.float32); m["zbar"] = o["zbar"]; m["x"] = o["x"]
    return kw, m
if __name__ == "__main__":
    grid = [dict(mo_ua=ua, pool_h=ph, rng=r) for ua in (5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3)
            for ph in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0) for r in (0, 1)]
    res = []
    with Pool(2) as P:
        for r in P.imap_unordered(one, grid): res.append(r); print(len(res), r[0], r[1]["regime"], round(r[1]["rms"], 1), round(r[1]["lvl"], 2), flush=True)
    pickle.dump(res, open("regime.pkl", "wb"))
