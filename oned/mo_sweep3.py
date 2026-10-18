import pickle, numpy as np
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter1d
from stair1d import run
from sweep2 import metrics2
def tstats(o):
    x, zb = o["x"], o["zb"]; dx = x[1]-x[0]
    ang = np.degrees(np.arctan(np.gradient(gaussian_filter1d(zb, 1, mode="nearest"), dx)))
    sel = (x > 3000) & (x < 7500); a = ang[sel]; t = np.abs(a) < 3
    return dict(tread_tilt=float(np.median(a[t])), tread_abs=float(np.median(np.abs(a[t]))),
                lvl05=float((np.abs(a) < 0.5).mean()), riser=float((a > 20).mean()))
def one(kw):
    o = run(**dict(dict(law="dns", thc=0.0, mo=True, mo_ua=1e-4, U=200.0, years=60.0, noise=1.0, rng=0), **kw))
    m = metrics2(o); m.update(tstats(o)); return kw, m
if __name__ == "__main__":
    grid = [dict(ds=ds, mo_smin=sm, mo_gD=gD, adapt=ad, M=M, pool_h=ph)
            for ds in (5.0, 20.0) for sm in (0.01, 0.02) for gD in (0.01, 0.03) for ad in (0.0, 2000.0) for M in (0.95, 1.0) for ph in (0.0, 1.0)]
    res = []
    with Pool(2) as P:
        for r in P.imap_unordered(one, grid): res.append(r); print(len(res), flush=True)
    pickle.dump(res, open("mo_sweep3.pkl", "wb"))
