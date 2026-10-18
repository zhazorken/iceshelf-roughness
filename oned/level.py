import numpy as np, pickle, sys
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter1d
from stair1d import run
from sweep2 import metrics2
def tread_stats(o):
    x, zb = o["x"], o["zb"]; dx = x[1] - x[0]
    g = np.gradient(gaussian_filter1d(zb, 2, mode="nearest"), dx); ang = np.degrees(np.arctan(g))
    sel = (x > 1000) & (x < 7000) & (np.abs(ang) < 10)          # tread-like cells
    a = ang[sel]
    return dict(tread_med=float(np.median(np.abs(a))), lvl05=float((np.abs(a) < 0.5).mean()),
                tilt=float(np.median(a)), tread_frac=float(sel.mean() / ((x > 1000) & (x < 7000)).mean()))
def one(kw):
    o = run(**dict(dict(law=0.02, thc=2.0, n=8, noise=1.0, years=12, adapt=2000.0, M=1.0), **kw))
    m = metrics2(o); m.update(tread_stats(o)); return kw, m
if __name__ == "__main__":
    grid = [dict(pool_h=ph, M=M, adapt=ad, rng=r) for ph in (0.0, 0.5, 2.0, 5.0) for M in (0.9, 1.0, 1.1) for ad in (0.0, 2000.0) for r in (0, 1)]
    res = []
    with Pool(2) as P:
        for r in P.imap_unordered(one, grid): res.append(r)
    pickle.dump(res, open("level.pkl", "wb"))
