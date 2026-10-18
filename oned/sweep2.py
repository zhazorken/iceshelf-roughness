import numpy as np, itertools, pickle, time
from multiprocessing import Pool
from sweep import metrics
from stair1d import run
def metrics2(o):
    m = metrics(o); x, zb = o["x"], o["zb"]; dx = x[1]-x[0]
    from scipy.ndimage import gaussian_filter1d
    g = np.gradient(gaussian_filter1d(zb, 2, mode="nearest"), dx); sel = (x > 1000) & (x < 7000)
    m["flat15"] = float((np.abs(g[sel]) < np.tan(np.deg2rad(1.5))).mean()); m["steep30"] = float((g[sel] > np.tan(np.deg2rad(30))).mean())
    return m
def one(kw):
    o = run(**dict(dict(noise=1.0, years=12, adapt=2000.0, thc=2.0, n=8), **kw)); return kw, metrics2(o)
if __name__ == "__main__":
    grid = [dict(law=a, M=M, dTdz_frac=tf, rng=r) for a in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.76, "sin23"]
            for M in [0.8, 1.0] for tf in [0.0, -0.5] for r in [0, 1]]
    grid += [dict(law=a, thc=0.0, M=1.0, rng=r) for a in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.76, "sin23"] for r in [0, 1]]
    t = time.time(); res = []
    with Pool(2) as P:
        for r in P.imap_unordered(one, grid): res.append(r)
    pickle.dump(res, open("sweep2.pkl", "wb")); print(len(res), f"{time.time()-t:.0f}s")
