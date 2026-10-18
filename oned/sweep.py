import numpy as np, itertools, pickle, time
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter1d, label
import scipy.ndimage as nd
from stair1d import run
def metrics(o):
    x, zb, zbar = o["x"], o["zb"], o["zbar"]; dx = x[1] - x[0]
    zs = gaussian_filter1d(zb, 2, mode="nearest"); g = np.gradient(zs, dx)
    lab, nl = label(g > np.tan(np.deg2rad(20)))
    idx = np.arange(1, nl + 1)
    h = (np.asarray(nd.maximum(zs, lab, idx)) - np.asarray(nd.minimum(zs, lab, idx))) if nl else np.array([])
    xc = np.asarray(nd.mean(x, lab, idx)) if nl else np.array([])
    big = h > 3
    def med(a, b):
        s = big & (xc >= a) & (xc < b); return float(np.median(h[s])) if s.sum() >= 2 else 0.0
    flat = float((np.abs(g[(x > 1000) & (x < 7000)]) < np.tan(np.deg2rad(1))).mean())
    thin = float(np.median((zb - zbar)[(x > 5000) & (x < 7500)]))
    return dict(h_1_3=med(1000, 3000), h_5_7=med(5000, 7000), nsteps=int(big.sum()), flat=flat, thin=thin)
def one(kw):
    try: o = run(**dict(dict(noise=1.0, years=12, dTdz_frac=-0.5), **kw)); return kw, metrics(o)
    except Exception as e: return kw, dict(err=repr(e))
if __name__ == "__main__":
    grid = []
    for law, thc, n, M, tfg, noise in itertools.product(["dns", 0.5, 0.25, "sin23"], [1.0, 2.0, 3.0], [2, 8], [0.8, 0.9, 1.0], [0.0, -0.5], [0.5, 2.0]):
        grid.append(dict(law=law, thc=thc, n=n, M=M, dTdz_frac=tfg, noise=noise))
    t = time.time(); res = []
    with Pool(2) as P:
        for r in P.imap_unordered(one, grid): res.append(r)
    pickle.dump(res, open("sweep.pkl", "wb")); print(len(res), f"{time.time()-t:.0f}s")
