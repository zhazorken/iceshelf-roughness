import numpy as np, pickle
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter1d, label
import scipy.ndimage as nd
from stair1d import run
from regime import BASE
def steps(o, x0, x1):
    x, zb = o["x"], o["zb"]; dx = x[1] - x[0]
    zs = gaussian_filter1d(zb, 1); g = np.gradient(zs, dx)
    lab, n = label(g > np.tan(np.deg2rad(15)))
    if n == 0: return np.nan, np.nan, 0
    idx = np.arange(1, n + 1)
    h = np.asarray(nd.maximum(zs, lab, idx)) - np.asarray(nd.minimum(zs, lab, idx)); xc = np.asarray(nd.mean(x, lab, idx))
    k = (h > 2) & (xc > x0) & (xc < x1); xs = np.sort(xc[k])
    return float(np.median(h[k])) if k.sum() else np.nan, float(np.median(np.diff(xs))) if k.sum() > 2 else np.nan, int(k.sum())
def one(kw):
    o = run(**dict(BASE, mo_ua=1e-4, pool_h=1.0, **dict(kw, years=1.5 * 8000 / kw.get("U", 200.0))))
    return kw, dict(near=steps(o, 1000, 3000), far=steps(o, 4500, 7500))
if __name__ == "__main__":
    grid = [dict(alpha=a, rng=r) for a in (2.0, 4.0, 8.0) for r in (0, 1)] + [dict(U=U, rng=r) for U in (100.0, 500.0) for r in (0, 1)]
    with Pool(2) as P: res = P.map(one, grid)
    pickle.dump(res, open("scales.pkl", "wb"))
    for kw, m in res: print(kw, "near h/L/n", m["near"], "far h/L/n", m["far"])
