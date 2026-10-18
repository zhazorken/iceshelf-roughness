import pickle, glob, numpy as np, sys
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def risers(d, theta=20, smooth=20):
    p = d["params"]; dx = p["dx"]; F = d["field"] + d["zbar"][None, :]
    zs = gaussian_filter(F, smooth / dx, mode=("wrap", "nearest"))
    gx = np.gradient(zs, dx, axis=1); mask = gx > np.tan(np.deg2rad(theta))
    lab, n = ndimage.label(mask, structure=[[0,0,0],[1,1,1],[0,0,0]])
    idx = np.arange(1, n + 1)
    h = np.asarray(ndimage.maximum(zs, lab, idx)) - np.asarray(ndimage.minimum(zs, lab, idx))
    xc = np.asarray(ndimage.mean(np.broadcast_to(d["x"], F.shape), lab, idx))
    return xc, h, F
def summary(names, out):
    fig, axs = plt.subplots(2, len(names), figsize=(3.2 * len(names), 5.5), squeeze=False)
    for j, n in enumerate(names):
        d = pickle.load(open(f"results_coarse/{n}.pkl", "rb")); xc, h, F = risers(d)
        big = h > 3; x = d["x"]
        bins = np.arange(0, x.max() + 1, 1000)
        hb = [np.percentile(h[big & (xc >= a) & (xc < a + 1000)], 90) if (big & (xc >= a) & (xc < a + 1000)).sum() > 3 else np.nan for a in bins[:-1]]
        nb = [(big & (xc >= a) & (xc < a + 1000)).sum() / (F.shape[0]) for a in bins[:-1]]
        print(f"{n:28s} h90 by km:", " ".join(f"{v:5.1f}" for v in hb), "| risers/km/row:", " ".join(f"{v:4.1f}" for v in nb))
        ax = axs[0, j]
        for iy in (10, 25, 40): ax.plot(x / 1e3, F[iy], lw=0.7)
        ax.set_title(n, fontsize=7)
        axs[1, j].plot((bins[:-1] + 500) / 1e3, hb, "o-"); axs[1, j].set_ylabel("riser h90 (m)"); axs[1, j].set_xlabel("km")
    fig.tight_layout(); fig.savefig(out, dpi=70)
if __name__ == "__main__":
    summary(sys.argv[2:], sys.argv[1])
