"""Ice-shelf base transects from BedMachine Antarctica v3 (500 m grid) for Pine Island and Dotson."""
import numpy as np
from scipy.ndimage import map_coordinates
D = np.load("obs/amundsen_bedmachine_v3.npz")
X, Y = D["x"], D["y"]; BASE = (D["surface"] - D["thickness"]).astype(float); FL = D["mask"] == 3
LINES = {"pig": ((-1591e3, -262e3), (-1613e3, -328e3)), "dotson": ((-1545e3, -633e3), (-1602e3, -660e3))}
def sample(field, xs, ys, order=1):
    ix = (xs - X[0]) / (X[1] - X[0]); iy = (ys - Y[0]) / (Y[1] - Y[0])
    return map_coordinates(field, [iy, ix], order=order, mode="nearest")
def transects(key, offsets=np.arange(-4e3, 4.01e3, 1e3), ds=250.0):
    (x0, y0), (x1, y1) = LINES[key]; L = np.hypot(x1 - x0, y1 - y0); ux, uy = (x1 - x0) / L, (y1 - y0) / L
    s = np.arange(0, L, ds); out = []
    for o in offsets:
        xs = x0 + ux * s - uy * o; ys = y0 + uy * s + ux * o
        z = sample(BASE, xs, ys); fl = sample(FL.astype(float), xs, ys, 0) > 0.5
        out.append(np.where(fl, z, np.nan))
    return s, np.array(out)
if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    for ax, k in zip(axs, LINES):
        s, Z = transects(k)
        for z in Z: ax.plot(s / 1e3, z, lw=0.6)
        ax.set_title(f"{k}: floating fraction {np.isfinite(Z).mean():.2f}")
    fig.tight_layout(); fig.savefig("/tmp/claude-0/-home-claude/bf9473f0-4bf8-5f9c-95fd-0632fbd61e01/scratchpad/tr.png", dpi=80)
