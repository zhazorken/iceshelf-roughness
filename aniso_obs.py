"""Plan-view anisotropy of BedMachine ice-shelf bases vs model runs at a 500 m footprint."""
import numpy as np, pickle, glob, os
from scipy.ndimage import gaussian_filter, map_coordinates
import obs_transects as OT
def aniso_box(key, half_w=6e3, L=40e3, ds=500.0):
    (x0, y0), (x1, y1) = OT.LINES[key]; Lt = np.hypot(x1 - x0, y1 - y0); ux, uy = (x1 - x0) / Lt, (y1 - y0) / Lt
    s = np.arange(1e3, min(L, Lt), ds); o = np.arange(-half_w, half_w + 1, ds)
    S, Oo = np.meshgrid(s, o); xs = x0 + ux * S - uy * Oo; ys = y0 + uy * S + ux * Oo
    ix = (xs - OT.X[0]) / (OT.X[1] - OT.X[0]); iy = (ys - OT.Y[0]) / (OT.Y[1] - OT.Y[0])
    Z = map_coordinates(OT.BASE, [iy, ix], order=1); F = map_coordinates(OT.FL.astype(float), [iy, ix], order=0) > 0.5
    Z = np.where(F, Z, np.nan); Z = Z - np.nanmean(Z, 0)[None, :]       # remove along-flow mean draft profile
    gx = np.diff(Z, axis=1) / ds; gy = np.diff(Z, axis=0) / ds
    return float(np.nanvar(gx) / np.nanvar(gy)), float(np.isfinite(Z).mean())
def aniso_model(r, ds=500.0, x0=1e3, x1=39e3):
    F = r["field"].astype(float); dx = float(r["params"]["dx"]); n = int(round(ds / dx))
    ny, nx = (F.shape[0] // n) * n, (F.shape[1] // n) * n
    B = F[:ny, :nx].reshape(ny // n, n, nx // n, n).mean((1, 3)); x = r["x"][:nx].reshape(-1, n).mean(1)
    B = B - B.mean(0)[None, :]; m = (x > x0) & (x < x1); B = B[:, m]
    gx = np.diff(B, axis=1) / ds; gy = np.diff(B, axis=0) / ds
    return float(np.var(gx) / np.var(gy))
if __name__ == "__main__":
    for k in OT.LINES: print(k, aniso_box(k))
    for f in sorted(glob.glob("results_coarse/build_b?.pkl")): print(os.path.basename(f), round(aniso_model(pickle.load(open(f, "rb"))), 2))
    import figures as F
    for nm in ["showcase__pig__shear__s1", "showcase__pig__conv__s1"]:
        z = np.load(f"results/{nm}.npz"); r = {k: z[k] for k in ("field", "x")}; r["params"] = {"dx": 10.0}; print(nm, round(aniso_model(r), 2))
    r = pickle.load(open("results_coarse/mix_convA03show.pkl", "rb")); print("pond", round(aniso_model(r), 2))
