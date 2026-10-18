"""Compare model along-flow relief with BedMachine transects at BedMachine-like resolution (500 m)."""
import numpy as np, pickle, glob, os
from scipy.signal import periodogram
from scipy.ndimage import uniform_filter1d
import obs_transects as OT
DS = 250.0; KMAX = 1 / 1000.0; KMIN = 1 / 20000.0; KRMS = (1 / 20000.0, 1 / 2000.0)
def psd_lines(Z, ds=DS):
    K = None; P = []
    for z in Z:
        z = z[np.isfinite(z)]
        if len(z) < 40: continue
        n = np.arange(len(z)); r = z - np.polyval(np.polyfit(n, z, 2), n)
        k, p = periodogram(r, fs=1 / ds, window="hann", detrend=False, nfft=160)
        K = k[1:]; P.append(p[1:])
    return K, np.array(P)
def band_rms(k, P, k0=KRMS[0], k1=KRMS[1]):
    s = (k >= k0) & (k <= k1); return np.sqrt(P[:, s].sum(1) * (k[1] - k[0]))
def model_lines(r, x0=0.5e3, x1=39.5e3, every=250.0):
    """Model centerlines smoothed to a 500 m footprint and resampled at 250 m, absolute depth."""
    F = r["field"].astype(float) if "field" in r else r["field40"].astype(float)
    dx = float(r["params"]["dx"]) if "field" in r else 40.0
    x = r["x"][:: max(1, int(round(dx / (r["x"][1] - r["x"][0]))))][: F.shape[1]]
    zb = np.interp(x, r["x"], r["zbar"]); F = F + zb[None, :]
    F = uniform_filter1d(F, max(1, int(round(500.0 / dx))), axis=1)
    m = (x >= x0) & (x <= x1); xs = np.arange(x0, x1, every)
    rows = F[:: max(1, int(round(500.0 / dx)))]
    return xs, np.array([np.interp(xs, x[m], row[m]) for row in rows])
def obs():
    out = {}
    for k in OT.LINES:
        s, Z = OT.transects(k)
        Z = Z[:, (s > 0.5e3) & (s < 39.5e3)]; s = s[(s > 0.5e3) & (s < 39.5e3)]
        out[k] = (s, Z)
    return out
if __name__ == "__main__":
    O = obs()
    for k, (s, Z) in O.items():
        kk, P = psd_lines(Z); print(k, "rms 1-20 km band %.1f m" % np.median(band_rms(kk, P)))
    import figures as F
    R = F.load(["results", "mac/results"])
    for g, lab, v in [("showcase", "pig", "shear"), ("showcase", "pig", "conv"), ("ens", "base", "dnsF"), ("crevasse height (steep walls)", "250", "shear"),
                      ("crevasse height (steep walls)", "250", "dnsF"), ("crevasse height (steep walls)", "250", "conv")]:
        rs = [r for r in R if r["group"] == g and r["label"] == lab and r["variant"] == v][:1]
        if not rs: continue
        xs, Z = model_lines(rs[0]); kk, P = psd_lines(Z); print(g, lab, v, "rms %.1f" % np.median(band_rms(kk, P)))
    for f in sorted(glob.glob("results_coarse/build_*.pkl")) + ["results_coarse/mix_convA03show.pkl"]:
        r = pickle.load(open(f, "rb")); xs, Z = model_lines(r); kk, P = psd_lines(Z); print(os.path.basename(f), "rms %.1f" % np.median(band_rms(kk, P)))
