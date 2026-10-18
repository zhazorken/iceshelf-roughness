"""Main figure: building melt-mechanism complexity toward the Pine Island Ice Shelf base (BedMachine)."""
import os, pickle, numpy as np, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from main_figs import *
from figures import save, INK, MUTED, GRID
import obs_compare as OC, obs_transects as OT

def logbin(k, p, n=10):
    e = np.logspace(np.log10(k[0]), np.log10(k[-1]), int(np.log10(k[-1] / k[0]) * n) + 1)
    i = np.digitize(k, e); u = np.unique(i)
    return np.array([k[i == j].mean() for j in u]), np.array([p[i == j].mean() for j in u])
hpf = lambda z, ds: z - gaussian_filter1d(z, 5000.0 / ds, mode="nearest")

O = OC.obs(); MOD = {}; SPREAD = {}
for k_, lab, c_ in BUILD:
    fs = [f for f in (f"results_coarse/build_{k_}.pkl", f"results_coarse/build_{k_}s2.pkl", f"results_coarse/build_{k_}s3.pkl") if os.path.exists(f)]
    if not fs: continue
    per = []; Pall = []
    for i_, f in enumerate(fs):
        r = pickle.load(open(f, "rb")); xs_, Z_ = OC.model_lines(r); kk, PP = OC.psd_lines(Z_)
        per.append(float(np.median(OC.band_rms(kk, PP)))); Pall.append(PP)
        if i_ == 0: xs, Z = xs_, Z_
    MOD[k_] = (xs, Z, kk, np.vstack(Pall), float(np.mean(per)), lab, c_); SPREAD[k_] = (min(per), max(per), len(per))
OBS = {}
for k_, lab, c_ in (("pig", "Pine Island (BedMachine)", "#111111"), ("dotson", "Dotson (BedMachine)", "#8a8a8a")):
    s_, Z = O[k_]; kk, PP = OC.psd_lines(Z); OBS[k_] = (s_, Z, kk, PP, float(np.median(OC.band_rms(kk, PP))), lab, c_)

fig = plt.figure(figsize=(7.2, 6.4))
# a: map of the Pine Island base with the transects
ax = fig.add_axes([0.06, 0.53, 0.3, 0.42])
x, y = OT.X / 1e3, OT.Y / 1e3; m = (x > -1650) & (x < -1560); n = (y > -345) & (y < -250)
b = np.where(OT.FL, OT.BASE, np.nan)[np.ix_(n, m)]
im = ax.pcolormesh(x[m], y[n], b, cmap="cividis", vmin=-1000, vmax=-250, shading="auto", rasterized=True)
(x0, y0), (x1, y1) = OT.LINES["pig"]; L = np.hypot(x1 - x0, y1 - y0); ux, uy = (x1 - x0) / L, (y1 - y0) / L
for o in np.arange(-4e3, 4.01e3, 1e3):
    ax.plot([(x0 - uy * o) / 1e3, (x0 + ux * 40e3 - uy * o) / 1e3], [(y0 + ux * o) / 1e3, (y0 + uy * 40e3 + ux * o) / 1e3],
            color="white", lw=0.5 if o else 1.1, alpha=0.9)
ax.annotate("GL", xy=(x0 / 1e3, y0 / 1e3), xytext=(x0 / 1e3 + 9, y0 / 1e3 + 2), fontsize=6.5, color=INK,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=INK))
ax.set_aspect("equal"); ax.tick_params(labelsize=5.5, length=2)
ax.set_xlabel("x (km, polar stereographic)", fontsize=6); ax.set_ylabel("y (km)", fontsize=6)
ax.set_title("Pine Island Ice Shelf base\n(BedMachine Antarctica v3)", fontsize=7, loc="left")
cb = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.06, pad=0.03); cb.set_label("ice base (m)", fontsize=6); cb.ax.tick_params(labelsize=5.5)
ax.text(-0.25, 1.08, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")

# b: relief along flow
ax = fig.add_axes([0.47, 0.53, 0.33, 0.42])
s_, Z, *_ , rp, _, _ = OBS["pig"]; zc = Z[len(Z) // 2]; ok = np.isfinite(zc); zz = hpf(zc[ok], OC.DS); mm = s_[ok] > 3e3
ax.plot(s_[ok][mm] / 1e3, zz[mm], color=INK, lw=1.0)
ax.text(40.8, 0, f"Pine Island, {rp:.0f} m", fontsize=6, va="center", color=INK, fontweight="semibold")
off = 0
for k_, (xs, Zm, kk, PP, rm, lab, c_) in MOD.items():
    off -= 105; z = Zm[len(Zm) // 2]; zz = hpf(z, 250.0); mm = xs > 3e3
    ax.plot(xs[mm] / 1e3, zz[mm] + off, color=c_, lw=0.9)
    ax.text(40.8, off, f"{lab}, {rm:.0f} m", fontsize=6, va="center", color=c_, fontweight="semibold")
ax.set_xlim(0, 40); ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.tick_params(labelsize=6, length=2)
ax.plot([1, 1], [off - 70, off - 20], color=INK, lw=1.2); ax.text(1.6, off - 45, "50 m", fontsize=6, va="center")
ax.set_xlabel("distance from grounding line (km)", fontsize=6.5)
ax.set_title("Relief along flow (5 km mean removed, 500 m footprint, offset)", fontsize=7, loc="left")
ax.text(-0.06, 1.08, "b", transform=ax.transAxes, fontsize=9, fontweight="bold")

# c: spectra
ax = fig.add_axes([0.09, 0.07, 0.4, 0.3])
for k_, (s_, Z, kk, PP, r_, lab, c_) in OBS.items():
    mk = kk <= OC.KMAX; ax.loglog(*logbin(kk[mk], PP[:, mk].mean(0)), color=c_, lw=2.2, label=lab)
for k_, (xs, Zm, kk, PP, rm, lab, c_) in MOD.items():
    mk = kk <= OC.KMAX; ax.loglog(*logbin(kk[mk], PP[:, mk].mean(0)), color=c_, lw=1.1)
ax.axvspan(1 / 20000, 1 / 2000, color="#f2f2f2", zorder=0, lw=0)
ax.set_xlabel("along-flow wavenumber (cycles m$^{-1}$)", fontsize=6.5); ax.set_ylabel("PSD (m$^2$ per cycle m$^{-1}$)", fontsize=6.5)
ax.grid(color=GRID, lw=0.4, which="major"); ax.tick_params(labelsize=6)
sec = ax.secondary_xaxis("top", functions=(lambda k: 1 / np.maximum(k, 1e-9), lambda l: 1 / np.maximum(l, 1e-9)))
sec.set_xlabel("wavelength (m)", fontsize=6); sec.tick_params(labelsize=5.5)
ax.legend(fontsize=5.6, loc="lower left")
ax.set_title("Spectra, first 40 km from the GL", fontsize=7, loc="left", pad=16)
fig.text(0.01, 0.43, "c", fontsize=9, fontweight="bold")

# d: rms ladder
ax = fig.add_axes([0.68, 0.07, 0.28, 0.3])
rows = [(lab, rm, c_) for (xs, Zm, kk, PP, rm, lab, c_) in MOD.values()]
keys = list(MOD.keys())
for i, ((lab, rm, c_), k_) in enumerate(zip(rows[::-1], keys[::-1])):
    ax.barh(i, rm, color=c_, height=0.65)
    lo_, hi_, n_ = SPREAD[k_]
    if n_ > 1: ax.plot([lo_, hi_], [i, i], color=INK, lw=0.9)
ax.axvline(OBS["pig"][4], color=INK, lw=1.2); ax.text(OBS["pig"][4], len(rows) - 0.35, " Pine Island", fontsize=6, color=INK, va="bottom")
ax.axvline(OBS["dotson"][4], color="#8a8a8a", lw=1.2, ls="--"); ax.text(OBS["dotson"][4], len(rows) - 0.35, " Dotson", fontsize=6, color="#8a8a8a", va="bottom")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=6)
for t, r in zip(ax.get_yticklabels(), rows[::-1]): t.set_color(r[2])
ax.set_xlabel("rms relief at 2 to 20 km (m)", fontsize=6.5); ax.tick_params(axis="x", labelsize=6); ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 34); ax.set_ylim(-0.6, len(rows) + 0.4); ax.grid(axis="x", color=GRID, lw=0.4)
ax.set_title("Adding mechanisms one at a time", fontsize=7, loc="right")
fig.text(0.53, 0.43, "d", fontsize=9, fontweight="bold")
save(fig, "Fig_pineisland")
