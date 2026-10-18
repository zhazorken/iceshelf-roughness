import sys, numpy as np, pickle
sys.path.insert(0, "oned")
import matplotlib.pyplot as plt
from main_figs import *
from figures import save, panel_label, INK, MUTED, GRID
from scipy.ndimage import gaussian_filter1d
from stair1d import run
SPY = 3.15576e7; GP, KAP, SCD = 9.81 * 7.8e-4 * 34, 0.4, 0.05
def mo(m0, sth, ua, gD=0.01, delta=0.2, smin=0.01):
    cth = np.sqrt(1 - sth ** 2); m = m0
    for _ in range(200):
        B = GP * max(m, 1e-14); us = np.sqrt(ua ** 2 + (SCD * np.sqrt(gD * sth)) ** 2)
        L = us ** 3 / (KAP * B * max(cth, 1e-6)); m = 0.5 * m + 0.5 * m0 * (smin + (1 - smin) / (1 + 5 * delta / L))
    return m
fig = plt.figure(figsize=(7.2, 5.2)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.55, wspace=0.32)
ax = fig.add_subplot(gs[0, 0]); ang = np.r_[0, np.logspace(-1, np.log10(90), 60)]
for ua, c in ((1e-4, "#1f5f99"), (3e-4, "#2a9d8f"), (1e-3, "#c98a14"), (2e-3, "#d1495b")):
    mm = [mo(20 * max(a / 90, ua / 2.97e-3) / SPY, np.sin(np.deg2rad(a)), ua) * SPY / 20 for a in ang]
    ax.plot(ang, mm, color=c, lw=1.2, label=f"$u_{{*a}}$ = {ua*1e3:g} mm/s")
ax.plot(ang, ang / 90, color=INK, ls=":", lw=0.9, label="DNS law, $u_*=0$, no suppression")
ax.set_xscale("symlog", linthresh=1); ax.set_xlim(0, 90); ax.set_ylim(0, 1.05)
ax.set_xlabel("interface slope (°)"); ax.set_ylabel("melt / quiescent wall melt")
ax.set_title("Monin-Obukhov suppression", pad=4); ax.legend(fontsize=5.6, loc="upper left"); panel_label(ax, "a")
base = dict(law="dnsmax", thc=0.0, mo=True, mo_ua=3e-4, U=200.0, years=60.0, noise=1.0, ds=5.0, mo_smin=0.01, mo_gD=0.01, adapt=2000.0, M=0.5)
cases = [("MO only", dict(), "#6a4c93"), ("MO + pooling (h = 1 m)", dict(pool_h=1.0), "#2a9d8f")]
outs = [(lab, run(**dict(base, **kw)), c) for lab, kw, c in cases]
ax = fig.add_subplot(gs[0, 1])
for lab, o, c in outs:
    x = o["x"]; g = np.degrees(np.arctan(np.gradient(gaussian_filter1d(o["zb"], 1), x[1] - x[0])))[(x > 3000) & (x < 7500)]
    ax.hist(g, bins=np.arange(-15, 45, 0.5), histtype="step", density=True, color=c, lw=1.1, label=lab)
ax.axvline(0, color=MUTED, lw=0.6); ax.set_xlim(-12, 40); ax.set_yscale("log")
ax.set_xlabel("along-flow base slope (°)"); ax.set_ylabel("probability density"); ax.legend(fontsize=5.8)
ax.set_title("Slope distribution, 3 to 7.5 km", pad=4); panel_label(ax, "b")
for j, (lab, o, c) in enumerate(outs):
    ax = fig.add_subplot(gs[1, j]); x = o["x"] / 1e3
    ax.plot(x, o["zb"], color=c, lw=0.8); ax.plot(x, o["zbar"], ":", color=MUTED, lw=0.6)
    ax.set_xlim(4, 6); sel = (x >= 4) & (x <= 6); ax.set_ylim(o["zb"][sel].min() - 5, o["zb"][sel].max() + 5)
    ax.set_xlabel("distance from GL (km)"); ax.set_ylabel("base depth (m)" if j == 0 else ""); ax.set_title(lab, pad=4)
    ax.grid(color=GRID, lw=0.4); panel_label(ax, "cd"[j])
save(fig, "FigS17_MO_terraces")
