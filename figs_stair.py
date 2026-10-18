import pickle, numpy as np, pandas as pd, sys
sys.path.insert(0, "oned")
import matplotlib.pyplot as plt
from main_figs import *
from figures import save, panel_label, INK, MUTED, GRID
from stair1d import run
from scipy.ndimage import gaussian_filter1d
fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 0.9, 1.0], hspace=0.65, wspace=0.3)
for j, (a, ttl) in enumerate([(0.02, "Flat ice melts at 2% of wall melt"), (0.2, "Flat ice melts at 20% of wall melt")]):
    o = run(law=a, thc=2.0, n=8, M=1.0, noise=1.0, years=16, adapt=2000.0, out_every=40)
    x = o["x"]; T = np.array([t for t, _ in o["snaps"]]); Z = np.array([zb for _, zb in o["snaps"]])
    G = np.gradient(gaussian_filter1d(Z, 2, axis=1), x[1] - x[0], axis=1)
    ax = fig.add_subplot(gs[0, j])
    ax.pcolormesh(x / 1e3, T, (G > np.tan(np.deg2rad(20))).astype(float), cmap="Greys", shading="auto", rasterized=True)
    ax.set_title(ttl, pad=4); ax.set_xlabel("distance from GL (km)")
    if j == 0: ax.set_ylabel("time (yr)")
    panel_label(ax, "ab"[j])
    ax = fig.add_subplot(gs[1, j])
    ax.plot(x / 1e3, Z[-1], color=INK, lw=0.8); ax.plot(x / 1e3, o["zbar"], ":", color=MUTED, lw=0.7)
    ax.set_xlim(0, 8); ax.set_xlabel("distance from GL (km)")
    if j == 0: ax.set_ylabel("base depth (m)")
    panel_label(ax, "cd"[j])
res = pickle.load(open("oned/sweep2.pkl", "rb")) + pickle.load(open("oned/sweep3.pkl", "rb"))
df = pd.DataFrame([dict(**k, **m) for k, m in res]); df["law"] = df.law.astype(str)
for c, v in (("thc", 2.0), ("dTdz_frac", 0.0), ("beta", 0.0), ("Tfb", 0.0), ("M", 1.0)): df[c] = df[c].fillna(v) if c in df else v
df = df[df.law != "sin23"]; df["a"] = df.law.astype(float)
ax = fig.add_subplot(gs[2, 0])
for (tf, lab, c) in ((0.0, "uniform thermal driving", "#c98a14"), (-0.5, "thermal driving halves over 500 m", "#1f5f99")):
    s = df[(df.thc == 2.0) & (df.dTdz_frac == tf) & (df.beta == 0) & (df.Tfb == 0) & (df.M == 1.0)].groupby("a")
    ax.plot(s.h_1_3.mean().index, s.h_1_3.mean(), "o--", color=c, ms=3, lw=0.9)
    ax.plot(s.h_5_7.mean().index, s.h_5_7.mean(), "o-", color=c, ms=3, lw=1.2, label=lab)
ax.set_xscale("log"); ax.set_xlabel("flat-ice melt / wall melt, $a$")
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
tk = [0.02, 0.05, 0.1, 0.2, 0.5]; ax.xaxis.set_major_locator(FixedLocator(tk)); ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in tk])); ax.xaxis.set_minor_locator(NullLocator()); ax.set_ylabel("median riser height (m)")
ax.plot([], [], "k--", lw=0.9, label="1 to 3 km"); ax.plot([], [], "k-", lw=1.2, label="5 to 7 km")
ax.legend(fontsize=5.8, loc="upper right"); ax.set_title("Steps grow only when $a\\lesssim0.05$", pad=4); panel_label(ax, "e")
ax = fig.add_subplot(gs[2, 1])
s = df[(df.Tfb == 0) & (df.dTdz_frac == 0) & (df.M == 1.0) & (df.thc == 2.0)]
for a, c in ((0.05, "#6a4c93"), (0.1, "#2a9d8f")):
    q = s[s.a == a].groupby("beta")[["h_1_3", "h_5_7"]].mean()
    ax.plot(q.index, q.h_5_7, "o-", color=c, ms=3, lw=1.2, label=f"a = {a:g}, 5 to 7 km")
    ax.plot(q.index, q.h_1_3, "o--", color=c, ms=3, lw=0.9, label=f"a = {a:g}, 1 to 3 km")
ax.set_xlabel("riser melt $\\propto h^{\\beta}$, exponent $\\beta$"); ax.set_ylabel("median riser height (m)")
ax.legend(fontsize=5.6, loc="upper left"); ax.set_title("Taller-faster risers coarsen", pad=4); panel_label(ax, "f")
save(fig, "FigS16_staircase_mechanism")
