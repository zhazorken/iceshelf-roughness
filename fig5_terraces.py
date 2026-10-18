"""Main Fig. 5: the terrace regime."""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "oned")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap
from matplotlib.patches import Rectangle
from main_figs import *
from cmcrameri import cm as cmc
from figures import save, panel_label, INK, MUTED, GRID
SPY = 3.15576e7; GP, KAP, SCD = 9.81 * 7.8e-4 * 34, 0.4, 0.05
RC = {"smooth": "#c98a14", "sawtooth": "#6a4c93", "terraces": "#2a9d8f"}
def mo(m0, sth, ua, gD=0.01, delta=0.2, smin=0.01):
    cth = np.sqrt(1 - sth ** 2); m = m0; us = np.sqrt(ua ** 2 + SCD ** 2 * gD * sth)
    for _ in range(200):
        L = us ** 3 / (KAP * GP * max(m, 1e-14) * max(cth, 1e-6)); m = 0.5 * m + 0.5 * m0 * (smin + (1 - smin) / (1 + 5 * delta / L))
    return m
res = [r for r in pickle.load(open("oned/regime_z.pkl", "rb")) if "tag" not in r[0]]
resM = [r for r in pickle.load(open("oned/regime_z.pkl", "rb")) if "tag" in r[0]]
UC = 2.97e-3
df = pd.DataFrame([dict(**k, **{kk: v for kk, v in m.items() if kk not in ("zb", "zbar", "x")}) for k, m in res])
def cls(r): return "smooth" if r.rms < 2.5 else ("terraces" if (abs(r.tilt) < 1.0 and r.back < 0.1) else "sawtooth")
df["reg"] = df.apply(cls, axis=1)
fig = plt.figure(figsize=(7.2, 4.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1], hspace=0.62, wspace=0.42)
# a: melt vs slope
ax = fig.add_subplot(gs[0, 0]); ang = np.r_[0, np.logspace(-1, np.log10(90), 80)]
for ua, c in ((1e-4, "#6a4c93"), (3e-4, "#2a9d8f"), (1e-3, "#1f5f99"), (2e-3, "#c98a14")):
    f0 = lambda a: max(a / 90, ua / UC)
    ax.plot(ang, [mo(30 * f0(a) / SPY, np.sin(np.deg2rad(a)), ua) * SPY / 30 for a in ang], color=c, lw=1.2,
            label=f"{ua*1e3:g} mm s$^{{-1}}$")
    ax.plot(ang, [f0(a) for a in ang], ":", color=c, lw=0.8)
ax.text(0.15, 0.76, "dotted: without stability closure", fontsize=5.4, color=INK)
ax.set_xscale("symlog", linthresh=1); ax.set_xlim(0, 90); ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticks([0, 1, 10, 90]); ax.set_xticklabels(["0", "1", "10", "90"])
ax.set_xlabel("interface slope (°)"); ax.set_ylabel("melt / wall melt"); ax.set_title("Meltwater-stabilized melt", pad=4)
ax.legend(fontsize=5.4, title="ambient $u_*$", title_fontsize=5.6, loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=2, columnspacing=0.8, handlelength=1.2); panel_label(ax, "a")
# b: regime map
ax = fig.add_subplot(gs[0, 1:]); uas = sorted(df.mo_ua.unique()); phs = sorted(df.pool_h.unique())
for i, ua in enumerate(uas):
    for j, ph in enumerate(phs):
        s = df[(df.mo_ua == ua) & (df.pool_h == ph)]; regs = s.reg.value_counts()
        r = regs.index[0]; ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, color=RC[r], alpha=0.85 if regs.iloc[0] == len(s) else 0.5, lw=0))
        ax.text(i, j, f"{s.rms.mean():.0f}", ha="center", va="center", fontsize=6, color="white")
ax.set_xlim(-0.5, len(uas) - 0.5); ax.set_ylim(-0.5, len(phs) - 0.5)
ax.set_xticks(range(len(uas))); ax.set_xticklabels([f"{u*1e3:g}" for u in uas]); ax.set_yticks(range(len(phs))); ax.set_yticklabels([f"{p:g}" for p in phs])
ax.set_xlabel("ambient friction velocity $u_{*a}$ (mm s$^{-1}$)"); ax.set_ylabel("pooling depth scale $h$ (m)")
for k, c in RC.items(): ax.plot([], [], "s", color=c, ms=6, label=k)
ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0), handletextpad=0.3); ax.set_title("Regimes (numbers: rms relief, m)", pad=4)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0); panel_label(ax, "b")
# c: example profiles
ax = fig.add_subplot(gs[1, :2]); off = 0
for (ua, ph, lab) in ((1e-3, 1.0, "smooth"), (3e-4, 0.0, "sawtooth"), (3e-4, 1.0, "terraces")):
    m = [m for k, m in res if k["mo_ua"] == ua and k["pool_h"] == ph and k["rng"] == 0][0]
    x = m["x"] / 1e3; sel = (x > 3.5) & (x < 6.5); z = m["zb"] - m["zbar"] + (m["zbar"] - m["zbar"][sel].mean())
    zz = m["zb"][sel] - m["zb"][sel].mean()
    ax.plot(x[sel], zz + off, color=RC[lab], lw=1.0); ax.text(6.52, zz[-1] + off, lab, color=RC[lab], fontsize=6.5, va="center")
    off += 140
ax.set_xlim(3.5, 6.9); ax.set_xlabel("distance from GL (km)"); ax.set_ylabel("base elevation (m, offset)")
ax.set_title("Base profiles on a 4° slope (1D, $M$ = 0.5)", pad=4); ax.grid(color=GRID, lw=0.4); panel_label(ax, "c")
# d: length scales vs observations
ax = fig.add_subplot(gs[1, 2]); sc = pickle.load(open("oned/scales_z.pkl", "rb"))
hh = np.linspace(1, 80, 50)
for a, ls in ((4, "-"), (8, "--")):
    ax.plot(hh, hh / np.tan(np.deg2rad(a)), ls, color=MUTED, lw=0.8)
    hl = 30 if a == 4 else 54; ax.text(hl, hl / np.tan(np.deg2rad(a)) + (12 if a == 4 else -55), f"$h\\cot{a}°$", fontsize=5.8, color=MUTED, ha="right" if a == 4 else "left")
ax.add_patch(Rectangle((5, 100), 45, 200, fc="#d9d9d9", ec="none", alpha=0.6)); ax.text(48, 292, "Pine Island\nterraces", fontsize=5.6, ha="right", va="top", color=INK)
for kw, m in sc:
    a = kw.get("alpha", 4.0); c = {2.0: "#c98a14", 4.0: "#2a9d8f", 8.0: "#1f5f99"}[a] if "U" not in kw else "#6a4c93"
    for key, mk in (("near", "o"), ("far", "s")):
        h, L, n = m[key]
        if np.isfinite(h) and np.isfinite(L): ax.plot(h, L, mk, color=c, ms=3.5, mfc=c if key == "far" else "white")
ax.plot([], [], "o", color=INK, mfc="white", ms=3.5, label="1 to 3 km"); ax.plot([], [], "s", color=INK, ms=3.5, label="4.5 to 7.5 km")
ax.plot([], [], "-", color="#2a9d8f", label="4° base"); ax.plot([], [], "-", color="#1f5f99", label="8° base"); ax.plot([], [], "-", color="#6a4c93", label="4°, melt ×0.5, ×2.5")
ax.set_xlim(0, 60); ax.set_ylim(0, 620); ax.set_xlabel("step height (m)"); ax.set_ylabel("tread length (m)")
ax.legend(fontsize=5.4, loc="upper left", bbox_to_anchor=(1.0, 1.0), handlelength=1.2); ax.set_title("Length scales", pad=4); panel_label(ax, "d")
save(fig, "Fig5_terraces")
