"""Main-text Figures 2-4 (polished layouts). Imported by figures.py."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
import shelfmelt as sm
import cmcrameri.cm as cmc
from figures import (COL, NAME, DIV, INK, MUTED, GRID, panel_label, save, sel, ms, hillshade, OAT_GROUPS, PRETTY)

COL.update({"plume": "#4f9a2f", "plumeF": "#4f9a2f", "convA": "#2a9d8f", "dnsQ": "#e07b00", "dnsP": "#2a9d8f", "dnsA": "#c98a14", "off": "#9a9a9a"})
SHORT = {"shear": "Shear", "conv": "sin$^{2/3}$ convection", "dns": "DNS law, tidal $u_*$", "dnsQ": "DNS law, quiet water",
         "dnsP": "DNS law, quiet + pooling", "dnsA": "DNS law + aging ponds", "off": "Offset-linear fit (old)", "theta": "DNS convection only ($u_*=0$)",
         "dnsF": "DNS law + flat strat.", "convP": "sin$^{2/3}$ + pooling", "plume": "Plume law",
         "plumeF": "Plume + flat strat.", "convF": "sin$^{2/3}$ + flat strat.", "shearF": "Shear + flat strat.",
         "mixed": "Shear + convection", "convA": "sin$^{2/3}$ + aging ponds", "shearA": "Shear + aging ponds",
          "plumeA": "Plume + aging ponds"}
MARK = {v: ("D" if v.endswith("F") or v.endswith("P") else "s" if v.endswith("A") else "o") for v in SHORT}

# ---------------------------------------------------------------- Figure 2
import os, pickle
TERR_RUN = os.environ.get("TERR_RUN", "results_coarse/terr2dZ_wide_Ly3000.pkl")
COL["terr"] = "#2a6f4e"; SHORT["terr"] = "All melt mechanisms"

def load_terr():
    if not os.path.exists(TERR_RUN): return None
    o = pickle.load(open(TERR_RUN, "rb")); o["params"]["dx"] = float(o["params"]["dx"]); return o

OBL = tuple(float(v) for v in os.environ.get("OBL", "2.5,5,0,3").split(","))
def detrend_x(r):
    """Relief about a plane fitted along flow (keeps terrace steps, removes the base slope)."""
    F = r["field"].astype(float) + r["zbar"][None, :]; x = r["x"]
    return F - np.polyval(np.polyfit(x, F.mean(0), 1), x)[None, :]

TERR_CASES = [("terr", "results_coarse/terr2dZ_wide_Ly3000.pkl", "#2a6f4e",
               "All melt mechanisms: DNS law, weak currents ($u_{*a}$ = 0.3 mm s$^{-1}$), boundary-layer stability, pooling"),
              ("saw", "results_coarse/terr2dZ_wide_Ly3000_ua0.3_ph0.pkl", "#6a4c93",
               "As (d) without pooling: scalloped"),
              ("smo", "results_coarse/terr2dZ_wide_Ly3000_ua1_ph1.pkl", "#c98a14",
               "As (d) with stronger currents ($u_{*a}$ = 1 mm s$^{-1}$): smooth")]

def melt_stats(r):
    F = r["field"].astype(float) + r["zbar"][None, :]; dx = float(r["params"]["dx"])
    sl = np.degrees(np.arctan(np.hypot(*np.gradient(F, dx)))); m = r["meltfield"]
    return float(np.median(m[sl < 2])), float(np.median(m[sl > 30])) if (sl > 30).any() else float("nan")

BUILD = [("b1", "shear, 40 m crevasses", "#1f5f99"), ("b2", "+ crevasses 40\u2013250 m", "#5a8fc4"),
         ("b3", "+ DNS law, weak currents", "#c98a14"), ("b4", "+ flat stratification", "#6a4c93"),
         ("b7", "+ ponds, 1% retained", "#b5487f"), ("b5", "+ ponds, 3% retained", "#8c2d6b")]
POND_ARROWS = [(29.55, 2.7, 32.0, 0.75, "preserved crevasses"), (35.0, 1.9, 32.0, 0.75, "preserved crevasses"), (15.5, 1.0, 11.5, 0.9, "convection-widened")]

def fig2(R):
    from matplotlib.colors import LogNorm
    order = [v for v in ["shear", "conv"] if sel(R, "showcase", "pig", v)]
    recs = {v: sel(R, "showcase", "pig", v)[0] for v in order + ["dns"] if sel(R, "showcase", "pig", v)}
    T = [(k, pickle.load(open(f, "rb")), c, t) for k, f, c, t in TERR_CASES if os.path.exists(f)]
    vlim = 50.0; W = 7.2
    fig = plt.figure(figsize=(W, 9.6))
    L, Rt = 0.08, 0.90          # common left and right edges for every map
    letters = iter("abcdefghij")
    box = dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.88)
    hrow = (Rt - L) * W * 6 / 40 / 9.6           # height (figure fraction) of a 40 x 6 km map at full width
    y = 0.975
    PF = "results_coarse/mix_convA03show.pkl"
    if os.path.exists(PF):
        recs["pond"] = pickle.load(open(PF, "rb")); recs["pond"]["params"]["dx"] = float(recs["pond"]["params"]["dx"])
        order = order + ["pond"]
    COL["pond"] = "#8c2d6b"
    SHORT["pond"] = "sin$^{2/3}$ + meltwater ponds (3% retained), 40\u2013150 m crevasses"
    for i, v in enumerate(order):
        y -= hrow + (0.02 if i else 0.0)
        ax = fig.add_axes([L, y, Rt - L, hrow])
        r = recs[v]; dx = r["params"]["dx"]
        Lx, Ly = r["x"][-1] / 1e3 + dx / 2e3, r["y"][-1] / 1e3 + dx / 2e3
        ax.imshow(hillshade(sm.highpass(r["field"].astype(float), dx), dx, vlim), extent=[0, Lx, 0, Ly], origin="lower",
                  aspect="auto", interpolation="antialiased")
        ax.text(0.006, 0.9, SHORT[v] if v != "shear" else "Shear melt (same base as the DNS law with tidal $u_*$)",
                transform=ax.transAxes, fontsize=6.8, fontweight="semibold", va="top", color=COL[v], bbox=box)
        ax.text(0.994, 0.9, f"G = {r['G']:.1f}   deepest incision {r['inc999']:.0f} m", transform=ax.transAxes,
                fontsize=6.2, va="top", ha="right", color=INK, bbox=box)
        ax.set_xlim(0, 40); ax.set_ylim(0, 6); ax.set_yticks([0, 3, 6]); ax.spines[:].set_visible(False)
        ax.tick_params(length=2, labelsize=6)
        if i < len(order) - 1: ax.set_xticklabels([])
        else: ax.set_xlabel("distance from grounding line (km), 40 km shelf, y in km", labelpad=1, fontsize=6.5)
        ax.text(-0.055, 0.5, next(letters), transform=ax.transAxes, fontsize=9, fontweight="bold", va="center")
        if v == "pond":
            for (xa, ya, xt, yt, t) in POND_ARROWS:
                ax.annotate(t, xy=(xa, ya), xytext=(xt, yt), fontsize=6.3, color=INK, ha="center", va="center", bbox=box,
                            arrowprops=dict(arrowstyle="->", lw=0.8, color=INK))
    y -= 0.07
    gap = 0.03; wp = (Rt - L - gap) / 2; hp = wp * W * 3 / 8 / 9.6
    norm = LogNorm(0.3, 100)
    for j, (k, r, c, title) in enumerate(T):
        dx = float(r["params"]["dx"]); Lx, Ly = r["x"][-1] / 1e3 + dx / 2e3, r["y"][-1] / 1e3 + dx / 2e3
        y -= hp + (0.045 if j else 0.0)
        ax = fig.add_axes([L, y, wp, hp])
        ax.imshow(hillshade(detrend_x(r), dx, vlim), extent=[0, Lx, 0, Ly], origin="lower", aspect="auto",
                  interpolation="antialiased")
        ax.text(0.0, 1.06, title, transform=ax.transAxes, fontsize=6.6, fontweight="semibold", color=c, va="bottom")
        ax.text(-0.17, 0.5, next(letters), transform=ax.transAxes, fontsize=9, fontweight="bold", va="center")
        ax.set_ylabel("y (km)", fontsize=6, labelpad=1); ax.set_yticks([0, 1, 2, 3])
        ax2 = fig.add_axes([L + wp + gap, y, wp, hp])
        im = ax2.imshow(np.maximum(r["meltfield"].astype(float), 0.05), extent=[0, Lx, 0, Ly], origin="lower", aspect="auto",
                        cmap="magma", norm=norm, interpolation="antialiased")
        fl, st = melt_stats(r)
        ax2.text(0.99, 0.94, f"flats {fl:.1f}, faces >30° {st:.0f} m yr$^{{-1}}$" if st == st else f"flats {fl:.1f} m yr$^{{-1}}$",
                 transform=ax2.transAxes, fontsize=6, ha="right", va="top", color=INK, bbox=box)
        ax2.text(-0.045, 0.5, next(letters), transform=ax2.transAxes, fontsize=9, fontweight="bold", va="center")
        ax2.set_yticks([0, 1, 2, 3]); ax2.set_yticklabels([])
        for a_ in (ax, ax2):
            a_.spines[:].set_visible(False); a_.tick_params(length=2, labelsize=6)
            if j < len(T) - 1: a_.set_xticklabels([])
            else: a_.set_xlabel("along-flow distance (km), 4° base", labelpad=1, fontsize=6.5)
    if T:
        mcax = fig.add_axes([Rt + 0.012, y, 0.012, 3 * hp + 0.09])
        mcb = fig.colorbar(im, cax=mcax); mcb.set_label("melt rate (m yr$^{-1}$)", fontsize=6.5); mcb.ax.tick_params(labelsize=6, length=2)
    y -= 0.04
    cax = fig.add_axes([0.62, y - 0.03, 0.22, 0.008])
    cb = fig.colorbar(cm.ScalarMappable(Normalize(-vlim, vlim), DIV), cax=cax, orientation="horizontal")
    cb.set_label("relief z\u2032 (m): a to c, 2-km mean removed;\nd, f, h, plane fit removed;\nred = melted upward into the ice", fontsize=6.0, labelpad=2)
    cb.ax.tick_params(labelsize=6, length=2)
    save(fig, "Fig2_morphology")

# ---------------------------------------------------------------- Figure 3
def fig3(R):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1.3, 1.2, 0.8], wspace=1.05))
    ax = axs[0]
    labs = sorted({r["label"] for r in sel(R, "contrast")}, key=float)
    rows = []
    for a in labs:
        rs = sel(R, "contrast", a); g = ms(rs, "G"); rows.append((np.mean([r["C10"] for r in rs]), g[0], g[2], g[3]))
    ax.axhspan(1, 12, color="#fbe9e7", lw=0, zorder=0); ax.axhspan(0.2, 1, color="#e7eff7", lw=0, zorder=0)
    ax.text(0.97, 0.96, "roughening", transform=ax.transAxes, ha="right", va="top", color="#b03a2e", fontsize=6.5,
            fontweight="semibold")
    ax.text(0.97, 0.04, "smoothing", transform=ax.transAxes, ha="right", color="#1f5f99", fontsize=6.5,
            fontweight="semibold")
    if rows:
        rows = np.array(sorted(rows))
        ax.fill_between(rows[:, 0], rows[:, 2], rows[:, 3], color="#9a9a9a", alpha=0.3, lw=0)
        ax.plot(rows[:, 0], rows[:, 1], "-o", color="#555555", lw=1.0, ms=3, mec="white", mew=0.5, zorder=3)
        ax.text(2.6, 1.25, "offset-linear family,\nvarying offset $a$", fontsize=5.8, color="#555555")
    # revised DNS law: ambient u* scaled from 0.1x to 2x the tidal profile
    sw = []
    for f, rs in [(float(l), sel(R, "dns u*", l, "dns")) for l in sorted({r["label"] for r in sel(R, "dns u*")}, key=float)] + \
                 [(1.0, sel(R, "ens", "base", "dns")), (0.2, sel(R, "ens", "base", "dnsQ"))]:
        if rs: sw.append((f, np.mean([r["C10"] for r in rs]), ms(rs, "G")[0]))
    if len(sw) > 1:
        sw = np.array(sorted(sw, key=lambda t: t[1]))
        ax.plot(sw[:, 1], sw[:, 2], "-", color=COL["dns"], lw=1.0, alpha=0.8, zorder=4)
        for f, Cc, Gg in sw:
            if f not in (1.0, 0.2): ax.plot(Cc, Gg, "o", ms=3, color=COL["dns"], mec="white", mew=0.4, zorder=5)
            if f == min(sw[:, 0]): ax.annotate(f"DNS law, $u_*$ \u00d7{f:g}", (Cc, Gg), textcoords="offset points", xytext=(4, 4), fontsize=5.6, color=COL["dns"])
    pts = {}
    for v in ["shear", "dns", "dnsQ", "conv", "theta", "plume", "dnsF", "plumeF", "convP", "convA"]:
        rs = sel(R, "ens", "base", v)
        if not rs: continue
        Cv = np.mean([r["C10"] for r in rs]) * (1.06 if v == "convA" else 1.0); g = ms(rs, "G"); pts[v] = (Cv, g[0])
        ax.errorbar(Cv, g[0], yerr=g[1], fmt=MARK[v], ms=5, color=COL[v], mec="white", mew=0.6,
                    elinewidth=0.8, capsize=0, zorder=6)
    lab = {"shear": ((4, -14), "shear; DNS law, tidal\nand quiet water"), "conv": ((6, -3), "sin$^{2/3}$"),
           "theta": ((-70, 8), "DNS convection,\n$u_*=0$"), "plume": ((7, -3), "plume law")}
    for v, (off, t) in lab.items():
        if v in pts: ax.annotate(t, pts[v], textcoords="offset points", xytext=off, fontsize=6.2, color=COL[v],
                                 fontweight="semibold")
    for v, base, t, off in (("dnsF", "dns", "+ flat strat.", (6, -3)), ("plumeF", "plume", "+ flat strat.", (6, -3)),
                            ("convP", "conv", "+ pooling", (5, -6)), ("convA", "conv", "+ aging ponds", (-62, -3))):
        if v in pts and base in pts:
            ax.annotate("", xy=pts[v], xytext=pts[base], arrowprops=dict(arrowstyle="-|>", lw=0.9, color=COL[v],
                                                                          shrinkA=4, shrinkB=4))
            ax.annotate(t, pts[v], textcoords="offset points", xytext=off, fontsize=6.0, color=COL[v],
                        fontweight="semibold")
    ax.axhline(1, color=INK, lw=0.6)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.25, 12); ax.set_xlim(0.97, 13)
    ax.set_xticks([1, 1.5, 2, 3, 5, 10]); ax.set_xticklabels(["1", "1.5", "2", "3", "5", "10"]); ax.text(1.21, 8.5, "1.2", fontsize=5.6, color=MUTED); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.axvline(1.2, color=MUTED, lw=0.6, ls="--")
    ax.set_yticks([0.3, 0.5, 1, 2, 5, 10]); ax.set_yticklabels(["0.3", "0.5", "1", "2", "5", "10"])
    ax.set_xlabel("low-angle contrast $C_{10}$ (melt at 10° / shelf melt)")
    ax.set_ylabel("roughness gain $G$")
    panel_label(ax, "a", x=-0.15); ax.set_title("Low-angle slope contrast sets the sign", fontsize=7.5)

    ax = axs[1]; handles = []
    for v in ["shear", "dns", "dnsQ", "convP", "conv", "plume", "dnsF"]:
        rs = sel(R, "ens", "base", v)
        if not rs: continue
        A = np.array([gaussian_filter1d(r["rms"], 4) for r in rs]); xs = rs[0]["x"] / 1e3
        ax.fill_between(xs, A.mean(0) - A.std(0), A.mean(0) + A.std(0), color=COL[v], alpha=0.18, lw=0)
        ls = "--" if v in ("convP",) else "-"
        ax.plot(xs, A.mean(0), color=COL[v], lw=1.1, ls=ls)
        handles.append(Line2D([], [], color=COL[v], lw=1.3, ls=ls, label=SHORT[v]))
    ax.set_xlim(0, 37); ax.set_ylim(bottom=0); ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(handles=handles, loc="upper left", fontsize=5.8, handlelength=1.6, labelspacing=0.25)
    ax.set_xlabel("distance from grounding line (km)"); ax.set_ylabel("rms relief, < 2 km scales (m)")
    panel_label(ax, "b", x=-0.14); ax.set_title("Roughness downstream", fontsize=7.5)

    ax = axs[2]
    order = [v for v in ["convP", "shear", "dns", "plume", "dnsQ", "conv", "theta", "dnsF"] if sel(R, "ens", "base", v)]
    ax.axvspan(1, 10, color="#fbe9e7", lw=0, zorder=0)
    for i, v in enumerate(order):
        vals = np.array([r["enh"] for r in sel(R, "ens", "base", v)])
        ax.plot(vals, i + np.linspace(-0.12, 0.12, len(vals)), "o", ms=2.3, color=COL[v], alpha=0.45, mew=0)
        ax.plot(vals.mean(), i, MARK[v], ms=5.5, color=COL[v], mec="white", mew=0.6)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([SHORT[v] for v in order], fontsize=5.8)
    ax.axvline(1, color=INK, lw=0.7); ax.grid(axis="x", color=GRID, lw=0.5)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.8, max(np.nanmax([r["enh"] for r in sel(R, "ens", "base", v)]) for v in order) * 1.08)
    ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xlabel("mean melt / smooth-shelf melt")
    ax.text(-1.0, 1.04, "c", transform=ax.transAxes, fontsize=9, fontweight="bold"); ax.set_title("Roughness raises mean melt", fontsize=7.5)
    save(fig, "Fig3_contrast")

# ---------------------------------------------------------------- Figure 4
def tornado(ax, R, v, key, xl, fmt=lambda s: s):
    base = sel(R, "ens", "base", v)
    if not base: return
    b0 = ms(base, key)[0]; rows = []
    for g in OAT_GROUPS:
        rs = [r for r in R if r["group"] == g and r["variant"] == v]
        if not rs: continue
        vals = sorted({r["label"] for r in rs}, key=float)
        ds = [(lab, ms([r for r in rs if r["label"] == lab], key)[0]) for lab in vals]
        rows.append((g, ds, max(abs(d - b0) for _, d in ds)))
    rows.sort(key=lambda t: t[2])
    span = max(max(abs(d - b0) for _, d in ds) for _, ds, _ in rows) or 1
    for i, (g, ds, _) in enumerate(rows):
        for j, (lab, d) in enumerate(ds):
            c = "#7fa7c9" if j == 0 else ("#c0392b" if j == len(ds) - 1 else "#d9a38f")
            ax.barh(i, d - b0, left=b0, height=0.64, color=c, lw=0, alpha=0.9, zorder=2 + (j == 0))
        lo, hi = ds[0], ds[-1]
        for (lab, d), side in ((lo, 0), (hi, 1)):
            ha = "left" if d >= b0 else "right"
            ax.text(d + (0.02 * span if d >= b0 else -0.02 * span), i + (0.22 if side else -0.22), fmt(lab),
                    va="center", ha=ha, fontsize=5.3, color=MUTED)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([PRETTY[g] for g, _, _ in rows], fontsize=6.2)
    ax.axvline(b0, color=INK, lw=0.7, zorder=4); ax.set_xlabel(xl)
    ax.set_xlim(b0 - 1.35 * span, b0 + 1.35 * span)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0); ax.grid(axis="x", color=GRID, lw=0.4)

def sci(s):
    f = float(s)
    return f"10$^{{{int(np.log10(f))}}}$" if f >= 1e5 else s

def rangeplot(ax, R, v, key, xl, ref, log=True, fixed=False, labels=True, annotate=True):
    """Low/high parameter values as dots joined by a bar on the absolute axis, with the baseline."""
    base = sel(R, "ens", "base", v)
    if not base: return
    b0 = ms(base, key)[0]; rows = []
    for g in OAT_GROUPS:
        rs = [r for r in R if r["group"] == g and r["variant"] == v]
        if not rs: continue
        vals = sorted({r["label"] for r in rs}, key=float)
        ds = [(lab, ms([r for r in rs if r["label"] == lab], key)[0]) for lab in vals]
        lo, hi = min(d for _, d in ds + [("b", b0)]), max(d for _, d in ds + [("b", b0)])
        rows.append((g, ds, (np.log(hi) - np.log(lo)) if log else hi - lo))
    if not fixed: rows.sort(key=lambda t: t[2])
    for i, (g, ds, _) in enumerate(rows):
        allv = [d for _, d in ds] + [b0]
        ax.plot([min(allv), max(allv)], [i, i], color="#bdbdbd", lw=3.2, solid_capstyle="round", zorder=1)
        for j, (lab, d) in enumerate(ds):
            c = "#2f6fa8" if j == 0 else ("#c0392b" if j == len(ds) - 1 else "#d9a38f")
            ax.plot(d, i, "o", ms=4.5, color=c, mec="white", mew=0.5, zorder=3)
            if annotate: ax.annotate(sci(lab), (d, i), textcoords="offset points", xytext=(0, 4.5 if j % 2 else -8.5), ha="center",
                        fontsize=5.2, color=c)
        ax.plot(b0, i, "|", ms=9, mew=1.4, color=INK, zorder=4)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([PRETTY[g] for g, _, _ in rows] if labels else [], fontsize=6.2)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    if log: ax.set_xscale("log")
    ax.axvline(ref, color=INK, lw=0.6, ls=":")
    ax.set_xlabel(xl); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.4)

def fig4(R):
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], width_ratios=[1, 1], hspace=0.5, wspace=0.75)
    order = [v for v in ["convA", "convP", "shear", "dns", "plume", "dnsQ", "conv", "theta", "plumeF", "dnsF"] if sel(R, "ens", "base", v)]
    for k, (key, xl, ttl, ref) in enumerate((("Ainc", "deepest incision / seeded crevasse depth", "Incisions deepen or heal", 1.0),)):
        ax = fig.add_subplot(gs[0, k])
        ax.axvspan(ref, 10, color="#fbe9e7", lw=0, zorder=0)
        for i, v in enumerate(order):
            vals = np.array([r[key] for r in sel(R, "ens", "base", v)])
            ax.plot(vals, i + np.linspace(-0.12, 0.12, len(vals)), "o", ms=2.3, color=COL[v], alpha=0.45, mew=0)
            ax.plot(vals.mean(), i, MARK[v], ms=5.5, color=COL[v], mec="white", mew=0.6)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([SHORT[v] for v in order] if k == 0 else [], fontsize=6.3)
        ax.set_xlabel(xl); ax.grid(axis="x", color=GRID, lw=0.5); ax.axvline(ref, color=INK, lw=0.7)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        hi = max(np.nanmax([r[key] for r in sel(R, "ens", "base", v)]) for v in order)
        ax.set_xlim(0 if key == "Ainc" else 0.8, hi * 1.08)
        ax.set_title(ttl, fontsize=7.5); panel_label(ax, "ab"[k], x=-0.04 if k else -0.62)
    ax = fig.add_subplot(gs[0, 1])
    for v in ["shear", "conv", "dnsF"]:
        rs = sel(R, "ens", "base", v)
        if not rs: continue
        h = np.mean([r["slope_hist"] for r in rs], 0); bb = rs[0]["slope_bins"]; c = 0.5 * (bb[1:] + bb[:-1])
        ax.plot(c, np.maximum(h, 1e-7), color=COL[v], lw=1.1, drawstyle="steps-mid", label=SHORT[v])
    ax.set_yscale("log"); ax.set_xlim(0, 60); ax.set_ylim(1e-5, 1)
    ax.set_xlabel("local slope (\u00b0), 20 to 38 km"); ax.set_ylabel("area fraction per degree", labelpad=1)
    ax.axvspan(30, 60, color="#f2f2f2", lw=0, zorder=0)
    ax.legend(loc="upper right", fontsize=5.6, handlelength=1.2, labelspacing=0.2)
    panel_label(ax, "b", x=-0.25); ax.set_title("Slope distribution", fontsize=7.5)
    axs = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    rangeplot(axs[0], R, "dnsF", "Ainc", "deepest incision / seeded crevasse depth", 1.0)
    rangeplot(axs[1], R, "dnsF", "G", "roughness gain G", 1.0)
    for ax in axs:
        ax.axvspan(1, 50, color="#fbe9e7", lw=0, zorder=0); ax.axvspan(0.05, 1, color="#e7eff7", lw=0, zorder=0)
    axs[0].set_xlim(0.4, 4); axs[1].set_xlim(0.3, 12)
    for ax, ticks in ((axs[0], [0.5, 1, 2, 4]), (axs[1], [0.5, 1, 2, 5, 10])):
        ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
        ax.xaxis.set_minor_formatter(NullFormatter())
    axs[0].set_title("What deepens incisions?", fontsize=7.5)
    axs[1].set_title("What roughens the base?", fontsize=7.5)
    axs[1].text(0.98, 0.02, "roughens →", transform=axs[1].transAxes, ha="right", fontsize=6, color="#b03a2e")
    axs[1].text(0.02, 0.02, "← smooths", transform=axs[1].transAxes, ha="left", fontsize=6, color="#1f5f99")
    panel_label(axs[0], "c", x=-0.62, y=1.04); panel_label(axs[1], "d", x=-0.62, y=1.04)
    fig.text(0.5, 0.0, "c, d: low (blue) and high (red) value of each parameter, grey bar = range, black tick = baseline "
             "(4 seeds). Values labeled at each dot.", fontsize=5.8, color=MUTED, ha="center")
    save(fig, "FigED_incisions")
