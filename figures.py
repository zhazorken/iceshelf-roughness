"""Main and SI figures. Usage: python figures.py [results_dir ...]  -> figs/*.png, figs/*.pdf"""
import os, sys, glob, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from matplotlib import cm
from scipy.ndimage import gaussian_filter1d
from cmcrameri import cm as cmc
import shelfmelt as sm
import experiments as ex
import cmocean

OUT = "figs"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "Inter", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#333333", "axes.labelcolor": "#222222",
    "xtick.color": "#333333", "ytick.color": "#333333", "savefig.dpi": 300,
    "axes.titleweight": "semibold", "axes.titlelocation": "left", "legend.frameon": False,
    "figure.dpi": 150, "pdf.fonttype": 42,
})
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
# validated categorical palette (fixed order; gold carries direct labels)
COL = {"dnsQ": "#e07b00", "plume": "#7a7a2a", "plumeF": "#4d4d1a", "shear": "#1f5f99", "conv": "#d1495b", "dns": "#c98a14", "dnsF": "#6a4c93",
       "convP": "#2a9d8f", "theta": "#a8572a", "mixed": "#d1495b", "convF": "#6a4c93", "shearF": "#1f5f99"}
NAME = {"plume": "Plume law", "plumeF": "Plume law + flat stratification", "shear": "Shear", "conv": "Convection (sin$^{2/3}\\theta$)", "dns": "DNS law (convective-shear), tidal $u_*$", "dnsQ": "DNS law, quiet water",
        "dnsF": "DNS law + flat stratification", "convP": "Convection + closed-basin pooling",
        "theta": "DNS convection only ($u_*=0$)", "mixed": "Shear + convection",
        "convF": "sin$^{2/3}$ + flat stratification", "shearF": "Shear + flat stratification"}
DIV = cmc.vik
LIGHT = LightSource(azdeg=315, altdeg=40)

def panel_label(ax, s, x=-0.02, y=1.02, **kw):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right", **kw)

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig); print("wrote", name)

# ---------------------------------------------------------------- data
def low_angle_contrast(p, regime, ang=10.0):
    """C10: melt on a 10-degree face divided by mean smooth-shelf melt (no stratification closure)."""
    x_, y_, zb_, u_ = sm.setup(p); dT = sm.thermal_driving(zb_, p)
    sth = np.sin(np.arctan(np.abs(np.gradient(zb_, p["dx"])))); f = sm.REGIMES[regime]
    return float(f(dT, np.full_like(sth, np.sin(np.deg2rad(ang))), u_, p).mean() / f(dT, sth, u_, p).mean())

def load(dirs):
    R = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(f, allow_pickle=False)
            rec = pickle.loads(z["meta"].tobytes())
            if any(r["name"] == rec["name"] for r in R): continue
            rec.update({k: z[k] for k in z.files if k != "meta"})
            for k in ("G", "Pi", "R0", "Rf", "mbar", "inc999", "thin999", "thinmax", "flat", "steep_d", "pond", "rug"):
                if k in rec: rec[k] = float(rec[k])
            rec["Hmin"] = 1 - rec["thinmax"]
            p = dict(sm.P0, **rec["params"]); rec["params"] = p
            try:
                rec["inc0"] = seeded_inc(p); rec["Ainc"] = rec["inc999"] / rec["inc0"]
            except Exception:
                rec["inc0"] = rec["Ainc"] = np.nan
            rec["C"] = 1.0 if rec["regime"] == "shear" else ex.slope_contrast(p, rec["regime"]) \
                if rec["regime"] in ("convection", "plume") else rec.get("C", np.nan)
            rec["C10"] = low_angle_contrast(p, rec["regime"])
            try:
                x_, y_, zb_, u_ = sm.setup(p)
                sth = np.sin(np.arctan(np.abs(np.gradient(zb_, p["dx"]))))
                sup = np.exp(-(sth / np.sin(np.deg2rad(p["strat_theta_c"]))) ** 2) if p["strat_theta_c"] > 0 else 0.0
                rec["m_smooth"] = float(np.mean(sm.REGIMES[rec["regime"]](sm.thermal_driving(zb_, p) * (1 - sup),
                                                                         sth, u_, p))) * sm.SPY
                rec["enh"] = rec["mbar"] / rec["m_smooth"]
            except Exception:
                rec["enh"] = np.nan
            R.append(rec)
    # the original GL-slope test let the downstream draft run to ~0 m; replace it in the one-at-a-time
    # set by the fixed-endpoint version (same draft at GL and front for every slope)
    if any(r["group"] == "GL slope (fixed ends)" for r in R):
        R = [r for r in R if r["group"] != "GL slope"]
        for r in R:
            if r["group"] == "GL slope (fixed ends)" and r["label"] in ("2", "8"):
                r = r.copy(); r["group"] = "GL slope"; R.append(r)
    return R

_SEED_CACHE = {}
def seeded_inc(p):
    """99.9th-percentile relief of the seeded (unmelted) base, same filter and region as inc999."""
    key = tuple(p[k] for k in ("Lx", "Ly", "dx", "crev_H", "crev_wall", "crev_spacing", "chan_amp", "chan_angle", "seed",
                               "noise_amp", "U_ice"))
    if key not in _SEED_CACHE:
        q = sm.fill_times(p); x, y, _, _ = sm.setup(q)
        z = sm.highpass(sm.Seeder(y, q).prefill(x), q["dx"])
        _SEED_CACHE[key] = float(np.percentile(z[:, x > min(5e3, 0.25 * q["Lx"])], 99.9))
    return _SEED_CACHE[key]

def sel(R, group, label=None, variant=None):
    return [r for r in R if r["group"] == group and (label is None or r["label"] == label)
            and (variant is None or r["variant"] == variant)]

def ms(rs, key):
    v = np.array([r[key] for r in rs]); return (v.mean(), v.std(), v.min(), v.max(), len(v)) if len(v) else (np.nan,) * 4 + (0,)

def hillshade(z, dx, vlim, vexag=3.0):
    return LIGHT.shade(z, cmap=DIV, norm=Normalize(-vlim, vlim), blend_mode="soft", vert_exag=vexag, dx=dx, dy=dx)

# ---------------------------------------------------------------- Figure 1: setup and melt laws
def fig1(R):
    p = dict(sm.P0); x, y, zbar, ustar = sm.setup(p)
    H = sm.thickness(zbar, p); zs = zbar + H
    fig = plt.figure(figsize=(7.2, 2.7))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.55, 1, 1], height_ratios=[1, 1], wspace=0.42, hspace=1.0)
    ax = fig.add_subplot(gs[:, 0])
    zz = np.linspace(-1000, 0, 300); X, Z = np.meshgrid(x / 1e3, zz)
    T = p["T_gl"] + p["dTdz"] * (Z - p["z_gl"])
    T = np.ma.masked_where(Z > np.interp(X * 1e3, x, zbar), T)
    pc = ax.pcolormesh(X, Z, T, cmap=cmocean.cm.thermal, shading="auto", vmin=-1.9, vmax=1.5, rasterized=True)
    ax.fill_between(x / 1e3, zbar, zs, color="#eef3f7", lw=0)
    ax.plot(x / 1e3, zbar, color=INK, lw=0.9); ax.plot(x / 1e3, zs, color=INK, lw=0.6)
    ax.axhline(0, color="#7aa6c2", lw=0.6, ls=(0, (3, 2)))
    ax.annotate("", xy=(30, 190), xytext=(20, 190), arrowprops=dict(arrowstyle="->", lw=0.8, color=INK))
    ax.text(19, 190, "ice flow, $U$ ", ha="right", va="center", fontsize=6.5)
    ax.annotate("grounding line", xy=(0.2, -880), xytext=(3, -700), fontsize=6.3, color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=INK))
    ax.text(25, -250, "ice shelf", fontsize=7, color=MUTED, ha="center")
    ax.set_xlim(0, 40); ax.set_ylim(-1000, 250)
    ax.set_xlabel("distance from grounding line (km)"); ax.set_ylabel("elevation (m)")
    cb = fig.colorbar(pc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.22, aspect=30)
    cb.set_label("ocean temperature (°C)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    panel_label(ax, "a", x=-0.12)
    ax.set_title("Idealized 40-km shelf, linear $T(z)$", fontsize=7.5)

    # b: slope laws
    ax = fig.add_subplot(gs[:, 1])
    th = np.linspace(0.05, 90, 400); sth = np.sin(np.deg2rad(th))
    q = dict(p)
    for law, c, lab in (("sin23", COL["conv"], "sin$^{2/3}\\theta$"),):
        ax.plot(th, sm.slope_factor(sth, dict(q, conv_law=law)), color=c, lw=1.4)
        xl = {"sin23": 55, "theta": 52, "dns": 2}[law]
        yl = sm.slope_factor(np.sin(np.deg2rad([xl])), dict(q, conv_law=law))[0]
        dy = {"sin23": -0.1, "theta": -0.09, "dns": 0.1}[law]
        ax.text(12, 0.9, lab, color=c, fontsize=6.5, fontweight="semibold", ha="left")
    for r_, ls_ in ((0.0, "-"), (0.2, "--"), (0.5, ":")):
        ax.plot(th, np.maximum(th / 90.0, r_), color=COL["dns"], lw=1.4, ls=ls_)
        if r_: ax.text(52 if r_ == 0.5 else 30, r_ + 0.02, f"$u_*/u_c$ = {r_:g}", fontsize=5.6, color=COL["dns"], ha="left", va="bottom")
    ax.text(40, 0.08, "DNS law: max of\n$\\theta$-convection, shear", color=COL["dns"], fontsize=6.3, fontweight="semibold")

    ax.axvspan(0.3, 4, color="#dddddd", alpha=0.6, lw=0)
    ax.annotate("shelf slopes", xy=(2, 0.12), xytext=(8, 1.02), fontsize=6, color=MUTED,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))
    ax.set_xlim(0, 90); ax.set_ylim(0, 1.08); ax.set_xticks([0, 30, 60, 90])
    ax.set_xlabel("interface slope θ (°)"); ax.set_ylabel("melt / quiescent vertical-wall melt")
    ax.set_title("Slope laws", fontsize=7.5); panel_label(ax, "b", x=-0.2)

    # c, d: grounding-line seeding and u*
    ax = fig.add_subplot(gs[0, 2])
    q = sm.fill_times(dict(p, Ly=4e3)); xx, yy, _, _ = sm.setup(q)
    sd = sm.Seeder(yy, q); zf = sd.prefill(xx[:300])
    ax.pcolormesh(xx[:300] / 1e3, yy / 1e3, zf, cmap=DIV, vmin=-40, vmax=40, shading="auto", rasterized=True)
    ax.set_ylabel("y (km)"); ax.set_aspect("equal")
    ax.set_title("GL seeding: crevasses + noise", fontsize=7.5)
    ax.text(6.1, 2, "40 m\ncrevasses", fontsize=5.8, color=MUTED, va="center"); panel_label(ax, "c", x=-0.2)
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(x / 1e3, ustar * 1e3, color=COL["shear"], lw=1.2)
    uc = p["dns_uc90"] * (sm.thermal_driving(zbar, p) / p["dns_dTref"]) ** (1 / 3)
    ax.plot(x / 1e3, uc * 1e3, color=COL["dns"], lw=1.0, ls="--")
    ax.text(39, uc[-1] * 1e3 - 0.35, "$u_c$ (wall)", color=COL["dns"], fontsize=5.8, ha="right", va="top")
    dT = sm.thermal_driving(zbar, p)
    ax.set_xlabel("distance from GL (km)"); ax.set_ylabel("$u_*$ (mm s$^{-1}$)")
    ax.text(38, 0.6, f"ΔT at base: {dT[0]:.1f} → {dT[-1]:.1f} °C", ha="right", fontsize=5.8, color=MUTED)
    ax.set_title("Friction velocity", fontsize=7.5)
    ax.set_ylim(0, 4.5); ax.set_xlim(0, 40); panel_label(ax, "d", x=-0.2)
    save(fig, "Fig1_setup")

# ---------------------------------------------------------------- Figure 2: showcase base morphology
def fig2(R):
    order = ["shear", "conv", "dnsF", "convP"]
    recs = {v: sel(R, "showcase", "pig", v) for v in order}
    order = [v for v in order if recs[v]]
    if not order: print("fig2: no showcase yet"); return
    vlim = 50.0
    fig = plt.figure(figsize=(7.2, 1.0 * len(order) + 2.4))
    gs = fig.add_gridspec(len(order) + 1, 2, height_ratios=[1] * len(order) + [1.9],
                          width_ratios=[1.6, 1], hspace=0.35, wspace=0.18)
    for i, v in enumerate(order):
        r = recs[v][0]; dx = r["params"]["dx"]
        z = sm.highpass(r["field"].astype(float), dx)
        ax = fig.add_subplot(gs[i, :])
        ax.imshow(hillshade(z, dx, vlim), extent=[0, r["x"][-1] / 1e3 + dx / 2e3, 0, r["y"][-1] / 1e3 + dx / 2e3],
                  origin="lower", aspect="equal", interpolation="antialiased")
        ax.set_yticks([0, 3, 6]); ax.tick_params(length=2)
        if i < len(order) - 1: ax.set_xticklabels([])
        ax.spines[:].set_visible(False)
        ax.text(0.006, 0.9, NAME[v], transform=ax.transAxes, fontsize=7, fontweight="semibold", va="top",
                color=INK, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
        ax.text(0.994, 0.9, f"G = {r['G']:.2f}   $H_{{min}}/H$ = {r['Hmin']:.2f}", transform=ax.transAxes,
                fontsize=6.5, va="top", ha="right", color=INK,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
        panel_label(ax, "abcd"[i], x=-0.01, y=0.55)
        if i == 0: ax.set_ylabel("y (km)")
    ax.set_xlabel("distance from grounding line (km)")
    # e: centerline sections
    ax = fig.add_subplot(gs[-1, 0])
    for k, v in enumerate(order):
        r = recs[v][0]; xs = r["x"] / 1e3; z = sm.highpass(r["field"].astype(float), r["params"]["dx"])
        j = z.shape[0] // 2; off = -90 * k
        ax.fill_between(xs, z[j] + off, off + 70, color=COL[v], alpha=0.10, lw=0)
        ax.plot(xs, z[j] + off, color=COL[v], lw=0.8)
        ax.text(40.3, off, NAME[v].split(" (")[0].split(" +")[0] if v not in ("dnsF", "convP") else
                ("+ flat strat." if v == "dnsF" else "+ pooling"), color=COL[v], fontsize=6.3, va="center")
    ax.set_xlim(0, 40); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.plot([1, 1], [-330, -280], color=INK, lw=1); ax.text(1.4, -305, "50 m", fontsize=6, va="center")
    ax.set_xlabel("distance from grounding line (km)")
    ax.set_title("Centerline sections (ice shaded)", fontsize=7.5); panel_label(ax, "e", x=-0.02)
    # f: oblique 3D close-up of terraces
    v3 = "dnsF" if "dnsF" in order else order[-1]
    r = recs[v3][0]; dx = r["params"]["dx"]
    z = sm.highpass(r["field"].astype(float), dx)
    i0, i1 = int(24e3 / dx), int(34e3 / dx); st = 3
    zc = z[::st, i0:i1:st]; xc = r["x"][i0:i1:st] / 1e3; yc = r["y"][::st] / 1e3
    Xc, Yc = np.meshgrid(xc, yc)
    ax = fig.add_subplot(gs[-1, 1], projection="3d")
    ax.plot_surface(Xc, Yc, zc, facecolors=hillshade(zc, dx * st, vlim, 2.0), rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    ax.view_init(elev=-25, azim=-60); ax.set_box_aspect((10, 6, 2.2))
    ax.set_zlim(-vlim * 1.6, vlim * 1.6); ax.set_axis_off()
    ax.set_title(f"{NAME[v3]}, 24–34 km (from below)", fontsize=7, pad=-4)
    fig.text(0.665, 0.29, "f", fontsize=9, fontweight="bold")
    cax = fig.add_axes([0.13, 0.035, 0.3, 0.012])
    cb = fig.colorbar(cm.ScalarMappable(Normalize(-vlim, vlim), DIV), cax=cax, orientation="horizontal")
    cb.set_label("basal relief z′ (m, 2-km mean removed); red = melted into ice", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    save(fig, "Fig2_morphology")

# ---------------------------------------------------------------- Figure 3: slope contrast controls sign
def fig3(R):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1.25, 1.2, 1], wspace=0.38))
    ax = axs[0]
    labs = sorted({r["label"] for r in sel(R, "contrast")}, key=float)
    C = []; Gm = []; Gl = []; Gh = []
    for a in labs:
        rs = sel(R, "contrast", a); C.append(np.mean([r["C"] for r in rs])); m = ms(rs, "G")
        Gm.append(m[0]); Gl.append(m[2]); Gh.append(m[3])
    ax.axhspan(1, 10, color="#fbe9e7", lw=0); ax.axhspan(0.3, 1, color="#e7eff7", lw=0)
    ax.text(1.06, 4.0, "roughening", color="#b03a2e", fontsize=6.5, fontweight="semibold")
    ax.text(1.06, 0.45, "smoothing", color="#1f5f99", fontsize=6.5, fontweight="semibold")
    if C:
        o = np.argsort(C); C, Gm, Gl, Gh = (np.array(a)[o] for a in (C, Gm, Gl, Gh))
        ax.fill_between(C, Gl, Gh, color="#999999", alpha=0.25, lw=0)
        ax.plot(C, Gm, "-", color="#555555", lw=1.0, zorder=3)
        ax.plot(C, Gm, "o", ms=3, color="#555555", mec="white", mew=0.5, zorder=4, label="$a+(1-a)\\theta$ family")
    pts = {}
    for v in ["shear", "dns", "conv", "theta", "dnsF", "convP"]:
        rs = sel(R, "ens", "base", v)
        if not rs: continue
        Cv = 1.0 if v == "shear" else np.mean([r["C"] for r in rs]); m = ms(rs, "G"); pts[v] = (Cv, m[0])
        ax.errorbar(Cv, m[0], yerr=m[1], fmt="o" if v not in ("dnsF", "convP") else "D", ms=5, color=COL[v],
                    mec="white", mew=0.6, elinewidth=0.8, capsize=0, zorder=6)
    for v, dyx in (("shear", (4, -10)), ("dns", (6, -2)), ("conv", (5, 4)), ("theta", (-30, 6))):
        if v in pts: ax.annotate(NAME[v].replace("Convection ", "").strip("()") if v != "shear" else "shear",
                                 pts[v], textcoords="offset points", xytext=dyx, fontsize=6.3, color=COL[v],
                                 fontweight="semibold")
    for v, base in (("dnsF", "dns"), ("convP", "conv")):
        if v in pts and base in pts:
            ax.annotate("", xy=pts[v], xytext=pts[base], arrowprops=dict(arrowstyle="-|>", lw=0.9, color=COL[v],
                                                                          shrinkA=4, shrinkB=4))
            ax.annotate("+ flat strat." if v == "dnsF" else "+ pooling", pts[v], textcoords="offset points",
                        xytext=(5, -3 if v == "dnsF" else -9), fontsize=6.3, color=COL[v], fontweight="semibold")
    ax.axhline(1, color=INK, lw=0.6)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.3, 10); ax.set_xlim(0.9, 150)
    ax.set_yticks([0.3, 0.5, 1, 2, 5, 10]); ax.set_yticklabels(["0.3", "0.5", "1", "2", "5", "10"])
    ax.set_xlabel("slope contrast $C$ = $m$(90°) / $m$(shelf slope)")
    ax.set_ylabel("roughness gain $G$ (downstream / seeded)")
    ax.legend(loc="upper left", fontsize=6, borderaxespad=0.2)
    panel_label(ax, "a", x=-0.16); ax.set_title("Slope contrast sets the sign", fontsize=7.5)

    ax = axs[1]
    for v in ["shear", "dns", "conv", "dnsF", "convP"]:
        rs = sel(R, "ens", "base", v)
        if not rs: continue
        A = np.array([gaussian_filter1d(r["rms"], 3) for r in rs]); xs = rs[0]["x"] / 1e3
        ax.fill_between(xs, A.mean(0) - A.std(0), A.mean(0) + A.std(0), color=COL[v], alpha=0.18, lw=0)
        ax.plot(xs, A.mean(0), color=COL[v], lw=1.1)
        ax.text(40.5, A.mean(0)[-40:].mean(), {"shear": "shear", "dns": "DNS", "conv": "sin$^{2/3}$",
                "dnsF": "DNS + flat", "convP": "pooling"}[v], color=COL[v], fontsize=6.2, va="center",
                fontweight="semibold")
    ax.set_xlim(0, 40); ax.set_ylim(bottom=0); ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_xlabel("distance from grounding line (km)"); ax.set_ylabel("rms roughness, < 2 km scales (m)")
    panel_label(ax, "b", x=-0.14); ax.set_title("Roughness downstream (4 seeds)", fontsize=7.5)

    ax = axs[2]
    for v in ["shear", "conv", "dnsF"]:
        rs = sel(R, "ens", "base", v)
        if not rs: continue
        h = np.mean([r["slope_hist"] for r in rs], 0); b = rs[0]["slope_bins"]; c = 0.5 * (b[1:] + b[:-1])
        ax.plot(c, np.maximum(h, 1e-7), color=COL[v], lw=1.1, drawstyle="steps-mid")
    ax.set_yscale("log"); ax.set_xlim(0, 70); ax.set_ylim(1e-5, 1)
    ax.set_xlabel("local slope (°)"); ax.set_ylabel("area fraction per degree")
    ax.axvspan(30, 70, color="#f2f2f2", lw=0); ax.text(50, 0.3, "steep faces", fontsize=6, color=MUTED, ha="center")
    panel_label(ax, "c", x=-0.2); ax.set_title("Slope distribution", fontsize=7.5)
    save(fig, "Fig3_contrast")

# ---------------------------------------------------------------- Figure 4: vulnerability
OAT_GROUPS = ["thermal forcing", "ambient stratification", "crevasse height", "crevasse spacing", "u* magnitude",
              "GL slope", "ice speed", "slope smoothing", "creep closure"]
PRETTY = {"thermal forcing": "Thermal forcing, $T_{GL}$ (°C)", "ambient stratification": "Ambient d$T$/d$z$ (°C/m)",
          "crevasse height": "GL crevasse height (m)", "crevasse spacing": "Crevasse spacing (m)",
          "u* magnitude": "$u_*$ multiplier", "GL slope": "GL basal slope (°)", "ice speed": "Ice speed (m/yr)",
          "slope smoothing": "Slope smoothing (m)", "creep closure": "Creep viscosity (Pa s)"}

def tornado(ax, R, v, key, title, xl):
    base = sel(R, "ens", "base", v)
    if not base: return
    b0 = ms(base, key)[0]; rows = []
    for g in OAT_GROUPS:
        rs = [r for r in R if r["group"] == g and r["variant"] == v]
        if not rs: continue
        vals = sorted({r["label"] for r in rs}, key=float)
        ds = [(lab, ms([r for r in rs if r["label"] == lab], key)[0] - b0) for lab in vals]
        rows.append((g, ds, max(abs(d) for _, d in ds)))
    rows.sort(key=lambda t: t[2])
    for i, (g, ds, _) in enumerate(rows):
        lo, hi = ds[0], ds[-1]
        for (lab, d), c in ((lo, "#7fa7c9"), (hi, "#c0392b")):
            ax.barh(i, d, left=b0, height=0.62, color=c, alpha=0.85, lw=0)
            ax.text(b0 + d + (0.004 if d >= 0 else -0.004) * (abs(b0) + 1) * (5 if key == "G" else 1), i, lab,
                    va="center", ha="left" if d >= 0 else "right", fontsize=5.6, color=MUTED)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([PRETTY[g] for g, _, _ in rows], fontsize=6.3)
    ax.axvline(b0, color=INK, lw=0.7); ax.set_xlabel(xl); ax.set_title(title, fontsize=7.5)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)

def fig4(R):
    fig = plt.figure(figsize=(7.2, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.15], wspace=0.75)
    ax = fig.add_subplot(gs[0])
    order = ["convP", "shear", "dns", "conv", "theta", "dnsF"]
    k = 0
    for v in order:
        rs = sel(R, "ens", "base", v)
        if not rs: continue
        vals = np.array([r["Hmin"] for r in rs])
        ax.plot(vals, np.full_like(vals, k) + np.linspace(-0.12, 0.12, len(vals)), "o", ms=2.5, color=COL[v], alpha=0.5)
        ax.plot(vals.mean(), k, "D" if v in ("dnsF", "convP") else "o", ms=5.5, color=COL[v], mec="white", mew=0.6)
        k += 1
    ax.set_yticks(range(k)); ax.set_yticklabels(
        [{"convP": "Pooling", "shear": "Shear", "dns": "DNS law", "conv": "sin$^{2/3}$", "theta": "θ law",
          "dnsF": "DNS + flat strat."}[v] for v in order if sel(R, "ens", "base", v)], fontsize=6.5)
    ax.set_xlabel("thinnest ice, $H_{min}/H$"); ax.invert_xaxis(); ax.grid(axis="x", color=GRID, lw=0.5)
    ax.axvspan(0.0, 0.7, color="#fbe9e7", lw=0, zorder=0)
    ax.set_xlim(1.0, min(0.45, ax.get_xlim()[1])); ax.set_title("Thinnest points (4 seeds)", fontsize=7.5)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    panel_label(ax, "a", x=-0.05)
    ax = fig.add_subplot(gs[1]); tornado(ax, R, "dnsF", "Hmin", "What thins it most? (DNS + flat)",
                                         "$H_{min}/H$ (baseline line)")
    panel_label(ax, "b", x=-0.05)
    ax = fig.add_subplot(gs[2]); tornado(ax, R, "dnsF", "G", "What roughens it most?", "roughness gain $G$")
    panel_label(ax, "c", x=-0.05)
    fig.text(0.42, -0.02, "bars: low (blue) and high (red) parameter value; labels give the value", fontsize=6,
             color=MUTED)
    save(fig, "Fig4_vulnerability")

# ---------------------------------------------------------------- SI figures
def figS_resolution(R):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(wspace=0.45))
    for v, ls in (("shear", "-"), ("conv", "-"), ("dnsF", "-")):
        rs = sorted(sel(R, "resolution", variant=v), key=lambda r: float(r["label"]))
        if not rs: continue
        dxs = [float(r["label"]) for r in rs]
        axs[0].plot(dxs, [r["Rf"] for r in rs], "o-", color=COL[v], ms=3.5, lw=1, label=NAME[v])
        axs[1].plot(dxs, [r["inc999"] for r in rs], "o-", color=COL[v], ms=3.5, lw=1)
        axs[2].plot(dxs, [r["Hmin"] for r in rs], "o-", color=COL[v], ms=3.5, lw=1)
    rs = sorted(sel(R, "resolution-scaled", variant="conv"), key=lambda r: float(r["label"]))
    if rs:
        dxs = [float(r["label"]) for r in rs]
        axs[0].plot(dxs, [r["Rf"] for r in rs], "s--", color=COL["conv"], mfc="white", ms=3.5, lw=1,
                    label="sin$^{2/3}$, smoothing = 2Δx")
        axs[1].plot(dxs, [r["inc999"] for r in rs], "s--", color=COL["conv"], mfc="white", ms=3.5, lw=1)
        axs[2].plot(dxs, [r["Hmin"] for r in rs], "s--", color=COL["conv"], mfc="white", ms=3.5, lw=1)
    for ax, yl, l in zip(axs, ["downstream rms (m)", "99.9% incision (m)", "$H_{min}/H$"], "abc"):
        ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 40]); ax.set_xticklabels(["5", "10", "20", "40"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("grid spacing Δx (m)"); ax.set_ylabel(yl); ax.invert_xaxis(); panel_label(ax, l, x=-0.2)
    axs[0].legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(1.9, -0.32), ncol=4)
    fig.suptitle("S1. Resolution sensitivity (20 × 1 km domain; slope smoothing fixed at 40 m unless noted)",
                 fontsize=7.5, x=0.02, ha="left", y=1.04)
    save(fig, "FigS1_resolution")

def figS_oat(R):
    import main_figs as MF
    fig, axs = plt.subplots(4, 3, figsize=(7.2, 9.6), gridspec_kw=dict(wspace=0.12, hspace=0.55))
    for i, v in enumerate(["shear", "conv", "plume", "dnsF"]):
        for j, (key, xl, log, ref) in enumerate((("G", "roughness gain G", True, 1.0), ("Hmin", "$H_{min}/H$", False, 1.0),
                                                  ("Ainc", "incision ratio A", True, 1.0))):
            MF.rangeplot(axs[i, j], R, v, key, xl, ref, log=log, fixed=True, labels=(j == 0), annotate=False)
            if j == 0: axs[i, j].set_title(NAME[v], fontsize=7.5, loc="left", x=-0.9)
            if log:
                from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
                axs[i, j].xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
                axs[i, j].xaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t:g}"))
                axs[i, j].xaxis.set_minor_formatter(NullFormatter())
    fig.subplots_adjust(left=0.25)
    save(fig, "FigS2_oat")

def figS_ens(R):
    vs = ["shear", "shearF", "mixed", "dns", "dnsQ", "dnsF", "conv", "convF", "theta", "convP"]
    fig, axs = plt.subplots(3, 3, figsize=(7.2, 6.0), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.6, wspace=0.15))
    for ax, v in zip(axs.flat, vs):
        rs = sel(R, "ens", "base", v)
        for r in rs: ax.plot(r["x"] / 1e3, gaussian_filter1d(r["rms"], 3), color=COL[v], lw=0.6, alpha=0.7)
        g = ms(rs, "G"); ax.set_title(f"{NAME[v]}\nG = {g[0]:.2f} ± {g[1]:.2f} (n={g[4]})", fontsize=6.5)
        ax.grid(color=GRID, lw=0.4)
    for ax in axs[-1]: ax.set_xlabel("km from GL")
    for ax in axs[:, 0]: ax.set_ylabel("rms (m)")
    fig.suptitle("S3. Seed ensembles: rms roughness for every physics variant", fontsize=7.5, x=0.02, ha="left")
    save(fig, "FigS3_ensembles")

def figS_channels(R):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True, gridspec_kw=dict(wspace=0.1))
    for ax, ang in zip(axs, ["0", "30"]):
        for v in ["shear", "conv", "dnsF"]:
            rs = sel(R, "channels", ang, v)
            for k, r in enumerate(rs):
                ax.plot(r["x"] / 1e3, gaussian_filter1d(r["chan"], 5), color=COL[v], lw=0.9,
                        label=NAME[v] if k == 0 else None)
        ax.axhline(1, color=INK, lw=0.5); ax.set_yscale("log"); ax.set_xlabel("km from GL")
        ax.set_title(f"channels at {ang}° to flow (10 m, 1 km wavelength)", fontsize=7)
    axs[0].set_ylabel("channel amplitude / seeded"); axs[0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(1.1, -0.28), ncol=3)
    fig.suptitle("S4. GL-seeded channels: preserved by shear, scrambled by sin$^{2/3}$ convection, oblique ones grow with flat stratification", fontsize=7.5, x=0.02,
                 ha="left", y=1.05)
    save(fig, "FigS4_channels")

def figS_profiles(R, group, fname, title):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, v in zip(axs, ["shear", "conv", "dnsF"]):
        rs = sorted([r for r in R if r["group"] == group and r["variant"] == v], key=lambda r: float(r["label"]))
        b = sel(R, "ens", "base", v)
        if b: ax.plot(b[0]["x"] / 1e3, gaussian_filter1d(np.mean([r["rms"] for r in b], 0), 3), color="#aaaaaa",
                      lw=1.6, label="baseline")
        labs = sorted({r["label"] for r in rs}, key=float)
        cols = cmc.batlow(np.linspace(0.15, 0.85, max(len(labs), 1)))
        for lab, c in zip(labs, cols):
            A = np.mean([r["rms"] for r in rs if r["label"] == lab], 0)
            txt = f"10$^{{{int(np.log10(float(lab)))}}}$" if float(lab) >= 1e5 else lab
            ax.plot(rs[0]["x"] / 1e3, gaussian_filter1d(A, 3), color=c, lw=1.1, label=txt)
        ax.set_title(NAME[v], fontsize=7); ax.set_xlabel("km from GL"); ax.legend(fontsize=5.8)
    axs[0].set_ylabel("rms roughness (m)")
    fig.suptitle(title, fontsize=7.5, x=0.02, ha="left", y=1.05)
    save(fig, fname)

def figS_strat(R):
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.2), gridspec_kw=dict(wspace=0.45, hspace=0.25), sharex="col")
    specs = [("flat-strat angle", "dnsF", "critical slope θ$_c$ (°)", 3.0),
             ("strat timescale", "dnsF", "relaxation time τ (yr)", 0.5),
             ("pool depth scale", "convP", "pond depth scale h (m)", 5.0)]
    for c, (g, v, xl, bv) in enumerate(specs):
        rs = [r for r in R if r["group"] == g]
        base = sel(R, "ens", "base", v)
        labs = sorted({r["label"] for r in rs}, key=float)
        pts = sorted([(bv, ms(base, "G")[0], ms(base, "Ainc")[0])] +
                     [(float(l), ms([r for r in rs if r["label"] == l], "G")[0],
                       ms([r for r in rs if r["label"] == l], "Ainc")[0]) for l in labs])
        for k, (key, yl) in enumerate(((1, "roughness gain G"), (2, "incision / seeded depth"))):
            ax = axs[k, c]
            ax.plot([p[0] for p in pts], [p[key] for p in pts], "-o", color=COL[v], ms=4, lw=1.1)
            ax.plot(bv, [p[key] for p in pts if p[0] == bv][0], "o", ms=7, mfc="none", mec=INK, mew=1)
            ax.axhline(1, color=MUTED, lw=0.6, ls=":")
            ax.set_xscale("log"); ax.grid(color=GRID, lw=0.4)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xticks([p[0] for p in pts])
            if c == 0: ax.set_ylabel(yl)
            if k == 1: ax.set_xlabel(xl)
        axs[0, c].set_title({"dnsF": "DNS law + flat strat.", "convP": "sin$^{2/3}$ + ponds"}[v], fontsize=7)
    fig.suptitle("S7. Stratification parameters (open circle = baseline)", fontsize=7.5, x=0.02, ha="left", y=1.0)
    save(fig, "FigS7_stratification")

def figS_snaps(R):
    rs = sel(R, "showcase", "pig", "conv")
    if not rs or "snaps" not in rs[0]: return
    r = rs[0]; S = r["snaps"]; t = r["snap_t"]; idx = np.linspace(0, len(t) - 1, 5).astype(int)
    fig, axs = plt.subplots(len(idx), 1, figsize=(7.2, 4.4), sharex=True, gridspec_kw=dict(hspace=0.15))
    for ax, i in zip(axs, idx):
        z = sm.highpass(S[i].astype(float), 40.0)
        ax.imshow(hillshade(z, 40.0, 50), extent=[0, 40, 0, 6], origin="lower", aspect="equal")
        ax.text(0.005, 0.85, f"t = {t[i]:.0f} yr", transform=ax.transAxes, fontsize=6.5,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1)); ax.set_yticks([])
    axs[-1].set_xlabel("km from GL")
    fig.suptitle("S8. Evolution of the sin^(2/3) showcase run", fontsize=7.5, x=0.02, ha="left")
    save(fig, "FigS8_evolution")

if __name__ == "__main__":
    dirs = sys.argv[1:] or ["results"]
    R = load(dirs); print(len(R), "results")
    import main_figs as MF
    for f in (fig1, MF.fig2, MF.fig3, MF.fig4, figS_resolution, figS_oat, figS_ens, figS_channels, figS_strat, figS_snaps):
        try: f(R)
        except Exception as e: print("FAILED", f.__name__, repr(e))
    try: figS_terrace(R)
    except Exception as e: print("FAILED terrace", repr(e))
    try:
        figS_profiles(R, "creep closure", "FigS5_creep", "S5. Viscous creep closure (η in Pa s)")
        figS_profiles(R, "ice speed", "FigS6_icespeed", "S6. Ice speed (m/yr); sets the advection number Π")
    except Exception as e: print("FAILED profiles", repr(e))

def figS_terrace(R):
    """S9: melt dominance and steps on a uniform 4-degree base."""
    import main_figs as MF
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw=dict(width_ratios=[1, 1, 1.6], wspace=0.4))
    def steep(r):
        f = r["field40"].astype(float); zb = r["zbar"][::4][None, :f.shape[1]] + f
        gx = np.gradient(zb, 40., axis=1); gy = (np.roll(zb, -1, 0) - np.roll(zb, 1, 0)) / 80.
        return (np.degrees(np.arctan(np.hypot(gx, gy))) > 15).mean()
    for v in ["plume", "plumeF", "dns", "dnsF"]:
        labs = sorted({r["label"] for r in sel(R, "terrace")}, key=float)
        M = [float(l) for l in labs]
        G = [ms(sel(R, "terrace", l, v), "G")[0] for l in labs]
        S = [np.mean([steep(r) for r in sel(R, "terrace", l, v)]) for l in labs]
        ls = "-" if v.endswith("F") else ":"
        axs[0].plot(M, G, ls, marker=MF.MARK[v], color=COL[v], ms=4, lw=1.1, label=MF.SHORT[v])
        axs[1].plot(M, S, ls, marker=MF.MARK[v], color=COL[v], ms=4, lw=1.1)
    axs[0].axhline(1, color=INK, lw=0.6); axs[0].set_xlabel("melt dominance M"); axs[0].set_ylabel("roughness gain G")
    axs[0].legend(fontsize=5.6, loc="upper center", bbox_to_anchor=(1.15, -0.3), ncol=2); axs[1].set_xlabel("melt dominance M"); axs[1].set_ylabel("area steeper than 15°")
    ax = axs[2]
    for k, (lab, v) in enumerate((("0.25", "plumeF"), ("1", "plumeF"), ("1", "plume"))):
        r = sel(R, "terrace", lab, v)[0]; z = r["zbar"] + r["centerline"]
        ax.plot(r["x"] / 1e3, z - 60 * k, color=["#9ac17a", COL["plumeF"], "#888888"][k], lw=1,
                label=f"{MF.SHORT[v]}, M = {lab}")
    ax.set_xlabel("km from GL"); ax.set_ylabel("draft (m, offset)"); ax.legend(fontsize=5.6)
    for a, l in zip(axs, "abc"): panel_label(a, l, x=-0.2)
    fig.suptitle("S9. Steps on a uniform 4° base need melt-dominated thinning and stratified flats", fontsize=7.5,
                 x=0.02, ha="left", y=1.06)
    save(fig, "FigS9_terraces")

def figS_v3(R):
    """S10 aging ponds, S11 slope-smoothing scale at 5 m, S12 GL slope with fixed endpoints."""
    import main_figs as MF
    COL.update({"shearA": COL["shear"], "convA": COL["conv"], "dnsA": COL["dns"], "plumeA": "#4f9a2f"})
    # S10
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.45))
    rows = [("shear", "shearA"), ("conv", "convA"), ("dns", "dnsA"), ("plume", "plumeA")]
    for i, (b, a) in enumerate(rows):
        for key, ax in (("G", axs[0]), ("inc999", axs[1])):
            rb, ra, rp = sel(R, "ens", "base", b), sel(R, "ens", "base", a), sel(R, "ens", "base", b + "P")
            if rb: ax.plot(ms(rb, key)[0], i + 0.15, "o", color=COL.get(b, "#4f9a2f"), ms=5, mfc="white", mew=1.2)
            if ra:
                ax.errorbar(ms(ra, key)[0], i - 0.15, xerr=ms(ra, key)[1], fmt="D", color=COL.get(b, "#4f9a2f"), ms=5,
                            mec="white", mew=0.6, elinewidth=0.8)
                if rb: ax.annotate("", xy=(ms(ra, key)[0], i - 0.15), xytext=(ms(rb, key)[0], i + 0.15),
                                   arrowprops=dict(arrowstyle="-|>", lw=0.7, color=MUTED, shrinkA=3, shrinkB=3))
            if rp: ax.plot(ms(rp, key)[0], i - 0.15, "s", color="#2a9d8f", ms=4)
    for ax, xl in zip(axs, ["roughness gain G", "deepest incision (m)"]):
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([MF.SHORT[b] for b, _ in rows], fontsize=6.5)
        ax.set_xlabel(xl); ax.grid(axis="x", color=GRID, lw=0.4); ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axs[0].axvline(1, color=INK, lw=0.6); axs[1].axvline(40, color=MUTED, lw=0.6, ls=":")
    axs[0].text(0.98, 0.02, "open circle: no ponds\ndiamond: aging ponds\nsquare: fixed-target ponds", transform=axs[0].transAxes,
                fontsize=5.6, color=MUTED, ha="right")
    fig.suptitle("S10. Ponds whose stratification builds with age (freshwater accumulates in closed basins)",
                 fontsize=7.5, x=0.02, ha="left", y=1.04)
    save(fig, "FigS10_agingponds")
    # S11
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for v in ["shear", "dns", "conv", "plume", "dnsF"]:
        rs = sel(R, "smoothing-5m", variant=v) + [dict(r, label="40") for r in sel(R, "resolution", "5", v)]
        rs = sorted({r["label"]: r for r in rs}.values(), key=lambda r: float(r["label"]))
        if rs: ax.plot([float(r["label"]) for r in rs], [r["G"] for r in rs], "-" + MF.MARK[v], color=COL[v], ms=4,
                       lw=1.1, label=MF.SHORT[v])
    ax.axhline(1, color=INK, lw=0.6); ax.set_xscale("log"); ax.set_xticks([10, 20, 40]); ax.set_xticklabels(["10", "20", "40"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("slope-smoothing (boundary-layer) scale (m)"); ax.set_ylabel("roughness gain G"); ax.legend(fontsize=5.6)
    ax.set_title("Slope scale at 5 m resolution", fontsize=7.5)
    save(fig, "FigS11_slopescale")
    # S12
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw=dict(wspace=0.45))
    for v in ["shear", "conv", "plume", "dnsF"]:
        labs = sorted({r["label"] for r in sel(R, "GL slope (fixed ends)", variant=v)}, key=float)
        if not labs: continue
        th = [float(l) for l in labs]
        for ax, key in zip(axs[:2], ["G", "inc999"]):
            ax.plot(th, [ms(sel(R, "GL slope (fixed ends)", l, v), key)[0] for l in labs], "-" + MF.MARK[v],
                    color=COL[v], ms=4, lw=1.1, label=MF.SHORT[v])
    axs[0].axhline(1, color=INK, lw=0.6); axs[0].set_ylabel("roughness gain G"); axs[1].set_ylabel("deepest incision (m)")
    for ax in axs[:2]: ax.set_xlabel("GL basal slope (°)")
    axs[1].legend(fontsize=5.6, loc="center right")
    p = dict(sm.P0)
    for th, c in zip([2, 4, 8, 12], cmc.batlow(np.linspace(0.15, 0.85, 4))):
        x, y, zb, u = sm.setup(dict(p, theta_gl=th, z_front=-367.57)); axs[2].plot(x / 1e3, zb, color=c, lw=1.1, label=f"{th}°")
    axs[2].set_xlabel("km from GL"); axs[2].set_ylabel("mean draft (m)"); axs[2].legend(fontsize=5.6, title="GL slope", title_fontsize=5.6)
    for a, l in zip(axs, "abc"): panel_label(a, l, x=-0.22)
    fig.suptitle("S12. GL slope with fixed draft at the GL (-900 m) and front (-368 m)", fontsize=7.5, x=0.02, ha="left", y=1.05)
    save(fig, "FigS12_glslope")

def figS_stair(R):
    """S13: search for staircase steps on the melt-dominated 4-degree base (M = 1)."""
    import main_figs as MF
    from matplotlib.colors import LightSource
    cases = [("none", "flat stratification only"), ("pool-h2-tau2", "+ fixed-target ponds"),
             ("layers-20", "+ 20 m layered melt"), ("age-hc5", "+ aging ponds (h$_c$ = 5 m)"),
             ("age-hc0.5", "+ aging ponds (h$_c$ = 0.5 m)")]
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1], hspace=0.5)
    ax = fig.add_subplot(gs[0])
    for i, (lab, txt) in enumerate(cases):
        rr = [r for r in sel(R, "stair", lab, "dnsF") if r["params"]["melt_mult"] > 20]
        if not rr: continue
        r = rr[0]; z = r["zbar"] + r["field"][r["field"].shape[0] // 2].astype(float) - 120 * i
        c = ["#9a9a9a", "#2a9d8f", "#c98a14", "#8e6bbf", "#4b2e83"][i]
        ax.plot(r["x"] / 1e3, z, color=c, lw=1.1)
        ax.text(5.05, z[-30:].mean(), f"{txt}\nG = {r['G']:.1f}", fontsize=5.8, color=c, va="center")
    ax.set_xlim(0, 5); ax.set_ylabel("draft (m), offset 120 m per case"); ax.set_xlabel("km from GL")
    ax.plot([0.1, 0.1], [ax.get_ylim()[0] + 10, ax.get_ylim()[0] + 60], color=INK, lw=1.2)
    ax.text(0.15, ax.get_ylim()[0] + 35, "50 m", fontsize=6, va="center")
    ax.set_title("Offset-linear law + flat stratification on a uniform 4° base, melt-dominated thinning (M = 1)", fontsize=7)
    panel_label(ax, "a", x=-0.07)
    ax = fig.add_subplot(gs[1])
    r = [r for r in sel(R, "stair", "age-hc0.5", "dnsF") if r["params"]["melt_mult"] > 20][0]
    zb = r["zbar"][None, :] + r["field"].astype(float)
    ax.imshow(LightSource(315, 35).shade(zb, cmap=cmc.lajolla, vert_exag=3, dx=10, dy=10, blend_mode="soft"),
              extent=[0, 5, 0, 1], origin="lower", aspect="auto")
    ax.set_xlabel("km from GL"); ax.set_ylabel("y (km)")
    ax.set_title("Aging ponds (h$_c$ = 0.5 m): plan view of absolute draft, hillshaded. Flats separated by "
                 "cross-flow scarps", fontsize=7)
    panel_label(ax, "b", x=-0.07)
    save(fig, "FigS13_staircase")
