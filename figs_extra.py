"""Extra SI figures: S14 (T(z) layering) and S15 (surface hydrofracture)."""
import pickle, glob, numpy as np, matplotlib.pyplot as plt
from main_figs import *          # style, COL, SHORT
from figures import save, panel_label, INK, MUTED, GRID

def d_at(D, P, q=0.01):
    i = np.argmax(P >= q); return D[i] if P[i] >= q else np.nan

def figS_surface():
    S = pickle.load(open("results_surface_frac.pkl", "rb")); D, res = S["D"], S["res"]
    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.22], wspace=0.28, hspace=0.55)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    lab = {"shear": "Shear", "dns": "DNS law", "conv": "sin$^{2/3}$ convection", "theta": "DNS convection, $u_*=0$",
           "plume": "Plume law", "dnsF": "DNS law + flat strat.", "convF": "sin$^{2/3}$ + flat strat.",
           "convA": "sin$^{2/3}$ + aging ponds"}
    col = dict(COL, convA="#2a9d8f", convF="#d1495b", dns="#c98a14")
    ls_ = {"convF": (0, (4, 1.5)), "dnsF": (0, (4, 1.5)), "convA": (0, (1, 1))}
    nz = lambda P: np.where(P > 0, P, np.nan)
    for ax in axs:
        ax.axvspan(0.02, 0.1, color="#e9eef4", zorder=0, lw=0)
        ax.set_yscale("log"); ax.set_ylim(1e-4, 1.3); ax.set_xlim(0, 1)
        ax.set_xlabel("crevasse penetration, $d/H$")
        ax.grid(True, color=GRID, lw=0.5, zorder=0)
    axs[0].text(0.06, 2e-4, "dry", ha="center", fontsize=6, color=MUTED)
    axs[0].set_ylabel("fraction of crevasses cutting through")
    h1 = []
    for k, nm in lab.items():
        h1 += axs[0].plot(D, nz(res[(k, False)]), color=col[k], lw=1.3, ls=ls_.get(k, "-"), label=nm)
        axs[2].plot(D, nz(res[(k, "local")]), color=col[k], lw=1.3, ls=ls_.get(k, "-"))
    h1 += axs[0].plot(D, nz(res[("seed40", False)]), color=INK, lw=0.8, ls=":", label="Seeded base, no melt")
    axs[2].plot(D, nz(res[("seed40", "local")]), color=INK, lw=0.8, ls=":")
    axs[0].set_title("40 m basal crevasses", pad=4); panel_label(axs[0], "a")
    axs[2].set_title("40 m, lateral-mean thinning removed", pad=4); panel_label(axs[2], "c")
    ax = axs[1]
    for k in ("shear", "conv", "plume", "dnsF"):
        for h, ls in ((250, "-"), (120, (0, (3, 1.5)))):
            ax.plot(D, nz(res[(f"crev{h}_{k}", False)]), color=col[k], lw=1.3, ls=ls)
    for h, ls in ((250, ":"), (120, (0, (1, 2)))):
        ax.plot(D, nz(res[(f"seed{h}", False)]), color=INK, lw=0.8, ls=ls)
    ax.set_title("120 m (dashed), 250 m (solid)", pad=4); panel_label(ax, "b")
    lg = fig.add_subplot(gs[1, :]); lg.axis("off")
    lg.legend(handles=h1, loc="center", ncol=5, fontsize=6.2, handlelength=2.4, columnspacing=1.4)
    save(fig, "FigS15_surface_fracture")
    print("d/H at which 1% cut through (random, aligned, local):")
    keys = list(lab) + ["seed40"] + [f"crev{h}_{v}" for h in (120, 250) for v in ("shear", "conv", "plume", "dnsF")] + ["seed120", "seed250"]
    for k in keys:
        print(f"  {k:14s} {d_at(D, res[(k, False)]):.2f} {d_at(D, res[(k, True)]):.2f} {d_at(D, res[(k, 'local')]):.2f}")


def _ld(name):
    import pickle
    z = np.load(f"results/{name}.npz"); r = pickle.loads(z["meta"].tobytes())
    r.update({k: z[k] for k in z.files if k != "meta"}); return r

def figS_tz():
    v = "dnsF"; rises = [0, 10, 20, 30, 50]
    cm_r = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, len(rises)))
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9], hspace=0.5, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    for c, r in zip(cm_r, rises):
        d = _ld(f"stair__tz0.1-rise{r}__M1__{v}__s1")
        ax.plot(d["tz_anom"], d["tz_z"], color=c, lw=1.0, label=f"r = {r} m")
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("$T'$ (°C)"); ax.set_ylabel("depth (m)"); ax.set_ylim(-930, -600)
    ax.set_title("M = 1, $\\gamma$ = 0.1", pad=4); ax.legend(fontsize=5.8, loc="upper left")
    panel_label(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    for c, r in ((cm_r[1], 10), (cm_r[4], 40)):
        d = _ld(f"stair__tz1-rise{r}__M0.6__{v}__s1")
        ax.plot(d["tz_anom"], d["tz_z"], color=c, lw=0.9, label=f"r = {r} m")
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("$T'$ (°C)"); ax.set_ylim(-930, -600); ax.set_yticklabels([])
    ax.set_title("M = 0.6, $\\gamma$ = 1", pad=4); ax.legend(fontsize=5.8, loc="upper left")
    panel_label(ax, "b")
    ax = fig.add_subplot(gs[0, 2])
    def flat(d):
        F = d["field"] + d["zbar"][None, :]; x = d["x"]
        g = np.degrees(np.arctan(np.abs(np.gradient(F, d["params"]["dx"], axis=1))))[:, (x > 1000) & (x < 4500)]
        return 100 * (g < 1).mean()
    base = flat(_ld(f"stair__none__M1__{v}__s1"))
    series = [("M = 1, $\\gamma$ = 0.1", [f"stair__tz0.1-rise{r}__M1__{v}__s1" for r in rises], rises, "#6a4c93", "-"),
              ("M = 1, $\\gamma$ = 0.3", ["stair__tz0.3__M1__dnsF__s1"] + [f"stair__tz0.3-rise{r}__M1__{v}__s1" for r in (10, 20, 50)], [0, 10, 20, 50], "#6a4c93", "--"),
              ("M = 0.7, $\\gamma$ = 1", [f"stair__tz1-rise{r}__M0.7__{v}__s1" for r in (10, 20, 40)], [10, 20, 40], "#c98a14", "-"),
              ("M = 0.5, $\\gamma$ = 1", [f"stair__tz1-rise{r}__M0.5__{v}__s1" for r in (10, 20, 40)], [10, 20, 40], "#c98a14", "--")]
    for lab, nms, rs, c, ls in series:
        ax.plot(rs, [flat(_ld(n)) for n in nms], color=c, ls=ls, marker="o", ms=2.5, lw=1, label=lab)
    ax.axhline(base, color=INK, lw=0.6, ls=":"); ax.text(48, base + 2, "no feedback", fontsize=5.8, color=MUTED, ha="right", va="bottom")
    ax.set_xlabel("rise height r (m)"); ax.set_ylabel("base flatter than 1° (%)")
    ax.set_title("Near-horizontal base", pad=4); ax.legend(fontsize=5.6, loc="center right")
    panel_label(ax, "c")
    for j, (nm, ttl) in enumerate([(f"stair__none__M1__{v}__s1", "No feedback, M = 1"),
                                   (f"stair__tz0.1-rise20__M1__{v}__s1", "r = 20 m, M = 1"),
                                   (f"stair__tz1-rise40__M0.7__{v}__s1", "r = 40 m, M = 0.7")]):
        d = _ld(nm); ax = fig.add_subplot(gs[1, j])
        F = d["field"] + d["zbar"][None, :]; x = d["x"] / 1e3
        for k, yy in enumerate((0.25, 0.5, 0.75)):
            iy = int(yy * F.shape[0]); ax.plot(x, F[iy], lw=0.8, color=plt.get_cmap("Greys")(0.45 + 0.2 * k))
        if "tz_anom" in d:
            T = d["tz_anom"]; zz = d["tz_z"]; cold = T < -0.5 * T.min() * -1 if False else T < 0.5 * T.min()
            for zc in zz[cold]: ax.axhspan(zc - 2.5, zc + 2.5, color="#cfe0f3", lw=0, zorder=0)
        ax.set_xlim(x[0], x[-1]); ax.set_ylim(-920, -620)
        ax.set_xlabel("distance from GL (km)")
        if j == 0: ax.set_ylabel("base depth (m)")
        else: ax.set_yticklabels([])
        ax.set_title(ttl, pad=4); panel_label(ax, "def"[j])
    save(fig, "FigS14_Tz_layering")

if __name__ == "__main__":
    import sys
    for a in sys.argv[1:]: globals()[a]()
