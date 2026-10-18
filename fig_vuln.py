"""Main figure: which melt mechanisms thin the ice most (vulnerability ranking)."""
import pickle, glob, numpy as np, matplotlib.pyplot as plt
from main_figs import *
from figures import save, panel_label, INK, MUTED, GRID
import shelfmelt as sm, vuln, surface_frac as SF

V = pickle.load(open("results_coarse/vuln.pkl", "rb"))
S = pickle.load(open("results_surface_frac.pkl", "rb")); D, res = S["D"], S["res"]
def d01(P, q=0.01):
    i = np.argmax(P >= q); return D[i] if P[i] >= q else np.nan

# seeded (no-melt) references
seed = {}
for h, pat in [(40, "mac/results/ens__base__shear__s1.npz"), (120, "results/crevasseheight(steepwalls)__120__shear__s1.npz"),
               (250, "results/crevasseheight(steepwalls)__250__shear__s1.npz")]:
    f = glob.glob(pat) or glob.glob(pat.replace("mac/results/", "results/"))
    z = np.load(f[0]); rec = pickle.loads(z["meta"].tobytes()); p = dict(sm.P0, **rec["params"])
    r = dict(params=p, x=z["x"], zbar=z["zbar"], field40=SF.seeded_field(p))
    seed[h] = vuln.incH(r)[0]
print("seeded", seed)

FAM = {"shear": "#1f5f99", "conv": "#d1495b", "plume": "#3a923a", "strat": "#6a4c93", "pond": "#2a9d8f"}
ROWS = [("shear", "Shear", "shear"), ("dns", "DNS law, tidal $u_*$", "shear"), ("dnsQ", "DNS law, quiet water", "shear"),
        ("conv", "sin$^{2/3}$ convection", "conv"), ("mixed", "shear + sin$^{2/3}$", "conv"),
        ("theta", "DNS convection only ($u_*=0$)", "conv"), ("plume", "Plume law", "plume"),
        ("shearF", "shear + flat stratification", "strat"), ("dnsF", "DNS law + flat stratification", "strat"),
        ("convF", "sin$^{2/3}$ + flat stratification", "strat"), ("plumeF", "plume + flat stratification", "strat"),
        ("dnsP", "DNS law + pooling", "pond"), ("convP", "sin$^{2/3}$ + pooling", "pond"),
        ("shearA", "shear + aging ponds", "pond"), ("convA", "sin$^{2/3}$ + aging ponds", "pond"), ("plumeA", "plume + aging ponds", "pond")]
ROWS = [r for r in ROWS if ("40", r[0]) in V]
ROWS.sort(key=lambda r: np.mean([v[2] for v in V[("40", r[0])]]))

fig = plt.figure(figsize=(7.2, 3.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.8, 0.9], wspace=0.75, left=0.25, right=0.98, top=0.88, bottom=0.14)
ax = fig.add_subplot(gs[0])
for i, (k, lab, fam) in enumerate(ROWS):
    a = np.array(V[("40", k)]); c = FAM[fam]
    ax.plot([a[:, 0].mean(), a[:, 2].mean()], [i, i], color=c, lw=0.8, alpha=0.5)
    ax.plot(a[:, 2], np.full(len(a), i), "o", ms=2.2, color=c, alpha=0.45, mew=0)
    ax.plot(a[:, 2].mean(), i, "D", ms=4.6, color=c, mec="white", mew=0.5, zorder=3)
    ax.plot(a[:, 0].mean(), i, "o", ms=3.6, mfc="white", mec=c, mew=0.9, zorder=3)
ax.axvline(seed[40], color=INK, ls=":", lw=0.9)
ax.text(seed[40] + 0.012, 1.0, "seeded 40 m crevasses", fontsize=5.4, color=INK, va="center")
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[1] for r in ROWS], fontsize=6.2)
for t, r in zip(ax.get_yticklabels(), ROWS): t.set_color(FAM[r[2]])
ax.set_xlim(0, 0.75); ax.set_ylim(-0.6, len(ROWS) - 0.4)
ax.set_xlabel("deepest incision / ice thickness")
ax.grid(axis="x", color=GRID, lw=0.4); ax.tick_params(axis="y", length=0)
for yy, mk, kw, t in ((0.41, "D", dict(color=MUTED, mec="white"), "below smooth-shelf base\n(dots: seeds)"),
                      (0.345, "o", dict(mfc="white", mec=MUTED, mew=0.9), "local relief only")):
    ax.plot([0.36], [yy], mk, ms=4.2, transform=ax.transAxes, clip_on=False, **kw)
    ax.text(0.39, yy, t, transform=ax.transAxes, fontsize=5.6, va="center")
ax.set_title("Ranking, 40 m crevasses (four seeds)", fontsize=7.2, loc="left")
ax.text(-0.66, 1.04, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")

ax = fig.add_subplot(gs[1])
H = [40, 120, 250]
for k, lab, c in (("dnsF", "DNS + flat strat.", FAM["strat"]), ("shear", "shear", FAM["shear"]), ("dnsQ", "DNS, quiet", "#7fa7d0"),
                  ("plume", "plume", FAM["plume"]), ("conv", "sin$^{2/3}$", FAM["conv"])):
    y = [np.mean([v[2] for v in V[(f"{h}s", k)]]) if (f"{h}s", k) in V else np.nan for h in H]
    ax.plot(H, y, "-o", color=c, ms=3, lw=1.1, label=lab)
ax.plot(H, [seed[h] for h in H], ":", color=INK, lw=0.9, label="seeded")
ax.set_xlabel("GL crevasse height (m)"); ax.set_ylabel("deepest incision / ice thickness")
ax.set_xticks(H); ax.set_ylim(0, 0.8); ax.grid(color=GRID, lw=0.4)
ax.legend(fontsize=5.6, loc="upper left", handlelength=1.6, borderaxespad=0.2)
ax.set_title("Taller crevasses", fontsize=7.2, loc="left")
ax.text(-0.42, 1.04, "b", transform=ax.transAxes, fontsize=9, fontweight="bold")

ax = fig.add_subplot(gs[2])
CR = [("shear", "Shear", "shear"), ("dns", "DNS, tidal", "shear"), ("dnsQ", "DNS, quiet", "shear"), ("convA", "sin$^{2/3}$ + aging ponds", "pond"),
      ("theta", "DNS conv. only", "conv"), ("plume", "Plume", "plume"), ("dnsF", "DNS + flat strat.", "strat"),
      ("conv", "sin$^{2/3}$", "conv"), ("convF", "sin$^{2/3}$ + flat strat.", "strat")]
CR = [c for c in CR if (c[0], False) in res]
vals = [(d01(res[(k, False)]), lab, FAM[f]) for k, lab, f in CR]
vals.sort(key=lambda v: -v[0])
for i, (v, lab, c) in enumerate(vals):
    ax.plot([v, 1], [i, i], color=c, lw=3.2, solid_capstyle="butt", alpha=0.85)
ax.axvline(d01(res[("seed40", False)]), color=INK, ls=":", lw=0.9)
ax.set_yticks(range(len(vals))); ax.set_yticklabels([v[1] for v in vals], fontsize=6)
for t, v in zip(ax.get_yticklabels(), vals): t.set_color(v[2])
ax.set_xlim(0.2, 1.0); ax.set_xlabel("surface-crevasse depth that\ncuts through 1% of columns, $d/H$")
ax.grid(axis="x", color=GRID, lw=0.4); ax.tick_params(axis="y", length=0)
ax.set_title("Surface-fracture threshold", fontsize=7.2, loc="left")
ax.text(-0.85, 1.04, "c", transform=ax.transAxes, fontsize=9, fontweight="bold")
save(fig, "Fig_vulnerability")
for r in ROWS: print(r[0], np.round(np.mean(V[("40", r[0])], 0), 3))
for v in vals: print(v[1], v[0])
