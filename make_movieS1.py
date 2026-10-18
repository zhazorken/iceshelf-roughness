"""Movie S1: evolving basal relief and melt rate for the cases of Figure 1 (shear, sin^(2/3), and three 4-degree-base cases)."""
import os, pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from matplotlib.colors import LogNorm, Normalize
import shelfmelt as sm
from figures import hillshade, DIV, INK

SHELF = [("shear", "Shear melt", "#1f5f99"), ("conv", "sin$^{2/3}\\theta$ convection", "#d1495b"), ("pond", "sin$^{2/3}\\theta$ + meltwater ponds (3% retained), 40-150 m crevasses", "#8c2d6b")]
FLANK = [("terr", "All melt mechanisms: DNS law, weak currents ($u_{*a}$ = 0.3 mm/s), boundary-layer stability, pooling", "#2a6f4e"),
         ("saw", "As above without pooling: scalloped", "#6a4c93"),
         ("smo", "As above with stronger currents ($u_{*a}$ = 1 mm/s): smooth", "#c98a14")]
FLANK = [c for c in FLANK if os.path.exists(f"results_coarse/movie_{c[0]}.pkl")]
R = {c: pickle.load(open(f"results_coarse/movie_{c}.pkl", "rb")) for c, _, _ in SHELF + FLANK}
VL = 50.0; NORM = LogNorm(0.3, 100)

def grid(r):
    nx = r["snaps"].shape[2]; s = len(r["x"]) // nx
    return r["x"][::s][:nx], r["y"][::s][: r["snaps"].shape[1]], r["zbar"][::s][:nx], float(r["x"][s] - r["x"][0])
def relief(c, i):
    r = R[c]; x, y, zb, dx = grid(r); z = r["snaps"][i].astype(float)
    if c not in ("shear", "conv", "pond"):
        F = z + zb[None, :]
        return F - np.polyval(np.polyfit(x, F.mean(0), 1), x)[None, :], dx
    return sm.highpass(z, dx), dx
nt = min(len(R[c]["snap_t"]) for c in R)

W = 10.0; L, Rt = 0.07, 0.88
h40 = (Rt - L) * W * 6 / 40              # inches
gap = 0.3; wp = (Rt - L - gap / W) / 2; h8 = wp * W * 3 / 8
H = 0.9 + 2 * len(SHELF) * h40 + (2 * len(SHELF) - 1) * 0.12 + 0.75 + len(FLANK) * (h8 + 0.45) + 0.35
fig = plt.figure(figsize=(W, H), dpi=150)
box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
ims = {}; y = H - 0.9
def setup(ax, c, f, ext):
    if f == "z":
        z, dx = relief(c, 0); ims[(c, f)] = ax.imshow(hillshade(z, dx, VL), extent=ext, origin="lower", aspect="auto")
    else:
        ims[(c, f)] = ax.imshow(np.maximum(R[c]["snap_melt"][0], 0.05), extent=ext, origin="lower", aspect="auto", cmap="magma", norm=NORM)
    ax.tick_params(labelsize=7, length=2); ax.spines[:].set_visible(False)
for k, (c, lab, col) in enumerate(SHELF):
    x, yy, zb, dx = grid(R[c]); ext = [0, x[-1] / 1e3 + dx / 2e3, 0, yy[-1] / 1e3 + dx / 2e3]
    for f in ("z", "m"):
        y -= h40 + (0.12 if (k or f == "m") else 0)
        ax = fig.add_axes([L, y / H, Rt - L, h40 / H]); setup(ax, c, f, ext); ax.set_yticks([0, 3, 6])
        ax.text(0.005, 0.93, f"{lab}: {'relief' if f == 'z' else 'melt rate'}", transform=ax.transAxes, fontsize=8,
                fontweight="semibold", va="top", color=col, bbox=box)
        if (k, f) == (len(SHELF) - 1, "m"): ax.set_xlabel("distance from grounding line (km), 40 km shelf, ice speed 1000 m/yr", fontsize=8)
        else: ax.set_xticklabels([])
y -= 0.75
for j, (c, lab, col) in enumerate(FLANK):
    x, yy, zb, dx = grid(R[c]); ext = [0, x[-1] / 1e3 + dx / 2e3, 0, yy[-1] / 1e3 + dx / 2e3]
    y -= h8 + (0.45 if j else 0)
    for i_, f in enumerate(("z", "m")):
        ax = fig.add_axes([L + i_ * (wp + gap / W), y / H, wp, h8 / H]); setup(ax, c, f, ext); ax.set_yticks([0, 1, 2, 3])
        if i_: ax.set_yticklabels([])
        else: ax.text(0, 1.04, lab, transform=ax.transAxes, fontsize=8, fontweight="semibold", color=col, va="bottom")
        ax.text(0.01, 0.93, "relief" if f == "z" else "melt rate", transform=ax.transAxes, fontsize=7.5, va="top", color=INK, bbox=box)
        if j == len(FLANK) - 1: ax.set_xlabel("along-flow distance (km), 4° base, ice speed 200 m/yr", fontsize=8)
        else: ax.set_xticklabels([])
c1 = fig.add_axes([Rt + 0.02, (H - 0.9 - 2.2 * h40) / H, 0.012, 1.8 * h40 / H])
fig.colorbar(cm.ScalarMappable(Normalize(-VL, VL), DIV), cax=c1).set_label("relief z′ (m)", fontsize=8)
c2 = fig.add_axes([Rt + 0.02, (y + 0.2) / H, 0.012, 1.8 * h40 / H])
fig.colorbar(cm.ScalarMappable(NORM, "magma"), cax=c2).set_label("melt rate (m/yr)", fontsize=8)
for cc in (c1, c2): cc.tick_params(labelsize=7)
ttl = fig.text(L, (H - 0.35) / H, "", fontsize=12, fontweight="semibold")
fig.text(L, (H - 0.6) / H, "Red: base melted upward into the ice. Relief: 2-km along-flow mean removed (40 km shelf), plane fit removed (4° base).",
         fontsize=8, color="#555555")

def draw(i):
    for (c, f), im in ims.items():
        if f == "z":
            z, dx = relief(c, i); im.set_data(hillshade(z, dx, VL))
        else:
            im.set_data(np.maximum(R[c]["snap_melt"][i], 0.05))
    ttl.set_text(f"Movie S1.   t = {R['shear']['snap_t'][i]:4.1f} yr")
    return list(ims.values())
if not os.environ.get("FRAME_ONLY"):
  ani = animation.FuncAnimation(fig, draw, frames=nt, blit=False)
  ani.save("figs/MovieS1.mp4", dpi=150, writer=animation.FFMpegWriter(fps=8, bitrate=8000, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]))
draw(nt - 1); fig.savefig("figs/MovieS1_lastframe.png", dpi=110)
print("frames", nt, "size", fig.get_size_inches())
