"""Along-flow roughness spectra: one line per across-flow position, starting at the grounding line."""
import pickle, numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from main_figs import *
from figures import save, panel_label, INK, MUTED, GRID
def psd_rows(F, dx, x0=0.0, x=None):
    if x is not None: F = F[:, x >= x0]
    k = None; P = []
    for row in F:
        r = row - np.polyval(np.polyfit(np.arange(len(row)), row, 1), np.arange(len(row)))
        k, p = periodogram(r, fs=1 / dx, window="hann", detrend=False); P.append(p[1:]); k = k[1:]
    return k, np.array(P)
def logbin(k, p, n=14):
    e = np.logspace(np.log10(k[0]), np.log10(k[-1]), int(np.log10(k[-1] / k[0]) * n) + 1)
    i = np.digitize(k, e); kk = np.array([k[i == j].mean() for j in np.unique(i)]); pp = np.array([p[i == j].mean() for j in np.unique(i)])
    return kk, pp
def band_rms(k, P, kmin, kmax):
    s = (k >= kmin) & (k <= kmax); dk = k[1] - k[0]
    return np.sqrt(P[:, s].sum(1) * dk)
fig = plt.figure(figsize=(7.2, 6.8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8], hspace=0.7, wspace=0.22, left=0.09, right=0.99)
# a: 8 km terrace-regime runs (plan view, 10 m)
ax = fig.add_subplot(gs[0, 0]); RC = {"terraces": "#2a9d8f", "sawtooth": "#6a4c93", "smooth": "#c98a14"}
rmsA = {}
for name, lab in (("terr2dZ_ua0.3_ph1_M0.5", "terraces"), ("terr2dZ_ua0.3_ph0_M0.5", "sawtooth"), ("terr2dZ_ua1_ph1_M0.5", "smooth")):
    try: d = pickle.load(open(f"results_coarse/{name}.pkl", "rb"))
    except FileNotFoundError: continue
    F = d["field"]
    k, P = psd_rows(F, 10.0)
    for p in P[::3]: ax.loglog(*logbin(k, p), color=RC[lab], lw=0.35, alpha=0.3)
    ax.loglog(*logbin(k, P.mean(0)), color=RC[lab], lw=1.5, label=lab)
ax.set_xlabel("along-flow wavenumber (cycles m$^{-1}$)"); ax.set_ylabel("PSD of basal relief (m$^2$ per cycle m$^{-1}$)")
ax.set_title("8 km of a 4° base, plan view (1 km wide)", pad=16); ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(0.0, 0.42)); panel_label(ax, "a", y=1.08)
# b: 40 km shelf runs (40 m sampling)
ax2 = fig.add_subplot(gs[0, 1]); C2 = {"shear": "#1f5f99", "conv": "#d1495b", "dnsF": "#6a4c93", "plume": "#4f9a2f", "dns": "#c98a14", "dnsQ": "#e07b00"}
N2 = {"shear": "Shear", "conv": "sin$^{2/3}$ convection", "dnsF": "DNS law + flat strat.", "plume": "Plume law", "dns": "DNS law, tidal $u_*$"}
import os
def ens(v, s=1):
    for d_ in ("results", "mac/results"):
        f = f"{d_}/ens__base__{v}__s{s}.npz"
        if os.path.exists(f): return np.load(f)
rmsB = {}
for v in ("shear", "dns", "conv", "plume", "dnsF"):
    d = ens(v); F = d["field40"].astype(float)
    k2, P2 = psd_rows(F, 40.0)
    for p in P2[::4]: ax2.loglog(*logbin(k2, p), color=C2[v], lw=0.35, alpha=0.3)
    ax2.loglog(*logbin(k2, P2.mean(0)), color=C2[v], lw=1.5, label=N2[v]); rmsB[v] = band_rms(k2, P2, 1 / 10000, 1 / 90)
ax2.set_xlabel("along-flow wavenumber (cycles m$^{-1}$)"); ax2.set_title("40 km shelf, 40 m sampling", pad=16)
panel_label(ax2, "b", y=1.08)
# arrows and observed feature bands
def arrow(a, lam, text, y, c=INK):
    a.annotate(text, xy=(1 / lam, y), xytext=(1 / lam * 1.6, y * 80), ha="left", fontsize=5.8, color=c,
               arrowprops=dict(arrowstyle="->", color=c, lw=0.8))
import numpy as np
dT = pickle.load(open("results_coarse/terr2dZ_ua0.3_ph1_M0.5.pkl", "rb")); kT, PT = psd_rows(dT["field"], 10.0)
yT = np.interp(1 / 800, *logbin(kT, PT.mean(0))); arrow(ax, 800, "terrace spacing", yT * 1.3, RC["terraces"])
dC = ens("dnsF"); kC, PC = psd_rows(dC["field40"].astype(float), 40.0)
yC = np.interp(1 / 1500, *logbin(kC, PC.mean(0))); arrow(ax2, 1500, "GL crevasse spacing", yC * 1.3)
dB = ens("conv"); kB, PB = psd_rows(dB["field40"].astype(float), 40.0)
yB = np.interp(1 / 400, *logbin(kB, PB.mean(0))); arrow(ax2, 400, "blisters", yB * 1.3, C2["conv"])
import obs_compare as OC
OBS = OC.obs(); OBSP = {}
for k_, lab_, c_ in (("pig", "Pine Island base (BedMachine)", "#111111"), ("dotson", "Dotson base (BedMachine)", "#8a8a8a")):
    kk_, PP_ = OC.psd_lines(OBS[k_][1]); m_ = kk_ <= OC.KMAX; OBSP[k_] = (kk_, PP_, lab_, c_)
for a_ in (ax, ax2):
    a_.axvspan(1 / 10000, 1 / 90, color="#f2f2f2", zorder=0, lw=0)
    a_.set_ylim(1e-6, 1e8)
    for k_, (kk_, PP_, lab_, c_) in OBSP.items():
        m_ = kk_ <= OC.KMAX; kb, pb = logbin(kk_[m_], PP_[:, m_].mean(0), 10)
        a_.loglog(kb, pb, color=c_, lw=2.0, ls="-", label=lab_ if a_ is ax2 else None, zorder=5)
    a_.grid(color=GRID, lw=0.4, which="major")
    sec = a_.secondary_xaxis("top", functions=(lambda k: 1 / np.maximum(k, 1e-9), lambda l: 1 / np.maximum(l, 1e-9)))
    sec.set_xlabel("wavelength (m)", fontsize=6.5); sec.tick_params(labelsize=6)
ax2.legend(fontsize=5.4, loc="lower left", ncol=1)
# c: rms in the radar band versus observations
ax3 = fig.add_subplot(gs[1, :]); _b = ax3.get_position(); ax3.set_position([0.33, _b.y0, 0.99 - 0.33, _b.height])
for h in (120, 250):
    for v in ("shear", "conv", "dnsF"):
        f_ = f"results/crevasseheight(steepwalls)__{h}__{v}__s1.npz"
        if not os.path.exists(f_): rmsB[f"{v}{h}"] = np.array([np.nan]); continue
        d = np.load(f_); kk_, PP_ = psd_rows(d["field40"].astype(float), 40.0)
        rmsB[f"{v}{h}"] = band_rms(kk_, PP_, 1 / 10000, 1 / 90)
rows = [("Model: shear (40 km)", rmsB["shear"], C2["shear"]), ("Model: DNS law, tidal $u_*$ (40 km)", rmsB["dns"], C2["dns"]),
        ("Model: plume (40 km)", rmsB["plume"], C2["plume"]),
        ("Model: DNS law + flat (40 km)", rmsB["dnsF"], C2["dnsF"]),
        ("Model: shear, 250 m GL crevasses", rmsB["shear250"], C2["shear"]), ("Model: sin$^{2/3}$, 250 m GL crevasses", rmsB["conv250"], C2["conv"]), ("Model: DNS law + flat, 250 m GL crevasses", rmsB["dnsF250"], C2["dnsF"])]
for k_, v_ in rmsB.items(): print(k_, round(float(np.median(v_)), 1))
for i, (lab, v, c) in enumerate(rows):
    ax3.plot(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(v)), "o", ms=1.6, color=c, alpha=0.35, mew=0)
    ax3.plot(np.median(v), i, "D", ms=5, color=c, mec="white", mew=0.6)
obsr = [("Ross (radar)", 12), ("Pine Island, along flow (radar)", 30), ("Pine Island, all tracks (radar)", 55), ("Thwaites (radar)", 90)]
for j, (lab, v) in enumerate(obsr):
    ax3.plot(v, len(rows) + j, "s", ms=5, color=INK)
labels = [r[0] for r in rows] + [o[0] for o in obsr]
ax3.set_yticks(range(len(labels))); ax3.set_yticklabels(labels, fontsize=6); ax3.set_xscale("log"); ax3.set_xlim(2, 130)
ax3.axhline(len(rows) - 0.5, color=MUTED, lw=0.6)
ax3.set_xlabel("rms basal relief in the 90 m to 10 km band (m); model lines start at the grounding line")
ax3.set_title("Roughness amplitude: model lines vs radar surveys (Watkins et al., 2021)", pad=4)
ax3.grid(axis="x", color=GRID, lw=0.4); ax3.spines["left"].set_visible(False); ax3.tick_params(axis="y", length=0)
fig.text(0.01, _b.y1 + 0.02, "c", fontsize=9, fontweight="bold")
save(fig, "Fig6_spectra")
