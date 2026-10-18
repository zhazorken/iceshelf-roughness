"""Cross-check every number in the Nature Geoscience text against values recomputed from the model output.
Usage: python crosscheck.py [tex]   ->  crosscheck.csv, crosscheck.md
Quantities are computed from results/ (npz), results_coarse/ (pickles) and oned/*.pkl; each claim names the text it
checks, the numbers printed in the text and the recomputed numbers. Status: ok (equal at the printed precision),
close (one unit in the last printed digit), MISMATCH, or missing (output not available)."""
import os, re, sys, glob, json, pickle, collections
import numpy as np
sys.path.insert(0, "oned")
import shelfmelt as sm, experiments as ex, figures as F
from scipy.signal import periodogram

TEX = sys.argv[1] if len(sys.argv) > 1 else "latex_ngeo/main.tex"
Q = {}                                      # name -> value (float or list)
def put(k, v): Q[k] = v

# ------------------------------------------------------------------ 40 km shelf runs
R = F.load(["results"])
grp = collections.defaultdict(list)
for r in R: grp[(r["group"], r["label"], r["variant"])].append(r)
def ens(v): return grp[("ens", "base", v)]
def mean(rs, k): return float(np.mean([r[k] for r in rs])) if rs else np.nan
for v in ["shear", "dns", "dnsQ", "conv", "theta", "plume", "mixed", "dnsF", "convF", "plumeF", "shearF", "convP", "dnsP",
          "convA", "shearA", "plumeA", "dnsA"]:
    rs = ens(v)
    put(f"G_{v}", mean(rs, "G")); put(f"Gsd_{v}", float(np.std([r["G"] for r in rs])) if rs else np.nan)
    put(f"C10_{v}", mean(rs, "C10")); put(f"enh_{v}", mean(rs, "enh")); put(f"n_{v}", len(rs))
put("G_aging_range", [min(Q[f"G_{v}A"] for v in ("conv", "shear", "plume", "dns")), max(Q[f"G_{v}A"] for v in ("conv", "shear", "plume", "dns"))])
put("G_aging_mean", float(np.mean([Q[f"G_{v}A"] for v in ("conv", "shear", "plume", "dns")])))
# DNS law, ambient u* sweep
sw = {}
for (g, l, v), rs in grp.items():
    if g == "dns u*": sw[float(l)] = (mean(rs, "G"), mean(rs, "C10"), len(rs))
sw[1.0] = (Q["G_dns"], Q["C10_dns"], Q["n_dns"]); sw[0.2] = (Q["G_dnsQ"], Q["C10_dnsQ"], Q["n_dnsQ"])
for f, (g, c, n) in sorted(sw.items()): put(f"dnsu_{f:g}", [g, c, n])
# offset-linear family: C10 range and G = 1 crossing
rows = sorted((mean(rs, "C10"), mean(rs, "G")) for (g, l, v), rs in grp.items() if g == "contrast")
if rows:
    C, G = np.array(rows).T
    put("contrast_C10_range", [float(C.min()), float(C.max())])
    i = np.flatnonzero((G[:-1] < 1) & (G[1:] >= 1))
    put("contrast_cross", [float(C[i[0]]), float(C[i[0] + 1])] if len(i) else [np.nan, np.nan])
    put("contrast_rows", [list(map(float, r)) for r in rows])
# convective transfer velocity on a vertical wall along the shelf
p = dict(ex.BASE); x, y, zb, u = sm.setup(p); dT = sm.thermal_driving(zb, p)
uc = p["dns_uc90"] * (dT / p["dns_dTref"]) ** (1 / 3)
put("uc_wall", [float(uc.min() * 1e3), float(uc.max() * 1e3)]); put("uc_10deg", [float(uc.min() * 1e3 / 9), float(uc.max() * 1e3 / 9)])
put("dT_gl_end", [float(dT[0]), float(dT[-1])])

# ------------------------------------------------------------------ deepest incisions (Fig. 5a, b)
import vuln
inc = collections.defaultdict(list)
for r in R:
    if r["group"] == "ens": inc[("40", r["variant"])].append(vuln.incH(r))
    elif r["group"] == "crevasse height (steep walls)": inc[(r["label"], r["variant"])].append(vuln.incH(r))
for k, v in inc.items():
    a = np.array(v); put(f"inc_{k[0]}_{k[1]}", [float(a[:, 0].mean()), float(a[:, 2].mean()), len(a)])
try:
    import surface_frac as SF
    rec = grp[("ens", "base", "shear")][0]; pp = dict(rec["params"]); seedf = SF.seeded_field(pp)
    put("inc_seed40", vuln.incH(dict(rec, field40=seedf))[0])
except Exception as e:
    put("inc_seed40", np.nan)

# ------------------------------------------------------------------ surface-crevasse threshold (Fig. 5c)
try:
    import surface_frac as SF
    D = np.linspace(0, 1, 101)
    def d01(recs):
        rr = []
        for i, r in enumerate(recs):
            H = sm.thickness(np.asarray(r["zbar"]), r["params"]); st = max(1, int(round(40.0 / r["params"]["dx"])))
            rr.append(SF.sample(r["field40"], H[::st], r["x"][::st], 20000, np.random.default_rng(i), False))
        rr = np.concatenate(rr); P = np.array([(rr < d).mean() for d in D]); j = np.argmax(P >= 0.01)
        return float(D[j]) if P[j] >= 0.01 else np.nan
    for v in ("shear", "dns", "dnsQ", "conv", "plume", "dnsF", "theta", "convF", "convA"):
        put(f"frac_{v}", d01(ens(v)))
    for v in ("shear", "conv", "plume", "dnsF", "dnsQ"):
        put(f"frac250_{v}", d01(grp[("crevasse height (steep walls)", "250", v)]))
except Exception as e:
    print("surface fracture failed", e)

# ------------------------------------------------------------------ spectra on the 40 km shelf (ED Fig. 2)
def psd_rows(Fz, dx):
    P = []
    for row in Fz:
        rr = row - np.polyval(np.polyfit(np.arange(len(row)), row, 1), np.arange(len(row)))
        k, pw = periodogram(rr, fs=1 / dx, window="hann", detrend=False); P.append(pw[1:])
    return k[1:], np.array(P)
def band(k, P, a, b):
    s = (k >= a) & (k <= b); return np.sqrt(P[:, s].sum(1) * (k[1] - k[0]))
for v in ("shear", "dns", "conv", "plume", "dnsF"):
    rs = [r for r in ens(v) if r["seed"] == 1]
    if not rs: continue
    k, P = psd_rows(rs[0]["field40"].astype(float), 40.0)
    put(f"rms_radar_{v}", float(np.median(band(k, P, 1 / 1e4, 1 / 90))))
    tot = P.sum(1); put(f"varfrac2km_{v}", float(np.median(P[:, k < 1 / 2000].sum(1) / tot)))
for v in ("shear", "conv", "dnsF"):
    rs = grp[("crevasse height (steep walls)", "250", v)]
    if rs:
        k, P = psd_rows(rs[0]["field40"].astype(float), 40.0); put(f"rms_radar250_{v}", float(np.median(band(k, P, 1 / 1e4, 1 / 90))))

# ------------------------------------------------------------------ plan-view anisotropy
def aniso(z, dx, x, x0=10e3, x1=38e3):
    hp = sm.highpass(z.astype(float), dx); m = (x > x0) & (x < x1); hp = hp[:, m]
    return float(np.var(np.diff(hp, axis=1) / dx) / np.var(np.diff(hp, axis=0) / dx))
for v in ("shear", "dns", "dnsQ", "conv", "theta", "plume", "dnsF", "convF", "mixed"):
    vals = []
    for r in ens(v):
        s = max(1, int(round(40.0 / r["params"]["dx"]))); xx = r["x"][::s][: r["field40"].shape[1]]
        vals.append(aniso(r["field40"], 40.0, xx))
    put(f"aniso_{v}", float(np.mean(vals)) if vals else np.nan)

# ------------------------------------------------------------------ BedMachine and the build-up (Fig. 4)
try:
    import obs_compare as OC, aniso_obs as AO, obs_transects as OT
    O = OC.obs()
    for kk in ("pig", "dotson"):
        k_, P_ = OC.psd_lines(O[kk][1]); put(f"obs_rms_{kk}", float(np.median(OC.band_rms(k_, P_))))
        put(f"obs_aniso_{kk}", AO.aniso_box(kk)[0])
    for b in ("b1", "b2", "b3", "b4", "b5", "b7", "b6", "b8"):
        vals, an = [], []
        for f in (f"results_coarse/build_{b}.pkl", f"results_coarse/build_{b}s2.pkl", f"results_coarse/build_{b}s3.pkl"):
            if not os.path.exists(f): continue
            r = pickle.load(open(f, "rb")); xs_, Z_ = OC.model_lines(r); k_, P_ = OC.psd_lines(Z_)
            vals.append(float(np.median(OC.band_rms(k_, P_)))); an.append(AO.aniso_model(r))
        if vals: put(f"build_{b}", [float(np.mean(vals)), min(vals), max(vals), len(vals)]); put(f"build_aniso_{b}", float(np.mean(an)))
except Exception as e:
    print("BedMachine/build-up failed", repr(e))

# ------------------------------------------------------------------ plan-view terraces (Fig. 1d-i)
import main_figs as M
def risers(r):
    Fz = r["field"].astype(float) + r["zbar"][None, :]; x = r["x"]; dx = x[1] - x[0]; H = []; S = []
    for row in Fz[::5]:
        st = np.abs(np.degrees(np.arctan(np.gradient(row, dx)))) > 15
        i0 = np.flatnonzero(np.diff(st.astype(int)) == 1) + 1; i1 = np.flatnonzero(np.diff(st.astype(int)) == -1) + 1; xs = []
        for i in i0:
            j = i1[i1 > i]
            if len(j) and abs(row[j[0]] - row[i - 1]) > 8 and x[i] > 1e3: H.append(abs(row[j[0]] - row[i - 1])); xs.append(x[i])
        S += list(np.diff(xs))
    return np.percentile(H, [25, 50, 75]) if H else [np.nan] * 3, np.percentile(S, [25, 50, 75]) if S else [np.nan] * 3
def terr_stats(f):
    if not os.path.exists(f): return None
    r = pickle.load(open(f, "rb")); fl, st = M.melt_stats(r); h, s = risers(r)
    return dict(G=float(r["G"]), flats=fl, faces=st, h=list(map(float, h)), s=list(map(float, s)))
TS = {}
for k, f0 in (("terr", "results_coarse/terr2dZ_wide_Ly3000.pkl"), ("saw", "results_coarse/terr2dZ_wide_Ly3000_ua0.3_ph0.pkl"),
              ("smo", "results_coarse/terr2dZ_wide_Ly3000_ua1_ph1.pkl")):
    for s, f in ((1, f0), (2, f"results_coarse/terrW_{k}_s2.pkl"), (3, f"results_coarse/terrW_{k}_s3.pkl")):
        try:
            t = terr_stats(f)
            if t: TS[(k, s)] = t
        except Exception as e:
            print("terr stats failed", f, e)
for k in ("terr", "saw", "smo"):
    ts = [TS[(k, s)] for s in (1, 2, 3) if (k, s) in TS]
    if not ts: continue
    put(f"terrW_G_{k}", [min(t["G"] for t in ts), max(t["G"] for t in ts), len(ts)])
    if k == "terr":
        put("terrW_h", [min(t["h"][0] for t in ts), max(t["h"][2] for t in ts)])
        put("terrW_s", [min(t["s"][0] for t in ts), max(t["s"][2] for t in ts)])
        put("terrW_flats", [min(t["flats"] for t in ts), max(t["flats"] for t in ts), TS[("terr", 1)]["flats"] if ("terr", 1) in TS else np.nan])
        put("terrW_faces", [min(t["faces"] for t in ts), max(t["faces"] for t in ts)])
for f, k in (("results_coarse/terr2dZ_ua0.3_ph1_M0.5.pkl", "res10"), ("results_coarse/terrRes_dx5_Ly1000.pkl", "res5")):
    t = terr_stats(f) if os.path.exists(f) else None
    if t: put(f"terr_{k}", [t["G"], t["flats"], t["faces"], t["h"][1]])
for f, k in (("results_coarse/attr_nomo_nopool.pkl", "attr_nomo_nopool"), ("results_coarse/attr_nomo_pool.pkl", "attr_nomo_pool")):
    if os.path.exists(f): put(k, float(pickle.load(open(f, "rb"))["G"]))
for f0 in ("results_coarse/terrW_terr_s2.pkl", "results_coarse/terr2dZ_wide_Ly3000.pkl"):
    if os.path.exists(f0):
        r = pickle.load(open(f0, "rb")); dx = float(r["params"]["dx"]); Fz = r["field"].astype(float) + r["zbar"][None, :]
        z = Fz - np.polyval(np.polyfit(r["x"], Fz.mean(0), 1), r["x"])[None, :]; put("aniso_terr", aniso(z, dx, r["x"], 1e3, 7.5e3)); break

# ------------------------------------------------------------------ ponds (Fig. 1c) and retention
pond = {}
for f in glob.glob("results_coarse/pond_*.pkl") + glob.glob("results_coarse/mix_convA03show.pkl") + glob.glob("results_coarse/prod_*.pkl"):
    pond[os.path.basename(f)[:-4]] = float(pickle.load(open(f, "rb"))["G"])
put("ponds", pond)
seeds3 = [pond.get(k) for k in ("mix_convA03show", "pond_convA_prod0.03_H150_s2", "pond_convA_prod0.03_H150_s3")]
if all(v is not None for v in seeds3): put("pond_seeds", [min(seeds3), max(seeds3)])
put("pond_ctrl", [pond.get("pond_conv_prodNone_H150_s1", np.nan), pond.get("pond_shear_prodNone_H150_s1", np.nan)])
put("prod_scan", [pond.get(f"prod_convA_prod{f:g}_Ly4000_dx20", np.nan) for f in (0.003, 0.01, 0.03, 0.1)])

# ------------------------------------------------------------------ along-flow (1D) terrace model
try:
    import pandas as pd
    res = pickle.load(open("oned/regime_z.pkl", "rb"))
    df = pd.DataFrame([dict(**k, **{kk: v for kk, v in m.items() if kk not in ("zb", "zbar", "x")}) for k, m in res if "tag" not in k])
    cls = lambda r: "smooth" if r.rms < 2.5 else ("terraces" if (abs(r.tilt) < 1.0 and r.back < 0.1) else "sawtooth")
    df["reg"] = df.apply(cls, axis=1)
    t = df[df.reg == "terraces"]; put("oned_terr_ua", sorted(set(map(float, t.mo_ua * 1e3)))); put("oned_terr_h", sorted(set(map(float, t.pool_h))))
    s = df[(df.pool_h == 0) & (df.reg == "sawtooth")]; put("oned_saw_tilt", [float(s.tilt.abs().min()), float(s.tilt.abs().max())])
    put("oned_terr_tilt", [float(t.tilt.abs().min()), float(t.tilt.abs().max())])
    put("oned_regmap", {f"{r.mo_ua*1e3:g},{r.pool_h:g}": r.reg for r in df.itertuples()})
    dM = pd.DataFrame([dict(**k, **{kk: v for kk, v in m.items() if kk not in ("zb", "zbar", "x")}) for k, m in res if "tag" in k])
    dM["reg"] = dM.apply(cls, axis=1); put("oned_M", {f"{r.M:g},{r.mo_ua*1e3:g}": r.reg for r in dM.itertuples()})
    sc = pickle.load(open("oned/scales_z.pkl", "rb"))
    h4 = [(m["near"][0], m["far"][0]) for k, m in sc if k.get("alpha") == 4.0]
    put("oned_step_h_near", [float(np.nanmin([a for a, b in h4])), float(np.nanmax([a for a, b in h4]))])
    put("oned_step_h_far", [float(np.nanmin([b for a, b in h4])), float(np.nanmax([b for a, b in h4]))])
    dz = pickle.load(open("oned/delta_z.pkl", "rb"))
    put("oned_delta", {f"{k.get('mo_delta', 0.2):g},{k['mo_ua']*1e3:g},{k.get('pool_h')},{k['rng']}": m["regime"] for k, m in dz})
except Exception as e:
    print("1D failed", repr(e))
# flat-to-wall melt ratio with the stability closure (Fig. 3a)
def mo_ratio(ua, ang=0.5):
    SPY = 3.15576e7; UC = 2.97e-3; GP, KAP, SCD = 9.81 * 7.8e-4 * 34, 0.4, 0.05
    sth = np.sin(np.deg2rad(ang)); f0 = max(ang / 90, ua / UC); m0 = 30 * f0 / SPY; cth = np.sqrt(1 - sth ** 2); m = m0
    us = np.sqrt(ua ** 2 + SCD ** 2 * 0.01 * sth)
    for _ in range(300):
        L = us ** 3 / (KAP * GP * max(m, 1e-14) * cth); m = 0.5 * m + 0.5 * m0 * (0.01 + 0.99 / (1 + 5 * 0.2 / L))
    mw = 30 / SPY      # wall: f0 = 1; stability closure is negligible on a vertical face
    return m / mw
put("mo_flat_wall", {f"{ua*1e3:g}": mo_ratio(ua) for ua in (1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3)})

json.dump(Q, open("crosscheck_values.json", "w"), indent=1, default=float)
print("computed", len(Q), "quantities")
