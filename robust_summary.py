import pickle, glob, os, numpy as np, main_figs as M
def risers(r):
    F = r["field"].astype(float) + r["zbar"][None, :]; x = r["x"]; dx = x[1] - x[0]; H = []; S = []
    for row in F[::5]:
        st = np.abs(np.degrees(np.arctan(np.gradient(row, dx)))) > 15
        i0 = np.flatnonzero(np.diff(st.astype(int)) == 1) + 1; i1 = np.flatnonzero(np.diff(st.astype(int)) == -1) + 1; xs = []
        for i in i0:
            j = i1[i1 > i]
            if len(j) and abs(row[j[0]] - row[i - 1]) > 8 and x[i] > 1e3: H.append(abs(row[j[0]] - row[i - 1])); xs.append(x[i])
        S += list(np.diff(xs))
    return np.percentile(H, [25, 50, 75]) if H else [np.nan] * 3, np.percentile(S, [25, 50, 75]) if S else [np.nan] * 3
def line(tag, r):
    fl, st = M.melt_stats(r); h, s = risers(r)
    print(f"{tag:28s} G {r['G']:.2f}  flats {fl:.1f} faces {st:.0f}  riser h {h[0]:.0f}-{h[2]:.0f} (med {h[1]:.0f})  spacing {s[0]:.0f}-{s[2]:.0f}")
for k, f0, _, _ in M.TERR_CASES:
    line(f"{k} s1", pickle.load(open(f0, "rb")))
    for s in (2, 3):
        f = f"results_coarse/terrW_{k}_s{s}.pkl"
        if os.path.exists(f): line(f"{k} s{s}", pickle.load(open(f, "rb")))
for f, t in (("results_coarse/terr2dZ_ua0.3_ph1_M0.5.pkl", "1 km wide, dx 10"), ("results_coarse/terrRes_dx5_Ly1000.pkl", "1 km wide, dx 5")):
    if os.path.exists(f): line(t, pickle.load(open(f, "rb")))
for f in sorted(glob.glob("results_coarse/pond_*.pkl")) + ["results_coarse/mix_convA03show.pkl", "results_coarse/mix_convA03.pkl"]:
    r = pickle.load(open(f, "rb")); print(f"{os.path.basename(f):40s} G {r['G']:.2f} inc999 {r['inc999']:.0f}")
