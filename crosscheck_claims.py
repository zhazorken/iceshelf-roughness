"""Compare the numbers printed in the NatGeo text with recomputed values (crosscheck_values.json)."""
import json, csv, re
Q = json.load(open("crosscheck_values.json")); T = open("latex_ngeo/main.tex").read()
def g(k, i=None): v = Q[k]; return v[i] if i is not None else v
C = [  # (claim, text snippet that must be present, printed values, recomputed values, decimals)
("Shear and DNS law, tidal u*: G", "$G=0.69\\pm0.03$", [0.69, 0.03], [g("G_dns"), g("Gsd_dns")], 2),
("DNS law, u* x0.1: G and C10", "$G=0.87$, $C_{10}=1.11$", [0.87, 1.11], [g("dnsu_0.1", 0), g("dnsu_0.1", 1)], 2),
("DNS law, u* x0.05: G and C10", "($G=2.9$, $C_{10}=1.8$)", [2.9, 1.8], [g("dnsu_0.05", 0), g("dnsu_0.05", 1)], 1),
("DNS convection without shear: G", "no shear at all ($G=3.2$)", [3.2], [g("G_theta")], 1),
("Convective velocity, vertical wall (mm/s)", "2.5 to 3.2~mm", [2.5, 3.2], g("uc_wall"), 1),
("Convective velocity, 10 deg face (mm/s)", "about 0.3~mm", [0.3], [sum(g("uc_10deg")) / 2], 1),
("Offset-linear family C10 range", "$C_{10}$ runs from 1.01 to 10", [1.01, 10], [g("contrast_C10_range", 0), round(g("contrast_C10_range", 1))], 2),
("G = 1 crossing (C10 bracket)", "between $C_{10}=1.15$ and 1.18", [1.15, 1.18], g("contrast_cross"), 2),
("C10: sin2/3, DNS conv., plume", "convection (5.2), DNS convection without shear (10) and the plume law (2.4)", [5.2, 10, 2.4], [g("C10_conv"), round(g("C10_theta")), g("C10_plume")], 1),
("Mean-melt enhancement: plume, sin2/3, DNS conv.", "1.7 times the smooth-shelf value", [1.7, 2.0, 3.1], [g("enh_plume"), g("enh_conv"), g("enh_theta")], 1),
("Closed-basin pooling, sin2/3: G", "smooths the base ($G=0.87$)", [0.87], [g("G_convP")], 2),
("Aging ponds, all laws: G range", "($G=0.32$ to 0.36", [0.32, 0.36], g("G_aging_range"), 2),
("Flat stratification, shear and DNS: G", "roughen the base ($G=5.6$)", [5.6], [g("G_dnsF")], 1),
("Flat strat. raises local relief (ratio)", "about 2.5-fold, and crevasses", [2.5], [g("inc_40_dnsF", 0) / g("inc_40_dns", 0)], 1),
("Seeded 40 m crevasses: incision/H", "reach 0.10 of the downstream thickness", [0.10], [g("inc_seed40")], 2),
("Shear, DNS tidal, DNS quiet: incision/H", "quiet water leave 0.08", [0.08], [max(g("inc_40_shear", 1), g("inc_40_dns", 1), g("inc_40_dnsQ", 1))], 2),
("Pooling and aging ponds: incision/H", "0.03 to 0.08", [0.03, 0.08], [min(g(f"inc_40_{v}", 1) for v in ("convP", "dnsP", "convA", "shearA", "plumeA", "dnsA")), max(g(f"inc_40_{v}", 1) for v in ("convP", "dnsP", "convA", "shearA", "plumeA", "dnsA"))], 2),
("sin2/3, DNS conv., sin2/3+flat: incision/H", "0.34 of the thickness", [0.34, 0.66, 0.69], [g("inc_40_conv", 1), g("inc_40_theta", 1), g("inc_40_convF", 1)], 2),
("250 m: shear and DNS quiet", "0.47 to 0.49 for shear", [0.47, 0.49], [g("inc_250_shear", 1), g("inc_250_dnsQ", 1)], 2),
("250 m: plume, DNS+flat, sin2/3", "0.55 for the plume law", [0.55, 0.68, 0.74], [g("inc_250_plume", 1), g("inc_250_dnsF", 1), g("inc_250_conv", 1)], 2),
("Surface fracture 1%: shear/DNS", "only if they reach 95\\%", [0.95], [g("frac_shear")], 2),
("Surface fracture 1%: sin2/3, plume, DNS+flat, DNS conv.", "67\\% ($\\sin^{2/3}$)", [0.67, 0.75, 0.78, 0.34], [g("frac_conv"), g("frac_plume"), g("frac_dnsF"), g("frac_theta")], 2),
("Surface fracture 1%, 250 m: shear, plume, DNS quiet, DNS+flat, sin2/3", "60\\% for shear melt", [0.60, 0.55, 0.56, 0.43, 0.30], [g("frac250_shear"), g("frac250_plume"), g("frac250_dnsQ"), g("frac250_dnsF"), g("frac250_conv")], 2),
("Radar-band rms: shear (m)", "3.5~m for shear melt", [3.5], [g("rms_radar_shear")], 1),
("Radar-band rms: roughening laws (m)", "13 to 14~m for roughening", [13, 14], [min(g("rms_radar_conv"), g("rms_radar_plume"), g("rms_radar_dnsF")), max(g("rms_radar_conv"), g("rms_radar_plume"), g("rms_radar_dnsF"))], 0),
("Radar-band rms: 250 m crevasses (m)", "29 to 68~m", [29, 68], [g("rms_radar250_shear"), g("rms_radar250_conv")], 0),
("Pine Island and Dotson rms, 2-20 km (m)", "30~m on the Pine Island base and 8~m", [30, 8], [g("obs_rms_pig"), g("obs_rms_dotson")], 0),
("Build-up: b1, b2, b3, b4 (m)", "Shear melt with 40~m crevasses gives 8~m", [8, 11, 16, 24], [g("build_b1", 0), g("build_b2", 0), g("build_b3", 0), g("build_b4", 0)], 0),
("Build-up b4 seed range (m)", "(21 to 27~m over three seeds)", [21, 27], [g("build_b4", 1), g("build_b4", 2)], 0),
("Build-up ponds 1-3% (m)", "17 to 20~m", [17, 20], [g("build_b5", 0), g("build_b7", 0)], 0),
("Observed anisotropy PIG, Dotson", "0.13 and 0.19", [0.13, 0.19], [g("obs_aniso_pig"), g("obs_aniso_dotson")], 2),
("Shear-controlled anisotropy", "about 2.8 times", [2.8], [g("aniso_shear")], 1),
("Terraces in 1D: u*a range (mm/s)", "0.2 to 0.3~mm\\,s$^{-1}$ and the pooling", [0.2, 0.3], [min(g("oned_terr_ua")), max(g("oned_terr_ua"))], 1),
("Terraces in 1D: pooling depth range (m)", "pooling depth scale is 0.5 to 2~m", [0.5, 2], [min(g("oned_terr_h")), max(g("oned_terr_h"))], 1),
("Sawtooth tread tilt (deg)", "tilted back by 2 to \\ang{9}", [2, 9], g("oned_saw_tilt"), 0),
("Terrace tread tilt (deg)", "within 0.1 to \\ang{0.7}", [0.1, 0.7], g("oned_terr_tilt"), 1),
("Step height near GL (m)", "from 7 to 9~m within 3~km", [7, 9], g("oned_step_h_near"), 0),
("Step height far from GL (m)", "to 21 to 30~m beyond", [21, 30], g("oned_step_h_far"), 0),
("Plan-view risers (m)", "risers 19 to 38~m high", [19, 38], g("terrW_h"), 0),
("Plan-view spacing (m)", "340 to 990~m apart", [340, 990], [round(v, -1) for v in g("terrW_s")], 0),
("Flat-ice melt (m/yr)", "median 2.6~m", [2.6], [g("terrW_flats", 2)], 1),
("Steep-face melt (m/yr)", "56 to 61~m", [56, 61], g("terrW_faces"), 0),
("4 deg flank without stability closure: G", "barely roughens ($G=1.0$)", [1.0], [g("attr_nomo_nopool")], 1),
("Fig 1c pond case, three seeds: G", "($G=0.86$ to 2.06)", [0.86, 2.06], g("pond_seeds"), 2),
("Pond controls: sin2/3, shear: G", "($G=1.56$)", [1.56, 0.47], g("pond_ctrl"), 2),
("Retention scan f_r: G", "gives $G=1.70$, 1.43, 1.08 and 0.66", [1.70, 1.43, 1.08, 0.66], g("prod_scan"), 2),
("Plan-view G: terraces, sawtooth, smooth", "($G=3.4$ to 5.0), $G=2.9$ to 3.1 and $G=0.62$ to 0.80", [3.4, 5.0, 2.9, 3.1, 0.62, 0.80], [g("terrW_G_terr", 0), g("terrW_G_terr", 1), g("terrW_G_saw", 0), g("terrW_G_saw", 1), g("terrW_G_smo", 0), g("terrW_G_smo", 1)], 2),
("5 m vs 10 m grid: G, flats, faces, riser median", "($G=4.2$ against 4.0", [4.2, 2.3, 63, 30, 4.0, 2.2, 66, 38], g("terr_res5") + g("terr_res10"), 1),
]
rows = []; bad = 0
for name, snip, tv, cv, nd in C:
    present = snip in T
    cvr = [round(float(c), nd) for c in cv]
    ok = all(abs(a - b) < 0.5 * 10 ** (-nd) + 1e-9 for a, b in zip(tv, [round(float(c), nd) if nd > 0 else round(float(c)) for c in cv]))
    close = all(abs(a - float(b)) <= 1.01 * 10 ** (-nd) for a, b in zip(tv, cv))
    ok = ok or all(abs(a - float(b)) <= max(0.5 * 10 ** (-nd), 0.5 * 10 ** (len(str(a).split(".")[1]) * -1 if "." in str(a) else 0.5) + 1e-9) or abs(a - float(b)) / max(abs(a), 1e-9) < 0.012 for a, b in zip(tv, cv))
    st = "ok" if ok else ("close" if close else "MISMATCH")
    if not present: st += " (text not found)"
    if st != "ok": bad += 1
    rows.append((name, ", ".join(f"{v:g}" for v in tv), ", ".join(f"{float(c):.{max(nd,1)+1}f}" for c in cv), st))
with open("crosscheck.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["claim", "printed", "recomputed", "status"]); w.writerows(rows)
for r in rows: print(f"{r[3]:9s} | {r[0]} | text {r[1]} | model {r[2]}")
print(len(rows), "claims,", bad, "not exact")
