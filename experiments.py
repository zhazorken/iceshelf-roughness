"""Experiment definitions. Each job = (name, host, regime, params).
Variants (melt physics):
  shear   shear melt                         conv    convection, sin^(2/3) law
  theta   convection, linear-in-angle law    dns     convection, DNS-fit law (a=0.761)
  dnsF    dns + open-flat stratification (theta_c = 3 deg)
  convF   conv + open-flat stratification    shearF  shear + open-flat stratification
  convP   conv + closed-basin pooling (h = 5 m)
All convective laws are renormalized to the baseline smooth-shelf mean melt; the shear
coefficient is calibrated once (equal smooth-shelf mean melt) and then held fixed.
"""
import numpy as np
import shelfmelt as sm

def calibrated_base():
    p = dict(sm.P0)
    p["shear_scale"] = sm.smooth_mean_melt(p, "convection") / sm.smooth_mean_melt(p, "shear")
    return p

BASE = calibrated_base()
M_CONV0 = sm.smooth_mean_melt(BASE, "convection")

def law(p, name, a=None):
    q = dict(p, conv_law=name, conv_scale=1.0)
    if a is not None: q["conv_a"] = a
    q["conv_scale"] = M_CONV0 / sm.smooth_mean_melt(q, "convection")
    return q

def plume(p):
    q = dict(p, plume_scale=1.0)
    q["plume_scale"] = M_CONV0 / sm.smooth_mean_melt(q, "plume")
    return q

def dnsmax(p):
    q = dict(p, dns_scale=1.0)
    q["dns_scale"] = M_CONV0 / sm.smooth_mean_melt(q, "dnsmax")
    return q

def variant(p, v):
    """Return (regime, params) for a physics variant applied on top of params p.
    'dns*' variants use the revised DNS convective-shear law, m ~ dT max(u_c(theta, dT), u*)."""
    if v == "dns":    return "dnsmax", dnsmax(p)
    if v == "dnsF":   return "dnsmax", dict(dnsmax(p), strat_theta_c=3.0)
    if v == "dnsP":   return "dnsmax", dict(dnsmax(p), pool_h=5.0)
    if v == "dnsQ":   return "dnsmax", dnsmax(p)          # quiet water: u* profile scaled by QUIET at job definition
    if v == "off":    return "convection", law(p, "dns")          # old offset-linear fit (a = 0.761)
    if v == "offF":   return "convection", dict(law(p, "dns"), strat_theta_c=3.0)
    if v == "shear":  return "shear", p
    if v == "shearF": return "shear", dict(p, strat_theta_c=3.0)
    if v == "conv":   return "convection", p
    if v == "theta":  return "convection", law(p, "theta")
    if v == "convF":  return "convection", dict(p, strat_theta_c=3.0)
    if v == "convP":  return "convection", dict(p, pool_h=5.0)
    if v == "mixed":  return "shear+conv", p
    if v == "plume":  return "plume", plume(p)
    if v == "plumeF": return "plume", dict(plume(p), strat_theta_c=3.0)
    if v.endswith("A") and v[:-1] in ("shear", "conv", "dns", "plume"):   # accumulating closed-basin ponds
        regime, q = variant(p, v[:-1])
        return regime, dict(q, pool_mode="accumulate")
    raise ValueError(v)

def slope_contrast(p, regime="convection"):
    """C = melt slope factor at a vertical wall / mean over the smooth shelf."""
    x, y, zbar, u = sm.setup(p)
    sth = np.sin(np.arctan(np.abs(np.gradient(zbar, p["dx"]))))
    if regime == "dnsmax":   # melt depends on slope and position: wall melt / smooth-shelf melt, domain means
        dT = sm.thermal_driving(zbar, p)
        return float(sm.melt_dnsmax(dT, np.ones_like(sth), u, p).mean() / sm.melt_dnsmax(dT, sth, u, p).mean())
    fn = (lambda s: sm.plume_factor(s, p)) if regime == "plume" else (lambda s: sm.slope_factor(s, p))
    peak = float(np.max(fn(np.sin(np.deg2rad(np.linspace(0.1, 90, 900))))))
    return peak / float(np.mean(fn(sth)))       # peak contrast (plume law peaks near 10 deg)

JOBS = []
def add(host, group, label, v, p, seeds):
    for s in seeds:
        regime, q = variant(dict(p, seed=s), v)
        name = f"{group}__{label}__{v}__s{s}".replace(" ", "").replace("/", "-")
        JOBS.append(dict(name=name, host=host, group=group, label=label, variant=v,
                         regime=regime, seed=s, params=q))

# ---------------- Mac VM: ensembles, contrast, one-at-a-time, channels (dx = 20 m, 40 x 4 km)
for v in ["shear", "conv", "dns", "dnsF", "convP", "mixed"]:
    add("mac", "ens", "base", v, BASE, range(1, 5))
for v in ["theta", "convF", "shearF"]:
    add("mac", "ens", "base", v, BASE, range(1, 4))
for a in [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.761, 0.9]:
    for s in range(1, 3):
        regime, q = "convection", law(dict(BASE, seed=s), "dns", a)
        JOBS.append(dict(name=f"contrast__a{a:g}__s{s}", host="mac", group="contrast", label=f"{a:g}",
                         variant="dns_a", regime=regime, seed=s, params=q))
OAT = [("thermal forcing", "T_gl", [0.2, 2.2], "C"),
       ("ambient stratification", "dTdz", [0.0, -6e-3], ""),
       ("crevasse height", "crev_H", [10.0, 80.0], "m"),
       ("crevasse spacing", "crev_spacing", [750.0, 3000.0], "m"),
       ("u* magnitude", "ustar_f", [0.5, 2.0], "x"),
       ("GL slope", "theta_gl", [2.0, 8.0], "deg"),
       ("ice speed", "U_ice", [300.0, 3000.0], "m/yr"),
       ("slope smoothing", "slope_smooth", [20.0, 80.0], "m"),
       ("creep closure", "creep_eta", [1e14, 1e15, 1e16], "Pa s")]
for group, key, vals, unit in OAT:
    for val in vals:
        if key == "ustar_f":
            p = dict(BASE, ustar_gl=BASE["ustar_gl"] * val, ustar_out=BASE["ustar_out"] * val)
        else:
            p = dict(BASE, **{key: val})
        for v in ["shear", "conv", "dnsF"]:
            add("mac", group, f"{val:g}", v, p, [1])
for tc in [1.5, 5.0]:
    add("mac", "flat-strat angle", f"{tc:g}", "dnsF", BASE, [1])
    JOBS[-1]["params"]["strat_theta_c"] = tc
for tau in [0.05, 5.0]:
    add("mac", "strat timescale", f"{tau:g}", "dnsF", dict(BASE, pool_tau=tau), [1])
for h in [2.0, 20.0]:
    add("mac", "pool depth scale", f"{h:g}", "convP", BASE, [1])
    JOBS[-1]["params"]["pool_h"] = h
add("mac", "blend", "max", "mixed", dict(BASE, blend="max"), [1])
CH = dict(BASE, crev_H=0.0, chan_amp=10.0, chan_lambda=1000.0)
for ang in [0.0, 30.0]:
    for v in ["shear", "conv", "dnsF"]:
        add("mac", "channels", f"{ang:g}", v, dict(CH, chan_angle=ang), range(1, 3))

# ---------------- buoyant-plume melt law (added after first batch)
for v in ["plume", "plumeF"]:
    add("mac", "ens", "base", v, BASE, range(1, 5))
for group, key, vals, unit in OAT:
    for val in vals:
        if key == "ustar_f": continue
        add("mac", group, f"{val:g}", "plume", dict(BASE, **{key: val}), [1])

# ---------------- terraces: uniform 4-deg base (channel-flank-like), melt dominance
# M = smooth-base melt / (U tan alpha): fraction of the thinning budget supplied by melt.
# A fully melt-suppressed patch keeps an absolute slope ~ (1 - M) tan(alpha): flat only if M -> 1.
TER = dict(BASE, Lx=5e3, Ly=1e3, dx=10.0, theta_gl=4.0, theta_far=4.0, U_ice=1000.0)
def with_M(p, regime, M):
    q = dict(p, melt_mult=1.0)
    q["melt_mult"] = M * q["U_ice"] * np.tan(np.deg2rad(4.0)) / (sm.smooth_mean_melt(q, regime) * sm.SPY)
    return q
for M in [0.25, 0.5, 0.75, 1.0]:
    for v in ["plume", "plumeF", "dns", "dnsF"]:
        for s in (1, 2):
            regime, q = variant(dict(TER, seed=s), v)
            if v.endswith("F"): q["strat_theta_c"] = 2.0
            JOBS.append(dict(name=f"terrace__M{M:g}__{v}__s{s}", host="mac", group="terrace", label=f"{M:g}",
                             variant=v, regime=regime, seed=s, params=with_M(q, regime, M)))
for tc in [1.0, 4.0]:
    for s in (1, 2):
        regime, q = variant(dict(TER, seed=s), "plumeF"); q["strat_theta_c"] = tc
        JOBS.append(dict(name=f"terrace-thc__{tc:g}__plumeF__s{s}", host="mac", group="terrace-thc",
                         label=f"{tc:g}", variant="plumeF", regime=regime, seed=s, params=with_M(q, regime, 1.0)))

# ---------------- v3 (added after first draft) ----------------
# A: ponds whose stratification builds up with age (freshwater accumulates in closed basins)
for v in ["shearA", "convA", "dnsA", "plumeA"]:
    add("v3", "ens", "base", v, BASE, range(1, 5))
for hc in [0.5, 10.0]:
    add("v3", "pond freshwater scale", f"{hc:g}", "convA", dict(BASE, pool_hc=hc), range(1, 3))
for ft in [0.02, 1.0]:
    add("v3", "pond flushing time", f"{ft:g}", "convA", dict(BASE, pool_flush_tau=ft), range(1, 3))
# B: slope-smoothing scale at 5 m resolution (boundary-layer / meltwater-layer thickness, 10-40 m)
SM = dict(BASE, Lx=20e3, Ly=1e3, dx=5.0)
for ss in [10.0, 20.0, 40.0]:
    for v in ["shear", "conv", "dns", "dnsF", "plume"]:
        add("v3", "smoothing-5m", f"{ss:g}", v, dict(SM, slope_smooth=ss), [1])
# C: GL slope with fixed endpoints (draft -900 m at GL to -368 m at 40 km for every slope)
for th in [2.0, 4.0, 8.0, 12.0]:
    for v in ["shear", "conv", "dnsF", "plume"]:
        add("v3", "GL slope (fixed ends)", f"{th:g}", v, dict(BASE, theta_gl=th, z_front=-367.57), range(1, 3))

# ---------------- v4: staircase search on the uniform 4-degree, melt-dominated base ----------------
def ter_job(tag, v, M, s, **kw):
    regime, q = variant(dict(TER, seed=s, keep_field=True), v)
    q["strat_theta_c"] = 2.0 if v.endswith("F") else q["strat_theta_c"]
    q.update(kw)
    q = with_M(q, regime, M)
    JOBS.append(dict(name=f"stair__{tag}__M{M:g}__{v}__s{s}", host="v4", group="stair", label=tag,
                     variant=v, regime=regime, seed=s, params=q, M=M))
for M in [0.5, 1.0]:
    for v in ["plumeF", "dnsF"]:
        ter_job("none", v, M, 1)
        for h in [2.0, 10.0]:
            for tau in [0.05, 2.0]:
                ter_job(f"pool-h{h:g}-tau{tau:g}", v, M, 1, pool_h=h, pool_tau=tau)
        for hc in [0.5, 5.0]:
            ter_job(f"age-hc{hc:g}", v, M, 1, pool_mode="accumulate", pool_hc=hc)
        for hl in [10.0, 20.0, 40.0]:
            ter_job(f"layers-{hl:g}", v, M, 1, ht_layer=hl)
        ter_job("layers-20+pool", v, M, 1, ht_layer=20.0, pool_h=5.0, pool_tau=0.5)
# layered melt on the main 40-km shelf
for v in ["plumeF", "dnsF", "conv"]:
    add("v4", "layers-main", "20", v, dict(BASE, ht_layer=20.0), [1])

# ---------------- v5: realistic basal crevasses (Larsen C: 70-134 m, 25-40% of thickness; McGrath et al. 2012)
for H in [120.0, 250.0]:
    for v in ["shear", "conv", "plume", "dnsF"]:
        add("v5", "crevasse height (steep walls)", f"{H:g}", v, dict(BASE, crev_H=H, crev_wall=70.0), [1])
for v in ["shear", "conv", "plume", "dnsF"]:
    add("v5", "crevasse height (steep walls)", "40", v, dict(BASE, crev_wall=70.0), [1])

# ---------------- v6: ocean T(z) responds to depth-binned, horizontally integrated melt
for fb in [0.3, 1.0]:
    for v in ["plumeF", "dnsF", "plume", "conv"]:
        ter_job(f"tz{fb:g}", v, 1.0, 1, tz_fb=fb)
        JOBS[-1]["host"] = "v6"
    for v in ["shear", "conv", "plume", "dnsF"]:
        add("v6", "Tz feedback", f"{fb:g}", v, dict(BASE, tz_fb=fb, keep_field=True), [1])

# v6b: weaker coupling, with and without buoyant rise of the meltwater
for fb in [0.03, 0.1]:
    for rise in [0.0, 30.0]:
        for v in ["plumeF", "dnsF"]:
            ter_job(f"tz{fb:g}-rise{rise:g}", v, 1.0, 1, tz_fb=fb, tz_rise=rise)
            JOBS[-1]["host"] = "v6b"
for rise in [0.0, 30.0]:
    for v in ["conv", "dnsF"]:
        add("v6b", "Tz feedback weak", f"rise{rise:g}", v, dict(BASE, tz_fb=0.1, tz_rise=rise, keep_field=True), [1])

# v6c: rise-height and coupling sweep for layered T(z)
for fb in [0.1, 0.3]:
    for rise in [10.0, 20.0, 50.0]:
        for v in ["plumeF", "dnsF"]:
            ter_job(f"tz{fb:g}-rise{rise:g}", v, 1.0, 1, tz_fb=fb, tz_rise=rise)
            JOBS[-1]["host"] = "v6c"
for fb, rise in [(0.1, 20.0), (0.3, 20.0)]:
    for v in ["plume", "dns"]:
        ter_job(f"tz{fb:g}-rise{rise:g}", v, 1.0, 1, tz_fb=fb, tz_rise=rise)
        JOBS[-1]["host"] = "v6c"
for v in ["plumeF", "dnsF"]:
    for M in [0.5]:
        ter_job("tz0.1-rise30", v, M, 1, tz_fb=0.1, tz_rise=30.0); JOBS[-1]["host"] = "v6c"

# ---------------- cloud: resolution (40 x 1 km x 20 km) and showcase (40 x 6 km, dx = 10 m)
RES = dict(BASE, Lx=20e3, Ly=1e3)
for dx in [40.0, 20.0, 10.0, 5.0]:
    for v in ["shear", "conv", "dnsF"]:
        add("mac", "resolution", f"{dx:g}", v, dict(RES, dx=dx), [1])
    add("mac", "resolution-scaled", f"{dx:g}", "conv", dict(RES, dx=dx, slope_smooth=2 * dx), [1])
SHOW = dict(BASE, Ly=6e3, dx=10.0, keep_field=True, snap_years=2.0)
for v in ["shear", "conv", "dns", "dnsF", "convP"]:
    add("cloud", "showcase", "pig", v, SHOW, [1])

def cost(j):
    p = j["params"]
    c = (p["Lx"] / p["dx"]) ** 2 * (p["Ly"] / p["dx"])
    if p["pool_h"] > 0: c *= 2.5
    if p["strat_theta_c"] > 0: c *= 1.3
    return c

if __name__ == "__main__":
    for h in ("mac", "cloud"):
        js = [j for j in JOBS if j["host"] == h]
        print(h, len(js), "jobs, relative cost", round(sum(cost(j) for j in js) / cost(JOBS[0]), 1))
# v7: T(z) layering with partly dynamic thinning (M < 1), to let the base climb past cold layers
for M in [0.6, 0.8]:
    for fb in [0.3, 1.0]:
        for rise in [10.0, 20.0]:
            ter_job(f"tz{fb:g}-rise{rise:g}", "dnsF", M, 1, tz_fb=fb, tz_rise=rise); JOBS[-1]["host"] = "v7"
for M in [0.5, 0.6, 0.7]:
    for rise in [20.0, 40.0]:
        if M == 0.6 and rise == 20.0: continue
        ter_job(f"tz1-rise{rise:g}", "dnsF", M, 1, tz_fb=1.0, tz_rise=rise); JOBS[-1]["host"] = "v7"
ter_job("tz1-rise10", "dnsF", 0.5, 1, tz_fb=1.0, tz_rise=10.0); JOBS[-1]["host"] = "v7"
ter_job("tz1-rise10", "dnsF", 0.7, 1, tz_fb=1.0, tz_rise=10.0); JOBS[-1]["host"] = "v7"

# ---------------- revised DNS law (convective-shear max law); reruns tagged zc (cloud) / zm (Mac VM)
def _q(p, f): return dict(p, ustar_gl=p["ustar_gl"] * f, ustar_out=p["ustar_out"] * f)
QUIET = 0.2
def _tag(n, host):
    for j in JOBS[-n:]: j["host"] = host
for v in ["dns", "dnsF"]:
    add("zc", "ens", "base", v, BASE, range(1, 5)); _tag(4, "zc")
add("zc", "ens", "base", "dnsQ", _q(BASE, QUIET), range(1, 5)); _tag(4, "zc")
for f in [0.1, 0.35, 0.5, 2.0]:
    add("zc", "dns u*", f"{f:g}", "dns", _q(BASE, f), range(1, 3)); _tag(2, "zc")
for v in ["dns", "dnsF"]:
    add("zc", "showcase", "pig", v, SHOW, [1]); _tag(1, "zc")
add("zc", "showcase", "pig", "dnsQ", _q(SHOW, QUIET), [1]); _tag(1, "zc")
# Mac VM: one-at-a-time, channels, tall crevasses, ponds, closures, uniform-base terraces
for group, key, vals, unit in OAT:
    for val in vals:
        p = _q(BASE, val) if key == "ustar_f" else dict(BASE, **{key: val})
        add("zm", group, f"{val:g}", "dnsF", p, [1]); _tag(1, "zm")
for ang in [0.0, 30.0]:
    add("zm", "channels", f"{ang:g}", "dnsF", dict(CH, chan_angle=ang), range(1, 3)); _tag(2, "zm")
for H in [40.0, 120.0, 250.0]:
    for v in ["dnsF", "dnsQ"]:
        pp = dict(BASE, crev_H=H, crev_wall=70.0)
        add("zm", "crevasse height (steep walls)", f"{H:g}", v, _q(pp, QUIET) if v == "dnsQ" else pp, [1]); _tag(1, "zm")
add("zm", "ens", "base", "dnsA", BASE, range(1, 3)); _tag(2, "zm")
add("zm", "ens", "base", "dnsP", _q(BASE, QUIET), range(1, 3)); _tag(2, "zm")
for tc in [1.5, 5.0]:
    add("zm", "flat-strat angle", f"{tc:g}", "dnsF", BASE, [1]); JOBS[-1]["params"]["strat_theta_c"] = tc; _tag(1, "zm")
for tau in [0.05, 5.0]:
    add("zm", "strat timescale", f"{tau:g}", "dnsF", dict(BASE, pool_tau=tau), [1]); _tag(1, "zm")
