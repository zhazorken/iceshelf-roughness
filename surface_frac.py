"""Randomized surface-crevasse (hydrofracture) test, applied to evolved basal fields.

Surface crevasses are transverse line segments (along y) placed at random
along flow downstream of x = 5 km. For each one we record the penetration
fraction needed to cut through the ice at that segment:
    r = min over segment of (H(x) - z'(x,y)) / H(x)
where z' is basal relief (positive up into the ice). A crevasse that
penetrates a fraction d/H of the ice thickness cuts through if d/H > r.

Two placements:
  random  : segment center random in y
  aligned : segment centered on the thickest-relief (thinnest-ice) point of
            its column; an upper bound for coupling between basal relief and
            surface fracture (e.g., fractures over basal channels).

Dimensions (literature ranges):
  length 0.2-2 km (uniform), width < 40 m grid cell (one column).
  Dry penetration about 10-40 m (Nye zero-stress depth for tensile stress
  100-200 kPa is about 20-45 m); water-filled crevasses can go deeper.
"""
import glob, os, pickle
import numpy as np
import shelfmelt as sm

RHO = None

def H_of(rec):
    p = dict(sm.P0, **rec["params"])
    return sm.thickness(np.asarray(rec["zbar"]), p), p

def sample(field, Hx, x40, n=20000, rng=None, aligned=False, xmin=5e3):
    rng = rng or np.random.default_rng(0)
    ny, nx = field.shape
    cols = np.where(x40 > xmin)[0]
    ix = rng.choice(cols, n)
    Lseg = rng.uniform(200, 2000, n)
    nseg = np.maximum(1, np.round(Lseg / 40.0).astype(int))
    out = np.empty(n)
    for k in range(n):
        col = field[:, ix[k]]
        if aligned:
            c = int(np.argmax(col))
        else:
            c = rng.integers(ny)
        rows = (c - nseg[k] // 2 + np.arange(nseg[k])) % ny
        out[k] = 1 - col[rows].max() / Hx[ix[k]]
    return out

def load_fields(pattern):
    recs = []; seen = set()
    pats = [pattern] + ([pattern.replace("mac/results/", "results/")] if pattern.startswith("mac/") else [])
    files = [f for p in pats for f in sorted(glob.glob(p))]
    for f in files:
        b = os.path.basename(f)
        if b in seen: continue
        seen.add(b)
        z = np.load(f)
        r = pickle.loads(z["meta"].tobytes())
        r.update(field40=z["field40"], zbar=z["zbar"], x=z["x"])
        recs.append(r)
    return recs

def seeded_field(p):
    q = sm.fill_times(p); x, y, _, _ = sm.setup(q)
    z = sm.Seeder(y, q).prefill(x)
    st = max(1, int(round(40.0 / q["dx"])))
    return z[::st, ::st]

def analyse(recs, aligned, n=20000):
    rs = []
    for i, r in enumerate(recs):
        H, p = H_of(r)
        st = max(1, int(round(40.0 / p["dx"])))
        Hx = H[::st]; x40 = r["x"][::st]
        rs.append(sample(r["field40"], Hx, x40, n, np.random.default_rng(i), aligned))
    return np.concatenate(rs)

if __name__ == "__main__":
    D = np.linspace(0, 1, 101)
    res = {}
    groups = {k: f"mac/results/ens__base__{k}__s*.npz" for k in
              ["shear", "dns", "dnsQ", "conv", "theta", "plume", "dnsF", "convF", "plumeF", "convP"]}
    groups["convA"] = "results/ens__base__convA__s*.npz"
    for h in (40, 120, 250):
        for v in ("shear", "conv", "plume", "dnsF", "dnsQ"):
            groups[f"crev{h}_{v}"] = f"results/crevasseheight(steepwalls)__{h}__{v}__s*.npz"
    for k, pat in groups.items():
        recs = load_fields(pat)
        if not recs: print("missing", k); continue
        for al in (False, True):
            r = analyse(recs, al)
            res[(k, al)] = np.array([(r < d).mean() for d in D])
        loc = [dict(r, field40=r["field40"] - r["field40"].mean(0, keepdims=True)) for r in recs]
        r = analyse(loc, False); res[(k, "local")] = np.array([(r < d).mean() for d in D])
        print(k, len(recs), "P_cut(0.5) rand %.4f  aligned %.4f   P_cut(0.7) %.3f %.3f" %
              (res[(k, False)][50], res[(k, True)][50], res[(k, False)][70], res[(k, True)][70]))
    # seeded references
    for h, pat in [(40, "mac/results/ens__base__shear__s*.npz"), (120, "results/crevasseheight(steepwalls)__120__shear__s*.npz"),
                   (250, "results/crevasseheight(steepwalls)__250__shear__s*.npz")]:
        recs = load_fields(pat)
        for r in recs:
            p = dict(sm.P0, **r["params"]); r["field40"] = seeded_field(p).astype(np.float32)
        for al in (False, True, "local"):
            if al == "local":
                recs = [dict(r, field40=r["field40"] - r["field40"].mean(0, keepdims=True)) for r in recs]
            rr = analyse(recs, al is True)
            res[(f"seed{h}", al)] = np.array([(rr < d).mean() for d in D])
        print("seed", h, res[(f"seed{h}", False)][50], res[(f"seed{h}", True)][50], res[(f"seed{h}", True)][70])
    with open("results_surface_frac.pkl", "wb") as f:
        pickle.dump(dict(D=D, res=res), f)
