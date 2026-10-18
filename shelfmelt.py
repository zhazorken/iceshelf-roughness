"""
shelfmelt v2: idealized 3D ice-shelf base evolution under shear / convective melt.

Base z_b(x,y) = zbar(x) + z'(x,y); x along flow from the grounding line (GL), y periodic.
  z'_t + U z'_x = m(z_b) sqrt(1+|grad z_b|^2) - w_ref(x) - creep(z')
- Ice advection: exact one-cell shift per step (no numerical diffusion).
- Melt front: Godunov Hamilton-Jacobi upwinding (keels round, apexes cusp).
- w_ref: melt of the smooth shelf (steady-shelf assumption: ice flux balances smooth-shelf melt).
- Optional physics: closed-basin meltwater pooling, drainage-limited stratification on open
  flats, viscous creep closure of basal topography (spectral, flat surface / bridged features).
Melt laws are placeholders for DNS fits: (dT, sin_theta, ustar, p) -> m [m/s].
"""
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

SPY = 3.15576e7
G = 9.81

P0 = dict(
    # domain ("mini Pine Island": 40 km along flow)
    Lx=40e3, Ly=4e3, dx=20.0,
    # mean base: slope theta(x) = far + (gl - far) exp(-x/L_theta)
    z_gl=-900.0, theta_gl=4.0, theta_far=0.3, L_theta=5e3,
    # ocean: T(z) = T_gl + dTdz (z - z_gl); Tf = lam1 S + lam2 + lam3 z
    T_gl=1.2, dTdz=-3.0e-3, S=34.5, lam1=-0.0573, lam2=0.0832, lam3=7.61e-4,
    rho_i=917.0, rho_w=1028.0, cw=3974.0, Lf=3.34e5,
    # shear regime: u*(x) weak in inner cavity, rising seaward
    ustar_gl=1.0e-3, ustar_out=4.0e-3, L_u=8e3, GammaT=0.011, shear_scale=1.0,
    # convective regime: m = Kconv dT^(4/3) f(theta) * conv_scale
    Kconv=5.0e-7, conv_law="sin23", conv_a=0.761, conv_scale=1.0,
    blend="cubic", melt_mult=1.0,
    dns_uc90=4.0e-3, dns_dTref=7.3, dns_Gamma=0.01, dns_scale=1.0, dns_umult=1.0,   # melt_mult scales every melt law (melt-dominance experiments)
    # local buoyant-plume melt (Lazeroms et al. 2018-type slope factor): m = M0 g(theta) dT^2
    plume_M0=10.0 / 3.15576e7, plume_E0=3.6e-2, plume_Cd=2.5e-3, plume_GTS=5.9e-4, plume_scale=1.0,
    # ice flow and GL seeding
    U_ice=1000.0,
    crev_H=40.0, crev_wall=45.0, crev_spacing=1500.0, crev_len=(600.0, 1800.0),
    noise_amp=2.0, noise_lmin=100.0, noise_lmax=1000.0,
    chan_amp=0.0, chan_lambda=1000.0, chan_angle=0.0,
    # meltwater stratification
    pool_h=0.0,          # closed-basin pooling depth scale (m); 0 = off
    strat_theta_c=0.0,   # open-flat stratification critical slope (deg); 0 = off
    pool_tau=0.5,        # relaxation time of thermal driving toward 0 (yr)
    # "accumulate" mode: meltwater produced inside closed basins stays there as a fresh layer of
    # thickness F (m, grows by local melt); outside basins it drains over pool_flush_tau (yr).
    # Suppression phi = 1 - exp(-F / pool_hc): stratification strengthens as the pond ages.
    pool_mode="relax", pool_hc=2.0, pool_flush_tau=0.1, z_front=None,
    pool_every_yr=0.1,   # basin recomputation interval (yr)
    # viscous creep closure: rate(k) = rho_w g k H^2 / (2 eta (1 + k^2 H^2)); 0 = off
    creep_eta=0.0, creep_every_yr=0.1,
    # Huppert-Turner-type layering: in a stratified ambient, melt on steep faces organizes into
    # horizontal layers of thickness ht_layer (m), fixed in depth; melt is modulated by a sawtooth
    # 1 + ht_amp (2 phase - 1), strongest just below each layer top. 0 = off.
    # ocean T(z) feedback: horizontally integrated melt in each depth bin cools that bin,
    # dT'/dt = -tz_fb (rho_i Lf / rho_w cw) Q_bin / (Ly Lx dz) - T'/tz_tau  (Q_bin = melt volume flux in bin)
    strat_n=2.0, adapt_ref=0.0, mo=False, mo_ua=1e-4, mo_gD=0.01, mo_delta=0.2, mo_smin=0.01, mo_iter=30, face_beta=0.0, face_theta=20.0, face_h0=10.0, face_clip=5.0, tz_fb=0.0, tz_dz=5.0, tz_tau=2.0,
    tz_rise=0.0,   # m: meltwater rises this far before intruding, so the cooling lands above its source
    ht_layer=0.0, ht_amp=0.8, ht_theta=5.0,   # ht_theta applies to the smoothed slope
    # numerics
    slope_smooth=40.0, cfl=0.25, seed=1,
    years=None, avg_years=None, diag_every=None,
    keep_field=False, snap_years=None, snap_dx=40.0, pool_prod=1.0, crev_Hmax=0.0,
)

# ---------------------------------------------------------------- base state
def setup(p):
    nx, ny = int(round(p["Lx"] / p["dx"])), int(round(p["Ly"] / p["dx"]))
    x = (np.arange(nx) + 0.5) * p["dx"]; y = (np.arange(ny) + 0.5) * p["dx"]
    if p.get("z_front") is not None:
        # fixed endpoints: draft goes z_gl -> z_front over Lx whatever the GL slope; the
        # e-folding length adjusts so the slope at the GL is theta_gl and far field theta_far
        sf = np.tan(np.deg2rad(p["theta_far"])); A = (p["z_front"] - p["z_gl"]) - sf * p["Lx"]
        L = A / (np.tan(np.deg2rad(p["theta_gl"])) - sf)
        zbar = p["z_gl"] + A * (1 - np.exp(-x / L)) + sf * x
    else:
        th = np.deg2rad(p["theta_far"] + (p["theta_gl"] - p["theta_far"]) * np.exp(-x / p["L_theta"]))
        zbar = p["z_gl"] + np.cumsum(np.tan(th)) * p["dx"]
    ustar = p["ustar_out"] - (p["ustar_out"] - p["ustar_gl"]) * np.exp(-x / p["L_u"])
    return x, y, zbar, ustar

def thermal_driving(z, p):
    T = p["T_gl"] + p["dTdz"] * (z - p["z_gl"])
    Tf = p["lam1"] * p["S"] + p["lam2"] + p["lam3"] * z
    return np.maximum(T - Tf, 0.0)

def thickness(zbar, p):
    return -zbar * p["rho_w"] / p["rho_i"]

def fill_times(p):
    p = dict(p); tres = p["Lx"] / p["U_ice"]
    if p["years"] is None: p["years"] = 1.5 * tres
    if p["avg_years"] is None: p["avg_years"] = 0.4 * tres
    if p["diag_every"] is None: p["diag_every"] = 0.01 * tres
    return p

# ---------------------------------------------------------------- melt laws
def slope_factor(sin_th, p):
    law = p["conv_law"]
    if law == "sin23":
        return np.maximum(sin_th, 1e-4) ** (2 / 3)
    th = np.arcsin(np.clip(sin_th, 0, 1)) / (np.pi / 2)
    if law == "theta":
        return np.maximum(th, 1e-4)
    if law == "dns":
        return p["conv_a"] + (1 - p["conv_a"]) * th
    raise ValueError(law)

def melt_shear(dT, sin_th, ustar, p):
    return p["melt_mult"] * p["shear_scale"] * p["rho_w"] * p["cw"] / (p["rho_i"] * p["Lf"]) * p["GammaT"] * ustar * dT

def melt_conv(dT, sin_th, ustar, p):
    return p["melt_mult"] * p["conv_scale"] * p["Kconv"] * dT ** (4 / 3) * slope_factor(sin_th, p)

def melt_combined(dT, sin_th, ustar, p):
    ms, mc = melt_shear(dT, sin_th, ustar, p), melt_conv(dT, sin_th, ustar, p)   # both include melt_mult
    return np.maximum(ms, mc) if p["blend"] == "max" else np.cbrt(ms ** 3 + mc ** 3)

def plume_factor(sin_th, p):
    """Lazeroms et al. (2018) slope dependence of plume-driven melt: plume speed grows with
    sqrt(sin theta) and entrainment with E0 sin theta. Normalized to 1 at a vertical wall."""
    def g(s):
        E, Cd, GTS = p["plume_E0"] * s, p["plume_Cd"], p["plume_GTS"]
        return np.sqrt(s / (Cd + E)) * np.sqrt(Cd / (Cd + E)) * E / (GTS + E)
    return g(np.maximum(sin_th, 1e-5)) / g(1.0)

def melt_plume(dT, sin_th, ustar, p):
    return p["melt_mult"] * p["plume_scale"] * p["plume_M0"] * plume_factor(sin_th, p) * dT ** 2

def mo_melt(m0, sth, p):
    """Monin-Obukhov suppression of melt (m/s). Meltwater buoyancy flux B = g' m stabilizes the boundary layer
    (component B cos theta); turbulence from an ambient friction velocity plus the along-slope gravity current of
    the meltwater layer, u = sqrt(g'D sin theta). S = smin + (1 - smin)/(1 + 5 delta/L), L = u*^3/(kappa B cos theta)."""
    GP, KAP, SCD = 9.81 * 7.8e-4 * 34.0, 0.4, 0.05
    cth = np.sqrt(np.maximum(1 - sth ** 2, 0)); m = m0
    us = np.sqrt(p["mo_ua"] ** 2 + SCD ** 2 * p["mo_gD"] * sth)
    for _ in range(p["mo_iter"]):
        L = us ** 3 / (KAP * GP * np.maximum(m, 1e-14) * np.maximum(cth, 1e-6))
        m = 0.5 * m + 0.5 * m0 * (p["mo_smin"] + (1 - p["mo_smin"]) / (1 + 5 * p["mo_delta"] / L))
    return m

def melt_dnsmax(dT, sin_th, ustar, p):
    """Zhao et al. (DNS, revised) convective-shear law: m = A Gamma_T dT max(u_c, u*), with the convective
    transfer velocity u_c = uc90 (theta/90 deg) (dT/dT_ref)^(1/3), i.e. convective melt ~ theta dT^(4/3).
    uc90 = 4.0 mm/s at dT_ref = 7.3 C matches the quiescent vertical-wall DNS melt (3.9 um/s) with Gamma_T = 0.01."""
    th = np.arcsin(np.clip(sin_th, 0, 1)) / (np.pi / 2)
    if p.get("dns_lit", False):   # literal form of the DNS paper: m ~ theta max(c theta drho^(1/3), u*), wall-normalized
        uc = p["dns_uc90"] * th * (np.maximum(dT, 0) / p["dns_dTref"]) ** (1 / 3)
        return (p["melt_mult"] * p["dns_scale"] * p["rho_w"] * p["cw"] / (p["rho_i"] * p["Lf"]) * p["dns_Gamma"]
                * dT * th * np.maximum(uc, ustar * p["dns_umult"]))
    uc = p["dns_uc90"] * th * (np.maximum(dT, 0) / p["dns_dTref"]) ** (1 / 3)
    return (p["melt_mult"] * p["dns_scale"] * p["rho_w"] * p["cw"] / (p["rho_i"] * p["Lf"]) * p["dns_Gamma"]
            * dT * np.maximum(uc, ustar * p["dns_umult"]))

REGIMES = {"dnsmax": melt_dnsmax, "shear": melt_shear, "convection": melt_conv, "shear+conv": melt_combined, "plume": melt_plume}

def smooth_mean_melt(p, regime):
    x, y, zbar, ustar = setup(p)
    sth = np.sin(np.arctan(np.abs(np.gradient(zbar, p["dx"]))))
    m = REGIMES[regime](thermal_driving(zbar, p), sth, ustar, p)
    if p.get("mo"): m = mo_melt(m, sth, p)
    return float(m.mean())

# ---------------------------------------------------------------- geometry operators
def slopes(zb, dx, smooth):
    zs = gaussian_filter(zb, smooth / dx, mode=("wrap", "nearest"), truncate=3.0) if smooth > 0 else zb
    gx = np.gradient(zs, dx, axis=1)
    gy = (np.roll(zs, -1, 0) - np.roll(zs, 1, 0)) / (2 * dx)
    g2 = gx * gx + gy * gy
    return np.sqrt(g2 / (1 + g2)), g2

def face_heights(zb, dx, smooth, theta):
    """Vertical extent of each contiguous steep segment along x (a riser), assigned to its cells; 0 elsewhere."""
    from scipy import ndimage
    zs = gaussian_filter(zb, smooth / dx, mode=("wrap", "nearest"), truncate=3.0) if smooth > 0 else zb
    gx = np.gradient(zs, dx, axis=1)
    mask = np.abs(gx) > np.tan(np.deg2rad(theta))
    lab, n = ndimage.label(mask, structure=[[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    if n == 0: return np.zeros_like(zb)
    idx = np.arange(1, n + 1)
    h = np.asarray(ndimage.maximum(zs, lab, idx)) - np.asarray(ndimage.minimum(zs, lab, idx))
    return np.where(lab > 0, np.concatenate([[0.0], h])[lab], 0.0)

def upwind_grad2(zb, dx):
    zp = np.pad(zb, ((0, 0), (1, 1)), mode="edge")
    Dmx = (zp[:, 1:-1] - zp[:, :-2]) / dx; Dpx = (zp[:, 2:] - zp[:, 1:-1]) / dx
    Dmy = (zb - np.roll(zb, 1, 0)) / dx;   Dpy = (np.roll(zb, -1, 0) - zb) / dx
    return (np.maximum(np.minimum(Dmx, 0) ** 2, np.maximum(Dpx, 0) ** 2)
            + np.maximum(np.minimum(Dmy, 0) ** 2, np.maximum(Dpy, 0) ** 2))

def pond_depth(zb):
    """Depth of buoyant meltwater trapped in local highs of the base (inverted basins),
    up to their spill level. Outlets at x boundaries; y periodic via wrap padding."""
    from skimage.morphology import reconstruction
    ny = zb.shape[0]; pad = max(ny // 4, 2)
    T = -np.pad(zb, ((pad, pad), (0, 0)), mode="wrap")
    seed = np.full_like(T, T.max()); seed[:, 0] = T[:, 0]; seed[:, -1] = T[:, -1]
    seed[0, :] = T[0, :]; seed[-1, :] = T[-1, :]
    return (reconstruction(seed, T, method="erosion") - T)[pad:pad + ny]

class Creep:
    """Spectral viscous relaxation of basal topography with a flat (bridging) surface.
    Half-space (Haskell) rate for kH >> 1, hydrostatic (no closure) for kH << 1."""
    def __init__(self, ny, nx, dx, H, eta, rho_w):
        ky = 2 * np.pi * np.fft.fftfreq(ny, dx)
        kx = 2 * np.pi * np.fft.rfftfreq(2 * nx, dx)       # even extension in x
        k = np.hypot(*np.meshgrid(kx, ky))
        self.rate = rho_w * G * k * H ** 2 / (2 * eta * (1 + (k * H) ** 2))
        self.nx = nx
    def apply(self, z, dt_s):
        ze = np.concatenate([z, z[:, ::-1]], axis=1)
        zh = np.fft.rfft2(ze) * np.exp(-self.rate * dt_s)
        return np.fft.irfft2(zh, s=ze.shape)[:, :self.nx]

# ---------------------------------------------------------------- GL seeding
class Seeder:
    def __init__(self, y, p):
        rng = np.random.default_rng(p["seed"])
        self.y, self.p, self.Ly = y, p, p["Ly"]
        self.hw = p["crev_H"] / np.tan(np.deg2rad(p["crev_wall"])) if p["crev_H"] > 0 else 0.0
        n = int(2 * (p["Lx"] + p["years"] * p["U_ice"]) / p["crev_spacing"]) + 8
        self.sk = -p["Lx"] - p["crev_spacing"] + p["crev_spacing"] * (np.arange(n) + rng.uniform(-0.2, 0.2, n))
        self.yk = rng.uniform(0, self.Ly, n); self.lk = rng.uniform(*p["crev_len"], n)
        nnx = int(round(p["Lx"] / p["dx"])); ny = len(y)
        kx = np.fft.fftfreq(nnx, p["dx"]); ky = np.fft.fftfreq(ny, p["dx"])
        K = np.hypot(*np.meshgrid(kx, ky))
        band = (K >= 1 / p["noise_lmax"]) & (K <= 1 / p["noise_lmin"])
        f = np.real(np.fft.ifft2(band * np.exp(2j * np.pi * rng.random(K.shape))))
        self.noise = p["noise_amp"] * f / max(f.std(), 1e-12)
        Hmax = max(p.get("crev_Hmax", 0.0), p["crev_H"])
        self.hk = rng.uniform(p["crev_H"], Hmax, n) if Hmax > p["crev_H"] else np.full(n, p["crev_H"])
        self.hwk = self.hk / np.tan(np.deg2rad(p["crev_wall"])) if p["crev_H"] > 0 else np.zeros(n)
        self.hw = float(self.hwk.max()) if p["crev_H"] > 0 else 0.0

    def column(self, s):
        p, col = self.p, np.zeros_like(self.y)
        if self.hw > 0:
            for k in np.nonzero(np.abs(self.sk - s) < self.hwk)[0]:
                dy = (self.y - self.yk[k] + self.Ly / 2) % self.Ly - self.Ly / 2
                taper = np.clip((self.lk[k] / 2 - np.abs(dy)) / 100.0, 0, 1)
                col = np.maximum(col, self.hk[k] * (1 - abs(s - self.sk[k]) / self.hwk[k]) * taper)
        if p["chan_amp"] > 0:
            ph = 2 * np.pi * (self.y - s * np.tan(np.deg2rad(p["chan_angle"]))) / p["chan_lambda"]
            col = col + p["chan_amp"] * np.cos(ph)
        return col + self.noise[:, int(np.floor(s / p["dx"])) % self.noise.shape[1]]

    def prefill(self, x):
        return np.stack([self.column(-xi) for xi in x], axis=1)

def highpass(z, dx, L=2000.0):
    return z - gaussian_filter1d(z, L / dx, axis=1, mode="nearest")

# ---------------------------------------------------------------- run
def run(p, regime):
    p = fill_times(p)
    melt_fn = REGIMES[regime]
    x, y, zbar, ustar = setup(p)
    dx, nx, ny = p["dx"], len(x), len(y)
    seed = Seeder(y, p)
    zp = seed.prefill(x)
    R0 = float(np.sqrt((highpass(zp, dx) ** 2).mean()))
    Zbar = np.broadcast_to(zbar, (ny, nx)); Ust = np.broadcast_to(ustar, (ny, nx))
    sb, g2b = slopes(np.array(Zbar), dx, 0)
    H1 = thickness(zbar, p); H = H1[None, :]

    def strat_target(sth):
        if p["strat_theta_c"] <= 0: return 0.0
        return np.exp(-(sth / np.sin(np.deg2rad(p["strat_theta_c"]))) ** p["strat_n"])
    accum = p["pool_mode"] == "accumulate"
    pool = p["pool_h"] > 0 or p["strat_theta_c"] > 0 or accum
    F = np.zeros((ny, nx)); in_pond = np.zeros((ny, nx), bool); m = np.zeros((ny, nx))
    phi = np.zeros((ny, nx)) + strat_target(sb)
    target = phi.copy()
    w_ref = melt_fn(thermal_driving(Zbar, p) * (1 - strat_target(sb)), sb, Ust, p)
    if p["mo"]: w_ref = mo_melt(w_ref, sb, p)
    w_ref = w_ref * np.sqrt(1 + g2b)

    tzf = p["tz_fb"] > 0
    tz_z0 = p["z_gl"] - 100.0
    Tanom = np.zeros(int((float(zbar.max()) + 200 - tz_z0) / p["tz_dz"]) + 1); ib = None
    # meltwater cannot cool a layer below its freezing point: cap the anomaly at -TF(z)
    TFbin = thermal_driving(tz_z0 + p["tz_dz"] * (np.arange(len(Tanom)) + 0.5), p)
    creep = Creep(ny, nx, dx, float(H1.mean()), p["creep_eta"], p["rho_w"]) if p["creep_eta"] > 0 else None
    dt = dx / p["U_ice"] * SPY
    nsteps = int(round(p["years"] * p["U_ice"] / dx))
    steps_per_yr = p["U_ice"] / dx
    every = max(1, int(round(p["diag_every"] * steps_per_yr)))
    pool_every = max(1, int(round(p["pool_every_yr"] * steps_per_yr)))
    creep_every = max(1, int(round(p["creep_every_yr"] * steps_per_yr)))
    snap_every = int(round(p["snap_years"] * steps_per_yr)) if p["snap_years"] else 0
    navg0 = nsteps - int(round(p["avg_years"] * steps_per_yr))
    xm, xt = x > min(5e3, 0.25 * p["Lx"]), (x > 0.5 * p["Lx"]) & (x < 0.95 * p["Lx"])
    prof = dict(rms=[], melt=[], steep=[])
    sc = dict(inc999=[], thin999=[], thinmax=[], flat=[], steep_d=[], pond=[], rug=[])
    hist_bins = np.linspace(0, 90, 91); hist = np.zeros(90)
    snaps = []
    for n in range(nsteps):
        zp = np.roll(zp, 1, axis=1); zp[:, 0] = seed.column(n * dx)
        if accum:
            F = np.roll(F, 1, axis=1); F[:, 0] = 0.0
            in_pond = np.roll(in_pond, 1, axis=1); in_pond[:, 0] = False
            if n % pool_every == 0:
                in_pond = pond_depth(Zbar + zp) > 0.5
            F = np.where(in_pond, F + p["pool_prod"] * m * p["rho_i"] / p["rho_w"] * dt, F * np.exp(-dt / SPY / p["pool_flush_tau"]))
            phi = np.maximum(1 - np.exp(-F / p["pool_hc"]), strat_target(slopes(Zbar + zp, dx, p["slope_smooth"])[0])
                             if p["strat_theta_c"] > 0 else 0.0)
        elif pool:
            phi = np.roll(phi, 1, axis=1); phi[:, 0] = phi[:, 1]
            if n % pool_every == 0:
                zb_ = Zbar + zp
                t1 = (1 - np.exp(-pond_depth(zb_) / p["pool_h"])) if p["pool_h"] > 0 else 0.0
                target = np.maximum(t1, strat_target(slopes(zb_, dx, p["slope_smooth"])[0]))
            phi += (target - phi) * (1 - np.exp(-dt / SPY / p["pool_tau"]))
        if p["face_beta"] != 0:
            hf = face_heights(Zbar + zp, dx, p["slope_smooth"], p["face_theta"])
            ffac = np.where(hf > 0, np.clip((np.maximum(hf, 1e-3) / p["face_h0"]) ** p["face_beta"],
                                            1 / p["face_clip"], p["face_clip"]), 1.0)
        tt = 0.0
        while tt < dt:
            zb = Zbar + zp
            sth, _ = slopes(zb, dx, p["slope_smooth"])
            if tzf:
                ib = np.clip(((zb - tz_z0) / p["tz_dz"]).astype(int), 0, len(Tanom) - 1)
                dT = np.maximum(thermal_driving(zb, p) + Tanom[ib], 0.0)
            else:
                dT = thermal_driving(zb, p)
            if pool: dT = dT * (1 - phi)
            m = melt_fn(dT, sth, Ust, p)
            if p["mo"]: m = mo_melt(m, sth, p)
            if p["face_beta"] != 0: m = m * ffac
            if p["ht_layer"] > 0:
                ph = np.mod((zb - p["z_gl"]) / p["ht_layer"], 1.0)
                m = m * np.where(sth > np.sin(np.deg2rad(p["ht_theta"])), 1 + p["ht_amp"] * (2 * ph - 1), 1.0)
            h = min(dt - tt, p["cfl"] * dx / max(float(m.max()), 1e-12))
            mv = m * np.sqrt(1 + upwind_grad2(zb, dx))
            if p["adapt_ref"] > 0 and tt == 0.0:
                # reference melt follows the lateral- and along-flow-mean melt (ice dynamics absorb mean thinning)
                lm = gaussian_filter1d(mv.mean(0), p["adapt_ref"] / dx, mode="nearest")
                wr = lm[None, :] + (w_ref - gaussian_filter1d(np.atleast_2d(w_ref).mean(0), p["adapt_ref"] / dx, mode="nearest")[None, :])
            elif p["adapt_ref"] <= 0:
                wr = w_ref
            zp = zp + (mv - wr) * h
            tt += h
        if tzf:
            Q = np.bincount(ib.ravel(), weights=(m * dx * dx).ravel(), minlength=len(Tanom))
            if p["tz_rise"] > 0:
                k = int(round(p["tz_rise"] / p["tz_dz"])); Q = np.concatenate([np.zeros(k), Q[:-k]])
            cool = p["tz_fb"] * p["rho_i"] * p["Lf"] / (p["rho_w"] * p["cw"]) * Q / (p["Ly"] * p["Lx"] * p["tz_dz"])
            Tanom = np.maximum(Tanom + dt * (-cool - Tanom / (p["tz_tau"] * SPY)), -TFbin)
        if creep is not None and n % creep_every == 0:
            zp = creep.apply(zp, creep_every * dt)
        if snap_every and (n + 1) % snap_every == 0:
            snaps.append(((n + 1) / steps_per_yr, zp.astype(np.float32), (m * SPY).astype(np.float32)))
        if n >= navg0 and n % every == 0:
            zb = Zbar + zp
            hp = highpass(zp, dx)
            _, g2 = slopes(zb, dx, 0)
            prof["rms"].append(np.sqrt((hp ** 2).mean(0)))
            prof["melt"].append(m.mean(0) * SPY)
            prof["steep"].append((g2 > np.tan(np.deg2rad(30)) ** 2).mean(0))
            sc["inc999"].append(np.percentile(hp[:, xm], 99.9))
            sc["thin999"].append(np.percentile((zp / H)[:, xm], 99.9))
            sc["thinmax"].append(float((zp / H)[:, xm].max()))
            sc["flat"].append(float((g2[:, xt] < np.tan(np.deg2rad(1)) ** 2).mean()))
            sc["steep_d"].append(float((g2[:, xt] > np.tan(np.deg2rad(30)) ** 2).mean()))
            sc["pond"].append(float((phi[:, xt] > 0.5).mean()) if pool else 0.0)
            sc["rug"].append(float(np.sqrt(1 + g2[:, xt]).mean() / np.sqrt(1 + g2b[:, xt]).mean()))
            ang = np.rad2deg(np.arctan(np.sqrt(g2[:, xt])))
            hist += np.histogram(ang, bins=hist_bins)[0]
    out = {k: np.mean(v, 0) for k, v in prof.items()}
    out.update({k: float(np.mean(v)) for k, v in sc.items()})
    b = (x > 0.75 * p["Lx"]) & (x < 0.95 * p["Lx"])
    out.update(x=x, y=y, zbar=zbar, ustar=ustar, R0=R0, Rf=float(out["rms"][b].mean()),
               slope_hist=hist / max(hist.sum(), 1), slope_bins=hist_bins,
               mbar=float(out["melt"].mean()), centerline=zp[ny // 2].astype(np.float32))
    out["G"] = out["Rf"] / R0
    if tzf:
        out["tz_anom"] = Tanom.astype(np.float32); out["tz_z"] = (tz_z0 + p["tz_dz"] * (np.arange(len(Tanom)) + 0.5)).astype(np.float32)
    Href = max(p["crev_H"], p["chan_amp"], p["noise_amp"])
    out["Pi"] = out["mbar"] * p["Lx"] / (p["U_ice"] * Href)
    if p["chan_amp"] > 0:
        k = int(round(p["Ly"] / p["chan_lambda"]))
        out["chan"] = (2 * np.abs(np.fft.fft(zp, axis=0)[k]) / ny / p["chan_amp"]).astype(np.float32)
    stride = max(1, int(round(40.0 / dx)))
    out["field40"] = zp[::stride, ::stride].astype(np.float32)
    if p["keep_field"]:
        out["field"] = zp.astype(np.float32)
        out["phi"] = phi.astype(np.float32) if pool else np.zeros(1, np.float32)
        out["meltfield"] = (m * SPY).astype(np.float32)
    if snaps:
        out["snap_t"] = np.array([s[0] for s in snaps])
        ss = max(1, int(round(p["snap_dx"] / dx)))
        out["snaps"] = np.stack([s[1][::ss, ::ss] for s in snaps])
        out["snap_melt"] = np.stack([s[2][::ss, ::ss] for s in snaps])
    return out
