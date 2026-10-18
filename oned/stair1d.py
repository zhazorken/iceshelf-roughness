"""1D (x-z) version of the basal evolution model for step-growth tests.
Ice frame shift by one cell per step (dt = dx/U); normal melt with Godunov upwind; slope seen by melt
smoothed over ds; suppression phi relaxes to exp(-(sin th/sin thc)^n) over tau; w_ref = smooth-ramp melt.
Optional: riser-height factor (h/h0)^beta, T(z) feedback with rise height r."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, label
SPY = 3.15576e7

def fac(sin_th, law, a=0.76):
    if isinstance(law, float):
        th = np.arcsin(np.clip(sin_th, 0, 1)) / (np.pi / 2); return law + (1 - law) * th
    th = np.arcsin(np.clip(sin_th, 0, 1)) / (np.pi / 2)
    if law == "dns": return a + (1 - a) * th
    if law == "sin23": return np.maximum(sin_th, 1e-4) ** (2 / 3)
    if law == "theta": return np.maximum(th, 1e-4)
    if law == "flat": return np.ones_like(th)

def run(Lx=8e3, dx=5.0, U=1000.0, alpha=4.0, M=1.0, law="dns", thc=2.0, n=8, tau=0.1, ds=20.0,
        beta=0.0, h0=10.0, face_th=20.0, years=None, seed_fn=None, noise=0.0, noise_l=(50, 400), rng=0,
        Tfb=0.0, rise=0.0, Tdz=5.0, Ttau=2.0, dTdz_frac=0.0, out_every=None, cfl=0.25, adapt=0.0, pool_h=0.0,
        mo=False, mo_ua=1e-4, mo_gD=4e-3, mo_delta=0.2, mo_smin=0.05, mo_iter=30):
    x = np.arange(0, Lx, dx); nx = len(x); s0 = np.tan(np.deg2rad(alpha)); zbar = -900 + s0 * x
    years = years or 2.0 * Lx / U
    sup = lambda sth: np.exp(-(sth / np.sin(np.deg2rad(thc))) ** n) if thc > 0 else 0 * sth
    sth0 = np.sin(np.arctan(s0))
    UC = 2.97e-3   # revised DNS law: convective transfer velocity at a vertical wall for TF ~ 3 C (4.0 mm/s at 7.3 C)
    F = (lambda s: np.maximum(np.arcsin(np.clip(s, 0, 1)) / (np.pi / 2), mo_ua / UC)) if law == "dnsmax" else (lambda s: fac(s, law))
    # ramp melt (vertical) = M U s0 sets amplitude; depth-dependence of TF ignored except via dTdz_frac
    A = M * U * s0 / (F(np.array([sth0]))[0] * (1 - sup(np.array([sth0]))[0]) * np.sqrt(1 + s0 ** 2))
    TF = lambda z: np.maximum(1 + dTdz_frac * (z + 900) / 500.0, 0.0)   # relative thermal driving (1 at GL depth)
    GP, KAP, SCD = 9.81 * 7.8e-4 * 34.0, 0.4, 0.05      # meltwater g' factor (m/s2), von Karman, sqrt(drag)
    def mo_melt(m0, sth):
        """Monin-Obukhov suppression of melt (m/s in, m/s out). B = g' m; u* from ambient + along-slope plume;
        stabilizing flux B cos(theta); S = smin + (1 - smin)/(1 + 5 delta/L)."""
        cth = np.sqrt(np.maximum(1 - sth ** 2, 0)); m = m0
        for _ in range(mo_iter):
            B = GP * np.maximum(m, 1e-14)
            up = np.sqrt(mo_gD * sth)          # gravity-current speed of the meltwater layer along the slope
            us = np.sqrt(mo_ua ** 2 + (SCD * up) ** 2)
            L = us ** 3 / (KAP * B * np.maximum(cth, 1e-6))
            S = mo_smin + (1 - mo_smin) / (1 + 5 * mo_delta / L)
            m = 0.5 * m + 0.5 * m0 * S
        return m
    if mo:
        target = M * U * s0 / SPY / np.sqrt(1 + s0 ** 2)
        Alo, Ahi = 1e-3, 1e5
        for _ in range(60):
            Am = np.sqrt(Alo * Ahi)
            val = float(mo_melt(np.array([Am * F(np.array([sth0]))[0] / SPY]), np.array([sth0]))[0])
            Alo, Ahi = (Am, Ahi) if val < target else (Alo, Am)
        A = np.sqrt(Alo * Ahi)
    w_ref = A * F(np.full(nx, sth0)) * (1 - sup(np.full(nx, sth0))) * TF(zbar) ** (4 / 3) * np.sqrt(1 + s0 ** 2)
    if mo:
        w_ref = mo_melt(A * F(np.full(nx, sth0)) * TF(zbar) ** (4 / 3) / SPY, np.full(nx, sth0)) * SPY * np.sqrt(1 + s0 ** 2)
    z = np.zeros(nx); phi = sup(np.full(nx, sth0)); dt = dx / U * SPY
    r = np.random.default_rng(rng)
    k = np.fft.rfftfreq(400000 // int(dx), dx); band = (k > 1 / noise_l[1]) & (k < 1 / noise_l[0])
    nz = np.fft.irfft(np.where(band, r.normal(size=len(k)) + 1j * r.normal(size=len(k)), 0))
    nz = noise * nz / max(nz.std(), 1e-12) if noise > 0 else 0 * nz
    zgrid = np.arange(-1000, 0, Tdz); Ta = np.zeros(len(zgrid))
    nsteps = int(round(years * U / dx)); snaps = []
    for it in range(nsteps):
        z = np.roll(z, 1); z[0] = nz[it % len(nz)] + (seed_fn(it * dx) if seed_fn else 0.0)
        phi = np.roll(phi, 1); phi[0] = phi[1]
        zb = zbar + z
        zs = zbar + (gaussian_filter1d(z, ds / dx, mode="nearest") if ds > 0 else z)
        g = np.gradient(zs, dx); sth = np.abs(g) / np.sqrt(1 + g * g)
        tgt = sup(sth)
        if pool_h > 0:
            Tq = -zb   # depth of the ceiling; inverted basins (local highs of zb) are minima of Tq
            W = np.minimum(np.maximum.accumulate(Tq), np.maximum.accumulate(Tq[::-1])[::-1])
            tgt = np.maximum(tgt, 1 - np.exp(-(W - Tq) / pool_h))
        phi += (tgt - phi) * (1 - np.exp(-dt / SPY / tau))
        ff = 1.0
        if beta != 0:
            mask = g > np.tan(np.deg2rad(face_th)); lab, nl = label(mask)
            if nl:
                idx = np.arange(1, nl + 1)
                import scipy.ndimage as nd
                hh = np.asarray(nd.maximum(zs, lab, idx)) - np.asarray(nd.minimum(zs, lab, idx))
                hf = np.where(lab > 0, np.concatenate([[0.0], hh])[lab], 0.0)
                ff = np.where(hf > 0, np.clip((np.maximum(hf, 1e-3) / h0) ** beta, 0.2, 5.0), 1.0)
        tt = 0.0
        while tt < dt:
            zb = zbar + z
            ib = np.clip(((zb - zgrid[0]) / Tdz).astype(int), 0, len(zgrid) - 1)
            tf = np.maximum(TF(zb) + Ta[ib], 0)
            m = A * F(sth) * (1 - phi) * tf ** (4 / 3) * ff / SPY
            if mo: m = mo_melt(m, sth)
            zp = np.pad(zb, 1, mode="edge")
            Dm = (zp[1:-1] - zp[:-2]) / dx; Dp = (zp[2:] - zp[1:-1]) / dx
            gr2 = np.maximum(np.minimum(Dm, 0) ** 2, np.maximum(Dp, 0) ** 2)
            h = min(dt - tt, cfl * dx / max(float(m.max()), 1e-15))
            mv = m * np.sqrt(1 + gr2)
            if adapt > 0 and tt == 0.0:
                wr = gaussian_filter1d(mv, adapt / dx, mode="nearest") + (w_ref / SPY - gaussian_filter1d(w_ref / SPY, adapt / dx, mode="nearest"))
            elif adapt <= 0: wr = w_ref / SPY
            z = z + (mv - wr) * h
            tt += h
        if Tfb > 0:
            Q = np.bincount(ib, weights=m * dx, minlength=len(zgrid)) * SPY   # m^2/yr per bin
            kk = int(round(rise / Tdz)); Q = np.concatenate([np.zeros(kk), Q[:len(Q) - kk]]) if kk else Q
            Ta = np.maximum(Ta + dt / SPY * (-Tfb * 21.0 * Q / (Lx * Tdz) - Ta / Ttau), -TF(zgrid))
        if out_every and it % out_every == 0: snaps.append((it * dx / U, zbar + z))
    return dict(x=x, zb=zbar + z, zbar=zbar, snaps=snaps, Ta=Ta, zgrid=zgrid, A=A)
