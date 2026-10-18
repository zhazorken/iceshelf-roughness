"""Cross-channel flank (y-z) evolution: ice flows along the steps, so no advection across them.
Base z_b(y) starts as a ramp of slope alpha from z0; melt normal to the interface:
m = Mw * f(theta) * (1 - phi) * (TF(z)/TF0)^(4/3) * riser factor,
phi relaxes to exp(-(sin th / sin thc)^n) (meltwater held under near-flat ice).
TF(z) = TF0 + G*(z0 - z)*... : thermal driving decreasing upward at dTFdz (C per m)."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, label
import scipy.ndimage as nd

def fac(sin_th, law, a=0.76):
    th = np.arcsin(np.clip(sin_th, 0, 1)) / (np.pi / 2)
    return {"dns": a + (1 - a) * th, "sin23": np.maximum(sin_th, 1e-4) ** (2 / 3),
            "theta": np.maximum(th, 1e-4), "flat": np.ones_like(th)}[law]

def run(L=3000.0, dx=2.0, alpha=8.0, z0=-900.0, years=200.0, Mw=15.0, law="dns", thc=2.0, n=8, tau=0.1,
        ds=10.0, TF0=3.8, dTFdz=0.004, beta=0.0, h0=10.0, face_th=20.0, noise=1.0, noise_l=(20, 200),
        rng=0, nout=8, cfl=0.25, Tfb=0.0, rise=0.0, Tdz=2.0, Ttau=2.0):
    y = np.arange(0, L, dx); s0 = np.tan(np.deg2rad(alpha))
    r = np.random.default_rng(rng)
    k = np.fft.rfftfreq(len(y), dx); band = (k > 1 / noise_l[1]) & (k < 1 / noise_l[0])
    nz = np.fft.irfft(np.where(band, r.normal(size=len(k)) + 1j * r.normal(size=len(k)), 0), n=len(y))
    z = z0 + s0 * y + (noise * nz / nz.std() if noise > 0 else 0)
    sup = lambda s: np.exp(-(s / np.sin(np.deg2rad(thc))) ** n)
    phi = sup(np.full(len(y), np.sin(np.arctan(s0))))
    TF = lambda zz: np.maximum(TF0 - dTFdz * (zz - z0), 0)
    zgrid = np.arange(z0 - 50, z0 + s0 * L + 50, Tdz); Ta = np.zeros(len(zgrid))
    t = 0.0; outs = [(0.0, z.copy())]; tout = np.linspace(0, years, nout + 1)[1:]; io = 0
    dtmax = 0.02
    while t < years:
        zs = gaussian_filter1d(z, ds / dx, mode="nearest")
        g = np.gradient(zs, dx); sth = np.abs(g) / np.sqrt(1 + g * g)
        ff = 1.0
        if beta != 0:
            mask = np.abs(g) > np.tan(np.deg2rad(face_th)); lab, nl = label(mask)
            if nl:
                idx = np.arange(1, nl + 1)
                hh = np.asarray(nd.maximum(zs, lab, idx)) - np.asarray(nd.minimum(zs, lab, idx))
                hf = np.where(lab > 0, np.concatenate([[0.0], hh])[lab], 0.0)
                ff = np.where(hf > 0, np.clip((np.maximum(hf, 1e-3) / h0) ** beta, 0.2, 5.0), 1.0)
        ib = np.clip(((z - zgrid[0]) / Tdz).astype(int), 0, len(zgrid) - 1)
        tf = np.maximum(TF(z) + Ta[ib], 0)
        m = Mw * fac(sth, law) * (1 - phi) * (tf / TF0) ** (4 / 3) * ff
        zp = np.pad(z, 1, mode="edge")
        Dm = (zp[1:-1] - zp[:-2]) / dx; Dp = (zp[2:] - zp[1:-1]) / dx
        gr2 = np.maximum(np.minimum(Dm, 0) ** 2, np.maximum(Dp, 0) ** 2)
        h = min(dtmax, cfl * dx / max(float(m.max()), 1e-9), years - t)
        z = z + m * np.sqrt(1 + gr2) * h
        phi += (sup(sth) - phi) * (1 - np.exp(-h / tau))
        if Tfb > 0:
            Q = np.bincount(ib, weights=m * dx, minlength=len(zgrid))
            kk = int(round(rise / Tdz)); Q = np.concatenate([np.zeros(kk), Q[:len(Q) - kk]]) if kk else Q
            Ta = np.maximum(Ta + h * (-Tfb * Q / (L * Tdz) - Ta / Ttau), -TF(zgrid))
        t += h
        if io < len(tout) and t >= tout[io] - 1e-9: outs.append((t, z.copy())); io += 1
    return dict(y=y, outs=outs, Ta=Ta, zgrid=zgrid)

def steps(y, z, theta=20.0, ds=4.0, dx=2.0):
    zs = gaussian_filter1d(z, ds / dx, mode="nearest"); g = np.gradient(zs, dx)
    lab, nl = label(np.abs(g) > np.tan(np.deg2rad(theta)))
    if nl == 0: return np.array([]), np.array([])
    idx = np.arange(1, nl + 1)
    hh = np.asarray(nd.maximum(zs, lab, idx)) - np.asarray(nd.minimum(zs, lab, idx))
    yc = np.asarray(nd.mean(y, lab, idx)); return yc, hh
