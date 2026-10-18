"""Most realistic shelf case: DNS law (theta convection + shear) at physical amplitude, quiet water,
Monin-Obukhov stability and closed-basin pooling."""
import sys, time, pickle, numpy as np
import shelfmelt as sm, experiments as ex
def params(dx=20.0, Ly=1e3, us=3e-4, ph=1.0, U=1000.0, thf=0.3, Lth=5e3, seed=1, keep=True, delta=0.2, cH=40.0, cw=45.0, ss=40.0):
    p = dict(ex.BASE, dx=dx, Ly=Ly, seed=seed, keep_field=keep, ustar_gl=us, ustar_out=us, U_ice=U,
             theta_far=thf, L_theta=Lth, crev_H=cH, crev_wall=cw, slope_smooth=ss)
    p.update(dns_scale=1.0, mo=True, mo_ua=us, mo_gD=0.01, mo_smin=0.01, mo_delta=delta, pool_h=ph, pool_tau=0.1)
    return "dnsmax", p
if __name__ == "__main__":
    dx, Ly, us, ph, U, thf, dl, cH, cw, ss = (float(a) for a in sys.argv[1:11])
    regime, q = params(dx, Ly, us, ph, U, thf, delta=dl, cH=cH, cw=cw, ss=ss)
    name = f"real_dx{dx:g}_Ly{Ly:g}_us{us*1e3:g}_ph{ph:g}_U{U:g}_thf{thf:g}_d{dl:g}_c{cH:g}_{cw:g}_ss{ss:g}"
    x, y, zb, u = sm.setup(q); sth = np.sin(np.arctan(np.abs(np.gradient(zb, dx))))
    m0 = sm.mo_melt(sm.melt_dnsmax(sm.thermal_driving(zb, q), sth, u, q), sth, q) * sm.SPY
    M = m0 / (U * np.abs(np.gradient(zb, dx)))
    print(name, "smooth melt GL/10/20/40 km", np.round(m0[[0, 500, 1000, -1]], 2), "M", np.round(M[[0, 500, 1000, -1]], 2), flush=True)
    t = time.time(); out = sm.run(q, regime); out["params"] = q
    pickle.dump(out, open(f"results_coarse/{name}.pkl", "wb"))
    mf = out["meltfield"]; print(name, f"{time.time()-t:.0f}s G {out['G']:.2f} melt median {np.median(mf):.1f} p95 {np.percentile(mf,95):.1f} p99.5 {np.percentile(mf,99.5):.1f}", flush=True)
