import pickle, time, sys, terr2d_z as T, shelfmelt as sm
Ly = float(sys.argv[1]) if len(sys.argv) > 1 else 3e3
ua = float(sys.argv[2]) if len(sys.argv) > 2 else 3e-4
ph = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
q, regime = T.job(Ly=Ly, ua=ua, ph=ph); t = time.time(); out = sm.run(q, regime); out["params"] = q
tag = "" if (ua == 3e-4 and ph == 1.0) else f"_ua{ua*1e3:g}_ph{ph:g}"
pickle.dump(out, open(f"results_coarse/terr2dZ_wide_Ly{Ly:.0f}{tag}.pkl", "wb")); print("done", time.time() - t)
