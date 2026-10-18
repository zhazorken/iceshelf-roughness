"""Robustness runs: extra seeds, controls and resolution for the v8 figures."""
import os, sys, pickle, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np, shelfmelt as sm, experiments as ex, runner, terr2d_z as T

def jobs():
    J = []
    for j in ex.JOBS:   # tall crevasses, second seed
        if j["group"] == "crevasse height (steep walls)" and j["label"] in ("120", "250") and j["seed"] == 1:
            p = dict(j["params"], seed=2); name = j["name"].replace("__s1", "__s2")
            J.append(("ens", dict(j, params=p, seed=2, name=name)))
    for s in (2, 3): J.append(("pond", dict(seed=s, Hmax=150.0, v="convA", prod=0.03)))
    for v in ("conv", "shear"): J.append(("pond", dict(seed=1, Hmax=150.0, v=v, prod=None)))
    for s in (2, 3):
        for k, kw in (("terr", {}), ("saw", dict(ph=0.0)), ("smo", dict(ua=1e-3))):
            J.append(("terr", dict(seed=s, key=k, kw=kw)))
    J.append(("res5", {}))
    return J

def work(item):
    kind, a = item; t = time.time()
    if kind == "ens":
        return runner.work(a)
    if kind == "pond":
        base = dict(ex.BASE, Ly=4e3, dx=20.0, seed=a["seed"], keep_field=True, crev_Hmax=a["Hmax"])
        regime, q = ex.variant(base, a["v"])
        if a["prod"] is not None: q.update(pool_prod=a["prod"])
        tag = f"pond_{a['v']}_prod{a['prod']}_H{a['Hmax']:g}_s{a['seed']}"
    elif kind == "terr":
        q, regime = T.job(Ly=3e3, seed=a["seed"], **a["kw"]); tag = f"terrW_{a['key']}_s{a['seed']}"
    else:
        q, regime = T.job(Ly=1e3); q.update(dx=5.0); tag = "terrRes_dx5_Ly1000"
    f = f"results_coarse/{tag}.pkl"
    if os.path.exists(f): return tag, 0, "skip"
    out = sm.run(q, regime); out["params"] = q
    pickle.dump(out, open(f, "wb"))
    return tag, time.time() - t, "G %.2f" % out["G"]

if __name__ == "__main__":
    J = jobs(); J.sort(key=lambda it: it[0] == "res5")    # long resolution run last
    print(len(J), "jobs", flush=True)
    with Pool(2) as P:
        for r in P.imap_unordered(work, J):
            print(r, flush=True)
