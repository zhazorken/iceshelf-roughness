import pickle
from multiprocessing import Pool
from level import one
if __name__ == "__main__":
    grid = [dict(law="dns", thc=0.0, mo=True, mo_ua=ua, mo_gD=gD, adapt=ad, pool_h=ph, M=1.0, rng=0)
            for ua in (1e-4, 5e-4, 1e-3) for gD in (1e-3, 4e-3, 1e-2) for ad in (0.0, 2000.0) for ph in (0.0, 2.0)]
    res = []
    with Pool(2) as P:
        for r in P.imap_unordered(one, grid): res.append(r); print(len(res), flush=True)
    pickle.dump(res, open("mo_sweep.pkl", "wb"))
