import pickle
from multiprocessing import Pool
from level import one
if __name__ == "__main__":
    grid = [dict(law="dns", thc=0.0, mo=True, mo_ua=ua, mo_gD=gD, mo_smin=sm, U=U, years=1.5 * 8000 / U, adapt=2000.0, M=1.0, rng=0)
            for U in (200.0, 1000.0) for ua in (1e-4, 5e-4) for gD in (4e-3, 1e-2, 3e-2) for sm in (0.02, 0.1)]
    res = []
    with Pool(2) as P:
        for r in P.imap_unordered(one, grid): res.append(r); print(len(res), flush=True)
    pickle.dump(res, open("mo_sweep2.pkl", "wb"))
