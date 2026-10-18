import pickle
from multiprocessing import Pool
from level import one
if __name__ == "__main__":
    grid = [dict(law=a, thc=tc, pool_h=ph, adapt=2000.0, M=1.0, rng=r) for a in (0.02, 0.1, 0.3, 0.76)
            for tc in (0.0, 0.5, 2.0) for ph in (0.5, 2.0) for r in (0, 1)]
    res = []
    with Pool(2) as P:
        for r in P.imap_unordered(one, grid): res.append(r)
    pickle.dump(res, open("level2.pkl", "wb"))
