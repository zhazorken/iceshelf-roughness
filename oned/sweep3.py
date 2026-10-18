import pickle, time
from multiprocessing import Pool
from sweep2 import metrics2
from stair1d import run
def one(kw):
    o = run(**dict(dict(noise=1.0, years=12, adapt=2000.0, thc=2.0, n=8, M=1.0), **kw)); return kw, metrics2(o)
if __name__ == "__main__":
    grid = [dict(law=a, beta=b, rng=r) for a in (0.05, 0.1) for b in (-0.25, 0.0, 0.33, 0.5) for r in (0, 1)]
    grid += [dict(law=a, Tfb=g, rise=rr, rng=r) for a in (0.05, 0.1) for g in (0.1, 1.0) for rr in (0.0, 20.0) for r in (0, 1)]
    res = []
    with Pool(2) as P:
        for r in P.imap_unordered(one, grid): res.append(r)
    pickle.dump(res, open("sweep3.pkl", "wb"))
