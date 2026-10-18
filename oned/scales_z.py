import pickle
from multiprocessing import Pool
from scales import steps
from stair1d import run
from regime_z import B
UA, PH = 3e-4, 1.0
def one(kw):
    o = run(**dict(B, mo_ua=UA, pool_h=PH, **dict(kw, years=1.5 * 8000 / kw.get("U", 200.0))))
    return kw, dict(near=steps(o, 1000, 3000), far=steps(o, 4500, 7500))
if __name__ == "__main__":
    grid = [dict(alpha=a, rng=r) for a in (2.0, 4.0, 8.0) for r in (0, 1)] + [dict(U=U, rng=r) for U in (100.0, 500.0) for r in (0, 1)]
    with Pool(1) as P: res = P.map(one, grid)
    pickle.dump(res, open("scales_z.pkl", "wb"))
    for kw, m in res: print(kw, "near h/L/n", m["near"], "far h/L/n", m["far"], flush=True)
