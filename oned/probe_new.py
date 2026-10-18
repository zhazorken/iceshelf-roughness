import sys, pickle
from regime import run, classify, BASE
res=[]
for ua in [3e-4, 1e-3, 2e-3]:
    for ph in [0.0, 1.0]:
        o=run(**dict(BASE, law="dnsmax", mo_ua=ua, pool_h=ph, rng=0)); m=classify(o)
        print(ua, ph, m["regime"], round(m["rms"],1), round(m["lvl"],2), round(m["tilt"],2), round(m["back"],2), flush=True)
        res.append((ua,ph,m,o["zb"],o["zbar"],o["x"]))
pickle.dump(res, open("probe_new.pkl","wb"))
