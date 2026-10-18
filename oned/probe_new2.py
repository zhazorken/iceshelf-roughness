import pickle
from regime import run, classify, BASE
res=[]
for ua,M,mo,ph in [(1e-4,0.3,True,1.0),(1e-4,0.5,True,1.0),(3e-4,0.5,True,1.0),(1e-4,0.5,False,1.0),(1e-4,0.95,False,1.0)]:
    o=run(**dict(BASE, law="dnsmax", mo_ua=ua, pool_h=ph, rng=0, M=M, mo=mo)); m=classify(o)
    print(ua, M, mo, ph, "A", round(o["A"]), m["regime"], round(m["rms"],1), round(m["lvl"],2), round(m["tilt"],2), round(m["back"],2), flush=True)
    res.append((ua,M,mo,ph,m,o["zb"],o["zbar"],o["x"]))
pickle.dump(res, open("probe_new2.pkl","wb"))
