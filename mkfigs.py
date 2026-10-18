import sys, figures as F, main_figs as MF
R = F.load(["results", "mac/results"]); print(len(R))
for name in sys.argv[1:]: getattr(MF, name)(R)
