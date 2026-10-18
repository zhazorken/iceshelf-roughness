import matplotlib.figure as mf
mf.Figure.suptitle = lambda self, *a, **k: None
import figures as F, main_figs as MF
R = F.load(["results", "mac/results"])
import re
for fn in (F.figS_resolution, F.figS_oat, F.figS_ens, F.figS_channels, F.figS_strat, F.figS_snaps, F.figS_terrace, F.figS_v3, F.figS_stair):
    try: fn(R)
    except Exception as e: print("FAILED", fn.__name__, repr(e))
F.figS_profiles(R, "creep closure", "FigS5_creep", "")
F.figS_profiles(R, "ice speed", "FigS6_icespeed", "")
