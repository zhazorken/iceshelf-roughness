import numpy as np, netCDF4
d = netCDF4.Dataset("BedMachineAntarctica-v3.nc")
x = d["x"][:]; y = d["y"][:]
ix = np.where((x > -1900e3) & (x < -1300e3))[0]; iy = np.where((y > -900e3) & (y < -100e3))[0]
sl = (slice(iy[0], iy[-1] + 1), slice(ix[0], ix[-1] + 1))
o = dict(x=np.asarray(x[ix], "f8"), y=np.asarray(y[iy], "f8"))
for v in ("surface", "thickness", "bed", "mask", "firn", "source"): o[v] = np.asarray(d[v][sl])
np.savez_compressed("amundsen_bedmachine_v3.npz", **o)
