# How the ocean roughens and smooths ice shelves: model and figure code

Kinematic model of an advecting ice-shelf base reshaped by slope-dependent melt (Zhao et al., submitted to Nature Geoscience).

## Requirements
Python 3.10 or newer with numpy, scipy, matplotlib, pandas. BedMachine comparison also needs netCDF4.

## Main pieces
- `shelfmelt.py`: the model (geometry, melt laws incl. the DNS convective-shear law, stratification closures, Monin-Obukhov closure, pooling, aging ponds, seeding, diagnostics).
- `experiments.py`: all experiment definitions (job list); `runner.py`, `rerun*.py`: run jobs in parallel.
- `terr2d_z.py`, `terr2d_wide.py`: plan-view terrace experiments; `oned/`: along-flow (1D) terrace model and regime maps.
- `mix_scan.py`, `prod_scan.py`, `buildup.py`, `robust_runs.py`, `movie_runs.py`: pond, build-up, robustness and movie runs.
- `obs_transects.py`, `obs_compare.py`, `aniso_obs.py`: BedMachine transects, spectra and anisotropy (needs `obs/amundsen_bedmachine_v3.npz`, cut from BedMachineAntarctica-v3.nc with `data/extract_amundsen.py`).
- Figures: `main_figs.py` (Figs 1 to 3 and Extended Data), `fig5_terraces.py`, `fig_pig.py`, `fig_vuln.py`, `fig_spectra.py`, `figures.py`, `figs_extra.py`, `make_movieS1.py`.

## Reproducing
1. Run the job list: `python3 rerun.py <nproc>` (and the scripts above for the 2D terrace, pond and build-up cases). All runs take a few hours on a laptop.
2. Make figures: `python3 mkfigs.py fig2 fig3 fig4`, then the stand-alone figure scripts.

BedMachine Antarctica v3: NSIDC-0756 (Morlighem et al., 2020).
