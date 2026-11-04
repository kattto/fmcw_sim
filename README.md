# fmcw_sim

Simulation and hardware processing code for FMCW radar parameter estimation. The project attempts to benchmark interpolated-FFT estimators (A&M and FIIB) against subspace methods (Root-MUSIC and ESPRIT) on the range axis, first in simulation and then on real captures from an Ancortek SDR-KIT 2400T2R4 (With plans to expand to other radar units).

## Layout

```
src/fmcw/
    constants.py, params.py                 shared constants and radar parameters
    processing.py, estimators.py            processing chain and estimators, used for both simulated and measured data
    signal.py, scene.py, link_budget.py     signal model, scene and link budget
    ancortek.py                             reader for recorded radar data (T2R4 Ancortek unit specifically)
scripts/
    computation/                    Monte Carlo runs and timing, with results saved to data/plotting_data/
    plots/                          figures generated from saved results
    tools/                          data preparation, such as converting Ancortek captures
data/                               not tracked by git
    plotting_data/                  saved experiment results
    radar_raw/                      raw .bin captures
    radar_processed/                converted .npz captures
docs/
```

Physics and computation live in `src/fmcw/` and the scripts only wire things together and plot.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Running

All scripts are run from the repo root so the relative data paths resolve.

Simulation results are generated with a compute step followed by a plotting step, so a figure can be regenerated without rerunning the Monte Carlo (Which can be very computationally heavy).

```
python scripts/computation/calculate_rmse_crb.py
python scripts/plots/plot_rmse_crb.py
```

Raw Ancortek captures are converted to `.npz` before processing. The converter reads the radar configuration from the file header, checks it against the sweep markers and saves the data cube in the same (N, M, L) layout that's used in the simulation.

```
python scripts/tools/convert_ancortek.py "data/radar_raw/corner reflector 1GHz/take_1.bin"
```

## Notes

Radar captures are not included in the repo because of their size.