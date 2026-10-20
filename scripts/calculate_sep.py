# Calculates RMSE data points at different target separations

import os

# Must be set before numpy is imported: one BLAS thread per worker process
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from fmcw.params import RadarParams
from fmcw.scene import Target
from fmcw.signal import get_IF_r_vec
from fmcw.processing import get_range_axis
from fmcw.estimators import *

N_TRIALS = 1000                                         # Monte Carlo trials
SEP_BINS = np.round(np.arange(0.75, 2.9, 0.05), 2)      # Separation range (bins), as in the FIIB paper
SNR = 30
SEED = 0
N_WORKERS = 7                                           # number of physical cores to use

# Multi-tone estimators to compare. Each takes the raw IF vector
# and returns two range estimates in bins.
ESTIMATORS = {
    "FIIB": lambda x: fiib(x, n_iter=200, L=2),
    "Root-MUSIC": lambda x: root_music(x, L=2),
    "ESPRIT": lambda x: esprit(x, L=2),
}

# The pair is centred on r_centre_bin. Each trial moves the centre by a random
# amount within +/- half a bin, so the RMSE is averaged over all sub-bin positions.
r_centre_bin = 15


def get_CRB_two_tone(N, snr_db, sep_bins, n_phases=64):
    n = np.arange(N)
    noise_var = 1.0 / (10 ** (snr_db / 10))        # total complex noise variance
    f = np.array([0.0, sep_bins])
    crb = []
    for dphi in np.linspace(0, 2 * np.pi, n_phases, endpoint=False):
        phi = np.array([0.0, dphi])
        s = np.exp(1j * (2 * np.pi * np.outer(n, f) / N + phi))   # (N, 2)
        # Derivatives w.r.t. [f1, f2, a1, a2, phi1, phi2], f in bins
        J = np.column_stack([
            1j * 2 * np.pi * n / N * s[:, 0],
            1j * 2 * np.pi * n / N * s[:, 1],
            s[:, 0], s[:, 1],
            1j * s[:, 0], 1j * s[:, 1],
        ])
        fim = (2 / noise_var) * np.real(J.conj().T @ J)
        crb.append(np.linalg.inv(fim)[0, 0])
    return np.mean(crb)


def run_sep(sep, seed):
    """runs batch"""
    params = RadarParams()
    bin_width = get_range_axis(params)[1]
    wavelength = params.lam
    rng = np.random.default_rng(seed)

    # Collect the squared error of every trial, for every estimator and target.
    squared_errors_1 = {name: [] for name in ESTIMATORS}
    squared_errors_2 = {name: [] for name in ESTIMATORS}

    for trial in range(N_TRIALS):
        # New random pair position and new noise on every trial.
        r_centre = r_centre_bin + rng.uniform(-0.5, 0.5)
        r1_bin = r_centre - sep / 2

        # Random relative phase: shifting target 2 by up to half a wavelength
        # changes its round-trip phase by up to 2*pi, but its range by << 1 bin.
        # The true bins below use the actual ranges, so this shift is accounted for.
        r2 = (r_centre + sep / 2) * bin_width + rng.uniform(0, wavelength / 2)
        r2_bin = r2 / bin_width

        targets = [
            Target(r=(r1_bin * bin_width), v=0.0, a=0.0, rcs=1.0),
            Target(r=r2, v=0.0, a=0.0, rcs=1.0),
        ]
        x = get_IF_r_vec(params, targets, noise=rng, snr=SNR)

        for name, estimator in ESTIMATORS.items():
            r_estimate = np.sort(estimator(x))      # match estimates to targets by order
            squared_errors_1[name].append((r_estimate[0] - r1_bin) ** 2)
            squared_errors_2[name].append((r_estimate[1] - r2_bin) ** 2)

    # RMSE = square root of the mean squared error over all trials.
    rmse_1 = {name: np.sqrt(np.mean(squared_errors_1[name])) for name in ESTIMATORS}
    rmse_2 = {name: np.sqrt(np.mean(squared_errors_2[name])) for name in ESTIMATORS}
    print(f"Separation {sep} bins done")
    return rmse_1, rmse_2


def main():
    params = RadarParams()
    N = params.N
    crb = [np.sqrt(get_CRB_two_tone(N, SNR, sep)) for sep in SEP_BINS]   # root-CRB (bins)

    # One independent, reproducible random stream per separation.
    seeds = np.random.SeedSequence(SEED).spawn(len(SEP_BINS))

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        # One task per separation; results come back in SEP_BINS order.
        results = list(pool.map(run_sep, SEP_BINS, seeds))
    print(f"Done in {time.perf_counter() - t0:.0f} s")

    # rmse_1["FIIB"] holds one RMSE value per separation for target 1 (the lower-range one).
    rmse_1 = {name: [r1[name] for r1, _ in results] for name in ESTIMATORS}
    rmse_2 = {name: [r2[name] for _, r2 in results] for name in ESTIMATORS}

    # np.savez(f"data/mc_sep_results.npz",
    #          sep_bins=SEP_BINS, crb=crb, snr=SNR,
    #          **{f"{k}_t1": np.array(v) for k, v in rmse_1.items()},
    #          **{f"{k}_t2": np.array(v) for k, v in rmse_2.items()})


if __name__ == "__main__":     # required on Windows, or every worker re-runs the script
    main()