# Timing benchmark for all estimators

import os

# Single BLAS thread so every estimator runs under the same conditions
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import numpy as np
from fmcw.params import RadarParams
from fmcw.scene import Target
from fmcw.signal import get_IF_r_vec
from fmcw.processing import get_range_axis
from fmcw.estimators import *

N_RUNS = 100            # runs per estimator
SNR = 30                # dB
SEED = 0
L = 2                   # targets for the multi-target estimators
SEP = 2.0               # target separation (bins)

ESTIMATORS = {
    "FFT peak":         lambda x: fft_peak(x),
    "A&M (2 it)":       lambda x: aboutanios_mulgrew(x, n_iter=2),
    "A&M (10 it)":      lambda x: aboutanios_mulgrew(x, n_iter=10),
    "A&M (50 it)":      lambda x: aboutanios_mulgrew(x, n_iter=50),
    "A&M (200 it)":     lambda x: aboutanios_mulgrew(x, n_iter=200),
    "FIIB (2 it)":      lambda x: fiib(x, n_iter=2, L=L),
    "FIIB (10 it)":     lambda x: fiib(x, n_iter=10, L=L),
    "FIIB (50 it)":     lambda x: fiib(x, n_iter=50, L=L),
    "FIIB (200 it)":    lambda x: fiib(x, n_iter=200, L=L),
    "Root-MUSIC":       lambda x: root_music(x, L=L),
    "ESPRIT":           lambda x: esprit(x, L=L),
}

params = RadarParams()
bin_width = get_range_axis(params)[1]
rng = np.random.default_rng(SEED)

# Pre-generate all signals so signal generation isn't timed and every estimator sees exactly the same data.
signals = []
for _ in range(N_RUNS):
    r1 = 15 + rng.uniform(-0.5, 0.5)
    targets = [
        Target(r=r1 * bin_width, v=0.0, a=0.0, rcs=1.0),
        Target(r=(r1 + SEP) * bin_width + rng.uniform(0, params.lam / 2), v=0.0, a=0.0, rcs=1.0),
    ]
    signals.append(get_IF_r_vec(params, targets, noise=rng, snr=SNR))

print(f"N = {params.N}, L = {L}, {N_RUNS} runs each, single-threaded\n")
print(f"{'Estimator':<16}{'mean (ms)':>12}{'std (ms)':>12}{'median (ms)':>14}{'min (ms)':>12}")
print("-" * 66)

for name, estimator in ESTIMATORS.items():
    estimator(signals[0])                     # warm-up, not timed

    times = np.empty(N_RUNS)
    for i, x in enumerate(signals):
        t0 = time.perf_counter()
        estimator(x)
        times[i] = (time.perf_counter() - t0) * 1e3

    print(f"{name:<16}{times.mean():>12.3f}{times.std():>12.3f}{np.median(times):>14.3f}{times.min():>12.3f}")