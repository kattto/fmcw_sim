# do later

import numpy as np
import matplotlib.pyplot as plt
from fmcw.params import RadarParams
from fmcw.scene import Target
from fmcw.signal import get_IF_r_vec
from fmcw.processing import get_range_axis, get_CRB
from fmcw.estimators import *

N_TRIALS = 1                    # Monte Carlo trials
SNR_DB = np.arange(-20, 41, 1)      # SNR Range
SEED = 0                            # RNG seed

# The estimators to compare. Takes the raw IF vector
# and returns a range estimate in bins.
ESTIMATORS = {
    "FFT peak": fft_peak,
    "A&M": aboutanios_mulgrew,
    "FIIB": fiib,
    "Root-MUSIC": root_music
}

params = RadarParams()
range_axis = get_range_axis(params)
bin_width = range_axis[1]
rng = np.random.default_rng(SEED)

# The target sits at the centre of r_bin. Each trial moves it by a random
# amount within +/- half a bin, so the RMSE is averaged over all sub-bin positions.
# v = 0 keeps the IF signal a pure tone, which is what the CRB assumes.
r_centre_bin = 15
target = Target(r=(r_centre_bin * bin_width), v=0.0, a=0.0, rcs=1.0)

crb = [get_CRB(params, snr, unit="bin") for snr in SNR_DB]

# rmse["A&M"] will hold one RMSE value per SNR, in the same order as SNR_DB.
rmse = {name: [] for name in ESTIMATORS}

for snr in SNR_DB:

    # Collect the squared error of every trial, for every estimator.
    squared_errors = {name: [] for name in ESTIMATORS}

    for trial in range(N_TRIALS):
        # New random target position and new noise on every trial.
        r_rng = r_centre_bin + rng.uniform(-0.5, 0.5)
        target.r = r_rng * bin_width
        x = get_IF_r_vec(params, [target], noise=rng, snr=snr)

        for name, estimator in ESTIMATORS.items():
            r_estimate = estimator(x)
            error = r_estimate - r_rng
            squared_errors[name].append(error ** 2)

    # RMSE = square root of the mean squared error over all trials.
    for name in ESTIMATORS:
        rmse[name].append(np.sqrt(np.mean(squared_errors[name])))


# ~~~~~~~~~~~~~~~~~~~~~~ Plotting ~~~~~~~~~~~~~~~~~~~~~~
fig, ax = plt.subplots()
ax.plot(SNR_DB, crb, "k--", label="CRB")
for name, values in rmse.items():
    ax.plot(SNR_DB, values, label=f"{name} RMSE")
ax.set_yscale("log")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Range RMSE (bins)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.show()
