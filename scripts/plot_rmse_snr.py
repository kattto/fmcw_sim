# Monte Carlo RMSE vs. SNR for single-target range estimation, plotted against the CRB.
# Each estimator takes the 1-D IF vector and returns a range in metres, so A&M and FIIB
# can be added later by writing one function and registering it in ESTIMATORS.

import numpy as np
import matplotlib.pyplot as plt
from fmcw.params import RadarParams
from fmcw.scene import Target
from fmcw.signal import get_IF_r_vec
from fmcw.processing import get_range_axis, get_CRB

N_TRIALS = 1000
SNR_DB = np.arange(-20, 41, 1)
SEED = 0


def fft_peak(x, range_axis):
    """Plain FFT peak picking, no zero padding or interpolation."""
    return range_axis[np.argmax(np.abs(np.fft.fft(x)))]


def aboutanios_mulgrew(x, range_axis, n_iter=2):
    """Aboutanios & Mulgrew (2005) iterative interpolation on Fourier coefficients."""
    N = len(x)
    n_axis = np.arange(N)
    bin_width = range_axis[1]                       # metres per FFT bin

    peak_bin = np.argmax(np.abs(np.fft.fft(x)))

    delta = 0.0
    for _ in range(n_iter):
        p_plus = (peak_bin + delta) + 1/2
        p_minus = (peak_bin + delta) - 1/2

        exp_plus = np.exp(-1j * 2 * np.pi * p_plus * n_axis / N)
        exp_minus = np.exp(-1j * 2 * np.pi * p_minus * n_axis / N)

        val_plus = np.dot(x, exp_plus)
        val_minus = np.dot(x, exp_minus)

        delta += 1/2 * np.real((val_plus + val_minus) / (val_plus - val_minus))

    return (peak_bin + delta) * bin_width


ESTIMATORS = {
    "FFT peak": fft_peak,
    "A&M": aboutanios_mulgrew,
}


params = RadarParams()
range_axis = get_range_axis(params)
rng = np.random.default_rng(SEED)

# Sit the target on a bin centre and draw the true range uniformly within +/- half a bin,
# so the estimate is averaged over all fractional-bin offsets. v = 0 keeps the IF signal
# a pure tone (Doppler would shift the beat frequency and bias the estimate).
r_centre = 15 * params.range_res
target = Target(r=r_centre, v=0.0, a=0.0, rcs=1.0)

crb = np.array([get_CRB(params, snr, unit="m") for snr in SNR_DB])
rmse = {name: np.zeros(len(SNR_DB)) for name in ESTIMATORS}

for i, snr in enumerate(SNR_DB):
    sq_err = {name: 0.0 for name in ESTIMATORS}
    for offset in rng.uniform(-params.range_res / 2, params.range_res / 2, N_TRIALS):
        target.r = r_centre + offset
        x = get_IF_r_vec(params, [target], noise=rng, snr=snr)
        for name, estimate in ESTIMATORS.items():
            sq_err[name] += (estimate(x, range_axis) - target.r) ** 2
    for name in ESTIMATORS:
        rmse[name][i] = np.sqrt(sq_err[name] / N_TRIALS)


# ~~~~~~~~~~~~~~~~~~~~~~ Plotting ~~~~~~~~~~~~~~~~~~~~~~
fig, ax = plt.subplots()
ax.plot(SNR_DB, crb, "k--", label="CRB")
for name, values in rmse.items():
    ax.plot(SNR_DB, values, label=f"{name} RMSE")
ax.set_yscale("log")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Range RMSE (m)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.show()
