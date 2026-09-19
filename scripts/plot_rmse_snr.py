# desc

import numpy as np
import matplotlib.pyplot as plt
from fmcw.params import RadarParams
from fmcw.scene import Target
from fmcw.signal import get_IF_cube
from fmcw.processing import *

params = RadarParams()
targets = [
    Target(r=19.56521739, v=2.0, a=0.0, rcs=1.0)
]
range_axis = get_range_axis(params)
velocity_axis = get_velocity_axis(params)
angle_axis = get_angle_axis(params)

rng = np.random.default_rng()

snr_range = range(-20, 20, 1)
crb_data = []
rmse_data = []
for snr in snr_range:
    crb_data.append(get_CRB(params, snr, unit="m"))
    
    offset_low = -params.range_res/2
    offset_high = params.range_res/2
    target_offsets = rng.uniform(low=offset_low, high=offset_high, size=10)
    s = 0
    for offset in target_offsets:
        targets[0].r = 19.56521739 + offset

        raw_cube = get_IF_cube(params, targets, noise=rng, snr=snr)
        rda_cube = get_rda_cube(params, raw_cube)
    
        ## Needs rework
        peaks = get_peaks(params, targets, rda_cube)
        range_meas = peaks[0][0]
        err = np.abs(range_axis[range_meas] - targets[0].r)
        s += err**2
        # print(err)
        
    rmse_data.append(np.sqrt(s/10))
        


# ~~~~~~~~~~~~~~~~~~~~~~ Plotting ~~~~~~~~~~~~~~~~~~~~~~
fig, ax = plt.subplots()
ax.plot(snr_range, crb_data, label="CRB")
ax.plot(snr_range, rmse_data, label="FFT RMSE")
ax.set_yscale("log")
ax.legend()
plt.show()

