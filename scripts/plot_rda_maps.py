# desc

import numpy as np
import matplotlib.pyplot as plt
from fmcw.params import RadarParams
from fmcw.scene import Target
from fmcw.signal import get_IF_cube
from fmcw.processing import get_rda_cube, get_range_axis, get_velocity_axis, get_angle_axis, get_peaks

params = RadarParams()
targets = [
    Target(r=20.0, v=2.0, a=0.0, rcs=1.0)
]
range_axis = get_range_axis(params)
velocity_axis = get_velocity_axis(params)
angle_axis = get_angle_axis(params)

rng = np.random.default_rng(0)
raw_cube = get_IF_cube(params, targets, noise=rng, snr=20)
rda_cube = get_rda_cube(params, raw_cube)

rda_cube_log = 20*np.log10(np.abs(rda_cube) + 1e-12)
# rda_cube_log = np.abs(rda_cube)

rd_map = np.max(rda_cube_log, 2)                   # Using maximum intensity projection (MIP)
ra_map = np.max(rda_cube_log, 1)                   #

rd_map -= np.max(rd_map)
ra_map -= np.max(ra_map)

# Find targets
get_peaks(params, targets, rda_cube)

# ~~~~~~~~~~~~~~~~~~~~~~ Plotting ~~~~~~~~~~~~~~~~~~~~~~
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
max_range_display = 75.0                                # Max range to display

# Range-Doppler 
im1 = ax1.imshow(rd_map,
                 aspect="auto",
                 origin="lower",
                 extent=(velocity_axis[0],
                         velocity_axis[-1],
                         range_axis[0],
                         range_axis[-1])
                )
ax1.set_ylim(0, max_range_display)
ax1.set_xlabel("Velocity (m/s)")
ax1.set_ylabel("Range (m)")
ax1.set_title("Range-Doppler Map")
fig.colorbar(im1, ax=ax1, label="dB")


# Range-Angle
im2 = ax2.pcolormesh(angle_axis,
                     range_axis,
                     ra_map)
# im2.set_mouseover(True)                     # Displays amplitude when hovering (Computationally heavy)
ax2.set_ylim(0, max_range_display)
ax2.set_xlabel("Angle (degrees)")
ax2.set_ylabel("Range (m)")
ax2.set_title("Range-Angle Map")
fig.colorbar(im2, ax=ax2, label="dB")


# Clean up layout and display
plt.tight_layout()
plt.show()
