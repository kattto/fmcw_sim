"""Range-Doppler map of a converted Ancortek capture."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fmcw.ancortek import load_capture

C = 3e8

CAPTURE = Path("data/radar_processed/1GHz_take_1.npz")
BACKGROUND = None #Path("data/radar_processed/1GHz_empty_background.npz")
RX = 1

parser = argparse.ArgumentParser()
# parser.add_argument("capture", type=Path)
# parser.add_argument("--background", type=Path, default=None)
# parser.add_argument("--rx", type=int, default=1)
args = parser.parse_args()

args.capture = CAPTURE
args.background = BACKGROUND
args.rx = RX

params, cubes, _ = load_capture(args.capture)
x = cubes["tx1"][:, :, args.rx - 1]                     # (N, M) for one RX

if args.background is not None:
    _, bg, _ = load_capture(args.background)
    x = x - bg["tx1"][:, :, args.rx - 1].mean(axis=1, keepdims=True)
else:
    x = x - x.mean(axis=0, keepdims=True)               # remove DC per sweep

N, M = x.shape
rd = np.fft.fftshift(np.fft.fft2(x))                    # range on axis 0, Doppler on axis 1

# axes
t_ramp = params["t_ramp"] if np.isfinite(params["t_ramp"]) else params["t_sweep"]
slope = params["bandwidth"] / t_ramp
range_axis = np.fft.fftshift(np.fft.fftfreq(N, 1 / params["fs"])) * C / (2 * slope)

prf = 1 / (params["t_sweep"] * params["n_tx"])
wavelength = C / ((params["f_start"] + params["f_stop"]) / 2)
vel_axis = np.fft.fftshift(np.fft.fftfreq(M, 1 / prf)) * wavelength / 2

rd_db = 20 * np.log10(np.abs(rd))
plt.imshow(rd_db, aspect="auto", origin="lower", vmin=rd_db.max() - 60,
           extent=[vel_axis[0], vel_axis[-1], range_axis[0], range_axis[-1]])
plt.xlabel("velocity (m/s)")
plt.ylabel("range (m)")
plt.title(f"RX{args.rx}, background subtracted: {args.background is not None}")
plt.colorbar(label="dB")
plt.show()