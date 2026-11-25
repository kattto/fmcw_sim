"""Range-Doppler map of a converted Ancortek capture."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fmcw.ancortek import load_capture

C = 3e8

# CAPTURE = Path("data/radar_processed/1GHz_empty_background.npz")
# BACKGROUND = None
CAPTURE = Path("data/radar_processed/1GHz_take_1.npz")
BACKGROUND = Path("data/radar_processed/1GHz_empty_background.npz")
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

rd_db = 20 * np.log10(np.abs(rd) / np.abs(rd).max())     # normalised so peak is 0 dB
# rd_db[rd_db < -80.0] = -240.0

fig, ax = plt.subplots(figsize=(11, 6))
im = ax.imshow(rd_db, aspect="auto", origin="lower", vmin=-60, vmax=0,
               extent=[vel_axis[0], vel_axis[-1], range_axis[0], range_axis[-1]])
ax.set_xlabel("velocity (m/s)")
ax.set_ylabel("range (m)")
ax.set_title(f"{args.capture.name}, RX{args.rx}, background subtracted: {args.background is not None}")
fig.colorbar(im, ax=ax, label="dB (relative to peak)")

# capture parameters
header = [
    ("n_rx", f"{params['n_rx']}"),
    ("n_tx", f"{params['n_tx']}"),
    ("f_start", f"{params['f_start'] / 1e9:.3f} GHz"),
    ("f_stop", f"{params['f_stop'] / 1e9:.3f} GHz"),
    ("t_sweep", f"{params['t_sweep'] * 1e6:.0f} us"),
    ("n_samp", f"{params['n_samp']}"),
]
derived = [
    ("bandwidth", f"{params['bandwidth'] / 1e6:.0f} MHz"),
    ("t_ramp", f"{t_ramp * 1e6:.1f} us"),
    ("fs", f"{params['fs'] / 1e3:.1f} kHz"),
    ("slope", f"{slope / 1e12:.3f} MHz/us"),
    ("lambda", f"{wavelength * 1e3:.2f} mm"),
    ("n_sweeps", f"{params['n_sweeps']} per TX"),
    ("duration", f"{params['duration']:.3f} s"),
]
processing = [
    ("T_c", f"{1 / prf * 1e3:.2f} ms"),
    ("CPI", f"{M / prf:.3f} s ({M} sweeps)"),
    ("range bin", f"{range_axis[1] - range_axis[0]:.3f} m"),
    ("R_max (one side)", f"{range_axis[-1]:.2f} m"),
    ("velocity bin", f"{(vel_axis[1] - vel_axis[0]) * 1e3:.2f} mm/s"),
    ("v_max", f"+/-{prf * wavelength / 4:.3f} m/s"),
]


def block(heading, rows):
    return heading + "\n" + "\n".join(f"  {k:<17}{v}" for k, v in rows)


info = "\n\n".join([block("header", header), block("derived", derived), block("processing", processing)])
fig.text(0.74, 0.5, info, va="center", ha="left", family="monospace", fontsize=8)
fig.subplots_adjust(left=0.07, right=0.72)
plt.show()