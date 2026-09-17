# desc

import numpy as np
from fmcw.constants import C


def get_rda_cube(params, data_raw):
    N, M, L = params.N, params.M, params.L
    window_3D = np.ones(N)[:, None, None] * np.ones(M)[None, :, None] * np.ones(L)[None, None, :]
    rda_raw = np.fft.fftn(data_raw * window_3D, s=(N, M, params.L_fft))
    rda_shifted = np.fft.fftshift(rda_raw, axes=1)          # centre doppler at zero
    rda_shifted = np.fft.fftshift(rda_shifted, axes=2)      # centre angle at zero
    return rda_shifted


def get_range_axis(params):
    return np.arange(params.N) * params.fs / params.N * C / (2.0 * params.K)
    
def get_velocity_axis(params):
    return np.arange(-params.M//2, params.M//2) * 1/(params.M * params.T_chirp) * C / (2.0 * params.fc)


def get_angle_axis(params):
    return np.degrees(np.arcsin(np.arange(-params.L_fft//2, params.L_fft//2) / params.L_fft * params.lam / params.dx))


def get_peaks(params, targets, rda_cube):
    rda_temp = np.max(np.abs(rda_cube), axis=2)     # Maximum intensity projection (MIP)
    x = 3                                           # Size of box around peaks considered part of the target
    y = 5                                           #
    peaks = []
    detected_targets = []
    range_axis = get_range_axis(params)
    velocity_axis = get_velocity_axis(params)
    for i in range(len(targets)):
        (r, d) = np.unravel_index(np.argmax(rda_temp), rda_temp.shape)
        peaks.append([r, d])
        d_extent = slice(max(d-x, 0), min(d+x+1, params.M))
        r_extent = slice(max(r-y, 0), min(r+y+1, params.N))
        detected_targets.append(rda_cube[r_extent, d_extent].copy())
        rda_temp[r_extent, d_extent] = 0.0
        print(f"Target {i+1}")
        print(f"    Range {range_axis[r]:.3f} m")
        print(f"    Velocity {velocity_axis[d]:.3f} m/s")
    
    return peaks, detected_targets


