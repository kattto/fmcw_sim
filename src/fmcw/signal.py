# desc

import numpy as np
from fmcw.constants import C
from fmcw.link_budget import get_rx_amplitude, get_noise_sigma

def get_IF_cube(params, targets, noise=None, snr=None):
    N, M, L = params.N, params.M, params.L
    Ts = 1.0 / params.fs
    
    # building the IF signal data
    data = np.zeros((N, M, L), dtype=np.complex128)
    
    n_axis = np.arange(N)[:, np.newaxis, np.newaxis]
    m_axis = np.arange(M)[np.newaxis, :, np.newaxis]
    l_axis = np.arange(L)[np.newaxis, np.newaxis, :]
    
    for target in targets:
        if snr is None:
            A = get_rx_amplitude(params, target)
        else:
            A = 1.0
            
        tau_0 = 2.0 * target.r / C
        tau = tau_0 + (2.0*target.v / C) * (m_axis * params.T_chirp + (n_axis * Ts))
        arg = 2.0 * np.pi * (params.fc * tau + params.K * tau*n_axis * Ts - 0.5*params.K*tau**2)
        delta_phi = 2.0 * np.pi * params.dx * l_axis * np.sin(np.deg2rad(target.a)) / params.lam
        data += A * np.exp(1j * (arg + delta_phi))
    
    if noise is not None:
        if snr is None:
            sigma = get_noise_sigma(params)
        else:
            sigma = 1.0 / np.sqrt(2.0 * 10**(snr / 10.0)) 
            
        noise_I = noise.normal(0, sigma, (N, M, L))
        noise_Q = noise.normal(0, sigma, (N, M, L))
        data += noise_I + 1j * noise_Q
            
    return data


def get_IF_r_vec(params, targets, noise=None, snr=None):
    N = params.N
    Ts = 1.0 / params.fs

    r_vec = np.zeros((N), dtype=np.complex128)
    n_axis = np.arange(N)
    m_axis = 0
    l_axis = 0

    for target in targets:
        if snr is None:
            A = get_rx_amplitude(params, target)
        else:
            A = 1.0
            
        tau_0 = 2.0 * target.r / C
        tau = tau_0 + (2.0*target.v / C) * (m_axis * params.T_chirp + (n_axis * Ts))
        arg = 2.0 * np.pi * (params.fc * tau + params.K * tau*n_axis * Ts - 0.5*params.K*tau**2)
        delta_phi = 2.0 * np.pi * params.dx * l_axis * np.sin(np.deg2rad(target.a)) / params.lam
        r_vec += A * np.exp(1j * (arg + delta_phi))

    if noise is not None:
        if snr is None:
            sigma = get_noise_sigma(params)
        else:
            sigma = 1.0 / np.sqrt(2.0 * 10**(snr / 10.0)) 
            
        noise_I = noise.normal(0, sigma, (N))
        noise_Q = noise.normal(0, sigma, (N))
        r_vec += noise_I + 1j * noise_Q
            
    return r_vec