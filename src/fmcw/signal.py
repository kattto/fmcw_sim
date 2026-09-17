# desc

import numpy as np
from fmcw.constants import C
from fmcw.link_budget import get_rx_amplitude, get_noise_sigma

def get_IF_cube(params, targets, noise=None, snr=None):
    N, M, L = params.N, params.M, params.L
    Ts = 1.0 / params.fs
    
    # building the IF signal data
    data_I = np.zeros((N, M, L), dtype=np.float64)            # float64 because precision is needed for phase fidelity
    data_Q = np.zeros((N, M, L), dtype=np.float64)            #
    n_axis = np.arange(N)
    for (i, target) in enumerate(targets):
        
        if snr is None:
            A = get_rx_amplitude(params, target)
        else:
            A = 1.0
            
        tau_0 = 2.0 * target.r / C
        for m in range(M):
            tau = tau_0 + (2.0*target.v / C) * (m * params.T_chirp + (n_axis * Ts))
            arg = 2.0 * np.pi * (params.fc * tau + params.K * tau*n_axis * Ts - 0.5*params.K*tau**2)
            for antenna in range(L):
                delta_phi = 2.0 * np.pi * params.dx * antenna * np.sin(np.deg2rad(target.a)) / params.lam
                data_I[:, m, antenna] += A * np.cos(arg + delta_phi)
                data_Q[:, m, antenna] += A * np.sin(arg + delta_phi)
    
    if noise is not None:
        if snr is None:
            sigma = get_noise_sigma(params)
        else:
            sigma = 1.0 / np.sqrt(2.0 * 10**(snr / 10.0)) 
            
        data_I += noise.normal(0, sigma, (N, M, L))
        data_Q += noise.normal(0, sigma, (N, M, L))
    
    return data_I + 1j*data_Q

    