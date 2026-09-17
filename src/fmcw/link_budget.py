# desc

import numpy as np
from fmcw.constants import K_BOLTZMANN, TEMP

# Returns amplitude of received wave according to the radar equation
def get_rx_amplitude(params, target):
    numerator = ( 10**(params.g_tx/10.0) * 10**(params.g_rx/10.0) * params.lam**2 * target.rcs )
    denominator =  ( (4.0*np.pi)**3 * target.r**4 * params.L_total)
    P_rx = 10**((params.P_tx - 30.0) / 10.0) * numerator / denominator    # -30 is the same as dividing by 1000 to convert mW -> W
    P_rx_amplified = P_rx * 10**(params.rx_gain/10.0)
    return np.sqrt(P_rx_amplified)          # Pk amplitude

# Returns the noise sigma (std dev) for gaussian noise
def get_noise_sigma(params):
    # Thermal noise model
    P_noise = K_BOLTZMANN * TEMP * 10**(params.NF / 10.0) * params.fs * 10**(params.rx_gain / 10.0)
    return np.sqrt(P_noise / 2.0)