# Desc

from dataclasses import dataclass
from fmcw.constants import C

@dataclass
class RadarParams:
    fc: float = 24e9
    bw: float = 115e6
    T_chirp: float = 500e-6
    fs: float = 1e6
    M: int = 64
    L: int = 8
    
    P_tx: float = 20
    g_tx: float = 7
    g_rx: float = 7
    rx_gain: float = 28
    NF: float = 12
    L_total: float = 1
    
    @property
    def lam(self):
        return C / self.fc
    
    @property
    def K(self):
        return self.bw / self.T_chirp
    
    @property
    def N(self):
        return int(self.fs * self.T_chirp)
    
    @property
    def unambig_range(self):
        return (self.fs * C) / (2.0 * self.K)
    
    @property
    def dx(self):
        return self.lam / 2.0
    
    @property
    def range_res(self):
        return C / (2.0 * self.bw)
    
    @property
    def L_fft(self):
        return max(self.L, 192)

               