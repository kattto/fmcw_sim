# desc

from dataclasses import dataclass

@dataclass
class Target:
    r: float        # range (m)
    v: float        # velocity (m/s)
    a: float        # angle (degrees, 0 is centre)
    rcs: float      # Radar cross section (m^2), equivalent to a gain