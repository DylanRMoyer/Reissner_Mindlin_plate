from dataclasses import dataclass

@dataclass
class PlateConfig:
    length: float
    width: float
    thickness: float
    nx: int
    ny: int
    rho: float
    mu: float
    lambda_: float
    constant_force: float = 1.0
