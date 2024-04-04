from NLSE import NLSE_3d
import numpy as np
import pyfftw
from scipy.constants import c, epsilon_0
import matplotlib.pyplot as plt

if NLSE_3d.__CUPY_AVAILABLE__:
    import cupy as cp
PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32


N = 256
n2 = -1.6e-9
D0 = 1e-16
vg = 1e-1 * c
waist = 2.23e-3
duration = 2e-6
waist2 = 70e-6
window = np.array([4 * waist, 10e-6])  # 4*waist transverse, 10e-6 s temporal
energy = 1.05 * duration
Isat = 10e4  # saturation intensity in W/m^2
L = 10e-3
alpha = 20


def main():
    print("Testing NLSE_3d class")
    for backend in ["GPU", "CPU"]:
        simu = NLSE_3d(
            alpha,
            energy,
            window,
            n2,
            D0,
            vg,
            None,
            L,
            NX=N,
            NY=N,
            NZ=128,
            Isat=Isat,
            backend=backend,
        )
        simu.delta_z = 1e-4
        E_0 = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(PRECISION_COMPLEX)
        E_0 *= np.exp(-(simu.TT**2) / duration**2)
        simu.V = -1e-4 * np.exp(-(simu.XX**2 + simu.YY**2) / waist2**2).astype(
            PRECISION_COMPLEX
        )
        simu.out_field(E_0, L, verbose=True, plot=True, precision="single")


if __name__ == "__main__":
    main()
