from NLSE import NLSE
import numpy as np
from NLSE.utils import __CUPY_AVAILABLE__

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32


def test_broadcasting():
    if NLSE.__CUPY_AVAILABLE__:
        N = 2048
        n2 = np.zeros((10, 1, 1))
        n2[:, 0, 0] = np.linspace(-1.6e-9, -1e-10, 10)
        waist = 2.23e-3
        waist2 = 70e-6
        window = 4 * waist
        puiss = 1.05
        Isat = 10e4  # saturation intensity in W/m^2
        L = 10e-3
        alpha = 20
        E_0 = np.ones((10, N, N), dtype=PRECISION_COMPLEX)
        simu = NLSE(alpha, puiss, window, n2, V=None, L=L, NX=N, NY=N, Isat=Isat)
        E_0 *= np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(PRECISION_COMPLEX)
        simu.V = -1e-4 * np.exp(-(simu.XX**2 + simu.YY**2) / waist2**2).astype(
            PRECISION_COMPLEX
        )
        simu.delta_z = 1e-4
        E = simu.out_field(E_0, L, verbose=True, plot=False, precision="single")
        assert E.shape == E_0.shape, "Output shape mismatch !"
    else:
        print("Broadcasting is only available for Cupy backend !")


def main():
    N = 2048
    n2 = np.zeros((10, 1, 1))
    n2[:, 0, 0] = np.linspace(-1.6e-9, -1e-10, 10)
    waist = 2.23e-3
    window = 4 * waist
    puiss = 1.05
    Isat = 10e4  # saturation intensity in W/m^2
    L = 10e-3
    alpha = 20
    E_0 = np.ones((10, N, N), dtype=PRECISION_COMPLEX)
    simu = NLSE(alpha, puiss, window, n2, V=None, L=L, NX=N, NY=N, Isat=Isat)
    E_0 *= np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(PRECISION_COMPLEX)
    simu.delta_z = 1e-4
    simu.out_field(E_0, L, verbose=True, plot=False, precision="single")


if __name__ == "__main__":
    main()
