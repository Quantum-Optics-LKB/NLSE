from NLSE.nlse import NLSE_1d
import numpy as np

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32


def main():
    N = 2048
    n2 = -1.6e-9
    waist = 2.23e-3
    waist2 = 70e-6
    window = N * 5.5e-6
    puiss = 1.05
    Isat = 10e4  # saturation intensity in W/m^2
    L = 10e-3
    alpha = 20
    for backend in ["CPU", "GPU"]:
        simu = NLSE_1d(
            alpha, puiss, window, n2, None, L, NX=N, Isat=Isat, backend=backend
        )
        simu.delta_z = 1e-6
        simu.puiss2 = 10e-3
        simu.n22 = 1e-10
        simu.k2 = 2 * np.pi / 795e-9
        E_0 = np.exp(-(simu.X**2) / waist**2).astype(PRECISION_COMPLEX)
        simu.V = -1e-4 * np.exp(-(simu.X**2) / waist2**2).astype(PRECISION_COMPLEX)
        simu.out_field(E_0, L, verbose=True, plot=False, precision="single")


if __name__ == "__main__":
    main()
