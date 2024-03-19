from NLSE.nlse import CNLSE_1d
import numpy as np

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32


def main():
    N = 2048
    n2 = -1.6e-9
    n12 = -1e-10
    waist = 2.23e-3
    waist2 = 70e-6
    window = N * 5.5e-6
    puiss = 1.05
    Isat = 10e4  # saturation intensity in W/m^2
    L = 10e-3
    alpha = 20
    for backend in ["CPU", "GPU"]:
        simu_c = CNLSE_1d(
            alpha, puiss, window, n2, n12, None, L, NX=N, Isat=Isat, backend=backend
        )
        simu_c.delta_z = 1e-6
        simu_c.puiss2 = 10e-3
        simu_c.n22 = 1e-10
        simu_c.k2 = 2 * np.pi / 795e-9
        E_0 = np.exp(-(simu_c.X**2) / waist**2).astype(PRECISION_COMPLEX)
        V = np.exp(-(simu_c.X**2) / waist2**2).astype(PRECISION_COMPLEX)
        E, V = simu_c.out_field(
            np.array([E_0, V]), L, verbose=True, plot=False, precision="single"
        )


if __name__ == "__main__":
    main()
