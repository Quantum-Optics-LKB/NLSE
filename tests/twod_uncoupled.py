from NLSE.nlse import NLSE
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
        simu = NLSE(
            alpha, puiss, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend=backend
        )
        E_0 = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(PRECISION_COMPLEX)
        simu.V = -1e-4 * np.exp(-(simu.XX**2 + simu.YY**2) / waist2**2).astype(
            PRECISION_COMPLEX
        )

        simu.out_field(E_0, L, verbose=True, plot=False, precision="single")


if __name__ == "__main__":
    main()
