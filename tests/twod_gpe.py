from NLSE import GPE
from scipy.constants import atomic_mass
import numpy as np

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32


def main():
    N = 2048
    N_at = 1e6
    g = 1e3 / (N_at / 1e-3**2)
    waist = 250e-6
    for backend in ["CPU", "GPU"]:
        simu_gpe = GPE(
            gamma=0,
            N=N_at,
            m=87 * atomic_mass,
            window=1e-3,
            g=g,
            V=None,
            NX=N,
            NY=N,
            backend=backend,
        )
        simu_gpe.delta_t = 1e-8
        psi_0 = np.exp(-(simu_gpe.XX**2 + simu_gpe.YY**2) / waist**2).astype(
            PRECISION_COMPLEX
        )
        simu_gpe.out_field(psi_0, 1e-6, verbose=True, plot=False, precision="single")


if __name__ == "__main__":
    main()
