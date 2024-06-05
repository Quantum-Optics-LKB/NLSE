from NLSE import GPE
from scipy.constants import atomic_mass
import numpy as np

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

N = 2048
N_at = 1e6
g = 1e3 / (N_at / 1e-3**2)
waist = 1e-3
window = 1e-3
m = 87 * atomic_mass


def main():
    print("Testing GPE class")
    for backend in ["CPU", "GPU"]:
        simu_gpe = GPE(
            gamma=0,
            N=N_at,
            window=window,
            g=g,
            V=None,
            m=m,
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
