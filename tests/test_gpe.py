from NLSE import GPE
from scipy.constants import atomic_mass, hbar
import numpy as np

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

N = 2048
N_at = 1e6
g = 1e3 / (N_at / 1e-3**2)
waist = 1e-3
window = 1e-3
m = 87 * atomic_mass


def test_build_propagator() -> None:
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
        prop = simu_gpe._build_propagator(simu_gpe.m)
        assert np.allclose(
            prop,
            np.exp(
                -1j
                * 0.5
                * hbar
                * (simu_gpe.Kxx**2 + simu_gpe.Kyy**2)
                / simu_gpe.m
                * simu_gpe.delta_t
            ),
        )


def test_prepare_output_array() -> None:
    for backend in ["CPU", "GPU"]:
        simu = GPE(
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
        E_in = np.ones((N, N), dtype=PRECISION_COMPLEX)
        A = simu._prepare_output_array(E_in, normalize=True)
        integral = (
            (A.real * A.real + A.imag * A.imag) * simu.delta_X * simu.delta_Y
        ).sum(axis=simu._last_axes)
        assert np.allclose(integral, simu.N)


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
