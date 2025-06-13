import numpy as np
from scipy.constants import atomic_mass, hbar

from NLSE import GPE

if GPE.__CUPY_AVAILABLE__:
    import cupy as cp
PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32
AVAILABLE_BACKENDS = ["CPU"]
if GPE.__CUPY_AVAILABLE__:
    AVAILABLE_BACKENDS.append("GPU")
# TODO: Write OpenCL tests
# if CNLSE.__PYOPENCL_AVAILABLE__:
#     AVAILABLE_BACKENDS.append("CL")

N = 2048
N_at = 1e6
g = 1e3 / (N_at / 1e-3**2)
waist = 1e-3
window = 1e-3
m = 87 * atomic_mass


def test_build_propagator() -> None:
    for backend in AVAILABLE_BACKENDS:
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
        prop = simu_gpe._build_propagator()
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
        ), f"Propagator is wrong. (Backend {backend})"


def test_prepare_output_array() -> None:
    for backend in AVAILABLE_BACKENDS:
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
        if backend == "CPU":
            E_in = np.random.random((N, N)) + 1j * np.random.random((N, N))
        elif backend == "GPU" and GPE.__CUPY_AVAILABLE__:
            E_in = cp.random.random((N, N)) + 1j * cp.random.random((N, N))
        A, A_sq = simu._prepare_output_array(E_in, normalize=True)
        assert A.flags.c_contiguous, (
            f"Output array is not C-contiguous. (Backend {backend})"
        )
        assert A_sq.flags.c_contiguous, (
            f"Output array is not C-contiguous. (Backend {backend})"
        )
        if backend == "CPU":
            assert A.flags.aligned, f"Output array is not aligned. (Backend {backend})"
            assert A_sq.flags.aligned, (
                f"Output array is not aligned. (Backend {backend})"
            )
        integral = (
            (A.real * A.real + A.imag * A.imag) * simu.delta_X * simu.delta_Y
        ).sum(axis=simu._last_axes)
        assert np.allclose(integral, simu.N), (
            f"Normalization failed. (Backend {backend})"
        )
        if backend == "CPU":
            assert isinstance(A, np.ndarray), (
                f"Output array type does not match backend. (Backend {backend})"
            )
            A /= np.max(np.abs(A))
            E_in /= np.max(np.abs(E_in))
            assert np.allclose(E_in, A), (
                f"Output array does not match input array. (Backend {backend})"
            )
        elif backend == "GPU" and GPE.__CUPY_AVAILABLE__:
            assert isinstance(A, cp.ndarray), (
                f"Output array type does not match backend. (Backend {backend})"
            )
            A /= cp.max(cp.abs(A))
            E_in /= cp.max(cp.abs(E_in))
            assert cp.allclose(E_in, A), (
                f"Output array does not match input array. (Backend {backend})"
            )


def test_out_field() -> None:
    for backend in AVAILABLE_BACKENDS:
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
        simu.delta_t = 1e-8
        psi_0 = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(PRECISION_COMPLEX)
        psi = simu.out_field(psi_0, 1e-6, verbose=True, plot=False, precision="single")
        norm = np.sum(np.abs(psi) ** 2 * simu.delta_X * simu.delta_Y)
        assert np.allclose(norm, simu.N, rtol=1e-4), (
            f"Norm not conserved. (Backend {backend})"
        )
