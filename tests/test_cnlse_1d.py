import numpy as np
from scipy.constants import c, epsilon_0

from NLSE import CNLSE_1d

if CNLSE_1d.__CUPY_AVAILABLE__:
    import cupy as cp
PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32
N = 2048
n2 = -1.6e-9
n12 = -1e-10
waist = 2.23e-3
waist2 = 70e-6
window = 4 * waist
power = 1.05
Isat = 10e4  # saturation intensity in W/m^2
L = 1e-3
alpha = 20


def test_build_propagator() -> None:
    for backend in ["CPU", "GPU"]:
        simu = CNLSE_1d(
            alpha,
            power,
            window,
            n2,
            n12,
            None,
            L,
            NX=N,
            Isat=Isat,
            backend=backend,
        )
        prop = simu._build_propagator()
        prop1 = np.exp(-1j * 0.5 * (simu.Kx**2) / simu.k * simu.delta_z)
        prop2 = np.exp(-1j * 0.5 * (simu.Kx**2) / simu.k2 * simu.delta_z)
        assert np.allclose(
            prop, np.array([prop1, prop2])
        ), f"Propagator is wrong. (Backend {backend})"


def test_prepare_output_array() -> None:
    for backend in ["CPU", "GPU"]:
        simu = CNLSE_1d(
            alpha,
            power,
            window,
            n2,
            n12,
            None,
            L,
            NX=N,
            Isat=Isat,
            backend=backend,
        )
        if backend == "CPU":
            A = np.ones((2, N), dtype=PRECISION_COMPLEX)
        elif backend == "GPU" and CNLSE_1d.__CUPY_AVAILABLE__:
            A = cp.ones((2, N), dtype=PRECISION_COMPLEX)
        out, out_sq = simu._prepare_output_array(A, normalize=True)
        assert (
            out.flags.c_contiguous
        ), f"Output array is not C-contiguous. (Backend {backend})"
        assert (
            out_sq.flags.c_contiguous
        ), f"Output array is not C-contiguous. (Backend {backend})"
        if backend == "CPU":
            assert (
                out.flags.aligned
            ), f"Output array is not aligned. (Backend {backend})"
            assert (
                out_sq.flags.aligned
            ), f"Output array is not aligned. (Backend {backend})"
        integral = (
            (out.real * out.real + out.imag * out.imag) * simu.delta_X**2
        ).sum(axis=simu._last_axes)
        integral *= c * epsilon_0 / 2
        assert np.allclose(
            integral,
            np.array([simu.power, simu.power2]),
            rtol=1e-4,
        ), f"Normalization failed. (Backend {backend})"
        assert out.shape == (
            2,
            N,
        ), f"Output array has wrong shape. (Backend {backend})"
        if backend == "CPU":
            assert isinstance(
                out, np.ndarray
            ), f"Ouptut array type does not match backend. (Backend {backend})"
            out /= np.max(np.abs(out))
            A /= np.max(np.abs(A))
            assert np.allclose(
                out, A
            ), f"Output array does not match input array. (Backend {backend})"
        elif backend == "GPU" and CNLSE_1d.__CUPY_AVAILABLE__:
            assert isinstance(
                out, cp.ndarray
            ), f"Ouptut array type does not match backend. (Backend {backend})"
            out /= cp.max(cp.abs(out))
            A /= cp.max(cp.abs(A))
            assert cp.allclose(
                out, A
            ), f"Output array does not match input array. (Backend {backend})"


def test_split_step() -> None:
    for backend in ["CPU", "GPU"]:
        simu = CNLSE_1d(
            alpha,
            power,
            window,
            n2,
            n12,
            None,
            L,
            NX=N,
            Isat=Isat,
            backend=backend,
        )
        simu.delta_z = 0
        simu.propagator = simu._build_propagator()
        E = np.ones((2, N), dtype=PRECISION_COMPLEX)
        A, A_sq = simu._prepare_output_array(E, normalize=False)
        simu.plans = simu._build_fft_plan(A)
        simu.propagator = simu._build_propagator()
        if backend == "GPU" and CNLSE_1d.__CUPY_AVAILABLE__:
            E = cp.asarray(E)
            simu._send_arrays_to_gpu()
        simu.split_step(
            E, A_sq, simu.V, simu.propagator, simu.plans, precision="double"
        )
        if backend == "CPU":
            assert np.allclose(
                E, np.ones((2, N), dtype=PRECISION_COMPLEX)
            ), f"Split step is not unitary. (Backend {backend})"
        elif backend == "GPU" and CNLSE_1d.__CUPY_AVAILABLE__:
            assert cp.allclose(
                E, cp.ones((2, N), dtype=PRECISION_COMPLEX)
            ), f"Split step is not unitary. (Backend {backend})"


def test_out_field() -> None:
    for backend in ["CPU", "GPU"]:
        simu = CNLSE_1d(
            0, power, window, n2, n12, None, L, NX=N, Isat=Isat, backend=backend
        )
        E0 = np.ones((2, N), dtype=PRECISION_COMPLEX)
        A = simu.out_field(
            E0, simu.delta_z, verbose=False, plot=False, precision="single"
        )
        rho = A.real * A.real + A.imag * A.imag
        print(rho)
        integral = (rho * simu.delta_X**2).sum(axis=simu._last_axes)
        integral *= c * epsilon_0 / 2
        assert A.shape == (
            2,
            N,
        ), f"Output array has wrong shape. (Backend {backend})"
        assert np.allclose(
            integral, [simu.power, simu.power2], rtol=1e-4
        ), f"Normalization failed. (Backend {backend})"
