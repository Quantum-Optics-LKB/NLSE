from NLSE import NLSE
import numpy as np
import pyfftw
from scipy.constants import c, epsilon_0

if NLSE.__CUPY_AVAILABLE__:
    import cupy as cp
    from pyvkfft.cuda import VkFFTApp
PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32


N = 2048
n2 = -1.6e-9
waist = 2.23e-3
waist2 = 70e-6
window = 4 * waist
puiss = 1.05
Isat = 10e4  # saturation intensity in W/m^2
L = 10e-3
alpha = 20


def test_build_propagator() -> None:
    for backend in ["CPU", "GPU"]:
        simu = NLSE(
            alpha, puiss, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend=backend
        )
        prop = simu._build_propagator()
        assert np.allclose(
            prop,
            np.exp(-1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k * simu.delta_z),
        ), f"Propagator is wrong. (Backend {backend})"


def test_build_fft_plan() -> None:
    for backend in ["CPU", "GPU"]:
        simu = NLSE(
            alpha, puiss, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend=backend
        )
        if backend == "CPU":
            A = np.random.random((N, N)) + 1j * np.random.random((N, N))
        elif backend == "GPU" and NLSE.__CUPY_AVAILABLE__:
            A = cp.random.random((N, N)) + 1j * cp.random.random((N, N))
        plans = simu._build_fft_plan(A)
        if backend == "CPU":
            assert len(plans) == 2, f"Number of plans is wrong. (Backend {backend})"
            assert isinstance(
                plans[0], pyfftw.FFTW
            ), f"Plan type is wrong. (Backend {backend})"
            assert plans[0].output_shape == (
                N,
                N,
            ), f"Plan shape is wrong. (Backend {backend})"
        elif backend == "GPU" and NLSE.__CUPY_AVAILABLE__:
            assert len(plans) == 1, f"Number of plans is wrong. (Backend {backend})"
            assert isinstance(
                plans[0], VkFFTApp
            ), f"Plan type is wrong. (Backend {backend})"
            assert plans[0].shape0 == (
                N,
                N,
            ), f"Plan shape is wrong. (Backend {backend})"


def test_prepare_output_array() -> None:
    for backend in ["CPU", "GPU"]:
        simu = NLSE(
            alpha, puiss, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend=backend
        )
        if backend == "CPU":
            A = np.random.random((N, N)) + 1j * np.random.random((N, N))
        elif backend == "GPU" and NLSE.__CUPY_AVAILABLE__:
            A = cp.random.random((N, N)) + 1j * cp.random.random((N, N))
        out = simu._prepare_output_array(A, normalize=True)
        integral = (
            (out.real * out.real + out.imag * out.imag) * simu.delta_X * simu.delta_Y
        ).sum(axis=simu._last_axes)
        integral *= c * epsilon_0 / 2
        assert np.allclose(
            integral, simu.puiss
        ), f"Normalization failed. (Backend {backend})"
        assert out.shape == (N, N), f"Output array has wrong shape. (Backend {backend})"
        if backend == "CPU":
            assert isinstance(
                out, np.ndarray
            ), f"Output array type does not match backend. (Backend {backend})"
            out /= np.max(np.abs(out))
            A /= np.max(np.abs(A))
            assert np.allclose(
                out, A
            ), f"Output array does not match input array. (Backend {backend})"
        elif backend == "GPU" and NLSE.__CUPY_AVAILABLE__:
            assert isinstance(
                out, cp.ndarray
            ), f"Output array type does not match backend. (Backend {backend})"
            out /= cp.max(cp.abs(out))
            A /= cp.max(cp.abs(A))
            assert cp.allclose(
                out, A
            ), f"Output array does not match input array. (Backend {backend})"


def test_send_arrays_to_gpu() -> None:
    if NLSE.__CUPY_AVAILABLE__:
        alpha = 20
        Isat = 10e4
        n2 = -1.6e-9
        V = np.random.random((N, N)) + 1j * np.random.random((N, N))
        alpha = np.repeat(alpha, 2)
        alpha = alpha[..., cp.newaxis, cp.newaxis]
        n2 = np.repeat(n2, 2)
        n2 = n2[..., cp.newaxis, cp.newaxis]
        Isat = np.repeat(Isat, 2)
        Isat = Isat[..., cp.newaxis, cp.newaxis]
        simu = NLSE(
            alpha, puiss, window, n2, V, L, NX=N, NY=N, Isat=Isat, backend="GPU"
        )
        simu.propagator = simu._build_propagator()
        simu._send_arrays_to_gpu()
        assert isinstance(
            simu.propagator, cp.ndarray
        ), "propagator is not a cp.ndarray. (Backend GPU)"
        assert isinstance(simu.V, cp.ndarray), "V is not a cp.ndarray. (Backend GPU)"
        assert isinstance(
            simu.alpha, cp.ndarray
        ), "alpha is not a cp.ndarray. (Backend GPU)"
        assert isinstance(simu.n2, cp.ndarray), "n2 is not a cp.ndarray. (Backend GPU)"
        assert isinstance(
            simu.I_sat, cp.ndarray
        ), "I_sat is not a cp.ndarray. (Backend GPU)"
    else:
        pass


def test_retrieve_arrays_from_gpu() -> None:
    if NLSE.__CUPY_AVAILABLE__:
        alpha = 20
        Isat = 10e4
        n2 = -1.6e-9
        V = np.random.random((N, N)) + 1j * np.random.random((N, N))
        alpha = np.repeat(alpha, 2)
        alpha = alpha[..., cp.newaxis, cp.newaxis]
        n2 = np.repeat(n2, 2)
        n2 = n2[..., cp.newaxis, cp.newaxis]
        Isat = np.repeat(Isat, 2)
        Isat = Isat[..., cp.newaxis, cp.newaxis]
        simu = NLSE(
            alpha, puiss, window, n2, V, L, NX=N, NY=N, Isat=Isat, backend="GPU"
        )
        simu.propagator = simu._build_propagator()
        simu._send_arrays_to_gpu()
        simu._retrieve_arrays_from_gpu()
        assert isinstance(
            simu.propagator, np.ndarray
        ), "propagator is not a np.ndarray. (Backend GPU)"
        assert isinstance(simu.V, np.ndarray), "V is not a np.ndarray. (Backend GPU)"
        assert isinstance(
            simu.alpha, np.ndarray
        ), "alpha is not a np.ndarray. (Backend GPU)"
        assert isinstance(simu.n2, np.ndarray), "n2 is not a np.ndarray. (Backend GPU)"
        assert isinstance(
            simu.I_sat, np.ndarray
        ), "I_sat is not a np.ndarray. (Backend GPU)"
    else:
        pass


def test_split_step() -> None:
    for backend in ["CPU", "GPU"]:
        simu = NLSE(
            alpha, puiss, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend=backend
        )
        simu.delta_z = 0
        simu.propagator = simu._build_propagator()
        E = np.ones((N, N), dtype=PRECISION_COMPLEX)
        A = simu._prepare_output_array(E, normalize=False)
        A_sq = A.copy().real
        simu.plans = simu._build_fft_plan(A)
        simu.propagator = simu._build_propagator()
        if backend == "GPU" and NLSE.__CUPY_AVAILABLE__:
            E = cp.asarray(E)
            simu._send_arrays_to_gpu()
        simu.split_step(
            E, A_sq, simu.V, simu.propagator, simu.plans, precision="double"
        )
        if backend == "CPU":
            assert np.allclose(
                E, np.ones((N, N), dtype=PRECISION_COMPLEX)
            ), f"Split step is not unitary. (Backend {backend})"
        elif backend == "GPU" and NLSE.__CUPY_AVAILABLE__:
            assert cp.allclose(
                E, cp.ones((N, N), dtype=PRECISION_COMPLEX)
            ), f"Split step is not unitary. (Backend {backend})"


# tests for convergence of the solver : the norm of the field should be conserved
def test_out_field() -> None:
    E = np.ones((N, N), dtype=PRECISION_COMPLEX)
    for backend in ["CPU", "GPU"]:
        simu = NLSE(
            0, puiss, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend=backend
        )
        E = simu.out_field(E, L, verbose=False, plot=False, precision="single")
        norm = np.sum(np.abs(E) ** 2 * simu.delta_X * simu.delta_Y)
        norm *= c * epsilon_0 / 2
        assert E.shape == (N, N), f"Output array has wrong shape. (Backend {backend})"
        assert np.allclose(
            norm, simu.puiss, rtol=1e-4
        ), f"Norm not conserved. (Backend {backend})"


# for integration testing
def main():
    print("Testing NLSE class")
    for backend in ["GPU", "CPU"]:
        simu = NLSE(
            alpha, puiss, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend=backend
        )
        simu.delta_z = 1e-4
        E_0 = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(PRECISION_COMPLEX)
        simu.V = -1e-4 * np.exp(-(simu.XX**2 + simu.YY**2) / waist2**2).astype(
            PRECISION_COMPLEX
        )
        simu.out_field(E_0, L, verbose=True, plot=True, precision="single")


if __name__ == "__main__":
    main()
