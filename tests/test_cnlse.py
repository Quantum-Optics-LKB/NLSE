from NLSE import CNLSE
import numpy as np
from scipy.constants import c, epsilon_0

if CNLSE.__CUPY_AVAILABLE__:
    import cupy as cp

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

N = 2048
n2 = -1.6e-9
n12 = -1e-10
waist = 2.23e-3
waist2 = 70e-6
window = 4 * waist
puiss = 1.05
Isat = 10e4  # saturation intensity in W/m^2
L = 1e-3
alpha = 20


def test_prepare_output_array() -> None:
    for backend in ["CPU", "GPU"]:
        simu = CNLSE(
            alpha,
            puiss,
            window,
            n2,
            n12,
            None,
            L,
            NX=N,
            NY=N,
            Isat=Isat,
            backend=backend,
        )
        if backend == "CPU":
            A = np.ones((2, N, N), dtype=PRECISION_COMPLEX)
        elif backend == "GPU" and CNLSE.__CUPY_AVAILABLE__:
            A = cp.ones((2, N, N), dtype=PRECISION_COMPLEX)
        out = simu._prepare_output_array(A, normalize=True)
        integral = (
            (out.real * out.real + out.imag * out.imag) * simu.delta_X * simu.delta_Y
        ).sum(axis=simu._last_axes)
        integral *= c * epsilon_0 / 2
        assert np.allclose(
            integral,
            np.array([simu.puiss, simu.puiss2]),
            rtol=1e-4,
        ), f"Normalization failed. (Backend {backend})"
        assert out.shape == (
            2,
            N,
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
        elif backend == "GPU" and CNLSE.__CUPY_AVAILABLE__:
            assert isinstance(
                out, cp.ndarray
            ), f"Ouptut array type does not match backend. (Backend {backend})"
            out /= cp.max(cp.abs(out))
            A /= cp.max(cp.abs(A))
            assert cp.allclose(
                out, A
            ), f"Output array does not match input array. (Backend {backend})"


def test_send_arrays_to_gpu() -> None:
    if CNLSE.__CUPY_AVAILABLE__:
        alpha = 20
        Isat = 10e4
        n2 = -1.6e-9
        n12 = -1e-10
        V = np.random.random((N, N)) + 1j * np.random.random((N, N))
        alpha = np.repeat(alpha, 2)
        alpha = alpha[..., np.newaxis, np.newaxis, np.newaxis]
        n2 = np.repeat(n2, 2)
        n2 = n2[..., np.newaxis, np.newaxis, np.newaxis]
        n12 = np.repeat(n2, 2)
        n12 = n12[..., np.newaxis, np.newaxis, np.newaxis]
        Isat = np.repeat(Isat, 2)
        Isat = Isat[..., np.newaxis, np.newaxis, np.newaxis]
        simu = CNLSE(
            alpha,
            puiss,
            window,
            n2,
            n12,
            V,
            L,
            NX=N,
            NY=N,
            Isat=Isat,
            backend="GPU",
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
            simu.n12, cp.ndarray
        ), "n12 is not a cp.ndarray. (Backend GPU)"
        assert isinstance(
            simu.I_sat, cp.ndarray
        ), "I_sat is not a cp.ndarray. (Backend GPU)"
    else:
        pass


def test_retrieve_arrays_from_gpu() -> None:
    if CNLSE.__CUPY_AVAILABLE__:
        alpha = 20
        Isat = 10e4
        n2 = -1.6e-9
        n12 = -1e-10
        V = np.random.random((N, N)) + 1j * np.random.random((N, N))
        alpha = np.repeat(alpha, 2)
        alpha = alpha[..., np.newaxis, np.newaxis, np.newaxis]
        n2 = np.repeat(n2, 2)
        n2 = n2[..., np.newaxis, np.newaxis, np.newaxis]
        n12 = np.repeat(n2, 2)
        n12 = n12[..., np.newaxis, np.newaxis, np.newaxis]
        Isat = np.repeat(Isat, 2)
        Isat = Isat[..., np.newaxis, np.newaxis, np.newaxis]
        simu = CNLSE(
            alpha,
            puiss,
            window,
            n2,
            n12,
            V,
            L,
            NX=N,
            NY=N,
            Isat=Isat,
            backend="GPU",
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
            simu.n12, np.ndarray
        ), "n12 is not a np.ndarray. (Backend GPU)"
        assert isinstance(
            simu.I_sat, np.ndarray
        ), "I_sat is not a np.ndarray. (Backend GPU)"
    else:
        pass


def test_take_components() -> None:
    for backend in ["CPU", "GPU"]:
        simu = CNLSE(
            alpha,
            puiss,
            window,
            n2,
            n12,
            None,
            L,
            NX=N,
            NY=N,
            Isat=Isat,
            backend=backend,
        )
        # create a larger array to test the fancy indexing
        A = np.ones((3, 2, N, N), dtype=PRECISION_COMPLEX)
        A1, A2 = simu._take_components(A)
        assert A1.shape[-2:] == (
            N,
            N,
        ), f"A1 has wrong last dimensions. (Backend {backend})"
        assert A2.shape[-2:] == (
            N,
            N,
        ), f"A2 has wrong last dimensions. (Backend {backend})"
        assert (
            A1.shape == A2.shape
        ), f"A1 and A2 have different shapes. (Backend {backend})"
        assert A1.shape[0] == 3, f"A1 has wrong first dimensions. (Backend {backend})"
        assert A2.shape[0] == 3, f"A2 has wrong first dimensions. (Backend {backend})"


def test_split_step() -> None:
    for backend in ["CPU", "GPU"]:
        simu = CNLSE(
            alpha,
            puiss,
            window,
            n2,
            n12,
            None,
            L,
            NX=N,
            NY=N,
            Isat=Isat,
            backend=backend,
        )
        simu.delta_z = 0
        simu.propagator = simu._build_propagator()
        E = np.ones((2, N, N), dtype=PRECISION_COMPLEX)
        A = simu._prepare_output_array(E, normalize=False)
        A_sq = A.copy().real
        simu.plans = simu._build_fft_plan(A)
        if backend == "GPU" and CNLSE.__CUPY_AVAILABLE__:
            E = cp.asarray(E)
            simu._send_arrays_to_gpu()
        simu.split_step(
            A,
            A_sq,
            simu.V,
            simu.propagator,
            simu.plans,
            precision="double",
        )
        if backend == "CPU":
            assert np.allclose(
                E, np.ones((2, N, N), dtype=PRECISION_COMPLEX)
            ), f"Split-step is not unitary. (Backend {backend})"
        elif backend == "GPU" and CNLSE.__CUPY_AVAILABLE__:
            assert cp.allclose(
                E, cp.ones((2, N, N), dtype=PRECISION_COMPLEX)
            ), f"Split-step is not unitary. (Backend {backend})"


# tests for convergence of the solver : the norm of the field should be conserved
def test_out_field() -> None:
    E = np.ones((2, N, N), dtype=PRECISION_COMPLEX)
    for backend in ["CPU", "GPU"]:
        simu = CNLSE(
            0, puiss, window, n2, n12, None, L, NX=N, NY=N, Isat=Isat, backend=backend
        )
        E = simu.out_field(E, L, verbose=False, plot=False, precision="single")
        norm = np.sum(
            np.abs(E) ** 2 * simu.delta_X * simu.delta_Y * c * epsilon_0 / 2,
            axis=simu._last_axes,
        )
        assert np.allclose(
            norm, [simu.puiss, simu.puiss2], rtol=1e-4
        ), "Norm not conserved."


def main():
    print("Testing CNLSE class")
    L = 10e-2
    for backend in ["GPU"]:
        simu_c = CNLSE(
            alpha,
            puiss,
            window,
            n2,
            n12,
            None,
            L,
            NX=N,
            NY=N,
            Isat=Isat,
            omega=None,
            backend=backend,
        )
        simu_c.delta_z = 0.5e-4
        simu_c.puiss2 = 10e-3
        simu_c.n22 = 1e-10
        simu_c.k2 = 2 * np.pi / 795e-9
        E_0 = np.exp(-(simu_c.XX**2 + simu_c.YY**2) / waist**2).astype(
            PRECISION_COMPLEX
        )
        V = np.exp(-(simu_c.XX**2 + simu_c.YY**2) / waist2**2).astype(PRECISION_COMPLEX)
        E, V = simu_c.out_field(
            np.array([E_0, V]),
            L,
            verbose=True,
            plot=True,
            precision="single",
        )


if __name__ == "__main__":
    main()
