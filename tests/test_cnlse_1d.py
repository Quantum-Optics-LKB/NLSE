from NLSE import CNLSE_1d
import numpy as np
from scipy.constants import c, epsilon_0

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
puiss = 1.05
Isat = 10e4  # saturation intensity in W/m^2
L = 1e-3
alpha = 20


def test_build_propagator() -> None:
    for backend in ["CPU", "GPU"]:
        simu = CNLSE_1d(
            alpha, puiss, window, n2, n12, None, L, NX=N, Isat=Isat, backend=backend
        )
        prop = simu._build_propagator(simu.k)
        assert np.allclose(
            prop, np.exp(-1j * 0.5 * (simu.Kx**2) / simu.k * simu.delta_z)
        ), f"Propagator is wrong. (Backend {backend})"


def test_prepare_output_array() -> None:
    for backend in ["CPU", "GPU"]:
        simu = CNLSE_1d(
            alpha,
            puiss,
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
        out = simu._prepare_output_array(A, normalize=True)
        integral = ((out.real * out.real + out.imag * out.imag) * simu.delta_X**2).sum(
            axis=simu._last_axes
        )
        integral *= c * epsilon_0 / 2
        assert np.allclose(
            integral,
            np.array([simu.puiss, simu.puiss2]),
            rtol=1e-4,
        ), f"Normalization failed. (Backend {backend})"
        assert out.shape == (2, N), f"Output array has wrong shape. (Backend {backend})"
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


def test_out_field() -> None:
    for backend in ["CPU", "GPU"]:
        simu = CNLSE_1d(
            0, puiss, window, n2, n12, None, L, NX=N, Isat=Isat, backend=backend
        )
        E0 = np.ones((2, N), dtype=PRECISION_COMPLEX)
        A = simu.out_field(E0, L, verbose=False, plot=False, precision="single")
        integral = ((A.real * A.real + A.imag * A.imag) * simu.delta_X**2).sum(
            axis=simu._last_axes
        )
        integral *= c * epsilon_0 / 2
        assert A.shape == (2, N), f"Output array has wrong shape. (Backend {backend})"
        assert np.allclose(
            integral, [simu.puiss, simu.puiss2], rtol=1e-4
        ), f"Normalization failed. (Backend {backend})"


def main() -> None:
    print("Testing CNLSE_1d class")
    for backend in ["CPU", "GPU"]:
        simu_c = CNLSE_1d(
            alpha,
            puiss,
            window,
            n2,
            n12,
            None,
            L,
            NX=N,
            Isat=Isat,
            omega=1,
            backend=backend,
        )
        simu_c.delta_z = 1e-5
        simu_c.puiss2 = 10e-3
        simu_c.n22 = 1e-10
        simu_c.k2 = 2 * np.pi / 795e-9
        E_0 = np.exp(-(simu_c.X**2) / waist**2).astype(PRECISION_COMPLEX)
        V = np.exp(-(simu_c.X**2) / waist2**2).astype(PRECISION_COMPLEX)
        E, V = simu_c.out_field(
            np.array([E_0, V]),
            L,
            verbose=True,
            plot=False,
            precision="single",
        )


if __name__ == "__main__":
    main()
