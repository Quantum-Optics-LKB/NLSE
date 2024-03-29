from NLSE import NLSE_1d
import numpy as np
from scipy.constants import c, epsilon_0

if NLSE_1d.__CUPY_AVAILABLE__:
    import cupy as cp
PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32
N = 2048
n2 = -1.6e-9
waist = 2.23e-3
waist2 = 70e-6
window = 4 * waist
puiss = 1.05
Isat = 10e4  # saturation intensity in W/m^2
L = 1e-3
alpha = 20


def test_build_propagator() -> None:
    for backend in ["CPU", "GPU"]:
        simu = NLSE_1d(
            alpha, puiss, window, n2, None, L, NX=N, Isat=Isat, backend=backend
        )
        prop = simu._build_propagator(simu.k)
        assert np.allclose(
            prop, np.exp(-1j * 0.5 * (simu.Kx**2) / simu.k * simu.delta_z)
        ), "Propagator is wrong."


def test_prepare_output_array() -> None:
    for backend in ["CPU", "GPU"]:
        simu = NLSE_1d(
            alpha,
            puiss,
            window,
            n2,
            None,
            L,
            NX=N,
            Isat=Isat,
            backend=backend,
        )
        if backend == "CPU":
            A = np.ones(N, dtype=PRECISION_COMPLEX)
        elif backend == "GPU" and NLSE_1d.__CUPY_AVAILABLE__:
            A = cp.ones(N, dtype=PRECISION_COMPLEX)
        out = simu._prepare_output_array(A, normalize=True)
        integral = (np.abs(out) ** 2 * simu.delta_X).sum() ** 2
        integral *= c * epsilon_0 / 2
        assert np.allclose(integral, simu.puiss), "Normalization failed."
        assert out.shape == (N, N), "Output array has wrong shape."
        if backend == "CPU":
            assert isinstance(
                out, np.ndarray
            ), "Output array type does not match backend."
            out /= np.max(np.abs(out))
            A /= np.max(np.abs(A))
            assert np.allclose(out, A), "Output array does not match input array."
        elif backend == "GPU" and NLSE_1d.__CUPY_AVAILABLE__:
            assert isinstance(
                out, cp.ndarray
            ), "Output array type does not match backend."
            out /= cp.max(cp.abs(out))
            A /= cp.max(cp.abs(A))
            assert cp.allclose(out, A), "Output array does not match input array."


def test_out_field() -> None:
    for backend in ["CPU", "GPU"]:
        simu = NLSE_1d(0, puiss, window, n2, None, L, NX=N, Isat=Isat, backend=backend)
        E0 = np.ones(N, dtype=PRECISION_COMPLEX)
        A = simu.out_field(E0, L, verbose=False, plot=True, precision="single")
        rho = A.real * A.real + A.imag * A.imag
        norm = (rho * simu.delta_X).sum(axis=simu._last_axes) ** 2
        norm *= c * epsilon_0 / 2
        print(norm)
        assert A.shape == (N,), "Output array has wrong shape."
        assert np.allclose(norm, puiss, rtol=1e-4), "Normalization failed."


def main():
    print("Testing NLSE_1d class")
    for backend in ["CPU", "GPU"]:
        simu = NLSE_1d(
            alpha, puiss, window, n2, None, L, NX=N, Isat=Isat, backend=backend
        )
        simu.delta_z = 1e-5
        simu.puiss2 = 10e-3
        simu.n22 = 1e-10
        simu.k2 = 2 * np.pi / 795e-9
        E_0 = np.exp(-(simu.X**2) / waist**2).astype(PRECISION_COMPLEX)
        simu.V = -1e-4 * np.exp(-(simu.X**2) / waist2**2).astype(PRECISION_COMPLEX)
        simu.out_field(E_0, L, verbose=True, plot=False, precision="single")


if __name__ == "__main__":
    main()
