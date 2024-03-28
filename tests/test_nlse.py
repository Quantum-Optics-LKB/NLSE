from NLSE import NLSE
import numpy as np
import pyfftw

if NLSE.__CUPY_AVAILABLE__:
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
L = 10e-3
alpha = 20


def test_build_propagator() -> None:
    for backend in ["CPU", "GPU"]:
        simu = NLSE(
            alpha, puiss, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend=backend
        )
        prop = simu._build_propagator(simu.k)
        if backend == "CPU":
            assert np.allclose(
                prop,
                np.exp(-1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k * simu.delta_z),
            )
        else:
            assert cp.allclose(
                prop,
                np.exp(-1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k * simu.delta_z),
            )


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
            assert len(plans) == 2
            assert isinstance(plans[0], pyfftw.FFTW)
            assert plans[0].output_shape == (N, N)
        else:
            assert len(plans) == 1
            assert isinstance(plans[0], cp.cuda.cufft.PlanNd)
            assert plans[0].shape == (N, N)


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
        if backend == "CPU":
            assert isinstance(out, np.ndarray)
            out /= np.max(np.abs(out))
            A /= np.max(np.abs(A))
            assert np.allclose(out, A)
        else:
            assert isinstance(out, cp.ndarray)
            assert out.shape == (N, N)
            out /= cp.max(cp.abs(out))
            A /= cp.max(cp.abs(A))
            assert cp.allclose(out, A)


def test_send_arrays_to_gpu() -> None:
    if NLSE.__CUPY_AVAILABLE__:
        global alpha
        global n2
        global Isat
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
        simu.propagator = simu._build_propagator(simu.k)
        simu._send_arrays_to_gpu()
        assert isinstance(simu.propagator, cp.ndarray)
        assert isinstance(simu.V, cp.ndarray)
        assert isinstance(simu.alpha, cp.ndarray)
        assert isinstance(simu.n2, cp.ndarray)
        assert isinstance(simu.I_sat, cp.ndarray)
    else:
        pass


def test_retrieve_arrays_from_gpu() -> None:
    pass


def test_split_step() -> None:
    pass


def test_plot_field() -> None:
    pass


def test_out_field() -> None:
    pass


# for integration testing
def main():
    print("Testing NLSE class")
    for backend in ["CPU", "GPU"]:
        simu = NLSE(
            alpha, puiss, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend=backend
        )
        simu.delta_z = 1e-4
        E_0 = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(PRECISION_COMPLEX)
        simu.V = -1e-4 * np.exp(-(simu.XX**2 + simu.YY**2) / waist2**2).astype(
            PRECISION_COMPLEX
        )
        simu.out_field(E_0, L, verbose=True, plot=False, precision="single")


if __name__ == "__main__":
    main()
