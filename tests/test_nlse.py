from NLSE import NLSE
import numpy as np

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
        assert prop == np.exp(
            -1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k * simu.delta_z
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


def test_prepare_output_array() -> None:
    pass


def test_send_arrays_to_gpu() -> None:
    pass


def test_retrieve_arrays_from_gpu() -> None:
    pass


def test_split_step() -> None:
    pass


def test_plot_field() -> None:
    pass


def test_out_field() -> None:
    pass


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
