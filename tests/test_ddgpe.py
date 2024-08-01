import numpy as np

from NLSE import DDGPE

if DDGPE.__CUPY_AVAILABLE__:
    import cupy as cp

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

N = 256
T = 1
h_bar = 0.654  # (meV*ps)
omega = 5.07 / h_bar  # (meV/h_bar) linear coupling (Rabi split)
omega_exc = 1484.44 / h_bar  # (meV/h_bar) exciton energy
omega_cav = 1482.76 / h_bar  # (meV/h_bar) cavity energy
detuning = 0.17 / h_bar
k_z = 27
gamma = 0 * 0.07 / h_bar
waist = 50
window = 256
g = 1e-2 / h_bar
puiss = detuning / g


def test_prepare_output_array() -> None:
    for backend in ["CPU", "GPU"]:
        simu = DDGPE(
            gamma,
            puiss,
            window,
            g,
            omega,
            T,
            omega_exc,
            omega_cav,
            detuning,
            k_z,
            NX=N,
            NY=N,
            backend=backend,
        )
        if backend == "CPU":
            A = np.ones((2, N, N), dtype=PRECISION_COMPLEX)
        elif backend == "GPU" and DDGPE.__CUPY_AVAILABLE__:
            A = cp.ones((2, N, N), dtype=PRECISION_COMPLEX)
        out, out_sq = simu._prepare_output_array(A, normalize=False)
        assert (
            out.flags.c_contiguous
        ), f"Output array is not C-contiguous. (Backend {backend})"
        assert (
            out_sq.flags.c_contiguous
        ), f"Output array is not C-contiguous. (Backend {backend})"
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
        elif backend == "GPU" and DDGPE.__CUPY_AVAILABLE__:
            assert isinstance(
                out, cp.ndarray
            ), f"Ouptut array type does not match backend. (Backend {backend})"
            out /= cp.max(cp.abs(out))
            A /= cp.max(cp.abs(A))
            assert cp.allclose(
                out, A
            ), f"Output array does not match input array. (Backend {backend})"


def test_send_arrays_to_gpu() -> None:
    if DDGPE.__CUPY_AVAILABLE__:
        omega_exc = 1484.44 / h_bar
        omega_cav = 1482.76 / h_bar
        detuning = 0.17 / h_bar
        k_z = 27
        gamma = 0 * 0.07 / h_bar
        omega = 5.07 / h_bar
        g = 1e-2 / h_bar
        V = np.random.random((N, N)) + 1j * np.random.random((N, N))
        omega_cav = np.repeat(omega_cav, 2)
        omega_cav = omega_cav[..., np.newaxis, np.newaxis, np.newaxis]
        omega_exc = np.repeat(omega_exc, 2)
        omega_exc = omega_exc[..., np.newaxis, np.newaxis, np.newaxis]
        gamma = np.repeat(gamma, 2)
        gamma = gamma[..., np.newaxis, np.newaxis, np.newaxis]
        omega = np.repeat(omega, 2)
        omega = omega[..., np.newaxis, np.newaxis, np.newaxis]
        g = np.repeat(g, 2)
        g = g[..., np.newaxis, np.newaxis, np.newaxis]
        simu = DDGPE(
            gamma,
            puiss,
            window,
            g,
            omega,
            T,
            omega_exc,
            omega_cav,
            detuning,
            k_z,
            V=V,
            NX=N,
            NY=N,
            backend="GPU",
        )
        simu.propagator = simu._build_propagator()
        simu._send_arrays_to_gpu()
        assert isinstance(
            simu.propagator, cp.ndarray
        ), "propagator is not a cp.ndarray. (Backend GPU)"
        assert isinstance(
            simu.V, cp.ndarray
        ), "V is not a cp.ndarray. (Backend GPU)"
        assert isinstance(
            simu.gamma, cp.ndarray
        ), "gamma is not a cp.ndarray. (Backend GPU)"
        assert isinstance(
            simu.g, cp.ndarray
        ), "g is not a cp.ndarray. (Backend GPU)"
        assert isinstance(
            simu.omega, cp.ndarray
        ), "omega is not a cp.ndarray. (Backend GPU)"
        assert isinstance(
            simu.omega_cav, cp.ndarray
        ), "omega cav is not a cp.ndarray. (Backend GPU)"
        assert isinstance(
            simu.omega_exc, cp.ndarray
        ), "omega exc is not a cp.ndarray. (Backend GPU)"
    else:
        pass


def test_retrieve_arrays_from_gpu() -> None:
    if DDGPE.__CUPY_AVAILABLE__:
        omega_exc = 1484.44 / h_bar
        omega_cav = 1482.76 / h_bar
        detuning = 0.17 / h_bar
        k_z = 27
        gamma = 0 * 0.07 / h_bar
        g = 1e-2 / h_bar
        omega = 5.07 / h_bar
        V = np.random.random((N, N)) + 1j * np.random.random((N, N))
        omega_cav = np.repeat(omega_cav, 2)
        omega_cav = omega_cav[..., np.newaxis, np.newaxis, np.newaxis]
        omega_exc = np.repeat(omega_exc, 2)
        omega_exc = omega_exc[..., np.newaxis, np.newaxis, np.newaxis]
        gamma = np.repeat(gamma, 2)
        gamma = gamma[..., np.newaxis, np.newaxis, np.newaxis]
        omega = np.repeat(omega, 2)
        omega = omega[..., np.newaxis, np.newaxis, np.newaxis]
        g = np.repeat(g, 2)
        g = g[..., np.newaxis, np.newaxis, np.newaxis]
        simu = DDGPE(
            gamma,
            puiss,
            window,
            g,
            omega,
            T,
            omega_exc,
            omega_cav,
            detuning,
            k_z,
            V=V,
            NX=N,
            NY=N,
            backend="GPU",
        )
        simu.propagator = simu._build_propagator()
        simu._send_arrays_to_gpu()
        simu._retrieve_arrays_from_gpu()
        assert isinstance(
            simu.propagator, np.ndarray
        ), "propagator is not a np.ndarray. (Backend GPU)"
        assert isinstance(
            simu.gamma, np.ndarray
        ), "gamma is not a np.ndarray. (Backend GPU)"
        assert isinstance(
            simu.g, np.ndarray
        ), "g is not a np.ndarray. (Backend GPU)"
        assert isinstance(
            simu.omega, np.ndarray
        ), "omega is not a np.ndarray. (Backend GPU)"
        assert isinstance(
            simu.omega_cav, np.ndarray
        ), "omega cav is not a np.ndarray. (Backend GPU)"
        assert isinstance(
            simu.omega_exc, np.ndarray
        ), "omega exc is not a np.ndarray. (Backend GPU)"
    else:
        pass


def test_build_propagator() -> None:
    for backend in ["GPU", "CPU"]:
        simu = DDGPE(
            gamma,
            puiss,
            window,
            g,
            omega,
            T,
            omega_exc,
            omega_cav,
            detuning,
            k_z,
            NX=N,
            NY=N,
        )
        prop = simu._build_propagator()
        assert np.allclose(
            prop[0],
            np.exp(
                -1j
                * (simu.omega_exc * (1 + 0 * simu.Kxx**2) - simu.omega_pump)
                * simu.delta_z
            ),
        ), f"Propagator1 is wrong. (Backend {backend})"
        assert np.allclose(
            prop[1],
            np.exp(
                -1j
                * (simu.omega_exc * (1 + 0 * simu.Kxx**2) - simu.omega_pump)
                * simu.delta_z
            ),
        ), f"Propagator2 is wrong. (Backend {backend})"


def test_take_components() -> None:
    for backend in ["CPU", "GPU"]:
        simu = DDGPE(
            gamma,
            puiss,
            window,
            g,
            omega,
            T,
            omega_exc,
            omega_cav,
            detuning,
            k_z,
            NX=N,
            NY=N,
            backend="GPU",
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
        assert (
            A1.shape[0] == 3
        ), f"A1 has wrong first dimensions. (Backend {backend})"
        assert (
            A2.shape[0] == 3
        ), f"A2 has wrong first dimensions. (Backend {backend})"


def callback_sample(
    simu: DDGPE,
    A: np.ndarray,
    z: float,
    i: int,
    save_every: int,
    sample1: list,
    sample2: list,
    sample3: list,
) -> None:
    if i % save_every == 0:
        sum_exc = (A[..., 0, :, :].real ** 2 + A[..., 0, :, :].imag ** 2).sum()
        sum_cav = (A[..., 1, :, :].real ** 2 + A[..., 1, :, :].imag ** 2).sum()
        sum_tot = sum_exc + sum_cav
        sample1[i // save_every] = sum_exc
        sample2[i // save_every] = sum_cav
        sample3[i // save_every] = sum_tot


def turn_on(
    F_laser_t: np.ndarray,
    time: np.ndarray,
    t_up=10,
):
    """A function to turn on the pump more or less adiabatically

    Args:
        F_laser_t (np.ndarray): self.F_pump_t as defined in class ggpe,
          cp.ones((int(self.t_max//self.dt)), dtype=cp.complex64)
        time (np.ndarray):  array with the value of the time at each discretized
          step.
        t_up (int, optional): time taken to reach the maximum intensity (=F).
          Defaults to 10.
    """
    F_laser_t[time < t_up] = np.exp(
        -1 * (time[time < t_up] - t_up) ** 2 / (t_up / 2) ** 2
    )
    F_laser_t[time >= t_up] = 1


def test_out_field() -> None:
    for backend in ["CPU", "GPU"]:
        simu = DDGPE(
            gamma,
            puiss,
            window,
            g,
            omega,
            T,
            omega_exc,
            omega_cav,
            detuning,
            k_z,
            NX=N,
            NY=N,
            backend=backend,
        )
        simu.delta_z = 0.1 / 32  # need to be adjusted automatically
        time = np.arange(
            0, T + simu.delta_z, step=simu.delta_z, dtype=np.float32
        )
        save_every = 1  # np.argwhere(time == 1)[0][0]
        sample1 = np.zeros(time.size // save_every, dtype=np.float32)
        sample2 = np.zeros(time.size // save_every, dtype=np.float32)
        sample3 = np.zeros(time.size // save_every, dtype=np.float32)
        E0 = np.zeros((2, simu.NY, simu.NX), dtype=np.complex64)
        F_pump = 0
        F_pump_r = F_pump * np.exp(
            -((simu.XX**2 + simu.YY**2) / waist**2)
        ).astype(np.complex64)
        F_pump_t = np.zeros(time.shape, dtype=np.complex64)
        F_probe = 0
        F_probe_r = F_probe * np.exp(
            -((simu.XX**2 + simu.YY**2) / waist**2)
        ).astype(np.complex64)
        F_probe_t = np.zeros(time.shape, dtype=np.complex64)
        turn_on(F_pump_t, time, t_up=20)
        callback = [callback_sample]
        if backend == "CPU":
            callback_args = [
                [
                    F_pump_r,
                    F_pump_t,
                    F_probe_r,
                    F_probe_t,
                ],
                [save_every, sample1, sample2, sample3],
            ]
        if backend == "GPU" and DDGPE.__CUPY_AVAILABLE__:
            callback_args = [
                [
                    cp.asarray(F_pump_r),
                    F_pump_t,
                    cp.asarray(F_probe_r),
                    F_probe_t,
                ],
                [save_every, sample1, sample2, sample3],
            ]
        simu.out_field(
            E0,
            T,
            simu.laser_excitation,
            plot=False,
            callback=callback,
            callback_args=callback_args,
        )
        # test stationarity here
