from NLSE import CNLSE, CNLSE_1d, NLSE, NLSE_1d, GPE
import numpy as np
from scipy.constants import c, epsilon_0

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

# TODO: Add assertions to check the norm


def test_nonlocality():
    N = 2048
    n2 = -1e-9
    n12 = -1e-10
    waist = 2.23e-3
    waist2 = 70e-6
    window = 4 * waist
    power = 1.05
    Isat = 10e4  # saturation intensity in W/m^2
    L = 1e-3
    alpha = 0
    nl_length = 60e-6
    for backend in ["CPU", "GPU"]:
        simu_c_1d = CNLSE_1d(
            alpha,
            power,
            window,
            n2,
            n12,
            None,
            L,
            NX=N,
            Isat=Isat,
            nl_length=nl_length,
            backend=backend,
        )
        simu_c_2d = CNLSE(
            alpha,
            power,
            window,
            n2,
            n12,
            None,
            L,
            NX=N,
            NY=N,
            Isat=Isat,
            nl_length=nl_length,
            backend=backend,
        )
        simu_1d = NLSE_1d(
            alpha,
            power,
            window,
            n2,
            None,
            L,
            NX=N,
            Isat=Isat,
            nl_length=nl_length,
            backend=backend,
        )
        simu_2d = NLSE(
            alpha,
            power,
            window,
            n2,
            None,
            L,
            NX=N,
            NY=N,
            Isat=Isat,
            nl_length=nl_length,
            backend=backend,
        )
        simu_gpe = GPE(
            alpha,
            power,
            window,
            n2,
            None,
            L,
            NX=N,
            NY=N,
            sat=Isat,
            nl_length=nl_length,
            backend=backend,
        )
        simu_c_1d.delta_z = 1e-5
        simu_c_1d.puiss2 = 10e-3
        simu_c_1d.n22 = 1e-10
        simu_c_1d.k2 = 2 * np.pi / 795e-9
        simu_c_2d.delta_z = simu_c_1d.delta_z
        simu_c_2d.puiss2 = simu_c_1d.puiss2
        simu_c_2d.n22 = simu_c_1d.n22
        simu_c_2d.k2 = simu_c_1d.k2
        simu_1d.delta_z = simu_c_1d.delta_z
        simu_2d.delta_z = simu_c_1d.delta_z
        E_0 = np.exp(-(simu_c_2d.XX**2 + simu_c_2d.YY**2) / waist**2).astype(
            PRECISION_COMPLEX
        )
        V0 = np.exp(-(simu_c_2d.XX**2 + simu_c_2d.YY**2) / waist2**2).astype(
            PRECISION_COMPLEX
        )
        E, V = simu_c_1d.out_field(
            np.array([E_0[N // 2, :], V0[N // 2, :]]),
            L,
            verbose=True,
            plot=False,
            precision="single",
        )
        arr = E.real * E.real + E.imag * E.imag
        arr *= c * epsilon_0 / 2 * simu_c_1d.delta_X**2
        norm = arr.sum(simu_c_1d._last_axes)
        assert np.allclose(
            norm, simu_c_1d.power, rtol=1e-3
        ), f"CNLSE_1d : Norm is not conserved ! (Backend {backend})"
        E = simu_1d.out_field(
            E_0[N // 2, :],
            L,
            verbose=True,
            plot=False,
            precision="single",
        )
        arr = E.real * E.real + E.imag * E.imag
        arr *= c * epsilon_0 / 2 * simu_1d.delta_X**2
        norm = arr.sum(simu_c_1d._last_axes)
        assert np.allclose(
            norm, simu_1d.power, rtol=1e-3
        ), f"NLSE_1d : Norm is not conserved ! (Backend {backend})"
        E, V = simu_c_2d.out_field(
            np.array([E_0, V0]),
            L,
            verbose=True,
            plot=False,
            precision="single",
        )
        arr = E.real * E.real + E.imag * E.imag
        arr *= c * epsilon_0 / 2 * simu_c_2d.delta_X * simu_c_2d.delta_Y
        norm = arr.sum(simu_c_2d._last_axes)
        assert np.allclose(
            norm, simu_c_2d.power, rtol=1e-3
        ), f"CNLSE : Norm is not conserved ! (Backend {backend})"
        E = simu_2d.out_field(
            E_0,
            L,
            verbose=True,
            plot=False,
            precision="single",
        )
        arr = E.real * E.real + E.imag * E.imag
        arr *= c * epsilon_0 / 2 * simu_2d.delta_X * simu_2d.delta_Y
        norm = arr.sum(simu_2d._last_axes)
        assert np.allclose(
            norm, simu_2d.power, rtol=1e-3
        ), f"NLSE : Norm is not conserved ! (Backend {backend})"
        E = simu_gpe.out_field(
            E_0,
            L,
            verbose=True,
            plot=False,
            precision="single",
        )
        arr = E.real * E.real + E.imag * E.imag
        arr *= simu_gpe.delta_X * simu_gpe.delta_Y
        norm = arr.sum(simu_gpe._last_axes)
        assert np.allclose(
            norm, simu_gpe.N, rtol=1e-3
        ), f"CNLSE : Norm is not conserved ! (Backend {backend})"
