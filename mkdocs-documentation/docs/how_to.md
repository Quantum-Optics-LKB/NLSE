# How to

## Benchmarking

If you want to benchmark the performance of the code you could use the function [`examples/benchmarks.py`](../examples/benchmarks.py) which is a simple script that runs the code multiple times and calculates the average time taken to run the code.

## Comparison with other projects

Compared to other related software like [`FourierGPE.jl`](https://github.com/AshtonSBradley/FourierGPE.jl/tree/master), `NLSE` is much faster as it uses a much simpler solver (the latter uses the awesome [`DifferentialEquations.jl`](https://docs.sciml.ai/DiffEqDocs/stable/)).

Of course this gives you less control over the numerical accuracy of your model, however we think it's worth it since performance often limits the possible physical scenarii.

A comparison between the two projects can be found in [`comparison_juliaGPE.py`](../examples/comparison_juliaGPE.py) in our examples.

## Testing

The tests have been written to work in tandem with [`pytest`](https://docs.pytest.org/en/8.2.x/) and are run on each repository push using a GitHub action.

These tests essentially check for the physical definitions of all the class methods.

Here is an example of such a test file:

```python
from NLSE import NLSE
import numpy as np
import pyfftw
from scipy.constants import c, epsilon_0

if NLSE.__CUPY_AVAILABLE__:
    import cupy as cp
    from pyvkfft.cuda import VkFFTApp as VkFFTApp_cuda
if NLSE.__PYOPENCL_AVAILABLE__:
    import pyopencl.array as cla
    from pyvkfft.opencl import VkFFTApp as VkFFTApp_cl
PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32


N = 2048
n2 = -1.6e-9
waist = 2.23e-3
waist2 = 70e-6
window = 4 * waist
power = 1.05
Isat = 10e4  # saturation intensity in W/m^2
L = 10e-3
alpha = 20


def test_build_propagator() -> None:
    for backend in ["CPU", "GPU"]:
        simu = NLSE(
            alpha, power, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend=backend
        )
        prop = simu._build_propagator()
        assert np.allclose(
            prop,
            np.exp(-1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k * simu.delta_z),
        ), f"Propagator is wrong. (Backend {backend})"


def test_build_fft_plan() -> None:
    for backend in ["CPU", "GPU"]:
        simu = NLSE(
            alpha, power, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend=backend
        )
        if backend == "CPU" or backend == "CL":
            A = np.random.random((N, N)) + 1j * np.random.random((N, N))
        elif backend == "GPU" and NLSE.__CUPY_AVAILABLE__:
            A = cp.random.random((N, N)) + 1j * cp.random.random((N, N))
        A = A.astype(PRECISION_COMPLEX)
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
                plans[0], VkFFTApp_cuda
            ), f"Plan type is wrong. (Backend {backend})"
            assert plans[0].shape0 == (
                N,
                N,
            ), f"Plan shape is wrong. (Backend {backend})"
        elif backend == "CL" and NLSE.__PYOPENCL_AVAILABLE__:
            assert len(plans) == 1, f"Number of plans is wrong. (Backend {backend})"
            assert isinstance(
                plans[0], VkFFTApp_cl
            ), f"Plan type is wrong. (Backend {backend})"
            assert plans[0].shape0 == (
                N,
                N,
            ), f"Plan shape is wrong. (Backend {backend})"


def test_prepare_output_array() -> None:
    for backend in ["CPU", "GPU"]:
        simu = NLSE(
            alpha, power, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend=backend
        )
        if backend == "CPU" or backend == "CL":
            A = np.random.random((N, N)) + 1j * np.random.random((N, N))
        elif backend == "GPU" and NLSE.__CUPY_AVAILABLE__:
            A = cp.random.random((N, N)) + 1j * cp.random.random((N, N))
        A = A.astype(PRECISION_COMPLEX)
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
        if simu.backend == "GPU" and NLSE.__CUPY_AVAILABLE__ or simu.backend == "CPU":
            integral = (
                (out.real * out.real + out.imag * out.imag)
                * simu.delta_X
                * simu.delta_Y
            ).sum(axis=simu._last_axes)
        if backend == "CL" and NLSE.__PYOPENCL_AVAILABLE__:
            arr = out.real * out.real + out.imag * out.imag
            arr *= simu.delta_X * simu.delta_Y
            integral = cla.sum(
                arr,
                dtype=arr.dtype,
                queue=simu._cl_queue,
            )
            integral = integral.get()
        integral *= c * epsilon_0 / 2
        assert np.allclose(
            integral, simu.power
        ), f"Normalization failed. (Backend {simu.backend}) : {integral} != {simu.power}"
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
            alpha, power, window, n2, V, L, NX=N, NY=N, Isat=Isat, backend="GPU"
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
            alpha, power, window, n2, V, L, NX=N, NY=N, Isat=Isat, backend="GPU"
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
            alpha, power, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend=backend
        )
        simu.delta_z = 0
        simu.propagator = simu._build_propagator()
        E = np.ones((N, N), dtype=PRECISION_COMPLEX)
        A, A_sq = simu._prepare_output_array(E, normalize=False)
        simu.plans = simu._build_fft_plan(A)
        simu.propagator = simu._build_propagator()
        if backend == "GPU" and NLSE.__CUPY_AVAILABLE__:
            E = cp.asarray(E)
        if (
            backend == "GPU"
            and NLSE.__CUPY_AVAILABLE__
            or backend == "CL"
            and NLSE.__PYOPENCL_AVAILABLE__
        ):
            simu._send_arrays_to_gpu()
        simu.split_step(
            A, A_sq, simu.V, simu.propagator, simu.plans, precision="double"
        )
        if backend == "CPU":
            assert np.allclose(
                A, np.ones((N, N), dtype=PRECISION_COMPLEX)
            ), f"Split step is not unitary. (Backend {backend})"
        elif backend == "GPU" and NLSE.__CUPY_AVAILABLE__:
            assert cp.allclose(
                A, cp.ones((N, N), dtype=PRECISION_COMPLEX)
            ), f"Split step is not unitary. (Backend {backend})"


# tests for convergence of the solver : the norm of the field should be conserved
def test_out_field() -> None:
    E = np.ones((N, N), dtype=PRECISION_COMPLEX)
    for backend in ["CPU", "GPU"]:
        simu = NLSE(
            0, power, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend=backend
        )
        E = simu.out_field(E, L, verbose=False, plot=False, precision="single")
        norm = np.sum(np.abs(E) ** 2 * simu.delta_X * simu.delta_Y)
        norm *= c * epsilon_0 / 2
        assert E.shape == (N, N), f"Output array has wrong shape. (Backend {backend})"
        assert np.allclose(
            norm, simu.power, rtol=1e-4
        ), f"Norm not conserved. (Backend {backend})"
``````

## Examples

Minimum working examples for each class can be found in the [examples](../examples) folder of the repo, as well as the code needed to benchmark performance on your machine and comparisons with other NLSE/GPE solvers.
