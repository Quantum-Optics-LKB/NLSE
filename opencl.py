import pyopencl as cl
from pyopencl import array as cla
from pyopencl import clmath
from pyvkfft import opencl, cuda
import numpy as np
import cupy as cp
from cupyx.scipy import fftpack
import time

queue = cl.CommandQueue(cl.create_some_context(interactive=False))
print(queue.device.name)
N = 2048
nb = 100
a = cla.zeros(queue=queue, shape=(N, N), dtype=np.complex64)
b = cla.zeros(queue=queue, shape=(N, N), dtype=np.complex64)
print(a.real)
a_np = np.zeros((N, N), dtype=np.complex64)
b_np = np.zeros((N, N), dtype=np.complex64)
a_cp = cp.zeros((N, N), dtype=np.complex64)
b_cp = cp.zeros((N, N), dtype=np.complex64)
t0 = time.perf_counter()
for i in range(nb):
    a *= clmath.exp(1j * b)
queue.finish()
t = time.perf_counter() - t0
print(f"OpenCL: {t/nb*1e3:.3f} ms")
t0 = time.perf_counter()
for i in range(nb):
    a_np *= np.exp(1j * b_np)
t = time.perf_counter() - t0
print(f"Numpy: {t/nb*1e3:.3f} ms")
t0 = cp.cuda.Event()
t1 = cp.cuda.Event()
t0.record()
for i in range(nb):
    a_cp *= cp.exp(1j * b_cp)
t1.record()
t1.synchronize()
print(f"Cupy: {cp.cuda.get_elapsed_time(t0, t1)/nb:.3f} ms")
app = opencl.VkFFTApp(a.shape, a.dtype, queue=queue, ndim=a.ndim, inplace=True)
t0 = time.perf_counter()
for i in range(nb):
    app.fft(a, a)
    app.ifft(a, a)
queue.finish()
t = time.perf_counter() - t0
print(f"OpenCL VkFFT: {t/nb*1e3:.3f} ms")
t0 = time.perf_counter()
for i in range(nb):
    np.fft.fft2(a_np)
    np.fft.ifft2(a_np)
t = time.perf_counter() - t0
print(f"Numpy FFT: {t/nb*1e3:.3f} ms")
plan = fftpack.get_fft_plan(a_cp)
t0 = cp.cuda.Event()
t1 = cp.cuda.Event()
t0.record()
for i in range(nb):
    plan.fft(a_cp, a_cp, direction=cp.cuda.cufft.CUFFT_FORWARD)
    plan.fft(a_cp, a_cp, direction=cp.cuda.cufft.CUFFT_INVERSE)
t1.record()
t1.synchronize()
print(f"Cupy FFT: {cp.cuda.get_elapsed_time(t0, t1)/nb:.3f} ms")
app = cuda.VkFFTApp(
    a_cp.shape,
    a_cp.dtype,
    stream=cp.cuda.get_current_stream(),
    ndim=a.ndim,
    inplace=True,
)
t0 = cp.cuda.Event()
t1 = cp.cuda.Event()
t0.record()
for i in range(nb):
    app.fft(a_cp, a_cp)
    app.ifft(a_cp, a_cp)
t1.record()
t1.synchronize()
print(f"CUDA VkFFT: {cp.cuda.get_elapsed_time(t0, t1)/nb:.3f} ms")
