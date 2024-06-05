## CPU

CPU kernels in the `NLSE` package are responsible for solving the nonlinear Schrödinger equation using CPU resources.

It uses the popular [Numba](https://numba.readthedocs.io/en/stable/user/index.html) library to just-in-time compile array functions.

These functions are not meant to be called directly and instead work as the backend specific implementations for the main classes methods.

Due to the manual indexing involved in the implementations, these kernels **do not support broadcasting** (*for now*).

CPU kernels are suitable for small to medium-sized problems or when GPU resources are not available. As it uses multithreading internally, they will benefit from higher core counts.

::: NLSE.kernels_cpu

## GPU (CUDA)

GPU kernels in the `NLSE` package are responsible for solving the nonlinear Schrödinger equation using GPU resources. 
They utilize the computational power of the graphics processing unit (GPU) to perform the necessary calculations.

These kernels use the [`cupy.fuse`](https://docs.cupy.dev/en/stable/reference/generated/cupy.fuse.html) API to just-in-time compile the array operations to
a single kernel.

The strategy here is to maximize GPU occupancy by grouping operations to a complex enough task such that we are not limited by memory bandwidth.
As with the CPU kernels, the paradigm is to try and mutate arrays in place as much as possible to avoid costly memory transfers.

To this end, most of these kernels have the following signature:
```python
@cp.fuse
def kernel(A: cp.ndarray, *args):
    # do something to A
    A += args[0]
    A *= args[1]
```

::: NLSE.kernels_gpu

## OpenCL (GPU or CPU)

In order to take advantage of the modern hardware, there is experimental support for OpenCL for the `NLSE` class.

The long-term goal is to use a unified interface for GPU and CPU (which is what OpenCL already does).

This uses [`PyOpenCL`](https://github.com/inducer/pyopencl) and its [`array`](https://github.com/inducer/pyopencl/blob/main/pyopencl/array.py) interface to follow the same approach as the other two implementations.

Due to the limited support for OpenCL on newer hardware (mostly Macs), this might not be the winning strategy so use at your own risk ! 😇

::: NLSE.kernels_cl