__BACKEND__ = "GPU"


try:
    import cupy

    __CUPY_AVAILABLE__ = True

except ImportError:
    print("CuPy not available, falling back to CPU BACKEND ...")
    __CUPY_AVAILABLE__ = False
    __BACKEND__ = "CPU"


try:
    import pyopencl

    __PYOPENCL_AVAILABLE__ = True

except ImportError:
    print("PyOpenCL not available, falling back to CPU BACKEND ...")
    __PYOPENCL_AVAILABLE__ = False
    __BACKEND__ = "CPU"
