__BACKEND__ = "GPU"


try:
    import cupy

    __CUPY_AVAILABLE__ = True

except ImportError:
    print("CuPy not available, falling back to CPU BACKEND ...")
    __CUPY_AVAILABLE__ = False
    __BACKEND__ = "CPU"


try:
    # for OpenCL backend you need to install OpenCL first
    # sudo apt install intel-opencl-icd opencl-headers ocl-icd-opencl-dev
    # or for AMD
    # sudo apt install opencl-headers ocl-icd-opencl-dev
    import pyopencl

    __PYOPENCL_AVAILABLE__ = True

except ImportError:
    print("PyOpenCL not available, falling back to CPU BACKEND ...")
    __PYOPENCL_AVAILABLE__ = False
    __BACKEND__ = "CPU"
