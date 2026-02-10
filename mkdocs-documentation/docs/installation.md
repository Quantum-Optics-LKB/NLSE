## Installation

First clone the repository:

```bash
git clone https://github.com/Quantum-Optics-LKB/NLSE.git
cd NLSE
```

Then pip install the package:

```bash
pip install .
```

#### If you are using an environement (for example on a shared GPU machine):
1) Create a dedicated env (example)
```
conda create -n nlse -c conda-forge python=3.11 pip -y
```

2) Activate it
```
conda activate nlse
```

3) Install your project
```
pip install .
```

## Requirements

### Supported platforms

This code has been tested on the three main platforms: Linux, MacOs and Windows.

### GPU computing

For optimal speed, this code uses your GPU (graphics card).
For this, you need specific libraries.
For Nvidia cards, you need a [CUDA](https://developer.nvidia.com/cuda-toolkit) install.
For AMD cards, you need a [ROCm](https://rocmdocs.amd.com/en/latest/) install.
Of course, you need to update your graphics driver to take full advantage of these.
In any case we use [CuPy](https://cupy.dev) for the Python interface to these libraries.

**The `cupy` dependency is not included in [`setup.py`](https://github.com/Quantum-Optics-LKB/NLSE/tree/main/setup.py) in order to not break installation on platforms that do not support it !**

### PyFFTW

If the code does not find Cupy, it will fall back to a CPU based implementation that uses the CPU : [PyFFTW](https://pyfftw.readthedocs.io/en/latest/).
To make the best out of your computer, this library is multithreaded.
By default it will use all available threads.
If this is not what you want, you can disable this by setting the variable `pyfftw.config.NUM_THREADS` to a number of your choosing.

**WARNING** : The default flag passed to `FFTW` for planning is `FFTW_PATIENT` which means that the first run of the code can take a long time.
This information is cached so subsequent runs just have to load the plans, removing this computation time.

Other than this, the code relies on these libraries :

- `numba` : for best CPU performance on Intel CPU's, with `icc_rt`
- `pickle`
- `numpy`
- `scipy`
- `matplotlib`

## Tests

Tests are included to check functionalities and benchmark performance.
You can run all tests by executing `pytest` at the root of the package (warning: this might take some time !).
It will test both CPU and GPU backends.

The benchmarks can be run using [`tests/benchmarks.py`](https://github.com/Quantum-Optics-LKB/NLSE/tree/main/tests/benchmarks.py) and compare a "naive" numpy implementation of the main solver loop to our solver.
On a Nvidia RTX4090 GPU and Ryzen 7950X CPU, we test our solver to the following results:
![benchmarks](https://github.com/Quantum-Optics-LKB/NLSE/tree/main/img/benchmarks.png)
