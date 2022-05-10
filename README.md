# NLSE

A small utility to easily simulate all sorts of non linear Schrödinger equation. It uses a [split-step spectral scheme](https://en.wikipedia.org/wiki/Split-step_method) to solve the equation.

## Requirements

### GPU computing

For optimal speed, this code uses your GPU (graphics card). For this, you need specific libraries. For Nvidia cards, you need a [CUDA](https://developer.nvidia.com/cuda-toolkit) install. For AMD cards, you need a [ROCm](https://rocmdocs.amd.com/en/latest/) install. Of course, you need to update your graphics driver to take full advantage of these. In any case we use [CuPy](cupy.dev) for the Python interface to these libraries.

### PyFFTW

If the code does not find Cupy, it will fall back to a CPU based implementation that uses the CPU : [PyFFTW](https://pyfftw.readthedocs.io/en/latest/). To make the best out of your computer, this library is multithreaded. By default it will use all available threads. If this is not what you want, you can disable this by setting the variable `pyfftw.config.NUM_THREADS` to a number of your choosing.

Other than this, the code relies on these libraries :
- `pickle`
- `numpy`
- `scipy`
- `matplotlib`

## How does it work ?

The code offers to solve a typical [non linear Schrödinger](https://en.wikipedia.org/wiki/Nonlinear_Schr%C3%B6dinger_equation) equation of the type :

<!-- $i\partial_{t}\psi = -\frac{1}{2}\nabla^2\psi+V\psi+g|\psi|^2\psi$ --> 
<img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=\bbox[white]{i%5Cpartial_%7Bt%7D%5Cpsi%20%3D%20-%5Cfrac%7B1%7D%7B2%7D%5Cnabla%5E2%5Cpsi%2BV%5Cpsi%2Bg%7C%5Cpsi%7C%5E2%5Cpsi}">.

In this particular instance, we solve in the formalism of the propagation of light in a non linear medium, such that the exact equation solved is :
<!-- $i\partial_{z}E = -\frac{1}{2k_0}\nabla_{\perp}^2 E-\frac{k_0}{2}\delta n(r) E - n_2 k_0|E|^2E$ --> 
<img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=\bbox[white]{i%5Cpartial_%7Bz%7DE%20%3D%20-%5Cfrac%7B1%7D%7B2k_0%7D%5Cnabla_%7B%5Cperp%7D%5E2%20E-%5Cfrac%7Bk_0%7D%7B2%7D%5Cdelta%20n(r)%20E%20-%20n_2%20k_0%7CE%7C%5E2E}">.

Here, the constants are defined as followed :
- <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=\bbox[white]{k_0}"> : is the electric field wavenumber in $m^{-1}$
- <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=\bbox[white]{%5Cdelta%20n(r)}"> : the "potential" i.e a local change in linear index of refraction. Dimensionless.
- <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=\bbox[white]{n_2}"> : the non linear coefficient in $W/m^2$.
  
Theses coefficients are defined at the instantiation of the `NLSE` class.\
The `E_out` method is the main function of the code that propagates the field for an arbitrary distance from an initial state `E_0` from z=0 (assumed to be the begining of the non linear medium) up to a specified distance z.\
A convenience function `slm` allows to simulate the application of a phase mask or intensity mask using a SLM or DMD of pixel pitch `d_slm` of the screen. This function resizes the picture to match the simulation pixel.