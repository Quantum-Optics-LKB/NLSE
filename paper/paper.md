---
title: 'NLSE: A Python package to solve the nonlinear Schrödinger equation'
tags:
  - Python
  - NLSE
  - Evolution
  - Nonlinear optics
  - Quantum fluids
authors:
  - name: Tangui Aladjidi
    orcid: 0000-0002-3109-9723
    affiliation: 1 # (Multiple affiliations must be quoted)
    corresponding: true # (This is how to denote the corresponding author)
  - name: Clara Piekarski
    orcid: 0000-0001-6871-6003
    affiliation: 1
  - name: Quentin Glorieux
    orcid: 0000-0003-0903-0233
    affiliation: 1
affiliations:
 - name: Laboratoire Kastler Brossel, Sorbonne University, CNRS, ENS-PSL University, Collège de France; 4 Place Jussieu, 75005 Paris, France
   index: 1
date: 15 March 2024
bibliography: paper.bib
---

# Summary

The nonlinear Schrödinger equation (NLSE) is a general nonlinear equation used to model the propagation of light in nonlinear media.
This equation is mathematically isomorphic to the Gross-Pitaevskii equation (GPE) [@pitaevskij_bose-einstein_2016] describing the evolution of cold atomic ensembles.
Recently, the growing field of quantum fluids of light [@carusotto_quantum_2013] has proven a fruitful testbed for several fundamental quantum and classical phenomena such as superfluidity [@michel_superfluid_2018] or turbulence [@bakerrasooliTurbulentDynamicsTwodimensional2023].
Providing a flexible, modern and performant framework to solve these equations is crucial to model realistic experimental scenarios.

# Statement of need

[`NumPy`](https://numpy.org/doc/stable/) [@harris2020array] is the default package for array operations in Python, which is the language of choice for most physicists.
However, the performance of `NumPy` for large arrays and Fourier transforms quickly bottlenecks homemade implementations of popular solvers like the split-step spectral scheme.

Over the years, there have been several packages striving to provide performant split-step solvers for NLSE type equations.
Here are a few examples:

- [`FourierGPE.jl`](https://github.com/AshtonSBradley/FourierGPE.jl/tree/master) for 1D to 3D Gross-Pitavskii equations in the context of cold atoms in Julia.
- [`GPUE`](https://github.com/GPUE-group/GPUE) [@Schloss2018] for 1D to 3D Gross-Pitaevskii equations accelerated on GPU, in C++ (currently unmaintained).
- [`py-fmas`](https://github.com/omelchert/py-fmas) for 1D NLSE in optical fibers, with a split-step method (currently unmaintained).

With our project, we bring similar performance to C++ and Julia implementations, while striving for accessibility and maintainability by using Python.
Using easy to extend object-oriented classes, users can readily input experimental parameters to quickly model real setups.

# Functionality

`NLSE` harnesses the power of pseudo-spectral schemes to solve efficiently the following general type of equation:

$$i\partial_t\psi = -\frac{1}{2m}\nabla^2\psi + V\psi + g|\psi|^2\psi.$$

To take advantage of the computing power of modern Graphics Processing Units (GPUs) for Fast Fourier Transforms (FFTs), the main workhorse of this code is the [`CuPy`](https://cupy.dev/) [@cupy_learningsys2017]  package that maps [`NumPy`](https://numpy.org/) functionalities onto the GPU using NVIDIA's [`CUDA`](https://developer.nvidia.com/cuda-downloads) API.
It also heavily uses just-in-time compilation using [`Numba`](https://numba.pydata.org/) [@lam2015numba] to optimize performance while having an easily maintainable Python codebase.
Compared to naive NumPy-based CPU implementations, this package provides a 100 to 10000 times speedup for typical sizes \autoref{fig:bench}.
While optimized for the use with GPU, NLSE also provides a performant CPU fallback layer.

The goal of this package is to provide a natural framework to model the propagation of light in nonlinear media or the temporal evolution of Bose gases. It can also be used to model the propagation of light in general.
It supports lossy, nonlinear and nonlocal media.

It provides several classes to model 1D, 2D or 3D propagation, and leverages the array functionalities of `NumPy` like broadcasting to allow scans of physical parameters to most faithfully replicate experimental setups.
The typical output of a simulation run is presented in \autoref{fig:output}.

This code was initially developed in @aladjidiFullOpticalControl2023 and used as the main simulation tool for several publications like @glorieuxHotAtomicVapors2023 and @bakerrasooliTurbulentDynamicsTwodimensional2023.

![Example of an output of the solver. A shearing layer is observed nucleating vortices, that are attracted towards the center due to an attractive potential. The density and phase of the field are represented as well as the momentum distribution  get a quick overview of the state of the field.\label{fig:output}](../img/output.png)

![Left: CPU vs GPU vs NumPy benchmark for 1 cm of propagation (200 evolution steps). Right: Comparison versus the `JuliaGPE.jl` package on the study of vortex precession. \label{fig:bench}](../img/benchmarks.png)

# Reproducibility

The code used to generate the figures can be found in the [`examples`](https://github.com/Quantum-Optics-LKB/NLSE/tree/main/examples) folder of the repository with the [`fig2_turbulence.py`](https://github.com/Quantum-Optics-LKB/NLSE/blob/main/examples/fig1_turbulence.py) and [`fig1_benchmarks.py`](https://github.com/Quantum-Optics-LKB/NLSE/blob/main/examples/fig2_benchmarks.py) scripts. 
Note that you will need Julia installed to run the `JuliaGPE.jl` script.

# Acknowledgements

We acknowledge contributions from Myrann Baker-Rasooli as our most faithful beta tester.

# Authors contribution

TA wrote the original code and is the main maintainer, CP extended the functionalities to include coupled systems. QG supervised the project.

# References
