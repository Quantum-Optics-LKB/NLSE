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
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Clara Piekarski
    orcid: 0000-0001-6871-6003
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Quentin Glorieux
    orcid: 0000-0003-0903-0233
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
affiliations:
 - name: Laboratoire Kastler Brossel, Sorbonne University, CNRS, ENS-PSL University, Collège de France; 4 Place Jussieu, 75005 Paris, France
   index: 1
date: 15 March 2024
bibliography: paper.bib

# Summary

The non-linear Shcrödinger equation (NLSE) is a general non-linear equation allowing to model the propagation of light in non-linear media.
This equation is mathematically isomorphic to the Gross-Pitaevskii equation (GPE) describing the evolution of cold atomic ensembles.
Providing a flexible, modern and performant framework to solve these equations is a crucial need to model realistic experimental scenarii.

# Statement of need

`NLSE` harnesses the power of pseudo-spectral schemes in order to solve efficiently the following general type of equation:
$$
i\partial_t\psi = -\frac{1}{2m}\nabla^2\psi + V\psi + g|\psi|^2\psi.
$$
In order to take advantge of the computing power of modern Graphical Processing Units (GPU) for Fast Fourier Transforms (FFT), the main workhorse of this code is the [`cupy`](https://cupy.dev/) package that maps [`numpy`](https://numpy.org/) functionalities onto the GPU using NVIDIA's [`CUDA`](https://developer.nvidia.com/cuda-downloads) API.
It also heavily uses just-in-time compilation using [`numba`](https://numba.pydata.org/) in order to optimize performance while having an easily maintainable Python codebase.
Compared to CPU implementations, this package provides a 10 to 100 times speedup for typical sizes.
While optimized for the use with GPU, it also provides a performant CPU fallback layer.

The goal of this package is to provide a natural framework for all physicists wishing to model the propagation of light in non-linear media or the temporal evolution of Bose gases. It can also be used to model the propagation of light in general.

It provides several classes to model 1D, 2D propagation, and leverages the array functionalities of `numpy` like broadcasting in order to allow easily scans of physical parameters to most closely replicate experimental setups.

This code has beenb developped during the author's PhD thesis `[@aladjidiFullOpticalControl2023]` and used as the main simulation tool for several publications `[@glorieuxHotAtomicVapors2023; @baker-rasooliTurbulentDynamicsTwodimensional2023]`.

![Example of an output of the solver. A shearing layer is observed nucleating vortices, that are attracted towards the center due to an attractive potential. The density and phase of the field are represented as well as the momentum distribution in order to get a quick overview of the state of the field.\label{fig:output}](../img/output.png)

# Acknowledgements

We acknowledge contributions from Myrann Baker-Rasooli as our most faithful beta tester.

# References