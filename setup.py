"""Init module for the NLSE package."""

from setuptools import setup

setup(
    name="NLSE",
    version="2.3.0",
    description="A package for solving the Nonlinear Schrödinger Equation"
    " (NLSE) using the Split-Step Fourier method.",
    url="https://github.com/Quantum-Optics-LKB/NLSE",
    author="Tangui Aladjidi",
    author_email="tangui.aladjidi@lkb.upmc.fr",
    license="GPLv3",
    packages=["NLSE"],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "tqdm",
        "pyfftw",
        "numba",
        "pyvkfft",
        "pyopencl",
    ],
    extra_requires=["cupy"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
        "Environment :: GPU",
        "Environment :: GPU :: NVIDIA CUDA :: 10",
        "Environment :: GPU :: NVIDIA CUDA :: 11",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
    ],
)
