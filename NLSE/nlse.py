#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Tangui Aladjidi / Clara Piekarski
"""NLSE Main module."""

import multiprocessing
import pickle
import time
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pyfftw
from scipy.constants import c, epsilon_0
from scipy import signal
from scipy import special
from . import kernels_cpu
from .utils import __BACKEND__, __CUPY_AVAILABLE__

if __CUPY_AVAILABLE__:
    import cupy as cp
    import cupyx.scipy.fftpack as fftpack
    import cupyx.scipy.signal as signal_cp
    from . import kernels_gpu

pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"
pyfftw.interfaces.cache.enable()


class NLSE:
    """A class to solve NLSE"""

    __CUPY_AVAILABLE__ = __CUPY_AVAILABLE__

    def __init__(
        self,
        alpha: float,
        puiss: float,
        window: float,
        n2: float,
        V: np.ndarray,
        L: float,
        NX: int = 1024,
        NY: int = 1024,
        Isat: float = np.inf,
        nl_length: float = 0,
        wvl: float = 780e-9,
        backend: str = __BACKEND__,
    ) -> object:
        """Instantiates the simulation.
        Solves an equation : d/dz psi = -1/2k0(d2/dx2 + d2/dy2) psi + k0 dn psi +
          k0 n2 psi**2 psi
        Args:
            alpha (float): alpha
            puiss (float): Power in W
            window (float): Computational window in the transverse plane in m.
            n2 (float): Non linear coeff in m^2/W
            V (np.ndarray): Potential.
            L (float): Length in m of the nonlinear medium
            NX (int, optional): Number of points in the x direction. Defaults to 1024.
            NY (int, optional): Number of points in the y direction. Defaults to 1024.
            Isat (float): Saturation intensity in W/m^2
            nl_length (float): Non local length in m
            wvl (float): Wavelength in m
            backend (str, optional): "GPU" or "CPU". Defaults to __BACKEND__.
        """
        # listof physical parameters
        self.backend = backend
        if self.backend == "GPU" and self.__CUPY_AVAILABLE__:
            self._kernels = kernels_gpu
            self._convolution = signal_cp.oaconvolve
        else:
            self.backend = "CPU"
            self._kernels = kernels_cpu
            self._convolution = signal.oaconvolve
        self.n2 = n2
        self.V = V
        self.wl = wvl
        self.k = 2 * np.pi / self.wl
        self.L = L  # length of the non linear medium
        self.alpha = alpha
        self.puiss = puiss
        self.I_sat = Isat
        # number of grid points in X (even, best is power of 2 or low prime factors)
        self.NX = NX
        self.NY = NY
        self.window = window
        Dn = self.n2 * self.puiss / self.window**2
        z_nl = 1 / (self.k * abs(Dn))
        if isinstance(z_nl, np.ndarray):
            z_nl = z_nl.min()
        self.delta_z = 1e-2 * z_nl
        # transverse coordinate
        self.X, self.delta_X = np.linspace(
            -self.window / 2,
            self.window / 2,
            num=NX,
            endpoint=False,
            retstep=True,
            dtype=np.float32,
        )
        self.Y, self.delta_Y = np.linspace(
            -self.window / 2,
            self.window / 2,
            num=NY,
            endpoint=False,
            retstep=True,
            dtype=np.float32,
        )
        # define last axes for broadcasting operations
        self._last_axes = (-2, -1)

        self.XX, self.YY = np.meshgrid(self.X, self.Y)
        # definition of the Fourier frequencies for the linear step
        self.Kx = 2 * np.pi * np.fft.fftfreq(self.NX, d=self.delta_X)
        self.Ky = 2 * np.pi * np.fft.fftfreq(self.NY, d=self.delta_Y)
        self.Kxx, self.Kyy = np.meshgrid(self.Kx, self.Ky)
        self.propagator = None
        self.plans = None
        self.nl_length = nl_length
        if self.nl_length > 0:
            d = self.nl_length // self.delta_X
            x = np.arange(-2 * d, 2 * d + 1)
            y = np.arange(-2 * d, 2 * d + 1)
            XX, YY = np.meshgrid(x, y)
            R = np.hypot(XX, YY)
            self.nl_profile = special.kn(0, R / d)
            self.nl_profile[
                self.nl_profile.shape[0] // 2, self.nl_profile.shape[1] // 2
            ] = np.nanmax(self.nl_profile[np.logical_not(np.isinf(self.nl_profile))])
            self.nl_profile /= self.nl_profile.sum()
        else:
            self.nl_profile = np.ones((self.NY, self.NX), dtype=np.float32)

    def _build_propagator(self) -> np.ndarray:
        """Build the linear propagation matrix.

        Returns:
            propagator (np.ndarray): the propagator matrix
        """
        propagator = np.exp(
            -1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k * self.delta_z
        ).astype(np.complex64)
        return propagator

    def _build_fft_plan(self, A: np.ndarray) -> list:
        """Builds the FFT plan objects for propagation

        Args:
            A (np.ndarray): Array to transform.
        Returns:
            list: A list containing the FFT plans
        """
        if self.backend == "GPU" and self.__CUPY_AVAILABLE__:
            plan_fft = fftpack.get_fft_plan(
                A,
                axes=self._last_axes,
                value_type="C2C",
            )
            return [plan_fft]
        else:
            # try to load previous fftw wisdom
            try:
                with open("fft.wisdom", "rb") as file:
                    wisdom = pickle.load(file)
                    pyfftw.import_wisdom(wisdom)
            except FileNotFoundError:
                print("No FFT wisdom found, starting over ...")
            plan_fft = pyfftw.FFTW(
                A,
                A,
                direction="FFTW_FORWARD",
                threads=multiprocessing.cpu_count(),
                axes=self._last_axes,
            )
            plan_ifft = pyfftw.FFTW(
                A,
                A,
                direction="FFTW_BACKWARD",
                threads=multiprocessing.cpu_count(),
                axes=self._last_axes,
            )
            with open("fft.wisdom", "wb") as file:
                wisdom = pyfftw.export_wisdom()
                pickle.dump(wisdom, file)
            return [plan_fft, plan_ifft]

    def _prepare_output_array(self, E_in: np.ndarray, normalize: bool) -> np.ndarray:
        """Prepare the output array depending on __BACKEND__.

        Args:
            E_in (np.ndarray): Input array
            normalize (bool): Normalize the field to the total power.
        Returns:
            np.ndarray: Output array
        """
        if self.backend == "GPU" and self.__CUPY_AVAILABLE__:
            A = cp.empty_like(E_in)
            E_in = cp.asarray(E_in)
        else:
            A = pyfftw.empty_aligned(
                E_in.shape, dtype=E_in.dtype, n=pyfftw.simd_alignment
            )
        if normalize:
            # normalization of the field
            integral = (
                (E_in.real * E_in.real + E_in.imag * E_in.imag)
                * self.delta_X
                * self.delta_Y
            ).sum(axis=self._last_axes)
            integral *= c * epsilon_0 / 2
            E_00 = (self.puiss / integral) ** 0.5
            A[:] = (E_00.T * E_in.T).T
        return A

    def _send_arrays_to_gpu(self) -> None:
        """
        Send arrays to GPU.
        """
        if self.V is not None:
            self.V = cp.asarray(self.V)
        self.nl_profile = cp.asarray(self.nl_profile)
        self.propagator = cp.asarray(self.propagator)
        # for broadcasting of parameters in case they are
        # not already on the GPU
        if isinstance(self.n2, np.ndarray):
            self.n2 = cp.asarray(self.n2)
        if isinstance(self.alpha, np.ndarray):
            self.alpha = cp.asarray(self.alpha)
        if isinstance(self.I_sat, np.ndarray):
            self.I_sat = cp.asarray(self.I_sat)

    def _retrieve_arrays_from_gpu(self) -> None:
        """
        Retrieve arrays from GPU.
        """
        if self.V is not None:
            self.V = self.V.get()
        self.nl_profile = self.nl_profile.get()
        self.propagator = self.propagator.get()
        if isinstance(self.n2, cp.ndarray):
            self.n2 = self.n2.get()
        if isinstance(self.alpha, cp.ndarray):
            self.alpha = self.alpha.get()
        if isinstance(self.I_sat, cp.ndarray):
            self.I_sat = self.I_sat.get()

    def split_step(
        self,
        A: np.ndarray,
        V: np.ndarray,
        propagator: np.ndarray,
        plans: list,
        precision: str = "single",
    ) -> None:
        """Split step function for one propagation step

        Args:
            A (np.ndarray): Field to propagate
            V (np.ndarray): Potential field (can be None).
            propagator (np.ndarray): Propagator matrix.
            plans (list): List of FFT plan objects. Either a single FFT plan for
            both directions
            (GPU case) or distinct FFT and IFFT plans for FFTW.
            precision (str, optional): Single or double application of the nonlinear
            propagation step.
            Defaults to "single".
        """
        if self.backend == "GPU" and self.__CUPY_AVAILABLE__:
            # on GPU, only one plan for both FFT directions
            plan_fft = plans[0]
        else:
            plan_fft, plan_ifft = plans
        if precision == "double":
            A_sq = A.real * A.real + A.imag * A.imag
            if self.nl_length > 0:
                A_sq = self._convolution(
                    A_sq, self.nl_profile, mode="same", axes=self._last_axes
                )
            if V is None:
                self._kernels.nl_prop_without_V(
                    A,
                    A_sq,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
            else:
                self.self._kernels.nl_prop(
                    A,
                    A_sq,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * V,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
        if self.backend == "GPU" and self.__CUPY_AVAILABLE__:
            plan_fft.fft(A, A, cp.cuda.cufft.CUFFT_FORWARD)
            # linear step in Fourier domain (shifted)
            cp.multiply(A, propagator, out=A)
            plan_fft.fft(A, A, cp.cuda.cufft.CUFFT_INVERSE)
        else:
            plan_fft(input_array=A, output_array=A)
            np.multiply(A, propagator, out=A)
            plan_ifft(input_array=A, output_array=A, normalise_idft=False)
        # fft normalization
        A *= 1 / np.prod(A.shape[self._last_axes[0] :])
        A_sq = A.real * A.real + A.imag * A.imag
        if self.nl_length > 0:
            A_sq = self._convolution(
                A_sq, self.nl_profile, mode="same", axes=self._last_axes
            )
        if precision == "double":
            if V is None:
                self._kernels.nl_prop_without_V(
                    A,
                    A_sq,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
            else:
                self._kernels.nl_prop(
                    A,
                    A_sq,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * V,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
        else:
            if V is None:
                self._kernels.nl_prop_without_V(
                    A,
                    A_sq,
                    self.delta_z,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
            else:
                self._kernels.nl_prop(
                    A,
                    A_sq,
                    self.delta_z,
                    self.alpha / 2,
                    self.k / 2 * V,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )

    def out_field(
        self,
        E_in: np.ndarray,
        z: float,
        plot=False,
        precision: str = "single",
        verbose: bool = True,
        normalize: bool = True,
        callback: callable = None,
        callback_args: tuple = (),
    ) -> np.ndarray:
        """Propagates the field at a distance z
        Args:
            E_in (np.ndarray): Normalized input field (between 0 and 1)
            z (float): propagation distance in m
            plot (bool, optional): Plots the results. Defaults to False.
            precision (str, optional): Does a "double" or a "single" application
            of the nonlinear term.
            This leads to a dz (single) or dz^3 (double) precision.
            Defaults to "single".
            verbose (bool, optional): Prints progress and time. Defaults to True.
            normalize (bool, optional): Normalize the field to the total power.
            Defaults to True.
            callback (callable, optional): Callback function. Defaults to None.
            *args: Additional arguments for the callback function.
        Returns:
            np.ndarray: Propagated field in proper units V/m
        """
        assert (
            E_in.shape[self._last_axes[0] :] == self.XX.shape[self._last_axes[0] :]
        ), "Shape mismatch"
        assert E_in.dtype in [
            np.complex64,
            np.complex128,
        ], "Type mismatch, E_in should be complex64 or complex128"
        Z = np.arange(
            self.delta_z, z + self.delta_z, step=self.delta_z, dtype=E_in.real.dtype
        )
        A = self._prepare_output_array(E_in, normalize)
        self.plans = self._build_fft_plan(A)
        # define propagator if not already done
        if self.propagator is None:
            self.propagator = self._build_propagator()
        if self.backend == "GPU" and self.__CUPY_AVAILABLE__:
            self._send_arrays_to_gpu()
        if self.V is None:
            V = self.V
        else:
            V = self.V.copy()
        if verbose:
            pbar = tqdm.tqdm(total=len(Z), position=4, desc="Iteration", leave=False)
        n2_old = self.n2
        if self.backend == "GPU" and self.__CUPY_AVAILABLE__:
            start_gpu = cp.cuda.Event()
            end_gpu = cp.cuda.Event()
            start_gpu.record()
        t0 = time.perf_counter()
        for i, z in enumerate(Z):
            if z > self.L:
                self.n2 = 0
            if verbose:
                pbar.update(1)
            self.split_step(A, V, self.propagator, self.plans, precision)
            if callback is not None:
                callback(self, A, z, i, *callback_args)
        t_cpu = time.perf_counter() - t0
        if verbose:
            pbar.close()

        if self.backend == "GPU" and self.__CUPY_AVAILABLE__:
            end_gpu.record()
            end_gpu.synchronize()
            t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
        if verbose:
            if self.backend == "GPU" and self.__CUPY_AVAILABLE__:
                print(
                    f"\nTime spent to solve : {t_gpu*1e-3} s (GPU) /"
                    f" {time.perf_counter()-t0} s (CPU)\n"
                )
            else:
                print(f"\nTime spent to solve : {t_cpu} s (CPU)\n")
        self.n2 = n2_old
        return_np_array = isinstance(E_in, np.ndarray)
        if self.backend == "GPU" and self.__CUPY_AVAILABLE__:
            if return_np_array:
                A = cp.asnumpy(A)
            self._retrieve_arrays_from_gpu()

        if plot:
            self.plot_field(A, z)
        return A

    def plot_field(self, A_plot: np.ndarray, z: float) -> None:
        """Plot a field for monitoring.

        Args:
            A_plot (np.ndarray): Field to plot.
            z (float): Propagation distance.
        """
        # if array is multi-dimensional, drop dims until the shape is 2D
        if A_plot.ndim > 2:
            while len(A_plot.shape) > 2:
                A_plot = A_plot[0]
        if self.__CUPY_AVAILABLE__ and isinstance(A_plot, cp.ndarray):
            A_plot = A_plot.get()
        fig, ax = plt.subplots(1, 3, layout="constrained")
        fig.suptitle(rf"Field at $z$ = {z:.2e} m")
        ext_real = [
            self.X[0] * 1e3,
            self.X[-1] * 1e3,
            self.Y[0] * 1e3,
            self.Y[-1] * 1e3,
        ]
        ext_fourier = [
            self.Kx[0] * 1e-3,
            self.Kx[-1] * 1e-3,
            self.Ky[0] * 1e-3,
            self.Ky[-1] * 1e-3,
        ]
        rho = np.abs(A_plot) ** 2 * 1e-4 * c / 2 * epsilon_0
        phi = np.angle(A_plot)
        im_fft = np.abs(np.fft.fftshift(np.fft.fft2(A_plot)))
        im0 = ax[0].imshow(rho, extent=ext_real)
        ax[0].set_title("Intensity")
        ax[0].set_xlabel("x (mm)")
        ax[0].set_ylabel("y (mm)")
        fig.colorbar(im0, ax=ax[0], shrink=0.6, label=r"Intensity ($W/cm^2$)")
        im1 = ax[1].imshow(
            phi, extent=ext_real, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi
        )
        ax[1].set_title("Phase")
        ax[1].set_xlabel("x (mm)")
        ax[1].set_ylabel("y (mm)")
        fig.colorbar(im1, ax=ax[1], shrink=0.6, label="Phase (rad)")
        im2 = ax[2].imshow(
            im_fft,
            extent=ext_fourier,
            cmap="nipy_spectral",
        )
        ax[2].set_title("Fourier space")
        ax[2].set_xlabel(r"$k_x$ ($mm^{-1}$)")
        ax[2].set_ylabel(r"$k_y$ ($mm^{-1}$)")
        fig.colorbar(im2, ax=ax[2], shrink=0.6, label="Intensity (a.u.)")
        plt.show()
