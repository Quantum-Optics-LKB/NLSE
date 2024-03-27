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
from scipy.constants import c, epsilon_0, hbar, atomic_mass
from scipy import signal
from scipy import special
from . import kernels_cpu

BACKEND = "GPU"

if BACKEND == "GPU":
    try:
        import cupy as cp
        import cupyx.scipy.fftpack as fftpack
        import cupyx.scipy.signal as signal_cp

        CUPY_AVAILABLE = True
        from . import kernels_gpu

    except ImportError:
        print("CuPy not available, falling back to CPU backend ...")
        CUPY_AVAILABLE = False
        BACKEND = "CPU"

pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"
pyfftw.interfaces.cache.enable()
np.random.seed(1)


class NLSE:
    """A class to solve NLSE"""

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
        backend: str = BACKEND,
    ) -> object:
        """Instantiates the simulation.
        Solves an equation : d/dz psi = -1/2k0(d2/dx2 + d2/dy2) psi + k0 dn psi +
          k0 n2 psi**2 psi
        Args:
            alpha (float): alpha
            puiss (float): Power in W
            waist (float): Waist size in m
            n2 (float): Non linear coeff in m^2/W
            V (np.ndarray): Potential
            Isat (float): Saturation intensity in W/m^2
            nl_length (float): Non linear length in m
            wvl (float): Wavelength in m
            backend (str, optional): "GPU" or "CPU". Defaults to BACKEND.
        """
        # listof physical parameters
        self.backend = backend
        if self.backend == "GPU" and CUPY_AVAILABLE:
            self._kernels = kernels_gpu
            self._convolution = signal_cp.fftconvolve
        else:
            self.backend = "CPU"
            self._kernels = kernels_cpu
            self._convolution = signal.fftconvolve
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
            x = np.arange(-3 * d, 3 * d + 1)
            y = np.arange(-3 * d, 3 * d + 1)
            XX, YY = np.meshgrid(x, y)
            R = np.hypot(XX, YY)
            self.nl_profile = 1 / (2 * np.pi * self.nl_length**2) * special.kn(0, R / d)
            self.nl_profile[
                self.nl_profile.shape[0] // 2, self.nl_profile.shape[1] // 2
            ] = np.nanmax(self.nl_profile[np.logical_not(np.isinf(self.nl_profile))])
            self.nl_profile /= self.nl_profile.sum()
        else:
            self.nl_profile = np.ones((self.NY, self.NX), dtype=np.float32)

    def _build_propagator(self, k: float) -> np.ndarray:
        """Builds the linear propagation matrix

        Args:
            precision (str, optional): "single" or "double" application of the
            propagator.
            Defaults to "single".
        Returns:
            propagator (np.ndarray): the propagator matrix
        """
        propagator = np.exp(-1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / k * self.delta_z)
        return propagator

    def _build_fft_plan(self, A: np.ndarray) -> list:
        """Builds the FFT plan objects for propagation

        Args:
            A (np.ndarray): Array to transform.
        Returns:
            list: A list containing the FFT plans
        """
        if self.backend == "GPU" and CUPY_AVAILABLE:
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
        """Prepare the output array depending on backend.

        Args:
            E_in (np.ndarray): Input array
            normalize (bool): Normalize the field to the total power.
        Returns:
            np.ndarray: Output array
        """
        if self.backend == "GPU" and CUPY_AVAILABLE:
            A = cp.empty_like(E_in)
            A[:] = cp.asarray(E_in)
        else:
            A = pyfftw.empty_aligned(E_in.shape, dtype=E_in.dtype)
            A[:] = E_in
        if normalize:
            # normalization of the field
            integral = (
                (A.real * A.real + A.imag * A.imag) * self.delta_X * self.delta_Y
            ).sum(axis=self._last_axes)
            E_00 = (2 * self.puiss / (c * epsilon_0 * integral)) ** 0.5
            A = (E_00.T * A.T).T
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
        if self.backend == "GPU" and CUPY_AVAILABLE:
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
        if self.backend == "GPU" and CUPY_AVAILABLE:
            plan_fft.fft(A, A, cp.cuda.cufft.CUFFT_FORWARD)
            # linear step in Fourier domain (shifted)
            cp.multiply(A, propagator, out=A)
            plan_fft.fft(A, A, cp.cuda.cufft.CUFFT_INVERSE)
            # fft normalization
            A /= np.prod(A.shape[self._last_axes[0] :])
        else:
            plan_fft(input_array=A, output_array=A)
            np.multiply(A, propagator, out=A)
            plan_ifft(input_array=A, output_array=A, normalise_idft=True)
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

    def plot_field(self, A_plot: np.ndarray) -> None:
        """Plot a field for monitoring.

        Args:
            A_plot (np.ndarray): Field to plot
        """
        # if array is multi-dimensional, drop dims until the shape is 2D
        if A_plot.ndim > 2:
            while len(A_plot.shape) > 2:
                A_plot = A_plot[0]
        if CUPY_AVAILABLE and isinstance(A_plot, cp.ndarray):
            A_plot = A_plot.get()
        fig, ax = plt.subplots(1, 3, layout="constrained")
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
        fig.colorbar(im0, ax=ax[0], shrink=0.6, label="Intensity (W/cm^2)")
        im1 = ax[1].imshow(phi, extent=ext_real, cmap="twilight_shifted")
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

    def out_field(
        self,
        E_in: np.ndarray,
        z: float,
        plot=False,
        precision: str = "single",
        verbose: bool = True,
        normalize: bool = True,
        callback: callable = None,
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
        Z = np.arange(0, z, step=self.delta_z, dtype=E_in.real.dtype)
        A = self._prepare_output_array(E_in, normalize)
        # define plans if not already done
        if self.plans is None:
            self.plans = self._build_fft_plan(A)
        # define propagator if not already done
        if self.propagator is None:
            self.propagator = self._build_propagator(self.k)
        if self.backend == "GPU" and CUPY_AVAILABLE:
            self._send_arrays_to_gpu()
        if self.V is None:
            V = self.V
        else:
            V = self.V.copy()
        if verbose:
            pbar = tqdm.tqdm(total=len(Z), position=4, desc="Iteration", leave=False)
        n2_old = self.n2
        if self.backend == "GPU" and CUPY_AVAILABLE:
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
                callback(self, A, z, i)
        t_cpu = time.perf_counter() - t0
        if verbose:
            pbar.close()

        if self.backend == "GPU" and CUPY_AVAILABLE:
            end_gpu.record()
            end_gpu.synchronize()
            t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
        if verbose:
            if self.backend == "GPU" and CUPY_AVAILABLE:
                print(
                    f"\nTime spent to solve : {t_gpu*1e-3} s (GPU) /"
                    f" {time.perf_counter()-t0} s (CPU)"
                )
            else:
                print(f"\nTime spent to solve : {t_cpu} s (CPU)")
        self.n2 = n2_old
        return_np_array = isinstance(E_in, np.ndarray)
        if self.backend == "GPU" and CUPY_AVAILABLE:
            if return_np_array:
                A = cp.asnumpy(A)
            self._retrieve_arrays_from_gpu()

        if plot:
            self.plot_field(A)
        return A


class NLSE_1d(NLSE):
    """A class to solve NLSE in 1d"""

    def __init__(
        self,
        alpha: float,
        puiss: float,
        window: float,
        n2: float,
        V: np.ndarray,
        L: float,
        NX: int = 1024,
        Isat: float = np.inf,
        nl_length: float = 0,
        wvl: float = 780e-9,
        backend: str = BACKEND,
    ) -> object:
        """Instantiates the simulation.
        Solves an equation : d/dz psi = -1/2k0(d2/dx2 + d2/dy2) psi + k0 dn psi +
          k0 n2 psi**2 psi
        Args:
            alpha (float): Transmission coeff
            puiss (float): Power in W
            waist (float): Waist size in m
            n2 (float): Non linear coeff in m^2/W
            V (np.ndarray) : Potential
            L (float): Length of the medium.
            Isat (float): Saturation intensity in W/m^2
            nl_length (float, optional): Non-local length in m. Defaults to 0.
            wvl (float, optional): Wavelength in m. Defaults to 780 nm.
            backend (str, optional): "GPU" or "CPU". Defaults to BACKEND.
        """
        super().__init__(
            alpha=alpha,
            puiss=puiss,
            window=window,
            n2=n2,
            V=V,
            L=L,
            NX=NX,
            Isat=Isat,
            nl_length=nl_length,
            wvl=wvl,
            backend=backend,
        )
        self._last_axes = (-1,)
        self.nl_profile = self.nl_profile[0]
        self.nl_profile /= self.nl_profile.sum()

    def _build_propagator(self, k: float) -> np.ndarray:
        """Builds the linear propagation matrix

        Args:
            precision (str, optional): "single" or "double" application of the
            propagator.
            Defaults to "single".
        Returns:
            propagator (np.ndarray): the propagator matrix
        """
        propagator = np.exp(-1j * 0.5 * (self.Kx**2) / k * self.delta_z)
        return propagator

    def plot_field(self, A_plot: np.ndarray) -> None:
        """Plot a field for monitoring.

        Args:
            A_plot (np.ndarray): Field to plot
        """
        fig, ax = plt.subplots(1, 2, layout="constrained")
        if A_plot.ndim == 2:
            for i in range(A_plot.shape[0]):
                ax[0].plot(
                    self.X * 1e3,
                    1e-4 * c / 2 * epsilon_0 * np.abs(A_plot[i, :]) ** 2,
                )
                ax[1].plot(self.X * 1e3, np.unwrap(np.angle(A_plot[i, :])))
        elif A_plot.ndim == 1:
            ax[0].plot(self.X * 1e3, 1e-4 * c / 2 * epsilon_0 * np.abs(A_plot) ** 2)
            ax[1].plot(self.X * 1e3, np.unwrap(np.angle(A_plot)))
        ax[0].set_title(r"$|\psi|^2$")
        ax[0].set_ylabel(r"Intensity $\frac{\epsilon_0 c}{2}|\psi|^2$ in $W/cm^2$")
        ax[1].set_title(r"Phase $\mathrm{arg}(\psi)$")
        ax[1].set_ylabel(r"Phase arg$(\psi)$")
        for a in ax:
            a.set_xlabel("Position x in mm")
        plt.show()


class CNLSE(NLSE):
    """A class to solve the coupled NLSE"""

    def __init__(
        self,
        alpha: float,
        puiss: float,
        window: float,
        n2: float,
        n12: float,
        V: np.ndarray,
        L: float,
        NX: int = 1024,
        NY: int = 1024,
        Isat: float = np.inf,
        nl_length: float = 0,
        wvl: float = 780e-9,
        omega: float = None,
        backend: str = BACKEND,
    ) -> object:
        """Instantiates the class with all the relevant physical parameters

        Args:
            alpha (float): alpha through the cell
            puiss (float): Optical power in W
            waist (float): Beam waist in m
            window (float): Computational window in m
            n2 (float): Non linear index of the 1 st component in m^2/W
            n12 (float): Inter component interaction parameter
            V (np.ndarray): Potential landscape in a.u
            L (float): Length of the cell in m
            NX (int, optional): Number of points along x. Defaults to 1024.
            NY (int, optional): Number of points along y. Defaults to 1024.
            Isat (float, optional): Saturation intensity, assumed to be the same
            for both components. Defaults to infinity.
            nl_length (float, optional): Nonlocal length. Defaults to 0.
            wvl (float, optional): Wavelength in m. Defaults to 780 nm.
            omega (float, optional): Rabi coupling. Defaults to None.
            backend (str, optional): "GPU" or "CPU". Defaults to BACKEND.
        Returns:
            object: CNLSE class instance
        """
        super().__init__(
            alpha=alpha,
            puiss=puiss,
            window=window,
            n2=n2,
            V=V,
            L=L,
            NX=NX,
            NY=NY,
            Isat=Isat,
            nl_length=nl_length,
            wvl=wvl,
            backend=backend,
        )
        self.n12 = n12
        self.I_sat2 = self.I_sat
        # initialize intra component 2 interaction parameter
        # to be the same as intra component 1
        self.n22 = self.n2
        # Rabi coupling
        self.omega = omega
        # same for the losses, this is to leave separate attributes so
        # the the user can chose whether or not to unbalence the rates
        self.alpha2 = self.alpha
        # wavenumbers
        self.k2 = self.k
        # powers
        self.puiss2 = self.puiss
        # waists
        self.propagator1 = None
        self.propagator2 = None

    def _prepare_output_array(self, E: np.ndarray, normalize: bool) -> np.ndarray:
        """Prepare the output array depending on backend."""
        if self.backend == "GPU" and CUPY_AVAILABLE:
            A = cp.empty_like(E)
            A[:] = cp.asarray(E)
            puiss_arr = cp.array([self.puiss, self.puiss2], dtype=E.dtype)
        else:
            A = pyfftw.empty_aligned(E.shape, dtype=E.dtype)
            A[:] = E
            puiss_arr = np.array([self.puiss, self.puiss2], dtype=E.dtype)
        if normalize:
            # normalization of the field
            integral = (
                (A.real * A.real + A.imag * A.imag) * self.delta_X * self.delta_Y
            ).sum(axis=self._last_axes)
            E_00 = (2 * puiss_arr / (c * epsilon_0 * integral)) ** 0.5
            A = (E_00.T * A.T).T
        return A

    def _send_arrays_to_gpu(self) -> None:
        """
        Send arrays to GPU.
        """
        if self.V is not None:
            self.V = cp.asarray(self.V)
        self.nl_profile = cp.asarray(self.nl_profile)
        self.propagator1 = cp.asarray(self.propagator1)
        self.propagator2 = cp.asarray(self.propagator2)
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
        self.propagator1 = self.propagator1.get()
        self.propagator2 = self.propagator2.get()
        if isinstance(self.n2, cp.ndarray):
            self.n2 = self.n2.get()
        if isinstance(self.alpha, cp.ndarray):
            self.alpha = self.alpha.get()
        if isinstance(self.I_sat, cp.ndarray):
            self.I_sat = self.I_sat.get()

    def _take_components(self, A: np.ndarray) -> tuple:
        """Take the components of the field.

        Args:
            A (np.ndarray): Field to retrieve the components of
        Returns:
            tuple: Tuple of the two components
        """
        A1 = A[..., 0, :, :]
        A2 = A[..., 1, :, :]
        return A1, A2

    def split_step(
        self,
        A: np.ndarray,
        V: np.ndarray,
        propagator1: np.ndarray,
        propagator2: np.ndarray,
        plans: list,
        precision: str = "single",
    ) -> None:
        """Split step function for one propagation step

        Args:
            A (np.ndarray): Fields to propagate of shape (2, NY, NX)
            A1_old (np.ndarray): Array to store copy of A1 at start of function
            to symetrize the evolution term
            V (np.ndarray): Potential field (can be None).
            propagator1 (np.ndarray): Propagator matrix for field 1.
            propagator2 (np.ndarray): Propagator matrix for field 2.
            plans (list): List of FFT plan objects. Either a single FFT plan for
            both directions
            (GPU case) or distinct FFT and IFFT plans for FFTW.
            precision (str, optional): Single or double application of the nonlinear
            propagation step.
            Defaults to "single".
        Returns:
            None
        """
        if self.backend == "GPU" and CUPY_AVAILABLE:
            # on GPU, only one plan for both FFT directions
            plan_fft = plans[0]
        else:
            plan_fft, plan_ifft = plans
        A1, A2 = self._take_components(A)
        if precision == "double":
            A_sq_1 = A1.real * A1.real + A1.imag * A1.imag
            A_sq_2 = A2.real * A2.real + A2.imag * A2.imag
            if self.nl_length > 0:
                A_sq_1 = self._convolution(
                    A_sq_1, self.nl_profile, mode="same", axes=self._last_axes
                )
                A_sq_2 = self._convolution(
                    A_sq_2, self.nl_profile, mode="same", axes=self._last_axes
                )

            if V is None:
                self._kernels.nl_prop_without_V_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    self.k / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
                self._kernels.nl_prop_without_V_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    self.delta_z / 2,
                    self.alpha2 / 2,
                    self.k2 / 2 * self.n22 * c * epsilon_0,
                    self.k2 / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat2 / (epsilon_0 * c),
                )
            else:
                self._kernels.nl_prop_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z,
                    self.alpha / 2,
                    self.k / 2 * V,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    self.k / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
                self._kernels.nl_prop_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    self.delta_z,
                    self.alpha2 / 2,
                    self.k2 / 2 * V,
                    self.k2 / 2 * self.n22 * c * epsilon_0,
                    self.k2 / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat2 / (epsilon_0 * c),
                )
            if self.omega is not None:
                A1_old = A1.copy()
                self._kernels.rabi_coupling(A1, A2, self.delta_z / 2, self.omega / 2)
                self._kernels.rabi_coupling(
                    A2, A1_old, self.delta_z / 2, self.omega / 2
                )
        if self.backend == "GPU":
            plan_fft.fft(A, A, cp.cuda.cufft.CUFFT_FORWARD)
            # linear step in Fourier domain (shifted)
            cp.multiply(A1, propagator1, out=A1)
            cp.multiply(A2, propagator2, out=A2)
            plan_fft.fft(A, A, cp.cuda.cufft.CUFFT_INVERSE)
            # fft normalization
            A /= A.shape[-2] * A.shape[-1]
        else:
            plan_fft, plan_ifft = plans
            plan_fft(input_array=A, output_array=A)
            np.multiply(A1, propagator1, out=A1)
            np.multiply(A2, propagator2, out=A2)
            plan_ifft(input_array=A, output_array=A, normalise_idft=True)
        A_sq_1 = A1.real * A1.real + A1.imag * A1.imag
        A_sq_2 = A2.real * A2.real + A2.imag * A2.imag
        if self.nl_length > 0:
            A_sq_1 = self._convolution(
                A_sq_1, self.nl_profile, mode="same", axes=self._last_axes
            )
            A_sq_2 = self._convolution(
                A_sq_2, self.nl_profile, mode="same", axes=self._last_axes
            )
        if precision == "double":
            if V is None:
                self._kernels.nl_prop_without_V_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    self.k / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
                self._kernels.nl_prop_without_V_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    self.delta_z / 2,
                    self.alpha2 / 2,
                    self.k2 / 2 * self.n22 * c * epsilon_0,
                    self.k2 / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat2 / (epsilon_0 * c),
                )
            else:
                self._kernels.nl_prop_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z,
                    self.alpha / 2,
                    self.k / 2 * V,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    self.k / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
                self._kernels.nl_prop_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    self.delta_z,
                    self.alpha2 / 2,
                    self.k2 / 2 * V,
                    self.k2 / 2 * self.n22 * c * epsilon_0,
                    self.k2 / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat2 / (epsilon_0 * c),
                )
            if self.omega is not None:
                A1_old = A1.copy()
                self._kernels.rabi_coupling(A1, A2, self.delta_z / 2, self.omega / 2)
                self._kernels.rabi_coupling(
                    A2, A1_old, self.delta_z / 2, self.omega / 2
                )
        else:
            if V is None:
                self._kernels.nl_prop_without_V_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    self.k / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
                self._kernels.nl_prop_without_V_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    self.delta_z,
                    self.alpha2 / 2,
                    self.k2 / 2 * self.n22 * c * epsilon_0,
                    self.k2 / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat2 / (epsilon_0 * c),
                )
            else:
                self._kernels.nl_prop_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z,
                    self.alpha / 2,
                    self.k / 2 * V,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    self.k / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
                self._kernels.nl_prop_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    self.delta_z,
                    self.alpha2 / 2,
                    self.k2 / 2 * V,
                    self.k2 / 2 * self.n22 * c * epsilon_0,
                    self.k2 / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat2 / (epsilon_0 * c),
                )
            if self.omega is not None:
                A1_old = A1.copy()
                self._kernels.rabi_coupling(A1, A2, self.delta_z, self.omega / 2)
                self._kernels.rabi_coupling(A2, A1_old, self.delta_z, self.omega / 2)

    def plot_field(self, A_plot: np.ndarray) -> None:
        """Plot the field.

        Args:
            A_plot (np.ndarray): The field to plot
        """
        # if array is multi-dimensional, drop dims until the shape is 2D
        if A_plot.ndim > 3:
            while len(A_plot.shape) > 3:
                A_plot = A_plot[0]
        if CUPY_AVAILABLE and isinstance(A_plot, cp.ndarray):
            A_plot = A_plot.get()
        fig, ax = plt.subplots(2, 2, layout="constrained")
        ext_real = [
            self.X[0] * 1e3,
            self.X[-1] * 1e3,
            self.Y[0] * 1e3,
            self.Y[-1] * 1e3,
        ]
        rho0 = np.abs(A_plot[0]) ** 2 * 1e-4 * c / 2 * epsilon_0
        phi0 = np.angle(A_plot[0])
        rho1 = np.abs(A_plot[1]) ** 2 * 1e-4 * c / 2 * epsilon_0
        phi1 = np.angle(A_plot[1])
        # plot amplitudes and phases
        im0 = ax[0, 0].imshow(rho0, extent=ext_real)
        ax[0, 0].set_title(r"$|\psi_1|^2$")
        ax[0, 0].set_xlabel("x (mm)")
        ax[0, 0].set_ylabel("y (mm)")
        fig.colorbar(im0, ax=ax[0, 0], shrink=0.6, label="Intensity (W/cm^2)")
        im1 = ax[0, 1].imshow(phi0, extent=ext_real, cmap="twilight_shifted")
        ax[0, 1].set_title(r"Phase $\mathrm{arg}(\psi_1)$")
        ax[0, 1].set_xlabel("x (mm)")
        ax[0, 1].set_ylabel("y (mm)")
        fig.colorbar(im1, ax=ax[0, 1], shrink=0.6, label="Phase (rad)")
        im2 = ax[1, 0].imshow(rho1, extent=ext_real)
        ax[1, 0].set_title(r"$|\psi_2|^2$")
        ax[1, 0].set_xlabel("x (mm)")
        ax[1, 0].set_ylabel("y (mm)")
        fig.colorbar(im2, ax=ax[1, 0], shrink=0.6, label="Intensity (W/cm^2)")
        im3 = ax[1, 1].imshow(phi1, extent=ext_real, cmap="twilight_shifted")
        ax[1, 1].set_title(r"Phase $\mathrm{arg}(\psi_2)$")
        ax[1, 1].set_xlabel("x (mm)")
        ax[1, 1].set_ylabel("y (mm)")
        fig.colorbar(im3, ax=ax[1, 1], shrink=0.6, label="Phase (rad)")
        plt.show()

    def out_field(
        self,
        E: np.ndarray,
        z: float,
        plot=False,
        precision: str = "single",
        verbose: bool = True,
        normalize: bool = True,
        callback: callable = None,
    ) -> np.ndarray:
        """Propagates the field at a distance z
        Args:
            E (np.ndarray): Fields tensor of shape (XX, 2, NY, NX).
            z (float): propagation distance in m
            plot (bool, optional): Plots the results. Defaults to False.
            precision (str, optional): Does a "double" or a "single" application
            of the nonlinear term.
            This leads to a dz (single) or dz^3 (double) precision.
            Defaults to "single".
            verbose (bool, optional): Prints progress and time. Defaults to True.
        Returns:
            np.ndarray: Propagated field in proper units V/m
        """
        assert (
            E.shape[self._last_axes[0] :] == self.XX.shape[self._last_axes[0] :]
        ), "Shape mismatch"
        assert E.shape[self._last_axes[0] - 1] == 2, "E should have 2 components"
        assert E.dtype in [
            np.complex64,
            np.complex128,
        ], "Precision mismatch, E should be np.complex64 or np.complex128"
        Z = np.arange(0, z, step=self.delta_z, dtype=E.real.dtype)
        A = self._prepare_output_array(E, normalize)
        if self.plans is None:
            self.plans = self._build_fft_plan(E)
        if self.propagator1 is None:
            self.propagator1 = self._build_propagator(self.k)
            self.propagator2 = self._build_propagator(self.k2)
        if self.backend == "GPU" and CUPY_AVAILABLE:
            self._send_arrays_to_gpu()
        if self.V is None:
            V = self.V
        else:
            V = self.V.copy()
        if verbose:
            pbar = tqdm.tqdm(total=len(Z), position=4, desc="Iteration", leave=False)
        if self.backend == "GPU" and CUPY_AVAILABLE:
            start_gpu = cp.cuda.Event()
            end_gpu = cp.cuda.Event()
            start_gpu.record()
        t0 = time.perf_counter()
        n2_old = self.n2
        for i, z in enumerate(Z):
            if z > self.L:
                self.n2 = 0
                self.n22 = 0
                self.n12 = 0
            if verbose:
                pbar.update(1)
            self.split_step(
                A,
                V,
                self.propagator1,
                self.propagator2,
                self.plans,
                precision,
            )
            if callback is not None:
                callback(self, A, z, i)
        t_cpu = time.perf_counter() - t0
        if self.backend == "GPU" and CUPY_AVAILABLE:
            end_gpu.record()
            end_gpu.synchronize()
            t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
        if verbose:
            pbar.close()
            if self.backend == "GPU" and CUPY_AVAILABLE:
                print(
                    f"\nTime spent to solve : {t_gpu*1e-3} s (GPU) /"
                    f" {time.perf_counter()-t0} s (CPU)"
                )
            else:
                print(f"\nTime spent to solve : {t_cpu} s (CPU)")
        self.n2 = n2_old
        return_np_array = isinstance(E, np.ndarray)
        if self.backend == "GPU" and CUPY_AVAILABLE:
            if return_np_array:
                A = cp.asnumpy(A)
            self._retrieve_arrays_from_gpu()
        if plot:
            self.plot_field(A)
        return A


class CNLSE_1d(CNLSE):
    """A class to solve the 1D coupled NLSE"""

    def __init__(
        self,
        alpha: float,
        puiss: float,
        window: float,
        n2: float,
        n12: float,
        V: np.ndarray,
        L: float,
        NX: int = 1024,
        Isat: float = np.inf,
        nl_length: float = 0,
        wvl: float = 780e-9,
        omega: float = None,
        backend: str = BACKEND,
    ) -> object:
        """Instantiates the class with all the relevant physical parameters

        Args:
            alpha (float): Alpha through the cell
            puiss (float): Optical power in W
            waist (float): Beam waist in m
            window (float): Computational window in m
            n2 (float): Non linear index of the 1 st component in m^2/W
            n12 (float): Inter component interaction parameter
            V (np.ndarray): Potential landscape in a.u
            L (float): Length of the cell in m
            NX (int, optional): Number of points along x. Defaults to 1024.
            Isat (float, optional): Saturation intensity, assumed to be the same
            for both components. Defaults to infinity.
            Isat (float, optional): Saturation intensity, assumed to be the same
            for both components. Defaults to infinity.
            nl_length (float, optional): Nonlocal length. Defaults to 0.
            wvl (float, optional): Wavelength in m. Defaults to 780 nm.
            omega (float, optional): Rabi coupling. Defaults to None.
            backend (str, optional): "GPU" or "CPU". Defaults to BACKEND.

        Returns:
            object: CNLSE class instance
        """
        super().__init__(
            alpha=alpha,
            puiss=puiss,
            window=window,
            n2=n2,
            n12=n12,
            V=V,
            L=L,
            NX=NX,
            Isat=Isat,
            nl_length=nl_length,
            wvl=wvl,
            omega=omega,
            backend=backend,
        )
        self._last_axes = (-1,)

    def _prepare_output_array(self, E: np.ndarray, normalize: bool) -> np.ndarray:
        """Prepare the output array depending on backend."""
        if self.backend == "GPU" and CUPY_AVAILABLE:
            A = cp.empty_like(E)
            A[:] = cp.asarray(E)
            puiss_arr = cp.array([self.puiss, self.puiss2], dtype=E.dtype)
        else:
            A = pyfftw.empty_aligned(E.shape, dtype=E.dtype)
            A[:] = E
            puiss_arr = np.array([self.puiss, self.puiss2], dtype=E.dtype)
        if normalize:
            # normalization of the field
            integral = ((A.real * A.real + A.imag * A.imag) * self.delta_X).sum(
                axis=self._last_axes
            ) ** 2
            E_00 = (2 * puiss_arr / (c * epsilon_0 * integral)) ** 0.5
            A = (E_00.T * A.T).T
        return A

    def _take_components(self, A: np.ndarray) -> tuple:
        """Take the components of the field.

        Args:
            A (np.ndarray): Field to retrieve the components of
        Returns:
            tuple: Tuple of the two components
        """
        A1 = A[..., 0, :]
        A2 = A[..., 1, :]
        return A1, A2

    def _build_propagator(self, k: float) -> np.ndarray:
        """Builds the linear propagation matrix

        Args:
            precision (str, optional): "single" or "double" application of the
            propagator.
            Defaults to "single".
        Returns:
            propagator (np.ndarray): the propagator matrix
        """
        propagator = np.exp(-1j * 0.5 * (self.Kx**2) / k * self.delta_z)
        return propagator

    def plot_field(self, A_plot: np.ndarray) -> None:
        """Plot the field.

        Args:
            A_plot (np.ndarray): The field to plot
        """
        # if array is multi-dimensional, drop dims until the shape is 2D
        if A_plot.ndim > 2:
            while len(A_plot.shape) > 2:
                A_plot = A_plot[0]
        if CUPY_AVAILABLE and isinstance(A_plot, cp.ndarray):
            A_plot = A_plot.get()
        A_1_plot = A_plot[0]
        A_2_plot = A_plot[1]
        fig, ax = plt.subplots(2, 2, layout="constrained")
        # plot amplitudes and phases
        ax[0, 0].plot(self.X * 1e3, np.abs(A_1_plot) ** 2 * epsilon_0 * c / 2 * 1e-4)
        ax[0, 0].set_title(r"$|\psi_1|^2$")
        ax[0, 0].set_xlabel("x in mm")
        ax[0, 0].set_ylabel(r"Intensity $\frac{\epsilon_0 c}{2}|\psi_1|^2$ in $W/cm^2$")
        ax[0, 1].plot(self.X * 1e3, np.unwrap(np.angle(A_1_plot)))
        ax[0, 1].set_title(r"$\mathrm{arg}(\psi_1)$")
        ax[0, 1].set_xlabel("x in mm")
        ax[0, 1].set_ylabel(r"Phase in rad")
        ax[1, 0].plot(self.X * 1e3, np.abs(A_2_plot) ** 2 * epsilon_0 * c / 2 * 1e-4)
        ax[1, 0].set_title(r"$|\psi_2|^2$")
        ax[1, 0].set_xlabel("x in mm")
        ax[1, 0].set_ylabel(r"Intensity $\frac{\epsilon_0 c}{2}|\psi_1|^2$ in $W/cm^2$")
        ax[1, 1].plot(self.X * 1e3, np.unwrap(np.angle(A_2_plot)))
        ax[1, 1].set_title(r"$\mathrm{arg}(\psi_2)$")
        ax[1, 1].set_xlabel("x in mm")
        ax[1, 1].set_ylabel(r"Phase in rad")
        plt.show()


# TODO(Tangui): Setup inheritance from NLSE
class GPE:
    """A class to solve GPE."""

    def __init__(
        self,
        gamma: float,
        N: float,
        window: float,
        g: float,
        V: np.ndarray,
        m: float = 87 * atomic_mass,
        NX: int = 1024,
        NY: int = 1024,
        sat: float = np.inf,
        nl_length: float = 0,
        backend: str = BACKEND,
    ) -> object:
        """Instantiate the simulation.

        Solves an equation : d/dt psi = -1/2m(d2/dx2 + d2/dy2) psi + V psi +
          g psi**2 psi
        Args:
            gamma (float): Losses in Hz
            N (float): Total number of atoms
            window (float): Window size in m
            g (float): Interaction energy in Hz*m^2
            V (np.ndarray): Potential in Hz
            m (float, optionnal): mass of one atom in kg. Defaults to 87*atomic_mass for
            Rubidium 87.
            NX (int, optional): Number of points in x. Defaults to 1024.
            NY (int, optional): Number of points in y. Defaults to 1024.
            sat (float): Saturation parameter in Hz/m^2.
            nl_length (float, optional): Non local length scale in m. Defaults to 0.
            backend (str, optional): "GPU" or "CPU". Defaults to BACKEND.
        """
        self.backend = backend
        if self.backend == "GPU" and CUPY_AVAILABLE:
            self._kernels = kernels_gpu
            self._convolution = signal_cp.fftconvolve
        else:
            self._kernels = kernels_cpu
            self._convolution = signal.fftconvolve
        # listof physical parameters
        self.g = g
        self.V = V
        self.gamma = gamma
        self.N = N
        self.m = m
        self.sat = sat
        # number of grid points in X (even, best is power of 2 or
        # low prime factors)
        self.NX = NX
        self.NY = NY
        self.window = window
        tau_nl = self.window**2 / (abs(self.g) * self.N)
        self.delta_t = tau_nl / 10
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
            x = np.arange(-3 * d, 3 * d + 1)
            y = np.arange(-3 * d, 3 * d + 1)
            XX, YY = np.meshgrid(x, y)
            R = np.hypot(XX, YY)
            self.nl_profile = 1 / (2 * np.pi * self.nl_length**2) * special.kn(0, R / d)
            self.nl_profile[
                self.nl_profile.shape[0] // 2, self.nl_profile.shape[1] // 2
            ] = np.nanmax(self.nl_profile[np.logical_not(np.isinf(self.nl_profile))])
            self.nl_profile /= self.nl_profile.sum()
        else:
            self.nl_profile = np.ones((self.NY, self.NX), dtype=np.float32)
        self._last_axes = (-2, -1)

    def build_propagator(self) -> np.ndarray:
        """Build the linear propagation matrix.

        Returns:
            propagator (np.ndarray): the propagator matrix
        """
        propagator = np.exp(
            -1j * 0.5 * hbar * (self.Kxx**2 + self.Kyy**2) / self.m * self.delta_t
        )
        if self.backend == "GPU" and CUPY_AVAILABLE:
            return cp.asarray(propagator)
        else:
            return propagator

    def build_fft_plan(self, A: np.ndarray) -> list:
        """Build the FFT plan objects for propagation.

        Args:
            A (np.ndarray): Array to transform.
        Returns:
            list: A list containing the FFT plans
        """
        if self.backend == "GPU" and CUPY_AVAILABLE:
            plan_fft = fftpack.get_fft_plan(
                A,
                shape=(A.shape[-2], A.shape[-1]),
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

    def split_step(
        self,
        A: np.ndarray,
        V: np.ndarray,
        propagator: np.ndarray,
        plans: list,
        precision: str = "single",
    ) -> None:
        """Split step function for one propagation step.

        Args:
            A (np.ndarray): Field to propagate
            V (np.ndarray): Potential field (can be None).
            propagator (np.ndarray): Propagator matrix.
            plans (list): List of FFT plan objects.
            Either a single FFT plan for both directions
            (GPU case) or distinct FFT and IFFT plans for FFTW.
            precision (str, optional): Single or double application of the
            nonlinear propagation step.
            Defaults to "single".
        """
        if self.backend == "GPU" and CUPY_AVAILABLE:
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
                    self.delta_t / 2,
                    self.gamma,
                    self.g,
                    self.sat,
                )
            else:
                self._kernels.nl_prop(
                    A,
                    A_sq,
                    self.delta_t / 2,
                    self.gamma,
                    V,
                    self.g,
                    self.sat,
                )
        if self.backend == "GPU" and CUPY_AVAILABLE:
            plan_fft.fft(A, A, cp.cuda.cufft.CUFFT_FORWARD)
            # linear step in Fourier domain (shifted)
            cp.multiply(A, propagator, out=A)
            plan_fft.fft(A, A, cp.cuda.cufft.CUFFT_INVERSE)
            A /= A.shape[-2] * A.shape[-1]
        else:
            plan_fft, plan_ifft = plans
            plan_fft(input_array=A, output_array=A)
            np.multiply(A, propagator, out=A)
            plan_ifft(input_array=A, output_array=A, normalise_idft=True)
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
                    self.delta_t / 2,
                    self.gamma,
                    self.g,
                    self.sat,
                )
            else:
                self._kernels.nl_prop(
                    A,
                    A_sq,
                    self.delta_t / 2,
                    self.gamma,
                    V,
                    self.g,
                    self.sat,
                )
        else:
            if V is None:
                self._kernels.nl_prop_without_V(
                    A,
                    A_sq,
                    self.delta_t,
                    self.gamma,
                    self.g,
                    self.sat,
                )
            else:
                self._kernels.nl_prop(
                    A,
                    A_sq,
                    self.delta_t,
                    self.gamma,
                    V,
                    self.g,
                    self.sat,
                )

    def out_field(
        self,
        psi: np.ndarray,
        T: float,
        plot=False,
        precision: str = "single",
        verbose: bool = True,
        normalize: bool = True,
        callback: callable = None,
    ) -> np.ndarray:
        """Propagate the field at a time T.

        Args:
            psi (np.ndarray): Normalized input field (between 0 and 1)
            T (float): propagation time in s
            plot (bool, optional): Plots the results. Defaults to False.
            precision (str, optional): Does a "double" or a "single"
            application of the nonlinear term.
            This leads to a dz (single) or dz^3 (double) precision.
            Defaults to "single".
            verbose (bool, optional): Prints progress and time.
            Defaults to True.
        Returns:
            np.ndarray: Propagated field in proper units atoms/m
        """
        assert psi.shape[-2] == self.NY and psi.shape[-1] == self.NX
        assert psi.dtype in [
            np.complex64,
            np.complex128,
        ], "Precision mismatch, E_in should be np.complex64 or np.complex128"
        Ts = np.arange(0, T, step=self.delta_t, dtype=psi.real.dtype)
        if self.backend == "GPU" and CUPY_AVAILABLE:
            if type(psi) is np.ndarray:
                # A = np.empty((self.NX, self.NY), dtype=PRECISION_COMPLEX)
                A = np.empty_like(psi)
                integral = np.sum(
                    np.abs(psi) ** 2 * self.delta_X * self.delta_Y, axis=self._last_axes
                )
                return_np_array = True
            elif type(psi) is cp.ndarray:
                # A = cp.empty((self.NX, self.NY), dtype=PRECISION_COMPLEX)
                A = cp.empty_like(psi)
                integral = cp.sum(
                    cp.abs(psi) ** 2 * self.delta_X * self.delta_Y, axis=self._last_axes
                )
                return_np_array = False
        else:
            return_np_array = True
            A = pyfftw.empty_aligned((self.NX, self.NY), dtype=psi.dtype)
            integral = np.sum(
                np.abs(psi) ** 2 * self.delta_X * self.delta_Y, axis=self._last_axes
            )
        if self.plans is None:
            self.plans = self.build_fft_plan(A)
        if normalize:
            # normalization of the field
            E_00 = np.sqrt(self.N / integral)
            A[:] = (E_00.T * psi.T).T
        else:
            A[:] = psi
        if self.propagator is None:
            self.propagator = self.build_propagator()
        if self.backend == "GPU" and CUPY_AVAILABLE:
            if type(self.V) is np.ndarray:
                V = cp.asarray(self.V)
            elif type(self.V) is cp.ndarray:
                V = self.V
            if self.V is None:
                V = self.V
            if type(A) is not cp.ndarray:
                A = cp.asarray(A)
        else:
            if self.V is None:
                V = self.V
            else:
                V = self.V.copy()

        if self.backend == "GPU" and CUPY_AVAILABLE:
            start_gpu = cp.cuda.Event()
            end_gpu = cp.cuda.Event()
            start_gpu.record()
        t0 = time.perf_counter()
        if verbose:
            pbar = tqdm.tqdm(total=len(Ts), position=4, desc="Iteration", leave=False)
        for _ in Ts:
            if verbose:
                pbar.update(1)
            self.split_step(A, V, self.propagator, self.plans, precision)
            if callback is not None:
                callback(self, A, _)
        if verbose:
            pbar.close()

        if self.backend == "GPU" and CUPY_AVAILABLE:
            end_gpu.record()
            end_gpu.synchronize()
            t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
        t_cpu = time.perf_counter() - t0
        if verbose:
            if self.backend == "GPU" and CUPY_AVAILABLE:
                print(
                    f"\nTime spent to solve : {t_gpu*1e-3} s (GPU)"
                    f" / {t_cpu} s (CPU)"
                )
            else:
                print(f"\nTime spent to solve : {t_cpu} s (CPU)")
        if self.backend == "GPU" and CUPY_AVAILABLE and return_np_array:
            A = cp.asnumpy(A)

        if plot:
            if not (return_np_array):
                if A.ndim == 2:
                    A_plot = A.get()
                elif A.ndim == 3:
                    A_plot = A[0, :, :].get()
            elif return_np_array or self.backend == "CPU":
                if A.ndim == 2:
                    A_plot = A
                elif A.ndim == 3:
                    A_plot = A[0, :, :]
            im_fft = np.abs(np.fft.fftshift(np.fft.fft2(A_plot))) ** 2
            ext_real = [
                self.X[0] * 1e3,
                self.X[-1] * 1e3,
                self.Y[0] * 1e3,
                self.Y[-1] * 1e3,
            ]
            ext_fft = [
                self.Kx[self.Kx.size // 2] * 1e-3,
                self.Kx[self.Kx.size // 2 - 1] * 1e-3,
                self.Ky[self.Ky.size // 2] * 1e-3,
                self.Ky[self.Ky.size // 2 - 1] * 1e-3,
            ]
            fig, ax = plt.subplots(1, 3)
            im = ax[0].imshow(np.abs(A_plot) ** 2, cmap="viridis", extent=ext_real)
            ax[0].set_title(r"Density in atoms/$m^2$")
            ax[0].set_xlabel("x (mm)")
            ax[0].set_ylabel("y (mm)")
            fig.colorbar(im, ax=ax[0], label="Density")
            im = ax[1].imshow(
                np.angle(A_plot),
                cmap="twilight_shifted",
                vmin=-np.pi,
                vmax=np.pi,
                extent=ext_real,
            )
            ax[1].set_title("Phase")
            ax[1].set_xlabel("x (mm)")
            ax[1].set_ylabel("y (mm)")
            fig.colorbar(im, ax=ax[1], label="Phase")
            im = ax[2].imshow(im_fft, cmap="nipy_spectral", extent=ext_fft)
            ax[2].set_title("Momentum space")
            ax[2].set_xlabel(r"kx ($mm^{-1}$)")
            ax[2].set_ylabel(r"ky ($mm^{-1}$)")
            fig.colorbar(im, ax=ax[2], label="Density")
            plt.show()
        return A
