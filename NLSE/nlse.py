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
from scipy.ndimage import zoom
from scipy import signal
from scipy import special
from typing import Any
from . import kernels_cpu

BACKEND = "GPU"
PRECISION = "single"
if PRECISION == "double":
    PRECISION_REAL = np.float64
    PRECISION_COMPLEX = np.complex128
else:
    PRECISION_REAL = np.float32
    PRECISION_COMPLEX = np.complex64

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
            self.kernels = kernels_gpu
            self.convolution = signal_cp.fftconvolve
        else:
            self.backend = "CPU"
            self.kernels = kernels_cpu
            self.convolution = signal.fftconvolve
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
            dtype=PRECISION_REAL,
        )
        self.Y, self.delta_Y = np.linspace(
            -self.window / 2,
            self.window / 2,
            num=NY,
            endpoint=False,
            retstep=True,
            dtype=PRECISION_REAL,
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
            self.nl_profile = np.ones((self.NY, self.NX), dtype=PRECISION_REAL)

    def plot_2d(
        self, ax, Z, X, AMP, title, cmap="viridis", label=r"$X$ (mm)", **kwargs
    ):
        """Plots a 2d amplitude on an equidistant Z * X grid.

        Args:
            ax (matplotlib.Axes): The ax instance on which the plot will be drawn
            Z (np.ndarray): the X axis values
            X (np.ndarray): the Y axis values
            AMP (np.ndarray): The 2D field to plot
            title (str): Title of the plot
            cmap (str, optional): Colormap. Defaults to 'viridis'.
            label (str, optional): Label for the x axis. Defaults to r'$ (mm)'.
            vmax (int, optional): Maximum value for cmap normalization. Defaults to 1.
        """
        im = ax.imshow(
            AMP,
            aspect="equal",
            origin="lower",
            extent=(Z[0], Z[-1], X[0], X[-1]),
            cmap=cmap,
            **kwargs,
        )
        ax.set_xlabel(label)
        ax.set_ylabel(r"$Y$ (mm)")
        ax.set_title(title)
        plt.colorbar(im)
        return

    def plot_1d(
        self, ax, T, labelT, AMP, labelAMP, PHASE, labelPHASE, Tmin, Tmax
    ) -> None:
        """Plots a 1D slice of a 2D field with amplitude and phase

        Args:
            ax (matplotlib.Axes): The ax instance on which the plot will be drawn
            T (np.ndarray): The x axis
            labelT (str):  x axis label
            AMP (np.ndarray): The field slice
            labelAMP (str): y axis label
            PHASE (np.ndarray): The corresponding phase for the slice
            labelPHASE (str): y axis label for the phase
            Tmin (float): x axis left limit
            Tmax (float): x axis right limit
        """
        ax.plot(T, AMP, "b")
        ax.set_xlim([Tmin, Tmax])
        ax.set_xlabel(labelT)
        ax.set_ylabel(labelAMP, color="b")
        ax.tick_params(axis="y", labelcolor="b")
        axbis = ax.twinx()
        axbis.plot(T, PHASE, "r:")
        axbis.set_ylabel(labelPHASE, color="r")
        axbis.tick_params(axis="y", labelcolor="r")

        return

    def plot_1d_amp(self, ax, T, labelT, AMP, labelAMP, Tmin, Tmax, color="b") -> None:
        """Plots a 1D slice of a 2D field

        Args:
            ax (matplotlib.Axes): The ax instance on which the plot will be drawn
            T (np.ndarray): The x axis
            labelT (str):  x axis label
            AMP (np.ndarray): The field slice
            labelAMP (str): y axis label
            Tmin (float): x axis left limit
            Tmax (float): x axis right limit
            color (float): The color of the label
        """
        ax.plot(T, AMP, color)
        ax.set_xlim([Tmin, Tmax])
        ax.set_xlabel(labelT)
        ax.set_ylabel(labelAMP, color=color)
        ax.tick_params(axis="y", labelcolor="b")

        return

    def slm(self, pattern: np.ndarray, d_slm: float) -> np.ndarray:
        """Resizes the SLM pattern according to the sampling size chosen

        Args:
            pattern (np.ndarray): Pattern displayed on the SLM
            d_slm (_type_): Pixel pitch of the SLM

        Returns:
            np.ndarray: Resized pattern.
        """
        phase = np.zeros((self.NY, self.NX))
        zoom_x = self.delta_X / d_slm
        zoom_y = self.delta_Y / d_slm
        phase_zoomed = zoom(pattern, (zoom_y, zoom_x), order=0)
        # compute center offset
        x_center = (self.NX - phase_zoomed.shape[1]) // 2
        y_center = (self.NY - phase_zoomed.shape[0]) // 2

        # copy img image into center of result image
        phase[
            y_center : y_center + phase_zoomed.shape[0],
            x_center : x_center + phase_zoomed.shape[1],
        ] = phase_zoomed
        return phase

    def build_propagator(self, k: float) -> np.ndarray:
        """Builds the linear propagation matrix

        Args:
            precision (str, optional): "single" or "double" application of the
            propagator.
            Defaults to "single".
        Returns:
            propagator (np.ndarray): the propagator matrix
        """
        propagator = np.exp(-1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / k * self.delta_z)
        if self.backend == "GPU" and CUPY_AVAILABLE:
            return cp.asarray(propagator)
        else:
            return propagator

    def build_fft_plan(self, A: np.ndarray) -> list:
        """Builds the FFT plan objects for propagation

        Args:
            A (np.ndarray): Array to transform.
        Returns:
            list: A list containing the FFT plans
        """
        if self.backend == "GPU" and CUPY_AVAILABLE:
            # plan_fft = fftpack.get_fft_plan(
            #     A, shape=A.shape, axes=(-2, -1), value_type='C2C')
            plan_fft = fftpack.get_fft_plan(
                A,
                shape=(A.shape[-2], A.shape[-1]),
                axes=(-2, -1),
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
                axes=(-2, -1),
            )
            plan_ifft = pyfftw.FFTW(
                A,
                A,
                direction="FFTW_BACKWARD",
                threads=multiprocessing.cpu_count(),
                axes=(-2, -1),
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
    ):
        """Split step function for one propagation step

        Args:
            A (np.ndarray): Field to propagate
            V (np.ndarray): Potential field (can be None).
            propagator (np.ndarray): Propagator matrix.
            plans (list): List of FFT plan objects. Either a single FFT plan for
            both directions
            (GPU case) or distinct FFT and IFFT plans for FFTW.
            precision (str, optional): Single or double application of the linear
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
                A_sq = self.convolution(
                    A_sq, self.nl_profile, mode="same", axes=(-2, -1)
                )
            if V is None:
                self.kernels.nl_prop_without_V(
                    A,
                    A_sq,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
            else:
                self.self.kernels.nl_prop(
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
            A /= A.shape[-2] * A.shape[-1]
        else:
            plan_fft(input_array=A, output_array=A)
            np.multiply(A, propagator, out=A)
            plan_ifft(input_array=A, output_array=A, normalise_idft=True)
        A_sq = A.real * A.real + A.imag * A.imag
        if self.nl_length > 0:
            A_sq = self.convolution(A_sq, self.nl_profile, mode="same", axes=(-2, -1))
        if precision == "double":
            if V is None:
                self.kernels.nl_prop_without_V(
                    A,
                    A_sq,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
            else:
                self.kernels.nl_prop(
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
                self.kernels.nl_prop_without_V(
                    A,
                    A_sq,
                    self.delta_z,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
            else:
                self.kernels.nl_prop(
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
    ) -> np.ndarray:
        """Propagates the field at a distance z
        Args:
            E_in (np.ndarray): Normalized input field (between 0 and 1)
            z (float): propagation distance in m
            plot (bool, optional): Plots the results. Defaults to False.
            precision (str, optional): Does a "double" or a "single" application
            of the propagator. This leads to a dz (single) or dz^3 (double) precision.
            Defaults to "single".
            verbose (bool, optional): Prints progress and time. Defaults to True.
        Returns:
            np.ndarray: Propagated field in proper units V/m
        """
        assert E_in.shape[-2] == self.NY and E_in.shape[-1] == self.NX
        assert (
            E_in.dtype == PRECISION_COMPLEX
        ), f"Precision mismatch, E_in should be {PRECISION_COMPLEX}"
        Z = np.arange(0, z, step=self.delta_z, dtype=PRECISION_REAL)
        if self.backend == "GPU" and CUPY_AVAILABLE:
            if type(E_in) == np.ndarray:
                # A = np.empty((self.NX, self.NY), dtype=PRECISION_COMPLEX)
                A = np.empty(E_in.shape, dtype=PRECISION_COMPLEX)
                integral = np.sum(
                    np.abs(E_in) ** 2 * self.delta_X * self.delta_Y,
                    axis=(-2, -1),
                )
                return_np_array = True
            elif type(E_in) == cp.ndarray:
                # A = cp.empty((self.NX, self.NY), dtype=PRECISION_COMPLEX)
                A = cp.empty(E_in.shape, dtype=PRECISION_COMPLEX)
                integral = cp.sum(
                    cp.abs(E_in) ** 2 * self.delta_X * self.delta_Y,
                    axis=(-2, -1),
                )
                return_np_array = False
        else:
            return_np_array = True
            A = pyfftw.empty_aligned((self.NX, self.NY), dtype=PRECISION_COMPLEX)
            integral = np.sum(
                np.abs(E_in) ** 2 * self.delta_X * self.delta_Y, axis=(-2, -1)
            )
        if self.plans is None:
            self.plans = self.build_fft_plan(A)
        if normalize:
            # normalization of the field
            E_00 = np.sqrt(2 * self.puiss / (c * epsilon_0 * integral))
            A[:] = (E_00.T * E_in.T).T
        else:
            A[:] = E_in
        if self.propagator is None:
            self.propagator = self.build_propagator(self.k)
        if self.backend == "GPU" and CUPY_AVAILABLE:
            self.nl_profile = cp.asarray(self.nl_profile)
            if type(self.V) == np.ndarray:
                V = cp.asarray(self.V)
            elif type(self.V) == cp.ndarray:
                # V = self.V.copy()
                V = self.V
            if self.V is None:
                V = self.V
            if type(A) != cp.ndarray:
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
        n2_old = self.n2
        if verbose:
            pbar = tqdm.tqdm(total=len(Z), position=4, desc="Iteration", leave=False)
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
            fig = plt.figure(3, [9, 8])

            # plot amplitudes and phases
            a1 = fig.add_subplot(221)
            self.plot_2d(
                a1,
                self.X * 1e3,
                self.Y * 1e3,
                np.abs(A_plot) ** 2 * epsilon_0 * c / 2 * 1e-4,
                r"I in $W/cm^{-2}$",
            )
            a2 = fig.add_subplot(222)
            self.plot_2d(
                a2,
                self.X * 1e3,
                self.Y * 1e3,
                np.angle(A_plot),
                r"arg$(\psi)$",
                cmap="twilight",
                vmin=-np.pi,
                vmax=np.pi,
            )

            a3 = fig.add_subplot(223)
            lim = 1
            im_fft = np.abs(np.fft.fftshift(np.fft.fft2(A_plot[lim:-lim, lim:-lim])))
            Kx_2 = 2 * np.pi * np.fft.fftfreq(self.NX - 2 * lim, d=self.delta_X)
            len_fft = len(im_fft[0, :])
            self.plot_2d(
                a3,
                np.fft.fftshift(Kx_2),
                np.fft.fftshift(Kx_2),
                im_fft,
                r"$|\mathcal{TF}(E_{out})|^2$",
                cmap="viridis",
                label=r"$K_y$",
            )

            a4 = fig.add_subplot(224)
            self.plot_1d_amp(
                a4,
                Kx_2[1 : -len_fft // 2] * 1e-3,
                r"$K_y (mm^{-1})$",
                im_fft[len_fft // 2, len_fft // 2 + 1 :],
                r"$|\mathcal{TF}(E_{out})|$",
                np.fft.fftshift(Kx_2)[len_fft // 2 + 1] * 1e-3,
                np.fft.fftshift(Kx_2)[-1] * 1e-3,
                color="b",
            )
            a4.set_yscale("log")
            a4.set_xscale("log")

            plt.tight_layout()
            plt.show()
        return A


class NLSE_1d:
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
        self.backend = backend
        if self.backend == "GPU" and CUPY_AVAILABLE:
            self.kernels = kernels_gpu
            self.convolution = signal_cp.fftconvolve
        else:
            self.backend = "CPU"
            self.kernels = kernels_cpu
            self.convolution = signal.fftconvolve
        # listof physical parameters
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
        self.window = window
        rho0 = puiss / L**2
        Dn = self.n2 * rho0
        z_nl = 1 / (self.k * abs(Dn))
        self.delta_z = 0.1 * z_nl
        # transverse coordinate
        self.X, self.delta_X = np.linspace(
            -self.window / 2,
            self.window / 2,
            num=NX,
            endpoint=False,
            retstep=True,
            dtype=PRECISION_REAL,
        )
        # definition of the Fourier frequencies for the linear step
        self.Kx = 2 * np.pi * np.fft.fftfreq(self.NX, d=self.delta_X)
        self.propagator = None
        self.plans = None
        self.nl_length = nl_length
        if self.nl_length > 0:
            d = self.nl_length // self.delta_X
            x = np.arange(-3 * d, 3 * d + 1)
            self.nl_profile = (
                1 / (np.sqrt(2 * np.pi) * self.nl_length) * special.kn(0, np.abs(x) / d)
            )
            self.nl_profile[x.size // 2] = np.nanmax(
                self.nl_profile[np.logical_not(np.isinf(self.nl_profile))]
            )
            self.nl_profile /= self.nl_profile.sum()
        else:
            self.nl_profile = np.ones(self.NX, dtype=PRECISION_REAL)

    def build_propagator(self, k: float) -> np.ndarray:
        """Builds the linear propagation matrix

        Args:
            precision (str, optional): "single" or "double" application of the
            propagator.
            Defaults to "single".
        Returns:
            propagator (np.ndarray): the propagator matrix
        """
        propagator = np.exp(-1j * 0.5 * (self.Kx**2) / k * self.delta_z)
        if self.backend == "GPU" and CUPY_AVAILABLE:
            return cp.asarray(propagator)
        else:
            return propagator

    def build_fft_plan(self, A: np.ndarray) -> list:
        """Builds the FFT plan objects for propagation

        Args:
            A (np.ndarray): Array to transform.
        Returns:
            list: A list containing the FFT plans
        """
        if self.backend == "GPU" and CUPY_AVAILABLE:
            plan_fft = fftpack.get_fft_plan(A, axes=-1, value_type="C2C")
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
                axes=(-1,),
            )
            plan_ifft = pyfftw.FFTW(
                A,
                A,
                direction="FFTW_BACKWARD",
                threads=multiprocessing.cpu_count(),
                axes=(-1,),
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
    ):
        """Split step function for one propagation step

        Args:
            A (np.ndarray): Field to propagate
            V (np.ndarray): Potential field (can be None).
            propagator (np.ndarray): Propagator matrix.
            plans (list): List of FFT plan objects. Either a single FFT plan for both
              directions
            (GPU case) or distinct FFT and IFFT plans for FFTW.
            precision (str, optional): Single or double application of the linear
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
                A_sq = self.convolution(A_sq, self.nl_profile, mode="same", axes=(-1,))
            if V is None:
                self.kernels.nl_prop_without_V(
                    A,
                    A_sq,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )

            else:
                self.kernels.nl_prop(
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
            A /= A.shape[-1]
        else:
            plan_fft(input_array=A, output_array=A)
            np.multiply(A, propagator, out=A)
            plan_ifft(input_array=A, output_array=A, normalise_idft=True)
        A_sq = A.real * A.real + A.imag * A.imag
        if self.nl_length > 0:
            A_sq = self.convolution(A_sq, self.nl_profile, mode="same", axes=(-1,))
        if precision == "double":
            if V is None:
                self.kernels.nl_prop_without_V(
                    A,
                    A_sq,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )

            else:
                self.kernels.nl_prop(
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
                self.kernels.nl_prop_without_V(
                    A,
                    A_sq,
                    self.delta_z,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )

            else:
                self.kernels.nl_prop(
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
    ) -> np.ndarray:
        """Propagates the field at a distance z
        Args:
            E_in (np.ndarray): Normalized input field (between 0 and 1)
            z (float): propagation distance in m
            plot (bool, optional): Plots the results. Defaults to False.
            precision (str, optional): Does a "double" or a "single" application
            of the propagator. This leads to a dz (single) or dz^3 (double) precision.
            Defaults to "single".
            verbose (bool, optional): Prints progress and time. Defaults to True.
            normalize (bool, optional): Normalizes the field to V/m. Defaults to True.
            Used to be able to reuse fields that have already been propagated.
        Returns:
            np.ndarray: Propagated field in proper units V/m
        """
        assert E_in.shape[-1] == self.NX, "Shape mismatch"
        assert (
            E_in.dtype == PRECISION_COMPLEX
        ), f"Precision mismatch, E_in should be {PRECISION_COMPLEX}"
        Z = np.arange(0, z, step=self.delta_z, dtype=PRECISION_REAL)
        if self.backend == "GPU" and CUPY_AVAILABLE:
            self.nl_profile = cp.asarray(self.nl_profile)
            if type(E_in) == np.ndarray:
                A = np.empty(E_in.shape, dtype=PRECISION_COMPLEX)
                integral = np.sum(np.abs(E_in) ** 2 * self.delta_X, axis=-1) ** 2
                return_np_array = True
            elif type(E_in) == cp.ndarray:
                A = cp.empty(E_in.shape, dtype=PRECISION_COMPLEX)
                integral = cp.sum(cp.abs(E_in) ** 2 * self.delta_X, axis=-1) ** 2
                return_np_array = False
        else:
            integral = np.sum(np.abs(E_in) ** 2 * self.delta_X, axis=-1) ** 2
            return_np_array = True
            A = pyfftw.empty_aligned(E_in.shape, dtype=PRECISION_COMPLEX)
        plans = self.build_fft_plan(A)
        if normalize:
            E_00 = np.sqrt(2 * self.puiss / (c * epsilon_0 * integral))
            A[:] = (E_00.T * E_in.T).T
        else:
            A[:] = E_in
        propagator = self.build_propagator(self.k)
        if self.backend == "GPU" and CUPY_AVAILABLE:
            if type(self.V) == np.ndarray:
                V = cp.asarray(self.V)
            elif type(self.V) == cp.ndarray:
                V = self.V.copy()
            if self.V is None:
                V = self.V
            if type(A) != cp.ndarray:
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
        n2_old = self.n2
        if verbose:
            pbar = tqdm.tqdm(total=len(Z), position=4, desc="Iteration", leave=False)
        for i, z in enumerate(Z):
            if z > self.L:
                self.n2 = 0
            if verbose:
                pbar.update(1)
            self.split_step(A, V, propagator, plans, precision)
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
        if self.backend == "GPU" and CUPY_AVAILABLE and return_np_array:
            A = cp.asnumpy(A)

        if plot:
            if not (return_np_array):
                A_plot = cp.asnumpy(A)
            elif return_np_array or self.backend == "CPU":
                A_plot = A.copy()
            fig, ax = plt.subplots(1, 2, layout="constrained")
            if A.ndim == 2:
                for i in range(A.shape[0]):
                    ax[0].plot(
                        self.X * 1e3,
                        1e-4 * c / 2 * epsilon_0 * np.abs(A_plot[i, :]) ** 2,
                    )
                    ax[1].plot(self.X * 1e3, np.unwrap(np.angle(A_plot[i, :])))
            elif A.ndim == 1:
                ax[0].plot(self.X * 1e3, 1e-4 * c / 2 * epsilon_0 * np.abs(A_plot) ** 2)
                ax[1].plot(self.X * 1e3, np.unwrap(np.angle(A_plot)))
            ax[0].set_title(r"$|\psi|^2$")
            ax[0].set_ylabel(r"Intensity $\frac{\epsilon_0 c}{2}|\psi|^2$ in $W/cm^2$")
            ax[1].set_title(r"Phase $\mathrm{arg}(\psi)$")
            ax[1].set_ylabel(r"Phase arg$(\psi)$")
            for a in ax:
                a.set_xlabel("Position x in mm")
            plt.show()
        return A


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
            precision (str, optional): Single or double application of the linear
            propagation step.
            Defaults to "single".
        Returns:
            None
        """
        A1 = A[..., 0, :, :]
        A2 = A[..., 1, :, :]
        if BACKEND == "GPU":
            # on GPU, only one plan for both FFT directions
            plan_fft = plans[0]
        else:
            plan_fft, plan_ifft = plans
        if precision == "double":
            A_sq_1 = A1.real * A1.real + A1.imag * A1.imag
            A_sq_2 = A2.real * A2.real + A2.imag * A2.imag
            if self.nl_length > 0:
                A_sq_1 = self.convolution(
                    A_sq_1, self.nl_profile, mode="same", axes=(-2, -1)
                )
                A_sq_2 = self.convolution(
                    A_sq_2, self.nl_profile, mode="same", axes=(-2, -1)
                )

            if V is None:
                self.kernels.nl_prop_without_V_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    self.k / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
                self.kernels.nl_prop_without_V_c(
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
                self.kernels.nl_prop_c(
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
                self.kernels.nl_prop_c(
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
                self.kernels.rabi_coupling(A1, A2, self.delta_z / 2, self.omega / 2)
                self.kernels.rabi_coupling(A2, A1_old, self.delta_z / 2, self.omega / 2)
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
            A_sq_1 = self.convolution(
                A_sq_1, self.nl_profile, mode="same", axes=(-2, -1)
            )
            A_sq_2 = self.convolution(
                A_sq_2, self.nl_profile, mode="same", axes=(-2, -1)
            )
        if precision == "double":
            if V is None:
                self.kernels.nl_prop_without_V_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    self.k / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
                self.kernels.nl_prop_without_V_c(
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
                self.kernels.nl_prop_c(
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
                self.kernels.nl_prop_c(
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
                self.kernels.rabi_coupling(A1, A2, self.delta_z / 2, self.omega / 2)
                self.kernels.rabi_coupling(A2, A1_old, self.delta_z / 2, self.omega / 2)
        else:
            if V is None:
                self.kernels.nl_prop_without_V_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    self.k / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
                self.kernels.nl_prop_without_V_c(
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
                self.kernels.nl_prop_c(
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
                self.kernels.nl_prop_c(
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
                self.kernels.rabi_coupling(A1, A2, self.delta_z, self.omega / 2)
                self.kernels.rabi_coupling(A2, A1_old, self.delta_z, self.omega / 2)

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
            of the propagator. This leads to a dz (single) or dz^3 (double) precision.
            Defaults to "single".
            verbose (bool, optional): Prints progress and time. Defaults to True.
        Returns:
            np.ndarray: Propagated field in proper units V/m
        """
        assert E.shape[-2] == self.NY and E.shape[-1] == self.NX, (
            f"Shape mismatch ! Simulation grid size is {(self.NY, self.NX)}"
            f" and array shape is {(E.shape[-2], E.shape[-1])}"
        )
        assert E.ndim >= 3, (
            "Input number of dimensions should at least be 3 !" " (2, NY, NX)"
        )
        assert (
            E.dtype == PRECISION_COMPLEX
        ), f"Precision mismatch, E should be {PRECISION_COMPLEX}"
        Z = np.arange(0, z, step=self.delta_z, dtype=PRECISION_REAL)
        if self.backend == "GPU" and CUPY_AVAILABLE:
            self.nl_profile = cp.asarray(self.nl_profile)
            if type(E) == np.ndarray:
                A = np.empty(E.shape, dtype=PRECISION_COMPLEX)
                integral = np.sum(
                    np.abs(E) ** 2 * self.delta_X * self.delta_Y, axis=(-1, -2)
                )
                puiss_arr = np.array([self.puiss, self.puiss2])
                return_np_array = True
            elif type(E) == cp.ndarray:
                A = cp.empty(E.shape, dtype=PRECISION_COMPLEX)
                integral = cp.sum(
                    np.abs(E) ** 2 * self.delta_X * self.delta_Y, axis=(-1, -2)
                )
                puiss_arr = cp.array([self.puiss, self.puiss2])
                return_np_array = False
        else:
            integral = np.sum(
                np.abs(E) ** 2 * self.delta_X * self.delta_Y, axis=(-1, -2)
            )
            puiss_arr = np.array([self.puiss, self.puiss2])
            return_np_array = True
            A = pyfftw.empty_aligned(E.shape, dtype=PRECISION_COMPLEX)
        # ndim logic ...
        A[:] = E
        if normalize:
            E_00 = np.sqrt(2 * puiss_arr / (c * epsilon_0 * integral))
            A[:] = (A.T * E_00.T).T
        if self.plans is None:
            self.plans = self.build_fft_plan(E)
        if self.propagator1 is None:
            self.propagator1 = self.build_propagator(self.k)
            self.propagator2 = self.build_propagator(self.k2)
        if self.backend == "GPU" and CUPY_AVAILABLE:
            if type(self.V) == np.ndarray:
                V = cp.asarray(self.V)
            elif type(self.V) == cp.ndarray:
                V = self.V.copy()
            if self.V is None:
                V = self.V
            if type(A) != cp.ndarray:
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
        n2_old = self.n2
        if verbose:
            pbar = tqdm.tqdm(total=len(Z), position=4, desc="Iteration", leave=False)
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
        if self.backend == "GPU" and CUPY_AVAILABLE and return_np_array:
            A = cp.asnumpy(A)

        if plot:
            if A.ndim == 3:
                if not (return_np_array):
                    A_1_plot = A[0, :, :].get()
                    A_2_plot = A[1, :, :].get()
                elif return_np_array or self.backend == "CPU":
                    A_1_plot = A[0, :, :].copy()
                    A_2_plot = A[1, :, :].copy()
            else:
                if not (return_np_array):
                    A_1_plot = A[-1, 0, :, :].get()
                    A_2_plot = A[-1, 1, :, :].get()
                elif return_np_array or self.backend == "CPU":
                    A_1_plot = A[-1, 0, :, :].copy()
                    A_2_plot = A[-1, 1, :, :].copy()
            fig = plt.figure(layout="constrained")
            # plot amplitudes and phases
            a1 = fig.add_subplot(221)
            self.plot_2d(
                a1,
                self.X * 1e3,
                self.Y * 1e3,
                np.abs(A_1_plot) ** 2,
                r"$|\psi_1|^2$",
            )

            a2 = fig.add_subplot(222)
            self.plot_2d(
                a2,
                self.X * 1e3,
                self.Y * 1e3,
                np.angle(A_1_plot),
                r"arg$(\psi_1)$",
                cmap="twilight_shifted",
                vmin=-np.pi,
                vmax=np.pi,
            )

            a3 = fig.add_subplot(223)
            self.plot_2d(
                a3,
                self.X * 1e3,
                self.Y * 1e3,
                np.abs(A_2_plot) ** 2,
                r"$|\psi_2|^2$",
            )

            a4 = fig.add_subplot(224)
            self.plot_2d(
                a4,
                self.X * 1e3,
                self.Y * 1e3,
                np.angle(A_2_plot),
                r"arg$(\psi_2)$",
                cmap="twilight_shifted",
                vmin=-np.pi,
                vmax=np.pi,
            )
            plt.show()
        # some more ndim logic to unpack along the right axes
        if A.ndim == 3:
            return A
        else:
            A0 = A[:, 0, :, :]
            A1 = A[:, 1, :, :]
            return [A0, A1]


class CNLSE_1d(NLSE_1d):
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
            V=V,
            L=L,
            NX=NX,
            Isat=Isat,
            nl_length=nl_length,
            wvl=wvl,
            backend=backend,
        )
        self.I_sat2 = self.I_sat
        self.n12 = n12
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
            precision (str, optional): Single or double application of the linear
            propagation step.
            Defaults to "single".
        Returns:
            None
        """
        A1 = A[..., 0, :]
        A2 = A[..., 1, :]
        if self.backend == "GPU" and CUPY_AVAILABLE:
            # on GPU, only one plan for both FFT directions
            plan_fft = plans[0]
        else:
            plan_fft, plan_ifft = plans
        if precision == "double":
            A_sq_1 = A1.real * A1.real + A1.imag * A1.imag
            A_sq_2 = A2.real * A2.real + A2.imag * A2.imag
            if self.nl_length > 0:
                A_sq_1 = self.convolution(
                    A_sq_1, self.nl_profile, mode="same", axes=(-1,)
                )
                A_sq_2 = self.convolution(
                    A_sq_2, self.nl_profile, mode="same", axes=(-1,)
                )

            if V is None:
                self.kernels.nl_prop_without_V_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    self.k / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
                self.kernels.nl_prop_without_V_c(
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
                self.kernels.nl_prop_c(
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
                self.kernels.nl_prop_c(
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
                self.kernels.rabi_coupling(A1, A2, self.delta_z / 2, self.omega / 2)
                self.kernels.rabi_coupling(A2, A1_old, self.delta_z / 2, self.omega / 2)
        if self.backend == "GPU" and CUPY_AVAILABLE:
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
            A_sq_1 = self.convolution(A_sq_1, self.nl_profile, mode="same", axes=(-1,))
            A_sq_2 = self.convolution(A_sq_2, self.nl_profile, mode="same", axes=(-1,))
        if precision == "double":
            if V is None:
                self.kernels.nl_prop_without_V_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    self.k / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
                self.kernels.nl_prop_without_V_c(
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
                self.kernels.nl_prop_c(
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
                self.kernels.nl_prop_c(
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
                self.kernels.rabi_coupling(A1, A2, self.delta_z / 2, self.omega / 2)
                self.kernels.rabi_coupling(A2, A1_old, self.delta_z / 2, self.omega / 2)
        else:
            if V is None:
                self.kernels.nl_prop_without_V_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    self.k / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
                self.kernels.nl_prop_without_V_c(
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
                self.kernels.nl_prop_c(
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
                self.kernels.nl_prop_c(
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
                self.kernels.rabi_coupling(A1, A2, self.delta_z, self.omega / 2)
                self.kernels.rabi_coupling(A2, A1_old, self.delta_z, self.omega / 2)

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
            E (np.ndarray): Fields tensor of shape (XX, 2, NX).
            z (float): propagation distance in m
            plot (bool, optional): Plots the results. Defaults to False.
            precision (str, optional): Does a "double" or a "single" application
            of the propagator. This leads to a dz (single) or dz^3 (double) precision.
            Defaults to "single".
            verbose (bool, optional): Prints progress and time. Defaults to True.
        Returns:
            np.ndarray: Propagated field in proper units V/m
        """
        assert E.shape[-1] == self.NX, (
            f"Shape mismatch ! Simulation grid size is {(self.NY,)}"
            " and array shape is {(E.shape[-1],)}"
        )
        assert E.ndim >= 2, (
            "Input number of dimensions should at least be 2 !" " (2, NX)"
        )
        assert (
            E.dtype == PRECISION_COMPLEX
        ), f"Precision mismatch, E_in should be {PRECISION_COMPLEX}"
        Z = np.arange(0, z, step=self.delta_z, dtype=PRECISION_REAL)
        if self.backend == "GPU" and CUPY_AVAILABLE:
            self.nl_profile = cp.asarray(self.nl_profile)
            if type(E) == np.ndarray:
                A = np.empty(E.shape, dtype=PRECISION_COMPLEX)
                integral = np.sum(np.abs(E) ** 2 * self.delta_X, axis=-1) ** 2
                return_np_array = True
            elif type(E) == cp.ndarray:
                A = cp.empty(E.shape, dtype=PRECISION_COMPLEX)
                integral = cp.sum(np.abs(E) ** 2 * self.delta_X, axis=-1) ** 2
                return_np_array = False
        else:
            return_np_array = True
            A = pyfftw.empty_aligned(E.shape, dtype=PRECISION_COMPLEX)
            integral = np.sum(np.abs(E) ** 2 * self.delta_X, axis=-1) ** 2
        A[:] = E
        if normalize:
            E_00 = np.sqrt(2 * self.puiss / (c * epsilon_0 * integral))
            A[:] = (A.T * E_00.T).T
        if self.plans is None:
            self.plans = self.build_fft_plan(E)
        if self.propagator1 is None:
            self.propagator1 = self.build_propagator(self.k)
            self.propagator2 = self.build_propagator(self.k2)
        if self.backend == "GPU" and CUPY_AVAILABLE:
            if type(self.V) == np.ndarray:
                V = cp.asarray(self.V)
            elif type(self.V) == cp.ndarray:
                V = self.V.copy()
            if self.V is None:
                V = self.V
            if type(A) != cp.ndarray:
                A = cp.asarray(A)
        else:
            if self.V is None:
                V = self.V
            else:
                V = self.V.copy()
        if self.backend == "GPU":
            start_gpu = cp.cuda.Event()
            end_gpu = cp.cuda.Event()
            start_gpu.record()
        t0 = time.perf_counter()
        n2_old = self.n2
        if verbose:
            pbar = tqdm.tqdm(total=len(Z), position=4, desc="Iteration", leave=False)
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
                    print(
                        f"\nTime spent to solve : {t_gpu*1e-3} s (GPU) /"
                        f" {time.perf_counter()-t0} s (CPU)"
                    )
                )
            else:
                print(f"\nTime spent to solve : {t_cpu} s (CPU)")
        self.n2 = n2_old
        if self.backend == "GPU" and CUPY_AVAILABLE and return_np_array:
            A = cp.asnumpy(A)

        if plot:
            if not (return_np_array):
                A_1_plot = A[0, :].get()
                A_2_plot = A[1, :].get()
            elif return_np_array or self.backend == "CPU":
                A_1_plot = A[0, :].copy()
                A_2_plot = A[1, :].copy()
            fig, ax = plt.subplots(2, 2, layout="constrained")
            # plot amplitudes and phases
            ax[0, 0].plot(
                self.X * 1e3, np.abs(A_1_plot) ** 2 * epsilon_0 * c / 2 * 1e-4
            )
            ax[0, 0].set_title(r"$|\psi_1|^2$")
            ax[0, 0].set_xlabel("x in mm")
            ax[0, 0].set_ylabel(
                r"Intensity $\frac{\epsilon_0 c}{2}|\psi_1|^2$ in $W/cm^2$"
            )
            ax[0, 1].plot(self.X * 1e3, np.unwrap(np.angle(A_1_plot)))
            ax[0, 1].set_title(r"$\mathrm{arg}(\psi_1)$")
            ax[0, 1].set_xlabel("x in mm")
            ax[0, 1].set_ylabel(r"Phase in rad")
            ax[1, 0].plot(
                self.X * 1e3, np.abs(A_2_plot) ** 2 * epsilon_0 * c / 2 * 1e-4
            )
            ax[1, 0].set_title(r"$|\psi_2|^2$")
            ax[1, 0].set_xlabel("x in mm")
            ax[1, 0].set_ylabel(
                r"Intensity $\frac{\epsilon_0 c}{2}|\psi_1|^2$ in $W/cm^2$"
            )
            ax[1, 1].plot(self.X * 1e3, np.unwrap(np.angle(A_2_plot)))
            ax[1, 1].set_title(r"$\mathrm{arg}(\psi_2)$")
            ax[1, 1].set_xlabel("x in mm")
            ax[1, 1].set_ylabel(r"Phase in rad")
        plt.show()
        return A


class GPE:
    """A class to solve GPE."""

    def __init__(
        self,
        gamma: float,
        N: float,
        m: float,
        window: float,
        g: float,
        V: np.ndarray,
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
            m (float): mass of one atom in kg
            window (float): Window size in m
            g (float): Interaction energy in Hz*m^2
            V (np.ndarray): Potential in Hz
            NX (int, optional): Number of points in x. Defaults to 1024.
            NY (int, optional): Number of points in y. Defaults to 1024.
            sat (float): Saturation parameter in Hz/m^2.
            nl_length (float, optional): Non local length scale in m. Defaults to 0.
            backend (str, optional): "GPU" or "CPU". Defaults to BACKEND.
        """
        self.backend = backend
        if self.backend == "GPU" and CUPY_AVAILABLE:
            self.kernels = kernels_gpu
        else:
            self.kernels = kernels_cpu
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
            dtype=PRECISION_REAL,
        )
        self.Y, self.delta_Y = np.linspace(
            -self.window / 2,
            self.window / 2,
            num=NY,
            endpoint=False,
            retstep=True,
            dtype=PRECISION_REAL,
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
            self.nl_profile = np.ones((self.NY, self.NX), dtype=PRECISION_REAL)

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
        if self.backend == "GPU":
            # plan_fft = fftpack.get_fft_plan(
            #     A, shape=A.shape, axes=(-2, -1), value_type='C2C')
            plan_fft = fftpack.get_fft_plan(
                A, shape=(A.shape[-2], A.shape[-1]), axes=(-2, -1), value_type="C2C"
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
                axes=(-2, -1),
            )
            plan_ifft = pyfftw.FFTW(
                A,
                A,
                direction="FFTW_BACKWARD",
                threads=multiprocessing.cpu_count(),
                axes=(-2, -1),
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
                A_sq = self.convolution(
                    A_sq, self.nl_profile, mode="same", axes=(-2, -1)
                )
            if V is None:
                self.kernels.nl_prop_without_V(
                    A,
                    A_sq,
                    self.delta_t / 2,
                    self.gamma,
                    self.g,
                    self.sat,
                )
            else:
                self.kernels.nl_prop(
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
            A_sq = self.convolution(A_sq, self.nl_profile, mode="same", axes=(-2, -1))
        if precision == "double":
            if V is None:
                self.kernels.nl_prop_without_V(
                    A,
                    A_sq,
                    self.delta_t / 2,
                    self.gamma,
                    self.g,
                    self.sat,
                )
            else:
                self.kernels.nl_prop(
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
                self.kernels.nl_prop_without_V(
                    A,
                    A_sq,
                    self.delta_t,
                    self.gamma,
                    self.g,
                    self.sat,
                )
            else:
                self.kernels.nl_prop(
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
        assert (
            psi.dtype == PRECISION_COMPLEX
        ), f"Precision mismatch, E_in should be {PRECISION_COMPLEX}"
        Ts = np.arange(0, T, step=self.delta_t, dtype=PRECISION_REAL)
        if self.backend == "GPU" and CUPY_AVAILABLE:
            if type(psi) is np.ndarray:
                # A = np.empty((self.NX, self.NY), dtype=PRECISION_COMPLEX)
                A = np.empty(psi.shape, dtype=PRECISION_COMPLEX)
                integral = np.sum(
                    np.abs(psi) ** 2 * self.delta_X * self.delta_Y, axis=(-2, -1)
                )
                return_np_array = True
            elif type(psi) is cp.ndarray:
                # A = cp.empty((self.NX, self.NY), dtype=PRECISION_COMPLEX)
                A = cp.empty(psi.shape, dtype=PRECISION_COMPLEX)
                integral = cp.sum(
                    cp.abs(psi) ** 2 * self.delta_X * self.delta_Y, axis=(-2, -1)
                )
                return_np_array = False
        else:
            return_np_array = True
            A = pyfftw.empty_aligned((self.NX, self.NY), dtype=PRECISION_COMPLEX)
            integral = np.sum(
                np.abs(psi) ** 2 * self.delta_X * self.delta_Y, axis=(-2, -1)
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


class NLSE_1d_adim(NLSE_1d):
    """A class to solve the 1D NLSE in adimensional units."""

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
        wvl: float = 1,
        backend: str = BACKEND,
    ) -> object:
        """Instantiate the simulation."""
        super().__init__(
            alpha=alpha,
            puiss=puiss,
            window=window,
            n2=n2,
            V=V,
            L=L,
            NX=NX,
            Isat=Isat,
            wvl=wvl,
            backend=backend,
        )
        self.m = 2 * np.pi / self.wl
        self.rho0 = self.puiss / self.window
        self.c = np.sqrt(self.n2 * self.rho0 / self.m)
        self.xi = 1 / np.sqrt(4 * self.n2 * self.rho0 * self.m)

    def build_propagator(self) -> np.ndarray:
        """Build the linear propagation matrix.

        Args:
            precision (str, optional): "single" or "double" application of the
            propagator.
            Defaults to "single".
        Returns:
            propagator (np.ndarray): the propagator matrix
        """
        propagator = np.exp(-1j * self.delta_z * (self.Kx**2) / (2 * self.m))
        if self.backend == "GPU":
            return cp.asarray(propagator)
        else:
            return propagator

    def split_step(
        self,
        A: np.ndarray,
        V: np.ndarray,
        propagator: np.ndarray,
        plans: list,
        precision: str = "single",
    ):
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
            if V is None:
                self.kernels.nl_prop_without_V(
                    A, self.delta_z / 2, self.alpha / 2, -self.n2, self.I_sat
                )
            else:
                self.kernels.nl_prop(
                    A, self.delta_z / 2, self.alpha / 2, V, -self.n2, self.I_sat
                )
        if self.backend == "GPU" and CUPY_AVAILABLE:
            plan_fft.fft(A, A, cp.cuda.cufft.CUFFT_FORWARD)
            # linear step in Fourier domain (shifted)
            cp.multiply(A, propagator, out=A)
            plan_fft.fft(A, A, cp.cuda.cufft.CUFFT_INVERSE)
            # fft normalization
            A /= A.shape[-1]
        else:
            plan_fft(input_array=A, output_array=A)
            np.multiply(A, propagator, out=A)
            plan_ifft(input_array=A, output_array=A, normalise_idft=True)
        if precision == "double":
            if V is None:
                self.kernels.nl_prop_without_V(
                    A, self.delta_z / 2, self.alpha / 2, -self.n2, self.I_sat
                )
            else:
                self.kernels.nl_prop(
                    A, self.delta_z, self.alpha / 2, V, -self.n2, self.I_sat
                )
        else:
            if V is None:
                self.kernels.nl_prop_without_V(
                    A, self.delta_z, self.alpha / 2, -self.n2, self.I_sat
                )
            else:
                self.kernels.nl_prop(
                    A, self.delta_z, self.alpha / 2, V, -self.n2, self.I_sat
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
    ) -> np.ndarray:
        """Propagate the field at a distance z.

        Args:
            E_in (np.ndarray): Normalized input field (between 0 and 1)
            z (float): propagation distance in m
            plot (bool, optional): Plots the results. Defaults to False.
            precision (str, optional): Does a "double" or a "single"
            application of the nonlinear term.
            This leads to a dz (single) or dz^3 (double) precision.
            Defaults to "single".
            verbose (bool, optional): Prints progress and time.
            Defaults to True.
            normalize (bool, optional): Normalizes the field to V/m.
            Defaults to True.
            Used to be able to reuse fields that have already been propagated.
        Returns:
            np.ndarray: Propagated field in proper units V/m
        """
        assert E_in.shape[-1] == self.NX
        assert (
            E_in.dtype == PRECISION_COMPLEX
        ), f"Precision mismatch, E_in should be {PRECISION_COMPLEX}"
        Z = np.arange(0, z, step=self.delta_z, dtype=PRECISION_REAL)
        if self.backend == "GPU" and CUPY_AVAILABLE:
            if type(E_in) is np.ndarray:
                A = np.empty(E_in.shape, dtype=PRECISION_COMPLEX)
                integral = np.sum(np.abs(E_in) ** 2 * self.delta_X, axis=-1)
                return_np_array = True
            elif type(E_in) is cp.ndarray:
                A = cp.empty(E_in.shape, dtype=PRECISION_COMPLEX)
                integral = cp.sum(cp.abs(E_in) ** 2 * self.delta_X, axis=-1)
                return_np_array = False
        else:
            integral = np.sum(np.abs(E_in) ** 2 * self.delta_X, axis=-1)
            return_np_array = True
            A = pyfftw.empty_aligned(E_in.shape, dtype=PRECISION_COMPLEX)
        plans = self.build_fft_plan(A)
        if normalize:
            E_00 = np.sqrt(self.puiss / integral)
            A[:] = (E_00.T * E_in.T).T
        else:
            A[:] = E_in
        propagator = self.build_propagator()
        if self.backend == "GPU" and CUPY_AVAILABLE:
            if type(self.V) is np.ndarray:
                V = cp.asarray(self.V)
            elif type(self.V) is cp.ndarray:
                V = self.V.copy()
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
        n2_old = self.n2
        if verbose:
            pbar = tqdm.tqdm(total=len(Z), position=4, desc="Iteration", leave=False)
        # dz = self.delta_z
        for i, z in enumerate(Z):
            # eps = 1-2*np.random.random()
            # self.delta_z = dz*(1+1e-2*eps)
            if z > self.L:
                self.n2 = 0
            if verbose:
                pbar.update(1)
            self.split_step(A, V, propagator, plans, precision)
            if callback is not None:
                callback(self, A, z, i)
        if self.backend == "GPU" and CUPY_AVAILABLE:
            end_gpu.record()
            end_gpu.synchronize()
            t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
        t_cpu = time.perf_counter() - t0
        if verbose:
            pbar.close()
            if self.backend == "GPU" and CUPY_AVAILABLE:
                print(
                    f"\nTime spent to solve : {t_gpu*1e-3} s (GPU)"
                    f" / {t_cpu} s (CPU)"
                )
            else:
                print(f"\nTime spent to solve : {t_cpu} s (CPU)")
        self.n2 = n2_old
        if self.backend == "GPU" and CUPY_AVAILABLE and return_np_array:
            A = cp.asnumpy(A)

        if plot:
            if not (return_np_array):
                A_plot = cp.asnumpy(A)
            elif return_np_array or self.backend == "CPU":
                A_plot = A.copy()
            fig, ax = plt.subplots(1, 2)
            if A.ndim == 2:
                for i in range(A.shape[0]):
                    ax[0].plot(self.X, np.unwrap(np.angle(A_plot[i, :])))
                    ax[1].plot(self.X, np.abs(A_plot[i, :]) ** 2)
            elif A.ndim == 1:
                ax[0].plot(self.X, np.unwrap(np.angle(A_plot)))
                ax[1].plot(self.X, np.abs(A_plot) ** 2)
            ax[0].set_title("Phase")
            ax[1].set_title(r"Density")
            plt.tight_layout()
            plt.show()
        return A

    def bogo_disp(self, q: Any) -> Any:
        """Return the dispersion.

        Return the Bogoliubov dispersion relation assuming
        a constant density.

        Args:
            q (Any): Wavenumber

        Returns:
            Any: The corresponding Bogoliubov frequency
        """
        # return self.c*np.abs(q)*np.sqrt(1 + self.xi**2 * q**2)
        return np.sqrt(
            (q**2 / (2 * self.m)) * (q**2 / (2 * self.m) + 2 * self.n2 * self.rho0)
        )

    def thermal_state(self, T: float, nb_real: int = 1) -> Any:
        """Define a thermal state of Bogoliubov excitations.

        Args:
            T (float): The temperature of the state
            nb_real (int, optional): Number of realizations. Defaults to 1.

        Returns:
            Any: The thermal state of shape (n_real, NX) psi_x and the
            corresponding Bogoliubov modes bq
        """
        eps_q = self.bogo_disp(self.Kx)
        eps_q[0] = 1
        E_q = self.Kx**2 / (2 * self.m)
        E_q[0] = 1
        var_X = T / eps_q
        var_X[0] = 0
        sigma = np.sqrt(0.5 * var_X)
        Re_X = np.random.normal(0, sigma, (nb_real, self.NX))
        Im_X = np.random.normal(0, sigma, (nb_real, self.NX))

        bq = Re_X + 1j * Im_X

        bmq = np.roll(bq[:, ::-1], 1, axis=1)
        theta_q = (1j / 2) * np.sqrt(eps_q / E_q) * (bq - np.conj(bmq))
        theta_q[..., 0] = 1.0

        delta_rho_q = np.sqrt(E_q / eps_q) * (np.conj(bmq) + bq)
        delta_rho_q[..., 0] = 1.0
        delta_rho_x = np.real(np.fft.ifft(delta_rho_q))
        theta_x = np.real(np.fft.ifft(theta_q))

        rho_x = self.rho0 + delta_rho_x
        psi_x = np.sqrt(rho_x) * np.exp(1j * theta_x)
        return psi_x, bq

    def get_bq(self, psi_x: Any) -> Any:
        """Retrieve the distribution of Bogoliubov modes from a field.

        Args:
            psi_x (Any): The field

        Returns:
            Any: The distribution of Bogoliubov modes in momentum space bq
        """
        eps_q = self.bogo_disp(self.Kx)
        eps_q[0] = 1
        E_q = self.Kx**2 / (2 * self.m)
        E_q[0] = 1
        theta_x = np.unwrap(np.angle(psi_x))
        delta_rho_x = np.abs(psi_x) ** 2
        theta_q = np.fft.fft(theta_x)
        delta_rho_q = np.fft.fft(delta_rho_x)
        bq = 0.5 * (
            -2 * 1j * np.sqrt(E_q / eps_q) * theta_q
            + np.sqrt(eps_q / E_q) * delta_rho_q
        )
        bq[..., 0] = 0
        return bq


if __name__ == "__main__":

    def normalize(arr: np.ndarray) -> np.ndarray:
        """Normalize an array.

        Args:
            arr (np.ndarray): Array to normalize.

        Returns:
            np.ndarray: Normalized array.
        """
        return (arr - np.amin(arr)) / (np.amax(arr) - np.amin(arr))

    def flatTop_tur(
        sx: int,
        sy: int,
        length: int = 150,
        width: int = 60,
        k_counter: int = 81,
        N_steps: int = 81,
    ) -> np.ndarray:
        """Collide two counterstreaming components.

        Generates the phase mask to create two counterstreaming colliding
        components.

        Args:
            sx (int): x Dimension of the mask : slm dimensions.
            sy (int): y Dimension of the mask : slm dimensions.
            length (int, optional): Length of the pattern. Defaults to 150.
            width (int, optional): Width of the pattern. Defaults to 60.
            k_counter (int, optional): Frequency of the blazed grating.
            Defaults to 81.
            N_steps (int, optional): Frequency of the vertical grating.
            Defaults to 81.

        Returns:
            _type_: _description_
        """
        output = np.zeros((sy, sx))
        Y, X = np.indices(output.shape)
        output[abs(X - output.shape[1] // 2) < length / 2] = 1
        output[abs(Y - output.shape[0] // 2) > width / 2] = 0

        grating_axe = X
        grating_axe = grating_axe % (sx / k_counter)
        grating_axe += abs(np.amin(grating_axe))
        grating_axe /= np.amax(grating_axe)

        grating_axe[X > output.shape[1] // 2] *= -1
        grating_axe[X > output.shape[1] // 2] += 1

        grating_axe_vert = Y
        grating_axe_vert = grating_axe_vert % (sy / N_steps)
        grating_axe_vert = normalize(grating_axe_vert)

        grating_axe = ((grating_axe + grating_axe_vert) % 1) * output
        return grating_axe

    def flatTop_super(
        sx: int,
        sy: int,
        length: int = 150,
        width: int = 60,
        k_counter: int = 81,
        N_steps: int = 81,
    ) -> np.ndarray:
        """Shear two components.

        Generates the phase mask to create two counterstreaming shearing
        components.

        Args:
            sx (int): x Dimension of the mask : slm dimensions.
            sy (int): y Dimension of the mask : slm dimensions.
            length (int, optional): Length of the pattern. Defaults to 150.
            width (int, optional): Width of the pattern. Defaults to 60.
            k_counter (int, optional): Frequency of the blazed grating.
            Defaults to 81.
            N_steps (int, optional): Frequency of the vertical grating.
            Defaults to 81.

        Returns:
            (np.ndarray): The generated phase mask.
        """
        output = np.zeros((sy, sx))
        Y, X = np.indices(output.shape)
        output[abs(X - output.shape[1] // 2) < length / 2] = 1
        output[abs(Y - output.shape[0] // 2) > width / 2] = 0

        grating_axe = X
        grating_axe = grating_axe % (sx / k_counter)
        grating_axe += abs(np.amin(grating_axe))
        grating_axe /= np.amax(grating_axe)

        grating_axe[Y > output.shape[0] // 2] *= -1
        grating_axe[Y > output.shape[0] // 2] += 1

        grating_axe_vert = Y
        grating_axe_vert = grating_axe_vert % (sy / N_steps)
        grating_axe_vert = normalize(grating_axe_vert)

        grating_axe = ((grating_axe + grating_axe_vert) % 1) * output
        return grating_axe

    trans = 0.5
    n2 = -1.6e-9
    n12 = -2e-10
    waist = 1e-3
    window = 2048 * 5.5e-6
    N = 1e6
    puiss = 500e-3
    Isat = 10e4  # saturation intensity in W/m^2
    L = 1e-3
    alpha = -np.log(trans) / L
    dn = 2.5e-4 * np.ones((2048, 2048), dtype=PRECISION_COMPLEX)
    simu = NLSE(
        alpha=alpha, puiss=puiss, window=window, n2=n2, V=dn, L=L, NX=2048, NY=2048
    )
    simu_c = CNLSE(alpha, puiss, window, n2, n12, None, L, NX=2048, NY=2048)
    simu_1d = NLSE_1d(alpha, puiss, window, n2, dn[1024, :], L, NX=2048)
    g = 1e3 / (N / 1e-3**2)
    print(f"{g=}")
    simu_gpe = GPE(
        gamma=0, N=N, m=87 * atomic_mass, window=1e-3, g=g, V=None, NX=4096, NY=4096
    )
    simu_gpe.delta_t = 1e-8
    simu.delta_z = 1e-4
    simu_1d.delta_z = 1e-4
    simu_c.delta_z = 0.5e-4
    simu_c.n22 = 0
    simu.I_sat = Isat
    simu_1d.I_sat = Isat
    phase_slm = 2 * np.pi * flatTop_super(1272, 1024, length=1000, width=600)
    phase_slm = simu.slm(phase_slm, 6.25e-6)
    E_in_0 = np.ones((simu.NY, simu.NX), dtype=PRECISION_COMPLEX) * np.exp(
        -(simu.XX**2 + simu.YY**2) / (2 * waist**2)
    )
    E_in_0_g = np.ones((simu_gpe.NY, simu_gpe.NX), dtype=PRECISION_COMPLEX) * np.exp(
        -(simu_gpe.XX**2 + simu_gpe.YY**2) / (2 * (1e-4) ** 2)
    )
    # A_gpe = simu_gpe.out_field(E_in_0_g, 7e-4, plot=True)
    simu.V *= np.exp(-(simu.XX**2 + simu.YY**2) / (2 * (waist / 3) ** 2))
    E_in_0 *= np.exp(1j * phase_slm)
    E_in_0 = np.fft.fftshift(np.fft.fft2(E_in_0))
    E_in_0[0 : E_in_0.shape[0] // 2 + 20, :] = 1e-10
    E_in_0[E_in_0.shape[0] // 2 + 225 :, :] = 1e-10
    E_in_0 = np.fft.ifft2(np.fft.ifftshift(E_in_0))
    E_in = np.zeros((2, simu.NY, simu.NX), dtype=PRECISION_COMPLEX)
    E_in[0, :, :] = np.exp(-(simu.XX**2 + simu.YY**2) / (2 * waist**2))
    E_in[1, :, :] = np.exp(-(simu.XX**2 + simu.YY**2) / (2 * (waist / 6) ** 2))
    A_c = simu_c.out_field(E_in, L, plot=True, verbose=True)
    A = simu.out_field(E_in_0, L, plot=True)
    A_1d = simu_1d.out_field(E_in_0[1024:1028, :], L, plot=True)
