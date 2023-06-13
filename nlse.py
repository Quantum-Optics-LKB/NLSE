#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Taladjidi / Clara

import multiprocessing
import pickle
import time
import progressbar
import tqdm
import matplotlib.pyplot as plt
import numba
import numpy as np
import pyfftw
from scipy.constants import c, epsilon_0
from scipy.ndimage import zoom

BACKEND = 'GPU'

if BACKEND == 'GPU':
    try:
        import cupy as cp
        import cupyx.scipy.fftpack as fftpack
        BACKEND = "GPU"

        @cp.fuse(kernel_name="nl_prop")
        def nl_prop(A: cp.ndarray, dz: float, alpha: float, V: cp.ndarray, g: float, Isat: float) -> None:
            """A fused kernel to apply real space terms

            Args:
                A (cp.ndarray): The field to propagate
                dz (float): Propagation step in m
                alpha (float): Losses
                V (cp.ndarray): Potential
                g (float): Interactions
                Isat (float): Saturation 
            """
            A_sq = cp.abs(A)**2
            A *= cp.exp(dz*(-alpha/(2*(1+A_sq/Isat)) + 1j * V + 1j*g *
                        A_sq/(1+A_sq/Isat)))

        @cp.fuse(kernel_name="nl_prop_without_V")
        def nl_prop_without_V(A: cp.ndarray, dz: float, alpha: float, g: float,
                              Isat: float) -> None:
            """A fused kernel to apply real space terms

            Args:
                A (cp.ndarray): The field to propagate
                dz (float): Propagation step in m
                alpha (float): Losses
                g (float): Interactions
                Isat (float): Saturation 
            """
            A_sq = cp.abs(A)**2
            A *= cp.exp(dz*(-alpha/(2*(1+A_sq/Isat)) + 1j*g *
                        A_sq/(1+A_sq/Isat)))
            
        @cp.fuse(kernel_name="nl_prop_c")
        def nl_prop_c(A1: cp.ndarray, A2: cp.ndarray, dz: float, alpha: float,
                      V: cp.ndarray, g11: float, g12: float,
                      Isat: float) -> None:
            """A fused kernel to apply real space terms
            Args:
                A1 (cp.ndarray): The field to propagate (1st component)
                A2 (cp.ndarray): 2nd component
                dz (float): Propagation step in m
                alpha (float): Losses
                V (cp.ndarray): Potential
                g11 (float): Intra-component interactions
                g12 (float): Inter-component interactions
                Isat (float): Saturation parameter of intra-component interaction
            """
            A_sq_1 = cp.abs(A1)**2
            A_sq_2 = cp.abs(A2)**2
            # Losses
            A1 *= cp.exp(dz*(-alpha/(2*(1+A_sq_1/Isat))))
            # Potential
            A1 *= cp.exp(dz*(1j * V))
            # Interactions
            A1 *= cp.exp(dz*(1j*(g11*A_sq_1/(1+A_sq_1/Isat) + g12*A_sq_2)))
                        

        @cp.fuse(kernel_name="nl_prop_without_V_c")
        def nl_prop_without_V_c(A1: cp.ndarray, A2: cp.ndarray, dz: float, alpha: float,
                                g11: float, g12: float,
                                Isat: float) -> None:
            """A fused kernel to apply real space terms
            Args:
                A1 (cp.ndarray): The field to propagate (1st component)
                A2 (cp.ndarray): 2nd component
                dz (float): Propagation step in m
                alpha (float): Losses
                g11 (float): Intra-component interactions
                g12 (float): Inter-component interactions
                Isat (float): Saturation parameter of intra-component interaction
            """
            
            A_sq_1 = cp.abs(A1)**2
            A_sq_2 = cp.abs(A2)**2
            # Losses
            A1 *= cp.exp(dz*(-alpha/(2*(1+A_sq_1/Isat))))
            # Interactions
            A1 *= cp.exp(dz*(1j*(g11*A_sq_1/(1+A_sq_1/Isat) + g12*A_sq_2)))

        @cp.fuse(kernel_name='vortex_cp')
        def vortex_cp(im: cp.ndarray, i: int, j: int, ii: cp.ndarray, jj: cp.ndarray,
                      ll: int) -> None:
            """Generates a vortex of charge l at a position (i,j) on the image im.

            Args:
                im (np.ndarray): Image
                i (int): position row of the vortex
                j (int): position column of the vortex
                ii (int): meshgrid position row (coordinates of the image)
                jj (int): meshgrid position column (coordinates of the image)
                l (int): vortex charge

            Returns:
                None
            """
            im += cp.angle(((ii-i)+1j*(jj-j))**ll)

    except ImportError:
        print("CuPy not available, falling back to CPU backend ...")
        pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
        BACKEND = "CPU"

        @numba.njit(parallel=True, fastmath=True, cache=True)
        def nl_prop_c(A1: np.ndarray, A2: np.ndarray, dz: float, alpha: float,
                      V: np.ndarray, g11: float, g12: float,
                      Isat: float) -> None:
            """A fused kernel to apply real space terms
            Args:
                A1 (cp.ndarray): The field to propagate (1st component)
                A2 (cp.ndarray): 2nd component
                dz (float): Propagation step in m
                alpha (float): Losses
                V (cp.ndarray): Potential
                g11 (float): Intra-component interactions
                g12 (float): Inter-component interactions
                Isat (float): Saturation parameter of intra-component interaction
            """
            for i in numba.prange(A.shape[0]):
                for j in numba.prange(A.shape[1]):
                    A_sq_1 = np.abs(A1[i, j])**2
                    A_sq_2 = np.abs(A2[i, j])**2
                    # Losses
                    A1[i, j] *= np.exp(dz*(-alpha/(2*(1+A_sq_1/Isat))))
                    # Potential
                    A1[i, j] *= np.exp(dz*(1j * V[i, j]))
                    # Interactions
                    A1[i, j] *= np.exp(dz*(1j*(g11*A_sq_1/(1+A_sq_1/Isat) + g12*A_sq_2)))
                        

        @numba.njit(parallel=True, fastmath=True, cache=True)
        def nl_prop_without_V_c(A1: np.ndarray, A2: np.ndarray, dz: float, alpha: float,
                                g11: float, g12: float,
                                Isat: float) -> None:
            """A fused kernel to apply real space terms
            Args:
                A1 (np.ndarray): The field to propagate (1st component)
                A2 (np.ndarray): 2nd component
                dz (float): Propagation step in m
                alpha (float): Losses
                g11 (float): Intra-component interactions
                g12 (float): Inter-component interactions
                Isat (float): Saturation parameter of intra-component interaction
            """
            for i in numba.prange(A.shape[0]):
                for j in numba.prange(A.shape[1]):
                    A_sq_1 = np.abs(A1[i, j])**2
                    A_sq_2 = np.abs(A2[i, j])**2
                    # Losses
                    A1[i, j] *= np.exp(dz*(-alpha/(2*(1+A_sq_1/Isat))))
                    # Interactions
                    A1[i, j] *= np.exp(dz*(1j*(g11*A_sq_1/(1+A_sq_1/Isat) + g12*A_sq_2)))

        @numba.njit(parallel=True, fastmath=True, cache=True)
        def nl_prop(A: np.ndarray, dz: float, alpha: float, V: np.ndarray, g: float,
                    Isat: float) -> None:
            """A compiled parallel implementation to apply real space terms

            Args:
                A (np.ndarray): The field to propagate
                dz (float): Propagation step in m
                alpha (float): Losses
                V (np.ndarray): Potential
                g (float): Interactions
            """
            for i in numba.prange(A.shape[0]):
                for j in numba.prange(A.shape[1]):
                    A_sq = np.abs(A[i, j])**2
                    A[i, j] *= np.exp(dz*(-alpha/2 + 1j *
                                          V[i, j] + 1j*g*A_sq/(1+A_sq/Isat)))

        @numba.njit(parallel=True, fastmath=True, cache=True)
        def nl_prop_1d(A: np.ndarray, dz: float, alpha: float, V: np.ndarray, g: float, 
                       Isat: float) -> None:
            """A compiled parallel implementation to apply real space terms

            Args:
                A (np.ndarray): The field to propagate
                dz (float): Propagation step in m
                alpha (float): Losses
                V (np.ndarray): Potential
                g (float): Interactions
            """
            for i in numba.prange(A.shape[0]):
                A_sq = np.abs(A[i])**2
                A[i] *= np.exp(dz*(-alpha/2 + 1j *
                                   V[i] + 1j*g*A_sq/(1+A_sq/Isat)))

        @numba.njit(parallel=True, fastmath=True, cache=True)
        def nl_prop_without_V(A: np.ndarray, dz: float, alpha: float, g: float,
                              Isat: float) -> None:
            """A compiled parallel implementation to apply real space terms

            Args:
                A (np.ndarray): The field to propagate
                dz (float): Propagation step in m
                alpha (float): Losses
                g (float): Interactions
            """
            for i in numba.prange(A.shape[0]):
                for j in numba.prange(A.shape[1]):
                    A_sq = np.abs(A[i, j])**2
                    A[i, j] *= np.exp(dz*(-alpha/2 + 1j *
                                      g*A_sq/(1+A_sq/Isat)))

        @numba.njit(parallel=True, fastmath=True, cache=True)
        def nl_prop_without_V_1d(A: np.ndarray, dz: float, alpha: float, g: float,
                                 Isat: float) -> None:
            """A compiled parallel implementation to apply real space terms

            Args:
                A (np.ndarray): The field to propagate
                dz (float): Propagation step in m
                alpha (float): Losses
                g (float): Interactions
            """
            for i in numba.prange(A.shape[0]):
                A_sq = np.abs(A[i])**2
                A[i] *= np.exp(dz*(-alpha/2 + 1j *
                                   g*A_sq/(1+A_sq/Isat)))

        @numba.njit(parallel=True, fastmath=True, cache=True)
        def vortex(im: np.ndarray, i: int, j: int, ii: np.ndarray, jj: np.ndarray, 
                   ll: int) -> None:
            """Generates a vortex of charge l at a position (i,j) on the image im.

            Args:
                im (np.ndarray): Image
                i (int): position row of the vortex
                j (int): position column of the vortex
                ii (int): meshgrid position row (coordinates of the image)
                jj (int): meshgrid position column (coordinates of the image)
                l (int): vortex charge

            Returns:
                None
            """
            for i in numba.prange(A.shape[0]):
                for j in numba.prange(A.shape[1]):
                    im[i, j] += np.angle(((ii[i, j]-i)+1j*(jj[i, j]-j))**ll)
else:
    pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def nl_prop_c(A1: np.ndarray, A2: np.ndarray, dz: float, alpha: float,
                    V: np.ndarray, g11: float, g12: float,
                    Isat: float) -> None:
        """A fused kernel to apply real space terms
        Args:
            A1 (cp.ndarray): The field to propagate (1st component)
            A2 (cp.ndarray): 2nd component
            dz (float): Propagation step in m
            alpha (float): Losses
            V (cp.ndarray): Potential
            g11 (float): Intra-component interactions
            g12 (float): Inter-component interactions
            Isat (float): Saturation parameter of intra-component interaction
        """
        for i in numba.prange(A.shape[0]):
            for j in numba.prange(A.shape[1]):
                A_sq_1 = np.abs(A1[i, j])**2
                A_sq_2 = np.abs(A2[i, j])**2
                # Losses
                A1[i, j] *= np.exp(dz*(-alpha/(2*(1+A_sq_1/Isat))))
                # Potential
                A1[i, j] *= np.exp(dz*(1j * V[i, j]))
                # Interactions
                A1[i, j] *= np.exp(dz*(1j*(g11*A_sq_1/(1+A_sq_1/Isat) + g12*A_sq_2)))
                    

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def nl_prop_without_V_c(A1: np.ndarray, A2: np.ndarray, dz: float, alpha: float,
                            g11: float, g12: float,
                            Isat: float) -> None:
        """A fused kernel to apply real space terms
        Args:
            A1 (np.ndarray): The field to propagate (1st component)
            A2 (np.ndarray): 2nd component
            dz (float): Propagation step in m
            alpha (float): Losses
            g11 (float): Intra-component interactions
            g12 (float): Inter-component interactions
            Isat (float): Saturation parameter of intra-component interaction
        """
        for i in numba.prange(A.shape[0]):
            for j in numba.prange(A.shape[1]):
                A_sq_1 = np.abs(A1[i, j])**2
                A_sq_2 = np.abs(A2[i, j])**2
                # Losses
                A1[i, j] *= np.exp(dz*(-alpha/(2*(1+A_sq_1/Isat))))
                # Interactions
                A1[i, j] *= np.exp(dz*(1j*(g11*A_sq_1/(1+A_sq_1/Isat) + g12*A_sq_2)))

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def nl_prop(A: np.ndarray, dz: float, alpha: float, V: np.ndarray, g: float,
                Isat: float) -> None:
        """A compiled parallel implementation to apply real space terms

        Args:
            A (np.ndarray): The field to propagate
            dz (float): Propagation step in m
            alpha (float): Losses
            V (np.ndarray): Potential
            g (float): Interactions
        """
        for i in numba.prange(A.shape[0]):
            for j in numba.prange(A.shape[1]):
                A_sq = np.abs(A[i, j])**2
                A[i, j] *= np.exp(dz*(-alpha/2 + 1j *
                                      V[i, j] + 1j*g*A_sq/(1+A_sq/Isat)))

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def nl_prop_1d(A: np.ndarray, dz: float, alpha: float, V: np.ndarray, g: float,
                   Isat: float) -> None:
        """A compiled parallel implementation to apply real space terms

        Args:
            A (np.ndarray): The field to propagate
            dz (float): Propagation step in m
            alpha (float): Losses
            V (np.ndarray): Potential
            g (float): Interactions
        """
        for i in numba.prange(A.shape[0]):
            A_sq = np.abs(A[i])**2
            A[i] *= np.exp(dz*(-alpha/2 + 1j *
                               V[i] + 1j*g*A_sq/(1+A_sq/Isat)))

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def nl_prop_without_V(A: np.ndarray, dz: float, alpha: float, g: float, 
                          Isat: float) -> None:
        """A compiled parallel implementation to apply real space terms

        Args:
            A (np.ndarray): The field to propagate
            dz (float): Propagation step in m
            alpha (float): Losses
            g (float): Interactions
        """
        for i in numba.prange(A.shape[0]):
            for j in numba.prange(A.shape[1]):
                A_sq = np.abs(A[i, j])**2
                A[i, j] *= np.exp(dz*(-alpha/2 + 1j *
                                      g*A_sq/(1+A_sq/Isat)))

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def nl_prop_without_V_1d(A: np.ndarray, dz: float, alpha: float, g: float, 
                             Isat: float) -> None:
        """A compiled parallel implementation to apply real space terms

        Args:
            A (np.ndarray): The field to propagate
            dz (float): Propagation step in m
            alpha (float): Losses
            g (float): Interactions
        """
        for i in numba.prange(A.shape[0]):
            A_sq = np.abs(A[i])**2
            A[i] *= np.exp(dz*(-alpha/2 + 1j *
                               g*A_sq/(1+A_sq/Isat)))

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def vortex(im: np.ndarray, i: int, j: int, ii: np.ndarray, jj: np.ndarray, 
               ll: int) -> None:
        """Generates a vortex of charge l at a position (i,j) on the image im.

        Args:
            im (np.ndarray): Image
            i (int): position row of the vortex
            j (int): position column of the vortex
            ii (int): meshgrid position row (coordinates of the image)
            jj (int): meshgrid position column (coordinates of the image)
            l (int): vortex charge

        Returns:
            None
        """
        for i in numba.prange(A.shape[0]):
            for j in numba.prange(A.shape[1]):
                im[i, j] += np.angle(((ii[i, j]-i)+1j*(jj[i, j]-j))**ll)


class NLSE:
    """A class to solve NLSE
    """

    def __init__(self, trans: float, puiss: float, waist: float, window: float, 
                 n2: float, V: np.ndarray, L: float, NX: int = 1024, 
                 NY: int = 1024, Isat: float = np.inf) -> object:
        """Instantiates the simulation.
        Solves an equation : d/dz psi = -1/2k0(d2/dx2 + d2/dy2) psi + k0 dn psi + k0 n2 psi**2 psi
        Args:
            trans (float): Transmission
            puiss (float): Power in W
            waist (float): Waist size in m
            n2 (float): Non linear coeff in m^2/W
            V (np.ndarray) : Potential
        """
        # listof physical parameters
        self.n2 = n2
        self.V = V
        self.waist = waist
        self.wl = 780e-9
        self.z_r = self.waist**2 * np.pi/self.wl
        self.k = 2 * np.pi / self.wl
        self.L = L  # length of the non linear medium
        self.alpha = -np.log(trans)/self.L
        self.puiss = puiss
        self.I_sat = Isat
        # number of grid points in X (even, best is power of 2 or low prime factors)
        self.NX = NX
        self.NY = NY
        self.window = window
        z_nl = 1/(self.k*abs(self.Dn))
        self.delta_z = min(0.1e-5*self.z_r, z_nl)
        # transverse coordinate
        self.X, self.delta_X = np.linspace(-self.window/2, self.window/2, num=NX,
                                           endpoint=False, retstep=True, dtype=np.float32)
        self.Y, self.delta_Y = np.linspace(-self.window/2, self.window/2, num=NY,
                                           endpoint=False, retstep=True, dtype=np.float32)

        self.XX, self.YY = np.meshgrid(self.X, self.Y)
        # definition of the Fourier frequencies for the linear step
        self.Kx = 2 * np.pi * np.fft.fftfreq(self.NX, d=self.delta_X)
        self.Ky = 2 * np.pi * np.fft.fftfreq(self.NY, d=self.delta_Y)
        self.Kxx, self.Kyy = np.meshgrid(self.Kx, self.Ky)

    @property
    def E_00(self):
        intens = 2*self.puiss/(np.pi*self.waist**2)
        return np.sqrt(2*intens/(c*epsilon_0))

    @property
    def Dn(self):
        intens = 2*self.puiss/(np.pi*self.waist**2)
        return self.n2*intens

    def plot_2d(self, ax, Z, X, AMP, title, cmap='viridis', label=r'$X$ (mm)'):
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
        im = ax.imshow(AMP, aspect='equal', origin='lower', extent=(
            Z[0], Z[-1], X[0], X[-1]), cmap=cmap)
        ax.set_xlabel(label)
        ax.set_ylabel(r'$Y$ (mm)')
        ax.set_title(title)
        plt.colorbar(im)
        return

    def plot_1d(self, ax, T, labelT, AMP, labelAMP, PHASE, labelPHASE, Tmin, Tmax) -> None:
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
        ax.plot(T, AMP, 'b')
        ax.set_xlim([Tmin, Tmax])
        ax.set_xlabel(labelT)
        ax.set_ylabel(labelAMP, color='b')
        ax.tick_params(axis='y', labelcolor='b')
        axbis = ax.twinx()
        axbis.plot(T, PHASE, 'r:')
        axbis.set_ylabel(labelPHASE, color='r')
        axbis.tick_params(axis='y', labelcolor='r')

        return

    def plot_1d_amp(self, ax, T, labelT, AMP, labelAMP, Tmin, Tmax, color='b') -> None:
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
        ax.tick_params(axis='y', labelcolor='b')

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
        zoom_x = self.delta_X/d_slm
        zoom_y = self.delta_Y/d_slm
        phase_zoomed = zoom(pattern, (zoom_y, zoom_x), order=0)
        # compute center offset
        x_center = (self.NX - phase_zoomed.shape[1]) // 2
        y_center = (self.NY - phase_zoomed.shape[0]) // 2

        # copy img image into center of result image
        phase[y_center:y_center+phase_zoomed.shape[0],
              x_center:x_center+phase_zoomed.shape[1]] = phase_zoomed
        return phase

    def build_propagator(self, k: float, precision: str = "single") -> np.ndarray:
        """Builds the linear propagation matrix

        Args:
            precision (str, optional): "single" or "double" application of the propagator.
            Defaults to "single".
        Returns:
            propagator (np.ndarray): the propagator matrix
        """
        if precision == "double":
            propagator = np.exp(-1j * 0.25 * (self.Kxx**2 + self.Kyy**2) /
                                k * self.delta_z)
        else:
            propagator = np.exp(-1j * 0.5 * (self.Kxx**2 + self.Kyy**2) /
                                k * self.delta_z)
        if BACKEND == "GPU":
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
        if BACKEND == "GPU":
            # plan_fft = fftpack.get_fft_plan(
            #     A, shape=A.shape, axes=(-2, -1), value_type='C2C')
            plan_fft = fftpack.get_fft_plan(
                A, shape=(A.shape[-2], A.shape[-1]), axes=(-2, -1), value_type='C2C')
            return [plan_fft]
        else:
            # try to load previous fftw wisdom
            try:
                with open("fft.wisdom", "rb") as file:
                    wisdom = pickle.load(file)
                    pyfftw.import_wisdom(wisdom)
            except FileNotFoundError:
                print("No FFT wisdom found, starting over ...")
            plan_fft = pyfftw.FFTW(A, A, direction="FFTW_FORWARD",
                                   flags=("FFTW_PATIENT",),
                                   threads=multiprocessing.cpu_count(),
                                   axes=(-2, -1))
            plan_ifft = pyfftw.FFTW(A, A, direction="FFTW_BACKWARD",
                                    flags=("FFTW_PATIENT",),
                                    threads=multiprocessing.cpu_count(),
                                    axes=(-2, -1))
            with open("fft.wisdom", "wb") as file:
                wisdom = pyfftw.export_wisdom()
                pickle.dump(wisdom, file)
            return [plan_fft, plan_ifft]

    def split_step(self, A: np.ndarray, V: np.ndarray, propagator: np.ndarray, 
                   plans: list, precision: str = "single"):
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
        if BACKEND == "GPU":
            # on GPU, only one plan for both FFT directions
            plan_fft = plans[0]
            plan_fft.fft(A, A, cp.cuda.cufft.CUFFT_FORWARD)
            # linear step in Fourier domain (shifted)
            cp.multiply(A, propagator, out=A)
            plan_fft.fft(A, A, cp.cuda.cufft.CUFFT_INVERSE)
            # fft normalization
            A /= A.shape[-2]*A.shape[-1]
            if V is None:
                nl_prop_without_V(A, self.delta_z, self.alpha,
                                  self.k/2*self.n2*c*epsilon_0, 
                                  2*self.I_sat/(epsilon_0*c))
            else:
                nl_prop(A, self.delta_z, self.alpha, self.k/2 *
                        V, self.k/2*self.n2*c*epsilon_0, 2*self.I_sat/(epsilon_0*c))
            if precision == "double":
                plan_fft.fft(A, A, cp.cuda.cufft.CUFFT_FORWARD)
                # linear step in Fourier domain (shifted)
                cp.multiply(A, propagator, out=A)
                plan_fft.fft(A, A, cp.cuda.cufft.CUFFT_INVERSE)
                A /= A.shape[-2]*A.shape[-1]
        else:
            plan_fft, plan_ifft = plans
            plan_fft(input_array=A, output_array=A)
            np.multiply(A, propagator, out=A)
            plan_ifft(input_array=A, output_array=A, normalise_idft=True)
            if V is None:
                nl_prop_without_V(A, self.delta_z, self.alpha,
                                  self.k/2*self.n2*c*epsilon_0, 
                                  2*self.I_sat/(epsilon_0*c))
            else:
                nl_prop(A, self.delta_z, self.alpha, self.k/2 *
                        V, self.k/2*self.n2*c*epsilon_0, 2*self.I_sat/(epsilon_0*c))
            if precision == "double":
                plan_fft(input_array=A, output_array=A)
                np.multiply(A, propagator, out=A)
                plan_ifft(input_array=A, output_array=A, normalise_idft=True)

    def out_field(self, E_in: np.ndarray, z: float, plot=False, 
                  precision: str = "single", verbose: bool = True) -> np.ndarray:
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
        Z = np.arange(0, z, step=self.delta_z, dtype=np.float32)
        if BACKEND == "GPU":
            if type(E_in) == np.ndarray:
                # A = np.empty((self.NX, self.NY), dtype=np.complex64)
                A = np.empty(E_in.shape, dtype=np.complex64)
                return_np_array = True
            elif type(E_in) == cp.ndarray:
                # A = cp.empty((self.NX, self.NY), dtype=np.complex64)
                A = cp.empty(E_in.shape, dtype=np.complex64)
                return_np_array = False
        else:
            return_np_array = True
            A = pyfftw.empty_aligned((self.NX, self.NY), dtype=np.complex64)
        plans = self.build_fft_plan(A)
        A[:] = self.E_00*E_in
        propagator = self.build_propagator(self.k, precision)
        if BACKEND == "GPU":
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

        if BACKEND == "GPU":
            start_gpu = cp.cuda.Event()
            end_gpu = cp.cuda.Event()
            start_gpu.record()
        t0 = time.perf_counter()
        n2_old = self.n2
        if verbose:
            pbar = tqdm.tqdm(total=len(Z), position=4, desc='Iteration', leave=False)
        for i, z in enumerate(Z):
            if z > self.L:
                self.n2 = 0
            if verbose:
                pbar.update(1)
            self.split_step(A, V, propagator, plans, precision)
        if verbose:
            pbar.close()

        if BACKEND == "GPU":
            end_gpu.record()
            end_gpu.synchronize()
            t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
        if verbose:
            if BACKEND == "GPU":
                print(
                    f"\nTime spent to solve : {t_gpu*1e-3} s (GPU) / {time.perf_counter()-t0} s (CPU)")
            else:
                print(
                    f"\nTime spent to solve : {time.perf_counter()-t0} s (CPU)")
        self.n2 = n2_old
        if BACKEND == "GPU" and return_np_array:
            A = cp.asnumpy(A)

        if plot:
            if not (return_np_array):
                A_plot = cp.asnumpy(A)
            elif return_np_array or BACKEND == 'CPU':
                A_plot = A.copy()
            fig = plt.figure(3, [9, 8])

            # plot amplitudes and phases
            a1 = fig.add_subplot(221)
            self.plot_2d(a1, self.X*1e3, self.Y*1e3, np.abs(A_plot),
                         r'$|\psi|$')

            a2 = fig.add_subplot(222)
            self.plot_2d(a2, self.X*1e3, self.Y*1e3,
                         np.angle(A_plot), r'arg$(\psi)$', cmap='twilight')

            a3 = fig.add_subplot(223)
            lim = 1
            im_fft = np.abs(np.fft.fftshift(
                np.fft.fft2(A_plot[lim:-lim, lim:-lim])))
            Kx_2 = 2 * np.pi * np.fft.fftfreq(self.NX-2*lim, d=self.delta_X)
            len_fft = len(im_fft[0, :])
            self.plot_2d(a3, np.fft.fftshift(Kx_2), np.fft.fftshift(Kx_2), im_fft,
                         r'$|\mathcal{TF}(E_{out})|^2$', cmap='nipy_spectral', label=r'$K_y$')

            a4 = fig.add_subplot(224)
            self.plot_1d_amp(a4, Kx_2[1:-len_fft//2]*1e-3, r'$K_y (mm^{-1})$', im_fft[len_fft//2, len_fft//2+1:],
                             r'$|\mathcal{TF}(E_{out})|$', np.fft.fftshift(Kx_2)[len_fft//2+1]*1e-3, np.fft.fftshift(Kx_2)[-1]*1e-3, color='b')
            a4.set_yscale('log')
            a4.set_xscale('log')

            plt.tight_layout()
            plt.show()
        return A


class NLSE_1d:
    """A class to solve NLSE in 1d
    """

    def __init__(self, alpha: float, puiss: float, waist: float, window: float, n2: float, V: np.ndarray, L: float, NX: int = 1024) -> None:
        """Instantiates the simulation.
        Solves an equation : d/dz psi = -1/2k0(d2/dx2 + d2/dy2) psi + k0 dn psi + k0 n2 psi**2 psi
        Args:
            alpha (float): Transmission coeff
            puiss (float): Power in W
            waist (float): Waist size in m
            n2 (float): Non linear coeff in m^2/W
            V (np.ndarray) : Potential
        """
        # listof physical parameters
        self.n2 = n2
        self.V = V
        self.waist = waist
        self.wl = 780e-9
        self.z_r = self.waist**2 * np.pi/self.wl
        self.k = 2 * np.pi / self.wl
        self.L = L  # length of the non linear medium
        self.alpha = alpha
        self.puiss = puiss
        self.I_sat = np.inf

        # number of grid points in X (even, best is power of 2 or low prime factors)
        self.NX = NX
        self.window = window
        z_nl = 1/(self.k*abs(self.Dn))
        self.delta_z = min(0.1e-5*self.z_r, z_nl)
        # transverse coordinate
        self.X, self.delta_X = np.linspace(-self.window/2, self.window/2, num=NX,
                                           endpoint=False, retstep=True, dtype=np.float32)
        # definition of the Fourier frequencies for the linear step
        self.Kx = 2 * np.pi * np.fft.fftfreq(self.NX, d=self.delta_X)

    @property
    def E_00(self):
        intens = self.puiss/(np.pi*self.waist**2)
        return np.sqrt(2*intens/(c*epsilon_0))

    @property
    def Dn(self):
        intens = self.puiss/(np.pi*self.waist**2)
        return self.n2*intens

    def build_propagator(self, precision: str = "single") -> np.ndarray:
        """Builds the linear propagation matrix

        Args:
            precision (str, optional): "single" or "double" application of the propagator.
            Defaults to "single".
        Returns:
            propagator (np.ndarray): the propagator matrix
        """
        if precision == "double":
            propagator = np.exp(-1j * 0.25 * (self.Kx**2) /
                                self.k * self.delta_z)
        else:
            propagator = np.exp(-1j * 0.5 * (self.Kx**2) /
                                self.k * self.delta_z)
        if BACKEND == "GPU":
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
        if BACKEND == "GPU":
            plan_fft = fftpack.get_fft_plan(
                A, axes=-1, value_type='C2C')
            return [plan_fft]
        else:
            # try to load previous fftw wisdom
            try:
                with open("fft.wisdom", "rb") as file:
                    wisdom = pickle.load(file)
                    pyfftw.import_wisdom(wisdom)
            except FileNotFoundError:
                print("No FFT wisdom found, starting over ...")
            plan_fft = pyfftw.FFTW(A, A, direction="FFTW_FORWARD",
                                   flags=("FFTW_PATIENT",),
                                   threads=multiprocessing.cpu_count(),
                                   axes=(-1,))
            plan_ifft = pyfftw.FFTW(A, A, direction="FFTW_BACKWARD",
                                    flags=("FFTW_PATIENT",),
                                    threads=multiprocessing.cpu_count(),
                                    axes=(-1,))
            with open("fft.wisdom", "wb") as file:
                wisdom = pyfftw.export_wisdom()
                pickle.dump(wisdom, file)
            return [plan_fft, plan_ifft]

    def split_step(self, A: np.ndarray, V: np.ndarray, propagator: np.ndarray, plans: list, precision: str = "single"):
        """Split step function for one propagation step

        Args:
            A (np.ndarray): Field to propagate
            V (np.ndarray): Potential field (can be None).
            propagator (np.ndarray): Propagator matrix.
            plans (list): List of FFT plan objects. Either a single FFT plan for both directions
            (GPU case) or distinct FFT and IFFT plans for FFTW.
            precision (str, optional): Single or double application of the linear propagation step.
            Defaults to "single".
        """
        if BACKEND == "GPU":
            # on GPU, only one plan for both FFT directions
            plan_fft = plans[0]
            plan_fft.fft(A, A, cp.cuda.cufft.CUFFT_FORWARD)
            # linear step in Fourier domain (shifted)
            cp.multiply(A, propagator, out=A)
            plan_fft.fft(A, A, cp.cuda.cufft.CUFFT_INVERSE)
            # fft normalization
            A /= A.shape[-1]
            if V is None:
                nl_prop_without_V(A, self.delta_z, self.alpha,
                                  self.k/2*self.n2*c*epsilon_0, 2*self.I_sat/(epsilon_0*c))
            else:
                nl_prop(A, self.delta_z, self.alpha, self.k/2 *
                        V, self.k/2*self.n2*c*epsilon_0, 2*self.I_sat/(epsilon_0*c))
            if precision == "double":
                plan_fft.fft(A, A, cp.cuda.cufft.CUFFT_FORWARD)
                # linear step in Fourier domain (shifted)
                cp.multiply(A, propagator, out=A)
                plan_fft.fft(A, A, cp.cuda.cufft.CUFFT_INVERSE)
                A /= A.shape[-1]
        else:
            plan_fft, plan_ifft = plans
            plan_fft(input_array=A, output_array=A)
            np.multiply(A, propagator, out=A)
            plan_ifft(input_array=A, output_array=A, normalise_idft=True)
            if V is None:
                nl_prop_without_V_1d(A, self.delta_z, self.alpha,
                                     self.k/2*self.n2*c*epsilon_0, 2*self.I_sat/(epsilon_0*c))
            else:
                nl_prop_1d(A, self.delta_z, self.alpha, self.k/2 *
                           V, self.k/2*self.n2*c*epsilon_0, 2*self.I_sat/(epsilon_0*c))
            if precision == "double":
                plan_fft(input_array=A, output_array=A)
                np.multiply(A, propagator, out=A)
                plan_ifft(input_array=A, output_array=A, normalise_idft=True)

    def out_field(self, E_in: np.ndarray, z: float, plot=False, precision: str = "single", verbose: bool = True) -> np.ndarray:
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
        assert E_in.shape[-1] == self.NX
        Z = np.arange(0, z, step=self.delta_z, dtype=np.float32)
        if BACKEND == "GPU":
            if type(E_in) == np.ndarray:
                A = np.empty(E_in.shape, dtype=np.complex64)
                return_np_array = True
            elif type(E_in) == cp.ndarray:
                A = cp.empty(E_in.shape, dtype=np.complex64)
                return_np_array = False
        else:
            return_np_array = True
            A = pyfftw.empty_aligned(E_in.shape, dtype=np.complex64)
        plans = self.build_fft_plan(A)
        A[:] = self.E_00*E_in
        propagator = self.build_propagator(precision)
        if BACKEND == "GPU":
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

        if BACKEND == "GPU":
            start_gpu = cp.cuda.Event()
            end_gpu = cp.cuda.Event()
            start_gpu.record()
        t0 = time.perf_counter()
        n2_old = self.n2
        if verbose:
            pbar = progressbar.ProgressBar(max_value=len(Z))
        for i, z in enumerate(Z):
            if z > self.L:
                self.n2 = 0
            if verbose:
                pbar.update(i+1)
            self.split_step(A, V, propagator, plans, precision)

        if BACKEND == "GPU":
            end_gpu.record()
            end_gpu.synchronize()
            t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
        if verbose:
            if BACKEND == "GPU":
                print(
                    f"\nTime spent to solve : {t_gpu*1e-3} s (GPU) / {time.perf_counter()-t0} s (CPU)")
            else:
                print(
                    f"\nTime spent to solve : {time.perf_counter()-t0} s (CPU)")
        self.n2 = n2_old
        if BACKEND == "GPU" and return_np_array:
            A = cp.asnumpy(A)

        if plot == True:
            if not (return_np_array):
                A_plot = cp.asnumpy(A)
            elif return_np_array or BACKEND == 'CPU':
                A_plot = A.copy()
            fig, ax = plt.subplots(1, 2)
            if A.ndim == 2:
                for i in range(A.shape[0]):
                    ax[0].plot(self.X, np.unwrap(np.angle(A_plot[i, :])))
                    ax[1].plot(self.X, 1e-4*c/2*epsilon_0 *
                               np.abs(A_plot[i, :])**2)
            elif A.ndim == 1:
                ax[0].plot(self.X, np.unwrap(np.angle(A_plot)))
                ax[1].plot(self.X, 1e-4*c/2*epsilon_0*np.abs(A_plot)**2)
            ax[0].set_title("Phase")
            ax[1].set_title(r"Intensity in $W/cm^2$")
            plt.tight_layout()
            plt.show()
        return A

class CNLSE(NLSE):
    """A class to solve the coupled NLSE

    """
    def __init__(self, trans: float, puiss: float, waist: float, window: float, 
                 n2: float, n12: float, V: np.ndarray, L: float, NX: int = 1024, 
                 NY: int = 1024, Isat: float = np.inf) -> object:
        """Instantiates the class with all the relevant physical parameters

        Args:
            trans (float): Transmission through the cell
            puiss (float): Optical power in W
            waist (float): Beam waist in m
            window (float): Computational window in m
            n2 (float): Non linear index of the 1 st component in m^2/W
            n12 (float): Inter component interaction parameter
            V (np.ndarray): Potential landscape in a.u
            L (float): Length of the cell in m
            NX (int, optional): Number of points along x. Defaults to 1024.
            NY (int, optional): Number of points along y. Defaults to 1024.

        Returns:
            object: CNLSE class instance
        """
        super().__init__(trans, puiss, waist, window, n2, V, L, NX, NY, Isat)
        self.n12 = n12
        # initialize intra component 2 interaction parameter
        # to be the same as intra component 1 
        self.n22 = self.n2
        # same for the losses, this is to leave separate attributes so
        # the the user can chose whether or not to unbalence the rates
        self.alpha1 = self.alpha
        self.alpha2 = self.alpha
        # wavenumbers
        self.k1 = self.k
        self.k2 = self.k
        # powers
        self.puiss2 = self.puiss
        # waists
        self.waist2 = self.waist

    @property
    def E_11(self):
        intens = 2*self.puiss2/(np.pi*self.waist2**2)
        return np.sqrt(2*intens/(c*epsilon_0))

    def split_step(self, A1: np.ndarray, A2: np.ndarray, V: np.ndarray, 
                   propagator1: np.ndarray, propagator2: np.ndarray, plans: list, 
                   precision: str = "single") -> None:
        """Split step function for one propagation step

        Args:
            A1 (np.ndarray): Field 1 to propagate. 
            A2 (np.ndarray): Field 2 to propagate
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
        A1_old = A1
        if BACKEND == "GPU":
            # on GPU, only one plan for both FFT directions
            plan_fft = plans[0]
            plan_fft.fft(A1, A1, cp.cuda.cufft.CUFFT_FORWARD)
            plan_fft.fft(A2, A2, cp.cuda.cufft.CUFFT_FORWARD)
            # linear step in Fourier domain (shifted)
            cp.multiply(A1, propagator1, out=A1)
            cp.multiply(A2, propagator2, out=A2)
            plan_fft.fft(A1, A1, cp.cuda.cufft.CUFFT_INVERSE)
            plan_fft.fft(A2, A2, cp.cuda.cufft.CUFFT_INVERSE)
            # fft normalization
            A1 /= A1.shape[-2]*A1.shape[-1]
            A2 /= A2.shape[-2]*A2.shape[-1]
            if V is None:
                nl_prop_without_V_c(A1, A2, self.delta_z, self.alpha1,
                                  self.k/2*self.n2*c*epsilon_0, 
                                  self.k/2*self.n12*c*epsilon_0, 
                                  2*self.I_sat/(epsilon_0*c))
                nl_prop_without_V_c(A2, A1_old, self.delta_z, self.alpha2,
                                  self.k/2*self.n22*c*epsilon_0, 
                                  self.k/2*self.n12*c*epsilon_0,
                                  2*self.I_sat/(epsilon_0*c))
            else:
                nl_prop_c(A1, A2, self.delta_z, self.alpha1, self.k/2 * V,
                                  self.k/2*self.n2*c*epsilon_0, 
                                  self.k/2*self.n12*c*epsilon_0, 
                                  2*self.I_sat/(epsilon_0*c))
                nl_prop_c(A2, A1_old, self.delta_z, self.alpha2, self.k/2 * V,
                                  self.k/2*self.n22*c*epsilon_0, 
                                  self.k/2*self.n12*c*epsilon_0,
                                  2*self.I_sat/(epsilon_0*c))
            if precision == "double":
                plan_fft.fft(A1, A1, cp.cuda.cufft.CUFFT_FORWARD)
                plan_fft.fft(A2, A2, cp.cuda.cufft.CUFFT_FORWARD)
                # linear step in Fourier domain (shifted)
                cp.multiply(A1, propagator1, out=A1)
                cp.multiply(A2, propagator2, out=A2)
                plan_fft.fft(A1, A1, cp.cuda.cufft.CUFFT_INVERSE)
                plan_fft.fft(A2, A2, cp.cuda.cufft.CUFFT_INVERSE)
                A1 /= A1.shape[-2]*A1.shape[-1]
                A2 /= A2.shape[-2]*A2.shape[-1]
        else:
            plan_fft, plan_ifft = plans
            plan_fft(input_array=A1, output_array=A1)
            plan_fft(input_array=A2, output_array=A2)
            np.multiply(A1, propagator1, out=A1)
            np.multiply(A2, propagator2, out=A2)
            plan_ifft(input_array=A1, output_array=A1, normalise_idft=True)
            plan_ifft(input_array=A2, output_array=A2, normalise_idft=True)
            if V is None:
                nl_prop_without_V_c(A1, A2, self.delta_z, self.alpha1,
                                  self.k/2*self.n2*c*epsilon_0, 
                                  self.k/2*self.n12*c*epsilon_0, 
                                  2*self.I_sat/(epsilon_0*c))
                nl_prop_without_V_c(A2, A1_old, self.delta_z, self.alpha2,
                                  self.k/2*self.n22*c*epsilon_0, 
                                  self.k/2*self.n12*c*epsilon_0,
                                  2*self.I_sat/(epsilon_0*c))
            else:
                nl_prop_c(A1, A2, self.delta_z, self.alpha1, self.k/2 * V,
                                  self.k/2*self.n2*c*epsilon_0, 
                                  self.k/2*self.n12*c*epsilon_0, 
                                  2*self.I_sat/(epsilon_0*c))
                nl_prop_c(A2, A1_old, self.delta_z, self.alpha2, self.k/2 * V,
                                  self.k/2*self.n22*c*epsilon_0, 
                                  self.k/2*self.n12*c*epsilon_0,
                                  2*self.I_sat/(epsilon_0*c))
            if precision == "double":
                plan_fft(input_array=A1, output_array=A1)
                plan_fft(input_array=A2, output_array=A2)
                np.multiply(A1, propagator1, out=A1)
                np.multiply(A2, propagator2, out=A2)
                plan_ifft(input_array=A1, output_array=A1, normalise_idft=True)
                plan_ifft(input_array=A2, output_array=A2, normalise_idft=True)

    def out_field(self, E1: np.ndarray, E2: np.ndarray,  z: float, plot=False,
                  precision: str = "single", verbose: bool = True) -> np.ndarray:
        """Propagates the field at a distance z
        Args:
            E1 (np.ndarray): Field component 1
            E2 (np.ndarray): Field component 2
            z (float): propagation distance in m
            plot (bool, optional): Plots the results. Defaults to False.
            precision (str, optional): Does a "double" or a "single" application
            of the propagator. This leads to a dz (single) or dz^3 (double) precision.
            Defaults to "single".
            verbose (bool, optional): Prints progress and time. Defaults to True.
        Returns:
            np.ndarray: Propagated field in proper units V/m
        """
        assert E1.shape[0] == self.NY and E1.shape[1] == self.NX
        Z = np.arange(0, z, step=self.delta_z, dtype=np.float32)
        if BACKEND == "GPU":
            if type(E1) == np.ndarray:
                A1 = np.empty((self.NX, self.NY), dtype=np.complex64)
                A2 = np.empty((self.NX, self.NY), dtype=np.complex64)
                return_np_array = True
            elif type(E1) == cp.ndarray:
                A1 = cp.empty((self.NX, self.NY), dtype=np.complex64)
                A2 = cp.empty((self.NX, self.NY), dtype=np.complex64)
                return_np_array = False
        else:
            return_np_array = True
            A1 = pyfftw.empty_aligned((self.NX, self.NY), dtype=np.complex64)
            A2 = pyfftw.empty_aligned((self.NX, self.NY), dtype=np.complex64)
        plans = self.build_fft_plan(A1)
        # A_p[:, :] = E_in*np.cos(theta)*self.E_00
        # A_m[:, :] = E_in*np.sin(theta)*np.exp(1j*phi)*self.E_00
        A1[:, :] = self.E_00*E1
        A2[:, :] = self.E_11*E2
        propagator1 = self.build_propagator(self.k1, precision)
        propagator2 = self.build_propagator(self.k2, precision)
        if BACKEND == "GPU":
            if type(self.V) == np.ndarray:
                V = cp.asarray(self.V)
            elif type(self.V) == cp.ndarray:
                V = self.V.copy()
            if self.V is None:
                V = self.V
            if type(A1) != cp.ndarray:
                A1 = cp.asarray(A1)
                A2 = cp.asarray(A2)
        else:
            if self.V is None:
                V = self.V
            else:
                V = self.V.copy()

        if BACKEND == "GPU":
            start_gpu = cp.cuda.Event()
            end_gpu = cp.cuda.Event()
            start_gpu.record()
        t0 = time.perf_counter()
        n2_old = self.n2
        if verbose:
            pbar = tqdm.tqdm(total=len(Z), position=4, desc='Iteration', leave=False)
        for i, z in enumerate(Z):
            if z > self.L:
                self.n2 = 0
                self.n22 = 0
                self.n12 = 0
            if verbose:
                pbar.update(1)
            self.split_step(A1, A2, V, propagator1, propagator2, plans, precision)

        if BACKEND == "GPU":
            end_gpu.record()
            end_gpu.synchronize()
            t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
        if verbose:
            if BACKEND == "GPU":
                print(
                    f"\nTime spent to solve : {t_gpu*1e-3} s (GPU) / {time.perf_counter()-t0} s (CPU)")
            else:
                print(
                    f"\nTime spent to solve : {time.perf_counter()-t0} s (CPU)")
        self.n2 = n2_old
        if BACKEND == "GPU" and return_np_array:
            A1 = cp.asnumpy(A1)
            A2 = cp.asnumpy(A2)

        if plot:
            if not(return_np_array):
                A_1_plot = cp.asnumpy(A1)
                A_2_plot = cp.asnumpy(A2)
            elif return_np_array or BACKEND == 'CPU':
                A_1_plot = A1.copy()
                A_2_plot = A2.copy()
            fig = plt.figure(layout='constrained')
            # plot amplitudes and phases
            a1 = fig.add_subplot(221)
            self.plot_2d(a1, self.X*1e3, self.Y*1e3, np.abs(A_1_plot)**2,
                         r'$|\psi_1|^2$')

            a2 = fig.add_subplot(222)
            self.plot_2d(a2, self.X*1e3, self.Y*1e3,
                         np.angle(A_1_plot), r'arg$(\psi_1)$', cmap='twilight')

            a3 = fig.add_subplot(223)
            self.plot_2d(a3, self.X*1e3, self.Y*1e3, np.abs(A_2_plot)**2,
                         r'$|\psi_2|^2$')

            a4 = fig.add_subplot(224)
            self.plot_2d(a4, self.X*1e3, self.Y*1e3,
                         np.angle(A_2_plot), r'arg$(\psi_2)$', cmap='twilight')
            plt.show()
        return A1, A2
    

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def flatTop_tur(sx: int, sy: int, length: int = 150, width: int = 60,
                k_counter: int = 81, N_steps: int = 81) -> np.ndarray:
    """Generates the phase mask to create two counterstreaming colliding components

    Args:
        sx (int): x Dimension of the mask : slm dimensions.
        sy (int): y Dimension of the mask : slm dimensions.
        length (int, optional): Length of the pattern. Defaults to 150.
        width (int, optional): Width of the pattern. Defaults to 60.
        k_counter (int, optional): Frequency of the blazed grating. Defaults to 81.
        N_steps (int, optional): Frequency of the vertical grating. Defaults to 81.

    Returns:
        _type_: _description_
    """
    output = np.zeros((sy, sx))
    Y, X = np.indices(output.shape)
    output[abs(X-output.shape[1]//2) < length/2] = 1
    output[abs(Y-output.shape[0]//2) > width/2] = 0

    grating_axe = X
    grating_axe = grating_axe % (sx/k_counter)
    grating_axe += abs(np.amin(grating_axe))
    grating_axe /= np.amax(grating_axe)

    grating_axe[X > output.shape[1]//2] *= -1
    grating_axe[X > output.shape[1]//2] += 1

    grating_axe_vert = Y
    grating_axe_vert = grating_axe_vert % (sy/N_steps)
    grating_axe_vert = normalize(grating_axe_vert)

    grating_axe = ((grating_axe+grating_axe_vert) % 1)*output
    return grating_axe


def flatTop_super(sx: int, sy: int, length: int = 150, width: int = 60,
                  k_counter: int = 81, N_steps: int = 81) -> np.ndarray:
    """Generates the phase mask to create two counterstreaming shearing components

    Args:
        sx (int): x Dimension of the mask : slm dimensions.
        sy (int): y Dimension of the mask : slm dimensions.
        length (int, optional): Length of the pattern. Defaults to 150.
        width (int, optional): Width of the pattern. Defaults to 60.
        k_counter (int, optional): Frequency of the blazed grating. Defaults to 81.
        N_steps (int, optional): Frequency of the vertical grating. Defaults to 81.

    Returns:
        _type_: _description_
    """
    output = np.zeros((sy, sx))
    Y, X = np.indices(output.shape)
    output[abs(X-output.shape[1]//2) < length/2] = 1
    output[abs(Y-output.shape[0]//2) > width/2] = 0

    grating_axe = X
    grating_axe = grating_axe % (sx/k_counter)
    grating_axe += abs(np.amin(grating_axe))
    grating_axe /= np.amax(grating_axe)

    grating_axe[Y > output.shape[0]//2] *= -1
    grating_axe[Y > output.shape[0]//2] += 1

    grating_axe_vert = Y
    grating_axe_vert = grating_axe_vert % (sy/N_steps)
    grating_axe_vert = normalize(grating_axe_vert)

    grating_axe = ((grating_axe+grating_axe_vert) % 1)*output
    return grating_axe


if __name__ == "__main__":
    trans = 0.5
    n2 = -1.6e-9
    n12 = 1e-9
    waist = 1e-3
    window = 2048*5.5e-6
    puiss = 500e-3
    Isat = 10e4  # saturation intensity in W/m^2
    L = 5e-2
    dn = 2.5e-4 * np.ones((2048, 2048), dtype=np.complex64)
    simu = NLSE(trans, puiss, waist, window, n2, dn,
                L, NX=2048, NY=2048)
    simu_c = CNLSE(trans, puiss, waist, window, n2, n12, None,
                L, NX=2048, NY=2048)
    simu_1d = NLSE_1d(trans, puiss, waist, window, n2, dn[1024, :],
                      L, NX=2048)
    simu.delta_z = 0.1e-3
    simu_1d.delta_z = 0.1e-3
    simu.I_sat = Isat
    simu_1d.I_sat = Isat
    phase_slm = 2*np.pi * \
        flatTop_super(1272, 1024, length=1000, width=600)
    phase_slm = simu.slm(phase_slm, 6.25e-6)
    E_in_0 = np.ones((simu.NY, simu.NX), dtype=np.complex64) * \
        np.exp(-(simu.XX**2 + simu.YY**2)/(2*simu.waist**2))
    simu.V *= np.exp(-(simu.XX**2 + simu.YY**2)/(2*(simu.waist/3)**2))
    E_in_0 *= np.exp(1j*phase_slm)
    E_in_0 = np.fft.fftshift(np.fft.fft2(E_in_0))
    E_in_0[0:E_in_0.shape[0]//2+20, :] = 1e-10
    E_in_0[E_in_0.shape[0]//2+225:, :] = 1e-10
    E_in_0 = np.fft.ifft2(np.fft.ifftshift(E_in_0))
    A_c = simu_c.out_field(E_in_0, E_in_0, L, plot=True, verbose=True)
    A = simu.out_field(E_in_0, L, plot=True)
    A_1d = simu_1d.out_field(E_in_0[1024:1028, :], L, plot=True)
