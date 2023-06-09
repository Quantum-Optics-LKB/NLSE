#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Taladjidi / Clara

import multiprocessing
import pickle
import sys
import time
import matplotlib.pyplot as plt
import numba
import progressbar
import numpy as np
import pyfftw
from scipy.constants import c, epsilon_0, hbar, mu_0
from scipy.ndimage import zoom

BACKEND = 'GPU'

if BACKEND == 'GPU':
    try:
        import cupy as cp
        import cupyx.scipy.fftpack as fftpack
        BACKEND = "GPU"

        @cp.fuse(kernel_name="nl_prop")
        def nl_prop(A1: cp.ndarray, A2: cp.ndarray, dz: float, alpha: float, V: cp.ndarray, g11: float, g12: float) -> None:
            """A fused kernel to apply real space terms
            Args:
                A1 (cp.ndarray): The field to propagate (1st component)
                A2 (cp.ndarray): 2nd component
                dz (float): Propagation step in m
                alpha (float): Losses
                V (cp.ndarray): Potential
                g11 (float): Intra-component interactions
                g12 (float): Inter-component interactions
            """
            
            A1 *= cp.exp(dz*(-alpha/2 + 1j * V + 1j*(g11*cp.abs(A1)**2 + g12*cp.abs(A2)**2)))

        @cp.fuse(kernel_name="nl_prop_without_V")
        def nl_prop_without_V(A1: cp.ndarray, A2: cp.ndarray, dz: float, alpha: float, g11: float, g12: float) -> None:
            """A fused kernel to apply real space terms
            Args:
                A1 (cp.ndarray): The field to propagate (1st component)
                A2 (cp.ndarray): 2nd component
                dz (float): Propagation step in m
                alpha (float): Losses
                g11 (float): Intra-component interactions
                g12 (float): Inter-component interactions
            """
            
            A1 *= cp.exp(dz*(-alpha/2 + 1j*(g11*cp.abs(A1)**2 + g12*cp.abs(A2)**2)))

        @cp.fuse(kernel_name='vortex_cp')
        def vortex_cp(im: cp.ndarray, i: int, j: int, ii: cp.ndarray, jj: cp.ndarray, l: int) -> None:
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
            im += cp.angle(((ii-i)+1j*(jj-j))**l)

    except ImportError:
        print("CuPy not available, falling back to CPU backend ...")
        pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
        BACKEND = "CPU"

        @numba.njit(parallel=True, fastmath=True)
        def nl_prop(A1: np.ndarray, A2: np.ndarray, dz: float, alpha: float, V: np.ndarray, g11: float, g12: float) -> None:
            """A compiled parallel implementation to apply real space terms
            Args:
                A1 (cp.ndarray): The field to propagate (1st component)
                A2 (cp.ndarray): 2nd component
                dz (float): Propagation step in m
                alpha (float): Losses
                V (cp.ndarray): Potential
                g11 (float): Intra-component interactions
                g12 (float): Inter-component interactions
            """
            for i in numba.prange(A1.shape[0]):
                for j in numba.prange(A1.shape[1]):
                    A1[i, j] *= np.exp(dz*(-alpha/2 + 1j *
                                          V[i, j] + 1j*(g11*abs(A1[i,j])**2 + g12*abs(A2[i,j])**2)))

        @numba.njit(parallel=True, fastmath=True)
        def nl_prop_1d(A1: np.ndarray, A2: np.ndarray, dz: float, alpha: float, V: np.ndarray, g11: float, g12: float) -> None:
            """A compiled parallel implementation to apply real space terms

            Args:
                A (np.ndarray): The field to propagate
                dz (float): Propagation step in m
                alpha (float): Losses
                V (np.ndarray): Potential
                g (float): Interactions
            """
            for i in numba.prange(A.shape[0]):
                A1[i] *= np.exp(dz*(-alpha/2 + 1j *
                                   V[i] + 1j*(g11*np.abs(A1[i])**2 + g12*np.abs(A2[i])**2)))

        @numba.njit(parallel=True, fastmath=True)
        def nl_prop_without_V(A1: np.ndarray, A2: np.ndarray, dz: float, alpha: float, g11: float, g12: float) -> None:
            """A compiled parallel implementation to apply real space terms
            Args:
                A1 (cp.ndarray): The field to propagate (1st component)
                A2 (cp.ndarray): 2nd component
                dz (float): Propagation step in m
                alpha (float): Losses
                g11 (float): Intra-component interactions
                g12 (float): Inter-component interactions
            """
            for i in numba.prange(A1.shape[0]):
                for j in numba.prange(A1.shape[1]):
                    A1[i, j] *= np.exp(dz*(-alpha/2 + 1j*(g11*abs(A1[i,j])**2 + g12*abs(A2[i,j])**2)))

        @numba.njit(parallel=True, fastmath=True)
        def nl_prop_without_V_1d(A1: np.ndarray, A2: np.ndarray, dz: float, alpha: float, g11: float, g12: float) -> None:
            """A compiled parallel implementation to apply real space terms
            Args:
                A1 (cp.ndarray): The field to propagate (1st component)
                A2 (cp.ndarray): 2nd component
                dz (float): Propagation step in m
                alpha (float): Losses
                g11 (float): Intra-component interactions
                g12 (float): Inter-component interactions
            """    
            for i in numba.prange(A1.shape[0]):
                A1[i] *= np.exp(dz*(-alpha/2 + 1j*(g11*abs(A1[i])**2 + g12*abs(A2[i])**2)))


        @numba.njit(parallel=True, fastmath=True)
        def vortex(im: np.ndarray, i: int, j: int, ii: np.ndarray, jj: np.ndarray, l: int) -> None:
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
                    im[i, j] += np.angle(((ii[i, j]-i)+1j*(jj[i, j]-j))**l)
else:
    pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()

    @numba.njit(parallel=True, fastmath=True)
    def nl_prop(A1: np.ndarray, A2: np.ndarray, dz: float, alpha: float, V: np.ndarray, g11: float, g12: float) -> None:
        """A compiled parallel implementation to apply real space terms
        Args:
            A1 (cp.ndarray): The field to propagate (1st component)
            A2 (cp.ndarray): 2nd component
            dz (float): Propagation step in m
            alpha (float): Losses
            V (cp.ndarray): Potential
            g11 (float): Intra-component interactions
            g12 (float): Inter-component interactions
        """
        
        for i in numba.prange(A1.shape[0]):
            for j in numba.prange(A1.shape[1]):
                A1[i, j] *= np.exp(dz*(-alpha/2 + 1j *
                                      V[i, j] + 1j*(g11*abs(A1[i, j])**2+ g12*abs(A2[i, j])**2)))

    @numba.njit(parallel=True, fastmath=True)
    def nl_prop_1d(A1: np.ndarray, A2: np.ndarray, dz: float, alpha: float, V: np.ndarray, g11: float, g12: float) -> None:
        """A compiled parallel implementation to apply real space terms
        Args:
            A1 (cp.ndarray): The field to propagate (1st component)
            A2 (cp.ndarray): 2nd component
            dz (float): Propagation step in m
            alpha (float): Losses
            V (cp.ndarray): Potential
            g11 (float): Intra-component interactions
            g12 (float): Inter-component interactions
        """
        for i in numba.prange(A1.shape[0]):
                A1[i] *= np.exp(dz*(-alpha/2 + 1j *
                                      V[i] + 1j*(g11*abs(A1[i])**2+ g12*abs(A2[i])**2)))                                 

    @numba.njit(parallel=True, fastmath=True)
    def nl_prop_without_V(A1: np.ndarray,A2: np.ndarray, dz: float, alpha: float, g11: float, g12: float) -> None:
        """A compiled parallel implementation to apply real space terms
        Args:
            A1 (cp.ndarray): The field to propagate (1st component)
            A2 (cp.ndarray): 2nd component
            dz (float): Propagation step in m
            alpha (float): Losses
            V (cp.ndarray): Potential
            g11 (float): Intra-component interactions
            g12 (float): Inter-component interactions
        """
        for i in numba.prange(A1.shape[0]):
            for j in numba.prange(A1.shape[1]):
                A1[i, j] *= np.exp(dz*(-alpha/2 + 1j*(g11*abs(A1[i, j])**2 + g12*abs(A2[i, j])**2)))

    @numba.njit(parallel=True, fastmath=True)
    def nl_prop_without_V_1d(A1: np.ndarray,A2: np.ndarray, dz: float, alpha: float, g11: float, g12: float) -> None:
        """A compiled parallel implementation to apply real space terms
        Args:
            A1 (cp.ndarray): The field to propagate (1st component)
            A2 (cp.ndarray): 2nd component
            dz (float): Propagation step in m
            alpha (float): Losses
            V (cp.ndarray): Potential
            g11 (float): Intra-component interactions
            g12 (float): Inter-component interactions
        """
        for i in numba.prange(A1.shape[0]):
                A1[i] *= np.exp(dz*(-alpha/2 + 1j*(g11*abs(A1[i])**2 + g12*abs(A2[i])**2)))

    @numba.njit(parallel=True, fastmath=True)
    def vortex(im: np.ndarray, i: int, j: int, ii: np.ndarray, jj: np.ndarray, l: int) -> None:
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
                im[i, j] += np.angle(((ii[i, j]-i)+1j*(jj[i, j]-j))**l)


class CNLSE:
    """A class to solve CNLSE
    """

    def __init__(self, trans: float, puiss: float, waist: float, window: float, n2: float, ratio_g12_g11: float, V: np.ndarray, L: float, NX: int = 1024, NY: int = 1024,Vramp:bool = False,n2ramp:bool = False,Lramp:float=1e-3) -> None:
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

        self.ratio_g12_g11 = ratio_g12_g11
        self.g11 = self.k/2*self.n2*c*epsilon_0  # linear case (single component, usual case)
        # self.g11 = self.g*2/3  # intra-component interaction, also circular case 
        self.g12 = ratio_g12_g11*self.g11
        self.g_circ = self.g11
        self.g_lin = (self.g11 + self.g12)/2

        # number of grid points in X (even, best is power of 2 or low prime factors)
        self.NX = NX
        self.NY = NY
        self.window = window
        self.delta_z = 1e-4*self.z_r
        if self.n2 !=0:
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

        self.Vramp, self.n2ramp = Vramp, n2ramp
        self.Lramp = Lramp

    @property
    def E_00(self):
        intens = self.puiss/(np.pi*self.waist**2)
        return np.sqrt(2*intens/(c*epsilon_0))

    @property
    def Dn(self):
        intens = self.puiss/(np.pi*self.waist**2)
        return self.n2*intens

    def plot_2d(self, ax, Z, X, AMP, title, cmap='viridis', label=r'$X$ (mm)', vmax=1):
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
            Z[0], Z[-1], X[0], X[-1]), cmap=cmap, vmax=vmax)
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

    def build_propagator(self, precision: str = "single") -> np.ndarray:
        """Builds the linear propagation matrix
        Args:
            precision (str, optional): "single" or "double" application of the propagator.
            Defaults to "single".
        Returns:
            propagator (np.ndarray): the propagator matrix
        """
        if precision == "double":
            propagator = np.exp(-1j * 0.25 * (self.Kxx**2 + self.Kyy**2) /
                                self.k * self.delta_z)
        else:
            propagator = np.exp(-1j * 0.5 * (self.Kxx**2 + self.Kyy**2) /
                                self.k * self.delta_z)



        if BACKEND == "GPU":
            return cp.asarray(propagator)
        else:
            return propagator

    def build_fft_plan(self, A:np.ndarray) -> list:
        """Builds the FFT plan objects for propagation
        Args:
            A (np.ndarray): Array to transform.
        Returns:
            list: A list containing the FFT plans
        """
        if BACKEND == "GPU":
            plan_fft = fftpack.get_fft_plan(
                A, shape=A.shape, axes=(0, 1), value_type='C2C')
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
                                   axes=(0, 1))
            plan_ifft = pyfftw.FFTW(A, A, direction="FFTW_BACKWARD",
                                    flags=("FFTW_PATIENT",),
                                    threads=multiprocessing.cpu_count(),
                                    axes=(0, 1))
            with open("fft.wisdom", "wb") as file:
                wisdom = pyfftw.export_wisdom()
                pickle.dump(wisdom, file)
            return [plan_fft, plan_ifft]

    def split_step(self, A_p: np.ndarray, A_m: np.ndarray, V: np.ndarray, propagator: np.ndarray, plans: list, precision:str="single"):
        """Split step function for one propagation step
        Args:
            A_p (np.ndarray): Field component on sigma_+ to propagate
            A_m (np.ndarray): Field component on sigma_- to propagate
            V (np.ndarray): Potential field (can be None).
            propagator (np.ndarray): Propagator matrix.
            plans (list): List of FFT plan objects. Either a single FFT plan for both directions
            (GPU case) or distinct FFT and IFFT plans for FFTW.
            precision (str, optional): Single or double application of the linear propagation step.
            Defaults to "single".
        """
        A_p_old = A_p
        if BACKEND == "GPU":
            # on GPU, only one plan for both FFT directions
            plan_fft = plans[0]
            plan_fft.fft(A_p, A_p, cp.cuda.cufft.CUFFT_FORWARD)
            plan_fft.fft(A_m, A_m, cp.cuda.cufft.CUFFT_FORWARD)
            # linear step in Fourier domain (shifted)
            cp.multiply(A_p, propagator, out=A_p)
            cp.multiply(A_m, propagator, out=A_m)
            plan_fft.fft(A_p, A_p, cp.cuda.cufft.CUFFT_INVERSE)
            plan_fft.fft(A_m, A_m, cp.cuda.cufft.CUFFT_INVERSE)
            # fft normalization
            A_p /= np.prod(A_p.shape)
            A_m /= np.prod(A_m.shape)
            
            if V is None:
                nl_prop_without_V(A_p, A_m, self.delta_z, self.alpha,
                                    self.g11, self.g12)
                nl_prop_without_V(A_m, A_p_old, self.delta_z, self.alpha,
                                    self.g11, self.g12)
            else:
                nl_prop(A_p, A_m, self.delta_z, self.alpha,
                        V, self.g11, self.g12)
                nl_prop(A_m, A_p_old, self.delta_z, self.alpha,
                        V, self.g11, self.g12)
            if precision == "double":
                plan_fft.fft(A_p, A_p, cp.cuda.cufft.CUFFT_FORWARD)
                plan_fft.fft(A_m, A_m, cp.cuda.cufft.CUFFT_FORWARD)
                # linear step in Fourier domain (shifted)
                cp.multiply(A_p, propagator, out=A_p)
                cp.multiply(A_m, propagator, out=A_m)
                plan_fft.fft(A_p, A_p, cp.cuda.cufft.CUFFT_INVERSE)
                plan_fft.fft(A_m, A_m, cp.cuda.cufft.CUFFT_INVERSE)
                A_p /= np.prod(A_p.shape)
                A_m /= np.prod(A_m.shape)
        else:
            plan_fft, plan_ifft = plans
            plan_fft(input_array=A_p, output_array=A_p)
            plan_fft(input_array=A_m, output_array=A_m)
            np.multiply(A_p, propagator, out=A_p)
            np.multiply(A_m, propagator, out=A_m)
            plan_ifft(input_array=A_p, output_array=A_p, normalise_idft=True)
            plan_ifft(input_array=A_m, output_array=A_m, normalise_idft=True)
            if V is None:
                nl_prop_without_V(A_p, A_m, self.delta_z, self.alpha,
                                    self.g11, self.g12)
                nl_prop_without_V(A_m, A_p_old, self.delta_z, self.alpha,
                                    self.g11, self.g12)
            else:
                nl_prop(A_p, A_m, self.delta_z, self.alpha,
                        V, self.g11, self.g12)
                nl_prop(A_m, A_p_old, self.delta_z, self.alpha,
                        V, self.g11, self.g12)
            if precision == "double":
                plan_fft(input_array=A_p, output_array=A_p)
                plan_fft(input_array=A_m, output_array=A_m)
                np.multiply(A_p, propagator, out=A_p)
                np.multiply(A_m, propagator, out=A_m)
                plan_ifft(input_array=A_p, output_array=A_p, normalise_idft=False)
                plan_ifft(input_array=A_m, output_array=A_m, normalise_idft=False)

    def out_field(self, E_p_in: np.ndarray, E_m_in: np.ndarray,  z: float, plot=False, precision: str = "single", verbose: bool = True) -> np.ndarray:
        """Propagates the field at a distance z
        Args:
            E_p_in (np.ndarray): Field component on sigma_+
            E_m_in (np.ndarray): Field component on sigma_-
            z (float): propagation distance in m
            plot (bool, optional): Plots the results. Defaults to False.
            precision (str, optional): Does a "double" or a "single" application
            of the propagator. This leads to a dz (single) or dz^3 (double) precision.
            Defaults to "single".
            verbose (bool, optional): Prints progress and time. Defaults to True.
        Returns:
            np.ndarray: Propagated field in proper units V/m
        """
        assert E_p_in.shape[0] == self.NY and E_p_in.shape[1] == self.NX
        Z = np.arange(0, z, step=self.delta_z, dtype=np.float32)
        if BACKEND == "GPU":
            if type(E_p_in) == np.ndarray:
                A_p = np.empty((self.NX, self.NY), dtype=np.complex64)
                A_m = np.empty((self.NX, self.NY), dtype=np.complex64)
                return_np_array = True
            elif type(E_p_in) == cp.ndarray:
                A_p = cp.empty((self.NX, self.NY), dtype=np.complex64)
                A_m = cp.empty((self.NX, self.NY), dtype=np.complex64)
                return_np_array = False
        else:
            return_np_array = True
            A_p = pyfftw.empty_aligned((self.NX, self.NY), dtype=np.complex64)
            A_m = pyfftw.empty_aligned((self.NX, self.NY), dtype=np.complex64)
        plans = self.build_fft_plan(A_p)
        # A_p[:, :] = E_in*np.cos(theta)*self.E_00
        # A_m[:, :] = E_in*np.sin(theta)*np.exp(1j*phi)*self.E_00
        A_p[:, :] = E_p_in
        A_m[:, :] = E_m_in
        propagator = self.build_propagator(precision)
        if BACKEND == "GPU":
            if type(self.V) == np.ndarray:
                V = cp.asarray(self.V)
            elif type(self.V) == cp.ndarray:
                V = self.V.copy()
            if self.V is None:
                V = self.V
            if type(A_p) != cp.ndarray:
                A_p = cp.asarray(A_p)
                A_m = cp.asarray(A_m)
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
        for i, z in enumerate(Z):
            if z > self.L:
                self.n2 = 0
            if verbose:
                sys.stdout.write(f"\rIteration {i+1}/{len(Z)}")

            ramp = z/self.Lramp
            ramp = 1/(1+np.exp(-10*(z/self.Lramp-0.5)))
            if self.Vramp:
                V = self.V*min(ramp,1.)
            if self.n2ramp:
                n2 = self.n2*min(ramp,1.)

            self.split_step(A_p, A_m, V, propagator, plans, precision)

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
            A_p = cp.asnumpy(A_p)
            A_m = cp.asnumpy(A_m)

        if plot == True:
            if not(return_np_array):
                A_p_plot = cp.asnumpy(A_p)
                A_m_plot = cp.asnumpy(A_m)
            elif return_np_array or BACKEND == 'CPU':
                A_p_plot = A_p.copy()
                A_m_plot = A_m.copy()
            fig = plt.figure()

            # plot amplitudes and phases
            a1 = fig.add_subplot(221)
            self.plot_2d(a1, self.X*1e3, self.Y*1e3, np.abs(A_p_plot)**2,
                         r'$|\psi_+|^2$', vmax=np.max(np.abs(A_p_plot)**2))

            a2 = fig.add_subplot(222)
            self.plot_2d(a2, self.X*1e3, self.Y*1e3,
                         np.angle(A_p_plot), r'arg$(\psi_+)$', cmap='twilight', vmax=np.pi)

            a3 = fig.add_subplot(223)
            self.plot_2d(a3, self.X*1e3, self.Y*1e3, np.abs(A_m_plot)**2,
                         r'$|\psi_-|^2$', vmax=np.max(np.abs(A_m_plot)**2))

            a4 = fig.add_subplot(224)
            self.plot_2d(a4, self.X*1e3, self.Y*1e3,
                         np.angle(A_m_plot), r'arg$(\psi_-)$', cmap='twilight', vmax=np.pi)

            # a3 = fig.add_subplot(223)
            # lim = 1
            # im_fft = np.abs(np.fft.fftshift(
            #     np.fft.fft2(np.abs(A_plot[lim:-lim, lim:-lim])**2)))
            # Kx_2 = 2 * np.pi * np.fft.fftfreq(self.NX-2*lim, d=self.delta_X)
            # len_fft = len(im_fft[0, :])
            # sefrom cnlse import CNLSElf.plot_2d(a3, np.fft.fftshift(Kx_2), np.fft.fftshift(Kx_2), np.log10(im_fft),
            #              r'$\mathcal{TF}(|E_{out}|^2)$', cmap='viridis', label=r'$K_y$', vmax=np.max(np.log10(im_fft)))

            # a4 = fig.add_subplot(224)
            # self.plot_1d_amp(a4, Kx_2[1:-len_fft//2]*1e-3, r'$K_y (mm^{-1})$', im_fft[len_fft//2, len_fft//2+1:],
            #                  r'$\mathcal{TF}(|E_{out}|^2)$', np.fft.fftshift(Kx_2)[len_fft//2+1]*1e-3, np.fft.fftshift(Kx_2)[-1]*1e-3, color='b')
            # a4.set_yscale('log')
            # a4.set_xscale('log')

            plt.tight_layout()
            plt.show()
        return A_p, A_m


class CNLSE_1d:
    """A class to solve CNLSE in 1d
    """

    def __init__(self, alpha: float, puiss: float, waist: float, window: float, n2: float,  V: np.ndarray, L: float, NX: int = 1024) -> None:
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

        # self.ratio_g12_g11 = ratio_g12_g11
        self.g11 = self.k/2*self.n2*c*epsilon_0  # linear case (single component, usual case)
        # self.g11 = self.g*2/3  # intra-component interaction, also circular case 
        # self.g12 = ratio_g12_g11*self.g11
        # self.g_circ = self.g11
        # self.g_lin = (self.g11 + self.g12)/2

        # number of grid points in X (even, best is power of 2 or low prime factors)
        self.NX = NX
        self.window = window
        self.delta_z = 1e-4*self.z_r
        if self.n2 !=0:
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

    def split_step(self, A_p: np.ndarray, A_m: np.ndarray, g12: float,V: np.ndarray, propagator: np.ndarray, plans: list, precision: str = "single"):
        """Split step function for one propagation step

        Args:
            A_p (np.ndarray): Field component on sigma_+ to propagate
            A_m (np.ndarray): Field component on sigma_- to propagate
            V (np.ndarray): Potential field (can be None).
            propagator (np.ndarray): Propagator matrix.
            plans (list): List of FFT plan objects. Either a single FFT plan for both directions
            (GPU case) or distinct FFT and IFFT plans for FFTW.
            precision (str, optional): Single or double application of the linear propagation step.
            Defaults to "single".
        """

        A_p_old = A_p
        if BACKEND == "GPU":
            # on GPU, only one plan for both FFT directions
            plan_fft = plans[0]
            plan_fft.fft(A_p, A_p, cp.cuda.cufft.CUFFT_FORWARD)
            plan_fft.fft(A_m, A_m, cp.cuda.cufft.CUFFT_FORWARD)
            # linear step in Fourier domain (shifted)
            cp.multiply(A_p, propagator, out=A_p)
            cp.multiply(A_m, propagator, out=A_m)
            plan_fft.fft(A_p, A_p, cp.cuda.cufft.CUFFT_INVERSE)
            plan_fft.fft(A_m, A_m, cp.cuda.cufft.CUFFT_INVERSE)
            # fft normalization
            A_p /= A_p.shape[-1]
            A_m /= A_m.shape[-1]
            if V is None:
                nl_prop_without_V(A_p, A_m, self.delta_z, self.alpha,
                                  self.g11, g12)
                nl_prop_without_V(A_m, A_p_old, self.delta_z, self.alpha,
                                  self.g11, g12)
            else:
                nl_prop(A_p, A_m, self.delta_z, self.alpha,V,
                                    self.g11, g12)
                nl_prop(A_m, A_p_old, self.delta_z, self.alpha,V,
                                    self.g11, g12)
            if precision == "double":
                plan_fft.fft(A_p, A_p, cp.cuda.cufft.CUFFT_FORWARD)
                plan_fft.fft(A_m, A_m, cp.cuda.cufft.CUFFT_FORWARD)
                # linear step in Fourier domain (shifted)
                cp.multiply(A_p, propagator, out=A_p)
                cp.multiply(A_m, propagator, out=A_m)
                plan_fft.fft(A_p, A_p, cp.cuda.cufft.CUFFT_INVERSE)
                plan_fft.fft(A_m, A_m, cp.cuda.cufft.CUFFT_INVERSE)
                A_p /= A_p.shape[-1]
                A_m /= A_m.shape[-1]
        else:
            plan_fft, plan_ifft = plans
            plan_fft(input_array=A_p, output_array=A_p)
            plan_fft(input_array=A_m, output_array=A_m)
            np.multiply(A_p, propagator, out=A_p)
            np.multiply(A_m, propagator, out=A_m)
            plan_ifft(input_array=A_p, output_array=A_p, normalise_idft=True)
            plan_ifft(input_array=A_m, output_array=A_m, normalise_idft=True)
            if V is None:
                nl_prop_without_V_1d(A_p, A_m, self.delta_z, self.alpha,
                                    self.g11, g12)
                nl_prop_without_V_1d(A_m, A_p_old, self.delta_z, self.alpha,
                                    self.g11, g12)
            else:
                nl_prop_1d(A_p, A_m, self.delta_z, self.alpha,
                        V, self.g11, g12)
                nl_prop_1d(A_m, A_p_old, self.delta_z, self.alpha,
                        V, self.g11, g12)
            if precision == "double":
                plan_fft(input_array=A_p, output_array=A_p)
                plan_fft(input_array=A_m, output_array=A_m)
                np.multiply(A_p, propagator, out=A_p)
                np.multiply(A_m, propagator, out=A_m)
                plan_ifft(input_array=A_p, output_array=A_p, normalise_idft=False)
                plan_ifft(input_array=A_m, output_array=A_m, normalise_idft=False)

    def out_field(self, E_p_in: np.ndarray, E_m_in: np.ndarray, z: float, ratio_g12_g11: float, plot=False, precision: str = "single", verbose: bool = True) -> np.ndarray:
        """Propagates the field at a distance z
        Args:
            E_p_in (np.ndarray): Field component on sigma_+
            E_m_in (np.ndarray): Field component on sigma_-
            z (float): propagation distance in m
            plot (bool, optional): Plots the results. Defaults to False.
            precision (str, optional): Does a "double" or a "single" application
            of the propagator. This leads to a dz (single) or dz^3 (double) precision.
            Defaults to "single".
            verbose (bool, optional): Prints progress and time. Defaults to True.
        Returns:
            np.ndarray: Propagated field in proper units V/m
        """
        assert E_p_in.shape[-1] == self.NX
        Z = np.arange(0, z, step=self.delta_z, dtype=np.float32)
        if BACKEND == "GPU":
            if type(E_p_in) == np.ndarray:
                A_p = np.empty(E_p_in.shape, dtype=np.complex64)
                A_m = np.empty(E_m_in.shape, dtype=np.complex64)
                return_np_array = True
            elif type(E_p_in) == cp.ndarray:
                A_p = cp.empty(E_p_in.shape, dtype=np.complex64)
                A_m = cp.empty(E_m_in.shape, dtype=np.complex64)
                return_np_array = False
        else:
            return_np_array = True
            A_p = pyfftw.empty_aligned(E_p_in.shape, dtype=np.complex64)
            A_m = pyfftw.empty_aligned(E_m_in.shape, dtype=np.complex64)
        plans = self.build_fft_plan(A_p)
        # A[:] = self.E_00*E_in
        A_p[:] = E_p_in
        A_m[:] = E_m_in
        propagator = self.build_propagator(precision)
        if BACKEND == "GPU":
            if type(self.V) == np.ndarray:
                V = cp.asarray(self.V)
            elif type(self.V) == cp.ndarray:
                V = self.V.copy()
            if self.V is None:
                V = self.V
            if type(A_p) != cp.ndarray:
                A_p = cp.asarray(A_p)
                A_m = cp.asarray(A_m)
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
        self.g12 = ratio_g12_g11*self.g11
        if verbose:
            pbar = progressbar.ProgressBar(max_value=len(Z))
        for i, z in enumerate(Z):
            if z > self.L:
                self.n2 = 0
            if verbose:
                pbar.update(i+1)
            self.split_step(A_p, A_m, self.g12, V, propagator, plans, precision)

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
            A_p = cp.asnumpy(A_p)
            A_m = cp.asnumpy(A_m)

        if plot == True:
            if not (return_np_array):
                A_p_plot = cp.asnumpy(A_p)
                A_m_plot = cp.asnumpy(A_m)
            elif return_np_array or BACKEND == 'CPU':
                A_p_plot = cp.asnumpy(A_p)
                A_m_plot = cp.asnumpy(A_m)
            fig, ax = plt.subplots(1, 2)
            ax[0].plot(self.X, np.unwrap(np.angle(A_p_plot)),label='+')
            ax[0].plot(self.X, np.unwrap(np.angle(A_m_plot)),label='-')
            ax[1].plot(self.X, 1e-4*c/2*epsilon_0*np.abs(A_p_plot)**2,label='+')
            ax[1].plot(self.X, 1e-4*c/2*epsilon_0*np.abs(A_m_plot)**2,label='-')
            ax[0].set_title("Phase")
            ax[0].legend()
            ax[1].set_title(r"Intensity in $W/cm^2$")
            ax[1].legend()
            plt.tight_layout()
            plt.show()
        return A_p, A_m


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))



##
if __name__ == "__main__":
    trans = 1
    alpha = 0
    n2 = -4e-10
    ratio_g12_g11 = 2
    waist = 1e-3
    window = 2048*5.5e-6
    puiss = 500e-3
    L = 5e-3
    dn = None

    theta = np.pi/3
    phi = np.pi/2

    # 1D case
    simu = CNLSE_1d(alpha, puiss, waist, window, n2, dn,
                L, NX=2048)
    E_in_0 = np.ones((simu.NX,), dtype=np.complex64) * \
        np.exp(-(simu.X**2)/(2*simu.waist**2))

    print(" 1D dz=",simu.delta_z)
    
    E_x_0, E_y_0 = E_in_0*simu.E_00*np.cos(theta), E_in_0*simu.E_00*np.sin(theta)*np.exp(1j*phi) # field components in (x,y) basis
    E_p_0, E_m_0 = (E_x_0+1j*E_y_0)/np.sqrt(2), (E_x_0-1j*E_y_0)/np.sqrt(2) # field components in circular basis (+,-)
    E1d_p, E1d_m = simu.out_field(E_p_0, E_m_0, L, plot=False, verbose=True)

    # 2D case
    simu = CNLSE(trans, puiss, waist, window, n2, ratio_g12_g11, dn,
                L, NX=2048, NY=2048)
    E_in_0 = np.ones((simu.NY,simu.NX), dtype=np.complex64) * \
        np.exp(-(simu.XX**2 + simu.YY**2)/(2*simu.waist**2))

    print("2D dz=",simu.delta_z)

    E_x_0, E_y_0 = E_in_0*simu.E_00*np.cos(theta), E_in_0*simu.E_00*np.sin(theta)*np.exp(1j*phi) # field components in (x,y) basis
    E_p_0, E_m_0 = (E_x_0+1j*E_y_0)/np.sqrt(2), (E_x_0-1j*E_y_0)/np.sqrt(2) # field components in circular basis (+,-)
    E2d_p, E2d_m = simu.out_field(E_p_0, E_m_0, L, plot=False, verbose=True)

    
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(simu.X,np.abs(E1d_p)**2,label='1D')
    plt.plot(simu.X,np.abs(E2d_p[:,1024])**2,label='2D')
    plt.xlabel('X')
    plt.ylabel('|E+|²')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(simu.X,np.abs(E1d_m)**2,label='1D')
    plt.plot(simu.X,np.abs(E2d_m[:,1024])**2,label='2D')
    plt.xlabel('X')
    plt.ylabel('|E-|²')
    plt.legend()
    # plt.show()
    plt.savefig('./cnlse_theta_0p33pi_1d2d_GPU.png',format='png')
