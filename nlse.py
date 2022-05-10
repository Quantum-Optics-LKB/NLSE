#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Taladjidi
Solves NLS equation with spectral operator splitting scheme
dA/dZ = i d^2A/dX^2 + i V(X,Z) A - i |A|^2 A
for given A(Z=0) for 0<Z<L
"""
import numba
import pyfftw
from scipy.ndimage import zoom
from scipy.constants import c, epsilon_0, hbar, mu_0
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import pickle
import time
import sys
BACKEND = "CPU"
try:
    import cupy as cp
    import cupyx.scipy.fftpack as fftpack
    BACKEND = "GPU"

    @cp.fuse(kernel_name="nl_prop")
    def nl_prop(A: cp.ndarray, dz: float, alpha: float, V: cp.ndarray, g: float):
        A *= cp.exp(dz*(-alpha/2 + 1j * V + 1j*g*cp.abs(A)**2))
except ImportError:
    print("CuPy not available, falling back to CPU backend ...")
    import pyfftw
    import numba
    pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
    BACKEND = "CPU"

    @numba.njit(parallel=True, fastmath=True)
    def nl_prop(A: np.ndarray, dz: float, alpha: float, V: np.ndarray, g: float):
        for i in numba.prange(A.shape[0]):
            for j in range(A.shape[1]):
                A[i, j] *= np.exp(dz*(-alpha/2 + 1j *
                                      V[i, j] + 1j*g*abs(A[i, j])**2))


class NLSE:
    """Non linear Schrödinger Equation simulation class
    """

    def __init__(self, trans: float, puiss: float, waist: float, window: float, n2: float, V: np.ndarray, L: float, NX: int = 1024, NY: int = 1024) -> None:
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
        m = hbar * self.k/c
        self.L = L  # length of the non linear medium
        self.alpha = -np.log(trans)/self.L
        intens = 2*puiss/(np.pi*waist**2)
        self.E_00 = np.sqrt(2*intens/(c*epsilon_0))

        # number of grid points in X (even, best is power of 2 or low prime factors)
        self.NX = NX
        self.NY = NY
        self.window = window
        # transverse coordinate
        self.X, self.delta_X = np.linspace(-self.window/2, self.window/2, num=NX,
                                           endpoint=False, retstep=True, dtype=np.float32)
        self.Y, self.delta_Y = np.linspace(-self.window/2, self.window/2, num=NY,
                                           endpoint=False, retstep=True, dtype=np.float32)

        self.XX, self.YY = np.meshgrid(self.X, self.Y)

    # plot 2D amplitude on equidistant ZxX grid
    def plot_2D(self, ax, Z, X, AMP, title, cmap='viridis', label=r'$X$ (mm)', vmax=1):
        im = ax.imshow(AMP, aspect='equal', origin='lower', extent=(
            Z[0], Z[-1], X[0], X[-1]), cmap=cmap, vmax=vmax)
        ax.set_xlabel(label)
        ax.set_ylabel(r'$Y$ (mm)')
        ax.set_title(title)
        plt.colorbar(im)

        return

    # plot 1D amplitude and phase
    def plot_1D(self, ax, T, labelT, AMP, labelAMP, PHASE, labelPHASE, Tmin, Tmax):
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

    # plot 1D amplitude and phase

    def plot_1D_amp(self, ax, T, labelT, AMP, labelAMP, Tmin, Tmax, color='b', label=''):
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
        zoom_x = d_slm/self.delta_X
        zoom_y = d_slm/self.delta_Y
        phase_zoomed = zoom(pattern, (zoom_y, zoom_x))
        # compute center offset
        x_center = (self.NX - phase_zoomed.shape[1]) // 2
        y_center = (self.NY - phase_zoomed.shape[0]) // 2

        # copy img image into center of result image
        phase[y_center:y_center+phase_zoomed.shape[0],
              x_center:x_center+phase_zoomed.shape[1]] = phase_zoomed
        return phase

    def E_out(self, E_in: np.ndarray, z: float, plot=False, precision: str = "single") -> np.ndarray:
        """Propagates the field at a distance z
        Args:
            E_in (np.ndarray): Normalized input field (between 0 and 1)
            z (float): propagation distance in m
            plot (bool, optional): Plots the results. Defaults to False.
            precision (str, optional): Does a "double" or a "single" application
            of the propagator. This leads to a dz (single) or dz^3 (double) precision.
            Defaults to "single".
        Returns:
            np.ndarray: Propagated field in proper units V/m
        """

        # normalized longitudinal coordinate
        delta_Z = 1e-5*self.z_r
        Z = np.arange(0, z, step=delta_Z, dtype=np.float32)
        # define fft plan
        if BACKEND == "GPU":
            A = np.empty((self.NX, self.NY), dtype=np.complex64)
            plan_fft = fftpack.get_fft_plan(
                A, shape=A.shape, axes=(0, 1), value_type='C2C')
        else:
            A = pyfftw.empty_aligned((self.NX, self.NY), dtype=np.complex64)
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
        A[:, :] = self.E_00*E_in
        # definition of the Fourier frequencies for the linear step
        Kx = 2 * np.pi * np.fft.fftfreq(self.NX, d=self.delta_X)
        Ky = 2 * np.pi * np.fft.fftfreq(self.NY, d=self.delta_Y)

        Kxx, Kyy = np.meshgrid(Kx, Ky)
        if precision == "double":
            propagator = np.exp(-1j * 0.25 * (Kxx**2 + Kyy**2) /
                                self.k * delta_Z)  # symetrized
        else:
            propagator = np.exp(-1j * 0.5 * (Kxx**2 + Kyy**2) /
                                self.k * delta_Z)
        if BACKEND == "GPU":
            propagator_cp = cp.asarray(propagator)
            self.V = cp.asarray(self.V)

            def split_step_cp(A):
                """computes one propagation step"""
                plan_fft.fft(A, A, cp.cuda.cufft.CUFFT_FORWARD)
                A *= propagator_cp  # linear step in Fourier domain (shifted)
                plan_fft.fft(A, A, cp.cuda.cufft.CUFFT_INVERSE)
                # fft normalization
                A /= np.prod(A.shape)
                nl_prop(A, delta_Z, self.alpha, self.k/2 *
                        self.V, self.k*self.n2*c*epsilon_0)
                if precision == "double":
                    plan_fft.fft(A, A, cp.cuda.cufft.CUFFT_FORWARD)
                    # linear step in Fourier domain (shifted)
                    A *= propagator_cp
                    plan_fft.fft(A, A, cp.cuda.cufft.CUFFT_INVERSE)
                    A /= np.prod(A.shape)
                return A
            start_gpu = cp.cuda.Event()
            end_gpu = cp.cuda.Event()
            start_gpu.record()
            A = cp.asarray(A)
            t0 = time.perf_counter()
            n2_old = self.n2
            for i, z in enumerate(Z):
                if z > self.L:
                    self.n2 = 0
                sys.stdout.write(f"\rIteration {i+1}/{len(Z)}")
                A[:, :] = split_step_cp(A)

            print()
            end_gpu.record()
            end_gpu.synchronize()
            t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
            self.n2 = n2_old
            A = cp.asnumpy(A)
            print(
                f"Time spent to solve : {t_gpu*1e-3} s (GPU) / {time.perf_counter()-t0} s (CPU)")
        else:
            def split_step(A):
                """computes one propagation step"""
                plan_fft(A)
                A *= propagator  # linear step in Fourier domain (shifted)
                plan_ifft(A)
                nl_prop(A, delta_Z, self.alpha, self.k/2 *
                        self.V, self.k*self.n2*c*epsilon_0)
                if precision == "double":
                    plan_fft(A)
                    A *= propagator  # linear step in Fourier domain (shifted)
                    plan_ifft(A)
                return A
            t0 = time.perf_counter()
            n2_old = self.n2
            for i, z in enumerate(Z):
                if z > self.L:
                    self.n2 = 0
                sys.stdout.write(f"\rIteration {i+1}/{len(Z)}")
                A[:, :] = split_step(A)
            print(
                f"\nTime spent to solve : {time.perf_counter()-t0} s (CPU)")
            with open("fft.wisdom", "wb") as file:
                wisdom = pyfftw.export_wisdom()
                pickle.dump(wisdom, file)
        if plot == True:
            fig = plt.figure(3, [9, 8])

            # plot amplitudes and phases
            a1 = fig.add_subplot(221)
            self.plot_2D(a1, self.X*1e3, self.Y*1e3, np.abs(A)**2,
                         r'$|\psi|^2$', vmax=np.max(np.abs(A)**2))

            a2 = fig.add_subplot(222)
            self.plot_2D(a2, self.X*1e3, self.Y*1e3,
                         np.angle(A), r'arg$(\psi)$', cmap='twilight', vmax=np.pi)

            a3 = fig.add_subplot(223)
            lim = 1
            im_fft = np.abs(np.fft.fftshift(
                np.fft.fft2(np.abs(A[lim:-lim, lim:-lim])**2)))
            Kx_2 = 2 * np.pi * np.fft.fftfreq(self.NX-2*lim, d=self.delta_X)
            len_fft = len(im_fft[0, :])
            self.plot_2D(a3, np.fft.fftshift(Kx_2), np.fft.fftshift(Kx_2), np.log10(im_fft),
                         r'$\mathcal{TF}(|E_{out}|^2)$', cmap='viridis', label=r'$K_y$', vmax=np.max(np.log10(im_fft)))

            a4 = fig.add_subplot(224)
            self.plot_1D_amp(a4, Kx_2[1:-len_fft//2]*1e-3, r'$K_x (mm^{-1})$', np.mean(im_fft[len_fft//2-10:len_fft//2+10, len_fft//2+1:], axis=0),
                             r'$\mathcal{TF}(|E_{out}|^2)$', np.fft.fftshift(Kx_2)[len_fft//2+1]*1e-3, np.fft.fftshift(Kx_2)[-1]*1e-3, color='b')
            a4.set_yscale('log')
            a4.set_xscale('log')

            plt.tight_layout()
            plt.show()
        return A


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
    n2 = -4e-10
    waist = 1e-3
    window = 2048*5.5e-6
    puiss = 500e-3
    L = 5e-2
    dn = 2.5e-4 * np.ones((2048, 2048), dtype=np.complex64)
    simu = NLSE(trans, puiss, waist, window, n2, dn,
                L, NX=2048, NY=2048)
    simu.V *= np.exp(-(simu.XX**2 + simu.YY**2)/(2*(simu.waist/3)**2))
    phase_slm = 2*np.pi * \
        flatTop_super(1272, 1024, length=1000, width=600)
    phase_slm = simu.slm(phase_slm, 6.25e-6)
    E_in_0 = np.ones((simu.NY, simu.NX), dtype=np.complex64) * \
        np.exp(-(simu.XX**2 + simu.YY**2)/(2*simu.waist**2))
    E_in_0 *= np.exp(1j*phase_slm)
    E_in_0 = np.fft.fftshift(np.fft.fft2(E_in_0))
    E_in_0[0:E_in_0.shape[0]//2+20, :] = 1e-10
    E_in_0[E_in_0.shape[0]//2+225:, :] = 1e-10
    E_in_0 = np.fft.ifft2(np.fft.ifftshift(E_in_0))
    A = simu.E_out(E_in_0, L, plot=True)
