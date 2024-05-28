from .nlse import NLSE
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0
import pyfftw
from typing import Union
from .utils import __BACKEND__, __CUPY_AVAILABLE__

if __CUPY_AVAILABLE__:
    import cupy as cp


class NLSE_3d(NLSE):
    """A class to solve the 3D NLSE i.e propagation of pulses
    of light in nonlinear media.
    """

    def __init__(
        self,
        alpha: float,
        energy: float,
        window: np.ndarray,
        n2: float,
        D0: float,
        vg: float,
        V: Union[np.ndarray, None],
        L: float,
        NX: int = 1024,
        NY: int = 1024,
        NZ: int = 1024,
        Isat: float = np.inf,
        nl_length: float = 0,
        wvl: float = 780e-9,
        backend: str = __BACKEND__,
    ) -> object:
        """Instantiates the simulation.
        Solves an equation : d/dz psi = -1/2k0(d2/dx2 + d2/dy2) psi +
          D0/2 (d2/dt2) psi + k0 dn psi +
          k0 n2 psi**2 psi
        Args:
            alpha (float): alpha
            energy (float): Total energy in J
            window (np.ndarray): Computanional window in the transverse plane (index 0) in m
            and longitudinal direction (index 1) in s.
            n2 (float): Non linear coeff in m^2/W.
            D0 (float): Dispersion in s^2/m.
            vg (float): Group velocity in m/s.
            V (np.ndarray): Potential.
            L (float): Length in m of the nonlinear medium
            NX (int, optional): Number of points in the x direction. Defaults to 1024.
            NY (int, optional): Number of points in the y direction. Defaults to 1024.
            NZ (int, optional): Number of points in the t direction. Defaults to 1024.
            Isat (float): Saturation intensity in W/m^2
            nl_length (float): Non local length in m
            wvl (float): Wavelength in m
            backend (str, optional): "GPU" or "CPU". Defaults to __BACKEND__.
        """
        super().__init__(
            alpha=alpha,
            puiss=energy,
            window=window[0],
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
        self.energy = self.puiss
        self.NZ = NZ
        self.window_t = window[1]
        self.puiss = self.energy / self.window_t
        Dn = self.n2 * self.puiss / self.window**2
        z_nl = 1 / (self.k * abs(Dn))
        if isinstance(z_nl, np.ndarray):
            z_nl = z_nl.min()
        self.delta_z = 0.5e-2 * z_nl
        self.T, self.delta_T = np.linspace(
            -self.window_t / 2, self.window_t / 2, self.NZ, retstep=True
        )
        self.omega = 2 * np.pi * np.fft.fftfreq(self.NZ, self.delta_T)
        self.D0 = D0
        self.vg = vg
        self.XX, self.YY, self.TT = np.meshgrid(self.X, self.Y, self.T)
        self.Kxx, self.Kyy, self.Omega = np.meshgrid(self.Kx, self.Ky, self.omega)
        self._last_axes = (-3, -2, -1)  # Axes are x, y, t

    def _build_propagator(self) -> np.ndarray:
        """Build the linear propagation matrix.

        Returns:
            np.ndarray: The propagator
        """
        prop_2d = super()._build_propagator()
        prop_t = np.exp(-1j * self.D0 / 2 * self.Omega**2)
        # prop_t *= np.exp(1 / self.vg * self.Omega)
        return prop_2d * prop_t

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
                * self.delta_T
            ).sum(axis=self._last_axes)
            integral *= c * epsilon_0 / 2
            E_00 = (self.energy / integral) ** 0.5
            A[:] = (E_00.T * E_in.T).T
        return A

    def plot_field(self, A_plot: np.ndarray, z: float) -> None:
        """Plot a field for monitoring.

        Args:
            A_plot (np.ndarray): Field to plot
            z (float): Propagation distance in m.
        """
        # if array is multi-dimensional, drop dims until the shape is 2D
        if A_plot.ndim > 3:
            while len(A_plot.shape) > 3:
                A_plot = A_plot[0]
        if self.__CUPY_AVAILABLE__ and isinstance(A_plot, cp.ndarray):
            A_plot = A_plot.get()
        fig, ax = plt.subplots(2, 2, layout="constrained")
        fig.suptitle(rf"Field at $z$ = {z:.2e} m")
        ext_real = [
            self.X[0] * 1e3,
            self.X[-1] * 1e3,
            self.Y[0] * 1e3,
            self.Y[-1] * 1e3,
        ]
        ext_time = [
            self.T[0] * 1e6,
            self.T[-1] * 1e6,
            self.X[0] * 1e3,
            self.X[-1] * 1e3,
        ]
        rho = np.abs(A_plot) ** 2 * 1e-4 * c / 2 * epsilon_0
        phi = np.angle(A_plot)
        rho_xy = rho[:, :, self.NZ // 2]
        phi_xy = phi[:, :, self.NZ // 2]
        rho_xt = rho[:, self.NY // 2, :]
        phi_xt = phi[:, self.NY // 2, :]
        im0 = ax[0, 0].imshow(rho_xy, extent=ext_real)
        ax[0, 0].set_title(r"Intensity in $xy$ plane at $t$=0")
        ax[0, 0].set_xlabel("x (mm)")
        ax[0, 0].set_ylabel("y (mm)")
        fig.colorbar(im0, ax=ax[0, 0], shrink=0.6, label="Intensity (W/cm^2)")
        im1 = ax[0, 1].imshow(
            phi_xy, extent=ext_real, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi
        )
        ax[0, 1].set_title(r"Phase in $xy$ plane at $t$=0")
        ax[0, 1].set_xlabel("x (mm)")
        ax[0, 1].set_ylabel("y (mm)")
        fig.colorbar(im1, ax=ax[0, 1], shrink=0.6, label="Phase (rad)")
        im2 = ax[1, 0].imshow(rho_xt, extent=ext_time, aspect="auto")
        ax[1, 0].set_title(r"Intensity in $xt$ plane at $y$=0")
        ax[1, 0].set_ylabel(r"$x$ ($mm$)")
        ax[1, 0].set_xlabel(r"$t$ ($\mu s$)")
        fig.colorbar(im2, ax=ax[1, 0], shrink=0.6, label="Intensity (a.u.)")
        im3 = ax[1, 1].imshow(
            phi_xt,
            extent=ext_time,
            cmap="twilight_shifted",
            vmin=-np.pi,
            vmax=np.pi,
            aspect="auto",
        )
        ax[1, 1].set_title(r"Phase in $xt$ plane at $y$=0")
        ax[1, 1].set_ylabel(r"$x$ ($mm$)")
        ax[1, 1].set_xlabel(r"$t$ ($\mu s$)")
        fig.colorbar(im3, ax=ax[1, 1], shrink=0.6, label="Intensity (a.u.)")
        plt.show()
