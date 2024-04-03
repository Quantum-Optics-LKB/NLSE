from .nlse import NLSE
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0
import pyfftw
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
        puiss: float,
        window: np.ndarray,
        n2: float,
        D0: float,
        vg: float,
        V: np.ndarray,
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
        Solves an equation : d/dz psi = -1/2k0(d2/dx2 + d2/dy2) psi + k0 dn psi +
          k0 n2 psi**2 psi
        Args:
            alpha (float): alpha
            puiss (float): Power in W
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
            nl_length (float): Non linear length in m
            wvl (float): Wavelength in m
            backend (str, optional): "GPU" or "CPU". Defaults to __BACKEND__.
        """
        super().__init__(
            alpha=alpha,
            puiss=puiss,
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
        self.NZ = NZ
        self.window_t = window[1]
        self.T, self.delta_T = np.linspace(
            -self.window_t / 2, self.window_t / 2, self.NZ, retstep=True
        )
        self.omega = 2 * np.pi * np.fft.fftfreq(self.NZ, self.delta_T)
        self.D0 = D0
        self.vg = vg
        self.Kxx, self.Kyy, self.Omega = np.meshgrid(self.Kx, self.Ky, self.omega)
        self._last_axes = (-3, -2, -1)  # Axes are x, y, t

    def _build_propagator(self, k: float) -> np.ndarray:
        prop_2d = super()._build_propagator(k)
        prop_t = np.exp(-1j * self.D0 / 2 * self.Omega**2)
        prop_t *= np.exp(1 / self.vg * self.Omega)
        return prop_2d * prop_t

    def plot_field(self, A_plot: np.ndarray) -> None:
        # TODO: Implement plot_field
        pass
