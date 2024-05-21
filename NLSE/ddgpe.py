from .cnlse import CNLSE
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar
import pyfftw
from .utils import __BACKEND__, __CUPY_AVAILABLE__

if __CUPY_AVAILABLE__:
    import cupy as cp


class DDGPE(CNLSE):
    """A class to solve the 2D driven dissipative Gross-Pitaevskii equation"""

    def __init__(
        self,
        alpha: float,
        puiss: float,
        window: float,
        n2: float,
        omega: float,
        V: np.ndarray,
        L: float,
        n12: float = 0,
        NX: int = 1024,
        NY: int = 1024,
        Isat: float = np.inf,
        nl_length: float = 0,
        wvl: float = 780e-9,
        backend: str = __BACKEND__,
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
            __BACKEND__ (str, optional): "GPU" or "CPU". Defaults to __BACKEND__.
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
            NY=NY,
            Isat=Isat,
            nl_length=nl_length,
            wvl=wvl,
            omega=omega,
            backend=backend,
        )
        self.alpha *= hbar
        self.omega *= hbar
        self.n2 *= hbar
        self.n22 = 0
        # k = 2 * np.pi / self.wvl = m

    @staticmethod
    def add_noise(simu: object, A: np.ndarray, t: float, i: int, *args, **kwargs):
        """Add noise to the propagation step.

        Follows the callback convention of NLSE.

        Args:
            simu (object): DDGPE object.
            A (np.ndarray): Field array.
            t (float): Propagation time in s.
            i (int): Propagation step.
        """
        # do something like A += np.random.normal(0, 1e-3, A.shape)
        # will certainly benefit from kernel fusion
        pass

    def _build_propagator(self) -> np.ndarray:
        # kinetic terms for the exciton dispersion and polariton dispersion

        # propagator1 = super()._build_propagator()
        # propagator2 = np.exp(
        #     -1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k2 * self.delta_z
        # ).astype(np.complex64)
        # return np.array([propagator1, propagator2])
        pass
