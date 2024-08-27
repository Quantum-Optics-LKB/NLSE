import matplotlib.pyplot as plt
import numpy as np
import pyfftw
from scipy.constants import c, epsilon_0

from .cnlse import CNLSE
from .utils import __BACKEND__, __CUPY_AVAILABLE__

if __CUPY_AVAILABLE__:
    import cupy as cp


class CNLSE_1d(CNLSE):
    """A class to solve the 1D coupled NLSE"""

    def __init__(
        self,
        alpha: float,
        power: float,
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
        backend: str = __BACKEND__,
    ) -> object:
        """Instantiates the class with all the relevant physical parameters

        Args:
            alpha (float): Alpha through the cell
            power (float): Optical power in W
            window (float): Computational window in m
            n2 (float): Non linear index of the 1 st component in m^2/W
            n12 (float): Inter component interaction parameter
            V (np.ndarray): Potential landscape in a.u
            L (float): Length of the cell in m
            NX (int, optional): Number of points along x. Defaults to 1024.
            Isat (float, optional): Saturation intensity, assumed to be the same
                for both components. Defaults to infinity.
            nl_length (float): Non local length in m.
                The non-local kernel is the instantiated as a Bessel function
                to model a diffusive non-locality stored in the nl_profile
                attribute.
            wvl (float, optional): Wavelength in m. Defaults to 780 nm.
            omega (float, optional): Rabi coupling. Defaults to None.
            backend (str, optional): "GPU" or "CPU". Defaults to __BACKEND__.

        Returns:
            object: CNLSE class instance
        """
        super().__init__(
            alpha=alpha,
            power=power,
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
        self.nl_profile = self.nl_profile[0]
        self.nl_profile /= self.nl_profile.sum()

    def _prepare_output_array(
        self, E: np.ndarray, normalize: bool
    ) -> np.ndarray:
        """Prepare the output arrays depending on __BACKEND__.

        Prepares the A and A_sq arrays to store the field and its modulus.
        Args:
            E_in (np.ndarray): Input array
            normalize (bool): Normalize the field to the total power.
        Returns:
            A (np.ndarray): Output field array
            A_sq (np.ndarray): Output field modulus squared array
        """
        if self.backend == "GPU" and self.__CUPY_AVAILABLE__:
            A = cp.empty_like(E)
            A_sq = cp.empty_like(E, dtype=E.real.dtype)
            E = cp.asarray(E)
            puiss_arr = cp.array([self.power, self.power2], dtype=E.dtype)
        else:
            A = pyfftw.empty_aligned(E.shape, dtype=E.dtype)
            A_sq = np.empty_like(E, dtype=E.real.dtype)
            puiss_arr = np.array([self.power, self.power2], dtype=E.dtype)
        if normalize:
            # normalization of the field
            integral = (
                (E.real * E.real + E.imag * E.imag) * self.delta_X**2
            ).sum(axis=self._last_axes)
            integral *= c * epsilon_0 / 2
            E_00 = (puiss_arr / integral) ** 0.5
            A[:] = (E_00.T * E.T).T
        else:
            A[:] = E
        return A, A_sq

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

    def _build_propagator(self) -> np.ndarray:
        """Builds the linear propagation matrix

        Args:
            precision (str, optional): "single" or "double" application of the
            propagator.
            Defaults to "single".
        Returns:
            propagator (np.ndarray): the propagator matrix
        """
        propagator1 = np.exp(-1j * 0.5 * (self.Kx**2) / self.k * self.delta_z)
        propagator2 = np.exp(-1j * 0.5 * (self.Kx**2) / self.k2 * self.delta_z)
        return np.array([propagator1, propagator2])

    def plot_field(self, A_plot: np.ndarray, z: float) -> None:
        """Plot a field for monitoring.

        Args:
            A_plot (np.ndarray): Field to plot
            z (float): Propagation distance in m.
        """
        # if array is multi-dimensional, drop dims until the shape is 2D
        if A_plot.ndim > 2:
            while len(A_plot.shape) > 2:
                A_plot = A_plot[0]
        if self.__CUPY_AVAILABLE__ and isinstance(A_plot, cp.ndarray):
            A_plot = A_plot.get()
        A_1_plot = A_plot[0]
        A_2_plot = A_plot[1]
        fig, ax = plt.subplots(2, 2, layout="constrained", figsize=(10, 10))
        fig.suptitle(rf"Field at $z$ = {z:.2e} m")
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
