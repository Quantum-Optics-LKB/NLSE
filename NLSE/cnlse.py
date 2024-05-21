from .nlse import NLSE
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0
import pyfftw
from .utils import __BACKEND__, __CUPY_AVAILABLE__

if __CUPY_AVAILABLE__:
    import cupy as cp


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
        """Prepare the output array depending on __BACKEND__."""
        if self.backend == "GPU" and self.__CUPY_AVAILABLE__:
            A = cp.empty_like(E)
            E = cp.asarray(E)
            puiss_arr = cp.array([self.puiss, self.puiss2], dtype=E.dtype)
        else:
            A = pyfftw.empty_aligned(E.shape, dtype=E.dtype, n=pyfftw.simd_alignment)
            puiss_arr = np.array([self.puiss, self.puiss2], dtype=E.dtype)
        if normalize:
            # normalization of the field
            integral = (
                (E.real * E.real + E.imag * E.imag) * self.delta_X * self.delta_Y
            ).sum(axis=self._last_axes)
            integral *= c * epsilon_0 / 2
            E_00 = (puiss_arr / integral) ** 0.5
            A[:] = (E_00.T * E.T).T
        return A

    def _send_arrays_to_gpu(self) -> None:
        """
        Send arrays to GPU.
        """
        super()._send_arrays_to_gpu()
        # for broadcasting of parameters in case they are
        # not already on the GPU
        if isinstance(self.n22, np.ndarray):
            self.n22 = cp.asarray(self.n22)
        if isinstance(self.n12, np.ndarray):
            self.n12 = cp.asarray(self.n12)

    def _retrieve_arrays_from_gpu(self) -> None:
        """
        Retrieve arrays from GPU.
        """
        super()._retrieve_arrays_from_gpu()
        if isinstance(self.n2, cp.ndarray):
            self.n22 = self.n22.get()
        if isinstance(self.n12, cp.ndarray):
            self.n12 = self.n12.get()

    def _build_propagator(self) -> np.ndarray:
        """Build the propagators.

        Returns:
            np.ndarray: A tuple of linear propagators for each component.
        """
        propagator1 = super()._build_propagator()
        propagator2 = np.exp(
            -1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k2 * self.delta_z
        ).astype(np.complex64)
        return np.array([propagator1, propagator2])

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
        propagator: np.ndarray,
        plans: list,
        precision: str = "single",
    ) -> None:
        """Split step function for one propagation step

        Args:
            A (np.ndarray): Fields to propagate of shape (2, NY, NX)
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
        if self.backend == "GPU" and self.__CUPY_AVAILABLE__:
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
        if self.backend == "GPU" and self.__CUPY_AVAILABLE__:
            plan_fft.fft(A, A)
            # linear step in Fourier domain (shifted)
            cp.multiply(A, propagator, out=A)
            plan_fft.ifft(A, A)
        else:
            plan_fft, plan_ifft = plans
            plan_fft(input_array=A, output_array=A)
            np.multiply(A, propagator, out=A)
            plan_ifft(input_array=A, output_array=A, normalise_idft=True)
        # fft normalization
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

    def plot_field(self, A_plot: np.ndarray, z: float) -> None:
        """Plot the field.

        Args:
            A_plot (np.ndarray): The field to plot
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
        rho0 = np.abs(A_plot[0]) ** 2 * 1e-4 * c / 2 * epsilon_0
        phi0 = np.angle(A_plot[0])
        rho1 = np.abs(A_plot[1]) ** 2 * 1e-4 * c / 2 * epsilon_0
        phi1 = np.angle(A_plot[1])
        # plot amplitudes and phases
        im0 = ax[0, 0].imshow(rho0, extent=ext_real)
        ax[0, 0].set_title(r"$|\psi_1|^2$")
        ax[0, 0].set_xlabel("x (mm)")
        ax[0, 0].set_ylabel("y (mm)")
        fig.colorbar(im0, ax=ax[0, 0], shrink=0.6, label=r"Intensity $(W/cm^2)$")
        im1 = ax[0, 1].imshow(
            phi0, extent=ext_real, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi
        )
        ax[0, 1].set_title(r"Phase $\mathrm{arg}(\psi_1)$")
        ax[0, 1].set_xlabel("x (mm)")
        ax[0, 1].set_ylabel("y (mm)")
        fig.colorbar(im1, ax=ax[0, 1], shrink=0.6, label="Phase (rad)")
        im2 = ax[1, 0].imshow(rho1, extent=ext_real)
        ax[1, 0].set_title(r"$|\psi_2|^2$")
        ax[1, 0].set_xlabel("x (mm)")
        ax[1, 0].set_ylabel("y (mm)")
        fig.colorbar(im2, ax=ax[1, 0], shrink=0.6, label=r"Intensity $(W/cm^2)$")
        im3 = ax[1, 1].imshow(
            phi1, extent=ext_real, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi
        )
        ax[1, 1].set_title(r"Phase $\mathrm{arg}(\psi_2)$")
        ax[1, 1].set_xlabel("x (mm)")
        ax[1, 1].set_ylabel("y (mm)")
        fig.colorbar(im3, ax=ax[1, 1], shrink=0.6, label="Phase (rad)")
        plt.show()
