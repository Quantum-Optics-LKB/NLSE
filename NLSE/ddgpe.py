from .cnlse import CNLSE
import numpy as np
import matplotlib.pyplot as plt
import pyfftw
from typing import Union
from .utils import __BACKEND__, __CUPY_AVAILABLE__

if __CUPY_AVAILABLE__:
    import cupy as cp


class DDGPE(CNLSE):
    """A class to solve the 2D driven dissipative Gross-Pitaevskii equation"""

    def __init__(
        self,
        gamma: float,
        puiss: float,
        window: float,
        g: float,
        omega: float,
        T: float,
        omega_exc: float,
        omega_cav: float,
        detuning: float,
        k_z: float,
        V: np.ndarray = None,
        g12: float = 0,
        NX: int = 1024,
        NY: int = 1024,
        Isat: float = np.inf,
        nl_length: float = 0,
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
            alpha=gamma,
            puiss=puiss,
            window=window,
            n2=-g,
            n12=g12,
            V=V,
            L=T,
            NX=NX,
            NY=NY,
            Isat=Isat,
            nl_length=nl_length,
            wvl=1e-30,
            omega=omega,
            backend=backend,
        )
        self.g = self.n2
        self.g12 = self.n12
        self.g2 = 0
        self.k_z = k_z
        self.gamma = gamma
        self.gamma2 = self.gamma
        self.omega_exc = omega_exc
        self.omega_cav = omega_cav
        self.detuning = detuning
        omega_lp = (omega_exc + omega_cav) / 2 - 0.5 * np.sqrt(
            (omega_exc - omega_cav) ** 2 + (omega) ** 2
        )
        self.omega_pump = omega_lp + detuning

    @staticmethod
    def add_noise(
        simu: object,
        A: np.ndarray,
        t: float,
        i: int,
        noise: float = 0,
        noise1: float = 0,
        **kwargs,
    ) -> None:
        """Add noise to the propagation step.

        Follows the callback convention of NLSE.

        Args:
            simu (object): DDGPE object.
            A (np.ndarray): Field array.
            t (float): Propagation time in s.
            i (int): Propagation step.
        """
        if simu.backend == "GPU" and simu.__CUPY_AVAILABLE__:
            rand1 = cp.random.normal(
                loc=0, scale=simu.delta_z, size=(simu.NY, simu.NX), dtype=np.float64
            ) + 1j * cp.random.normal(
                loc=0, scale=simu.delta_z, size=(simu.NY, simu.NX), dtype=np.float64
            )
            rand2 = cp.random.normal(
                loc=0, scale=simu.delta_z, size=(simu.NY, simu.NX), dtype=np.float64
            ) + 1j * cp.random.normal(
                loc=0, scale=simu.delta_z, size=(simu.NY, simu.NX), dtype=np.float64
            )
            A[..., 0, :, :] += (
                noise
                * cp.sqrt(simu.gamma / (4 * (simu.delta_X * simu.delta_Y)))
                * rand1
            )
            A[..., 1, :, :] += (
                noise
                * cp.sqrt((simu.gamma2) / (4 * (simu.delta_X * simu.delta_Y)))
                * rand2
            )

        else:
            rand1 = np.random.normal(
                loc=0, scale=simu.delta_z, size=(simu.NY, simu.NX)
            ) + 1j * np.random.normal(
                loc=0, scale=simu.delta_z, size=(simu.NY, simu.NX)
            )
            rand2 = np.random.normal(
                loc=0, scale=simu.delta_z, size=(simu.NY, simu.NX)
            ) + 1j * np.random.normal(
                loc=0, scale=simu.delta_z, size=(simu.NY, simu.NX)
            )
            A[..., 0, :, :] += (
                noise
                * np.sqrt(simu.gamma / (4 * (simu.delta_X * simu.delta_Y)))
                * rand1
            )
            A[..., 1, :, :] += (
                noise
                * np.sqrt((simu.gamma2) / (4 * (simu.delta_X * simu.delta_Y)))
                * rand2
            )

    @staticmethod
    def laser_excitation(
        simu: object,
        A: np.ndarray,
        t: float,
        i: int,
        F_pump: float,
        F_pump_r: np.ndarray,
        F_pump_t: float,
        F_probe: float = 0,
        F_probe_r: np.ndarray = 0,
        F_probe_t: np.ndarray = 0,
        **kwargs,
    ) -> None:
        """Add the pump and probe laser.

        Args:
            simu (object): _description_
            A (np.ndarray): _description_
            t (float): _description_
            i (int): _description_
        """
        if simu.backend == "GPU" and simu.__CUPY_AVAILABLE__:
            A[..., 1, :, :] -= cp.asarray(
                F_pump * F_pump_r * F_pump_t[i] * simu.delta_z * 1j
            )
            if F_probe != 0:
                A[..., 1, :, :] -= cp.asarray(
                    F_probe * F_probe_r * F_probe_t[i] * simu.delta_z * 1j
                )
        else:
            A[..., 1, :, :] -= F_pump * F_pump_r * F_pump_t[i] * simu.delta_z * 1j
            if F_probe != 0:
                A[..., 1, :, :] -= (
                    F_probe * F_probe_r * F_probe_t[i] * simu.delta_z * 1j
                )

    @staticmethod
    def boundary_losses(
        simu: object, A: np.ndarray, t: float, i: int, v_gamma: np.ndarray, **kwargs
    ) -> None:
        """A fused kernel to apply the potential term

        Args:
            phi_cav (cp.ndarray): Photon field in ph,exc basis
            dz (float): Propagation step in ps
            v_gamma (np.ndarray): Loss at the edges of the grid and optical defects
        """
        if simu.backend == "GPU" and simu.__CUPY_AVAILABLE__:
            A[..., 1, :, :] *= cp.exp(-simu.delta_z * 0.5 * cp.asarray(v_gamma))
        else:
            A[..., 1, :, :] *= np.exp(-simu.delta_z * 0.5 * v_gamma)

    def _build_propagator(self) -> np.ndarray:
        """Build the propagators.

        Returns:
            np.ndarray: A tuple of linear propagators for each component.
        """
        propagator1 = np.exp(
            -1j
            * (self.omega_exc * (1 + 0 * self.Kxx**2) - self.omega_pump)
            * self.delta_z
        ).astype(np.complex64)
        propagator2 = np.exp(
            -1j
            * (
                self.omega_cav * np.sqrt(1 + (self.Kxx**2 + self.Kyy**2) / self.k_z**2)
                - self.omega_pump
            )
            * self.delta_z
        ).astype(np.complex64)
        return np.array([propagator1, propagator2])
        pass

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
            A[:] = cp.asarray(E_in)
        else:
            A = pyfftw.empty_aligned(E_in.shape, dtype=E_in.dtype)
            A[:] = E_in
        if normalize:
            pass
        return A

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
                    self.gamma / 2,
                    self.g,
                    self.g12,
                    self.I_sat,
                    self.I_sat2,
                )
                self._kernels.nl_prop_without_V_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    self.delta_z / 2,
                    self.gamma2 / 2,
                    self.g,
                    self.g12,
                    self.I_sat2,
                    self.I_sat,
                )
            else:
                self._kernels.nl_prop_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z,
                    self.gamma / 2,
                    V,
                    self.g,
                    self.g12,
                    self.I_sat,
                    self.I_sat2,
                )
                self._kernels.nl_prop_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    self.delta_z,
                    self.gamma2 / 2,
                    V,
                    self.g2,
                    self.g12,
                    self.I_sat2,
                    self.I_sat,
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
                    self.gamma / 2,
                    self.g,
                    self.g12,
                    self.I_sat,
                    self.I_sat2,
                )
                self._kernels.nl_prop_without_V_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    self.delta_z / 2,
                    self.gamma2 / 2,
                    self.g,
                    self.g12,
                    self.I_sat2,
                    self.I_sat,
                )
            else:
                self._kernels.nl_prop_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z,
                    self.gamma / 2,
                    self.k / 2 * V,
                    self.g,
                    self.g12,
                    self.I_sat,
                    self.I_sat2,
                )
                self._kernels.nl_prop_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    self.delta_z,
                    self.gamma2 / 2,
                    V,
                    self.g2,
                    self.g12,
                    self.I_sat2,
                    self.I_sat,
                )
        else:
            if V is None:
                self._kernels.nl_prop_without_V_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z,
                    self.alpha / 2,
                    self.g,
                    self.g12,
                    self.I_sat,
                    self.I_sat2,
                )
                self._kernels.nl_prop_without_V_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    self.delta_z,
                    self.gamma2 / 2,
                    self.g2,
                    self.g12,
                    self.I_sat2,
                    self.I_sat,
                )
            else:
                self._kernels.nl_prop_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z,
                    self.gamma / 2,
                    V,
                    self.g,
                    self.g12,
                    self.I_sat,
                    self.I_sat2,
                )
                self._kernels.nl_prop_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    self.delta_z,
                    self.gamma2 / 2,
                    V,
                    self.g2,
                    self.g12,
                    self.I_sat2,
                    self.I_sat,
                )
            if self.omega is not None:
                self._kernels.rabi_coupling(A, self.delta_z, self.omega / 2)

    def plot_field(self, A_plot: np.ndarray, t: float) -> None:
        """Plot the field for monitoring.

        Args:
            A_plot (np.ndarray): The field
            t (float): The time at which the field was sampled.
        """
        # if array is multi-dimensional, drop dims until the shape is 2D
        if A_plot.ndim > 3:
            while len(A_plot.shape) > 3:
                A_plot = A_plot[0]
        if self.__CUPY_AVAILABLE__ and isinstance(A_plot, cp.ndarray):
            A_plot = A_plot.get()
        fig, ax = plt.subplots(2, 2, layout="constrained")
        fig.suptitle(rf"Field at $t$ = {t:} ps")
        ext_real = [
            self.X[0],
            self.X[-1],
            self.Y[0],
            self.Y[-1],
        ]
        rho0 = np.abs(A_plot[0]) ** 2
        phi0 = np.angle(A_plot[0])
        rho1 = np.abs(A_plot[1]) ** 2
        phi1 = np.angle(A_plot[1])
        # plot amplitudes and phases
        im0 = ax[0, 0].imshow(rho0, extent=ext_real)
        ax[0, 0].set_title(r"$|\psi_x|^2$")
        ax[0, 0].set_xlabel("x (mm)")
        ax[0, 0].set_ylabel("y (mm)")
        fig.colorbar(im0, ax=ax[0, 0], shrink=0.6, label=r"Density")
        im1 = ax[0, 1].imshow(
            phi0, extent=ext_real, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi
        )
        ax[0, 1].set_title(r"Phase $\mathrm{arg}(\psi_x)$")
        ax[0, 1].set_xlabel("x (mm)")
        ax[0, 1].set_ylabel("y (mm)")
        fig.colorbar(im1, ax=ax[0, 1], shrink=0.6, label="Phase (rad)")
        im2 = ax[1, 0].imshow(rho1, extent=ext_real)
        ax[1, 0].set_title(r"$|\psi_c|^2$")
        ax[1, 0].set_xlabel("x (mm)")
        ax[1, 0].set_ylabel("y (mm)")
        fig.colorbar(im2, ax=ax[1, 0], shrink=0.6, label=r"Density")
        im3 = ax[1, 1].imshow(
            phi1, extent=ext_real, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi
        )
        ax[1, 1].set_title(r"Phase $\mathrm{arg}(\psi_c)$")
        ax[1, 1].set_xlabel("x (mm)")
        ax[1, 1].set_ylabel("y (mm)")
        fig.colorbar(im3, ax=ax[1, 1], shrink=0.6, label="Phase (rad)")
        plt.show()

    def out_field(
        self,
        E_in: np.ndarray,
        F_pump: float,
        F_pump_r: np.ndarray,
        F_pump_t: np.ndarray,
        z: float,
        plot: bool = False,
        precision: str = "single",
        verbose: bool = True,
        normalize: bool = True,
        callback: Union[list[callable], callable] = [],
        callback_args: Union[list[tuple], tuple] = [],
    ) -> np.ndarray:
        callback.append(self.laser_excitation)
        callback_args.append([F_pump, F_pump_r, F_pump_t])
        super().out_field(
            E_in=E_in,
            z=z,
            plot=plot,
            precision=precision,
            verbose=verbose,
            normalize=normalize,
            callback=callback,
            callback_args=callback_args,
        )
