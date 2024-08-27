from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pyfftw

from .cnlse import CNLSE
from .utils import __BACKEND__, __CUPY_AVAILABLE__

if __CUPY_AVAILABLE__:
    import cupy as cp


class DDGPE(CNLSE):
    """A class to solve the 2D driven dissipative Gross-Pitaevskii equation"""

    def __init__(
        self,
        gamma: float,
        power: float,
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
            gamma (float): Losses coefficient in s^-1
            power (float): Optical power in W
            window (float): Computational window in m
            g (float): Interaction parameter
            n12 (float): Inter component interaction parameter
            V (np.ndarray): Potential landscape in a.u
            L (float): Length of the cell in m
            NX (int, optional): Number of points along x. Defaults to 1024.
            NY (int, optional): Number of points along y. Defaults to 1024.
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
            alpha=gamma,
            power=power,
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
        if self.backend == "GPU" and self.__CUPY_AVAILABLE__:
            self._random = cp.random.normal
        else:
            self._random = np.random.normal

    @staticmethod
    def add_noise(
        simu: object,
        A: np.ndarray,
        t: float,
        i: int,
        noise: float = 0,
    ) -> None:
        """Add noise to the propagation step.

        Follows the callback convention of NLSE.

        Args:
            simu (object): DDGPE object.
            A (np.ndarray): Field array.
            t (float): Propagation time in s.
            i (int): Propagation step.
        """
        rand1 = simu._random(
            loc=0, scale=simu.delta_z, size=(simu.NY, simu.NX)
        ) + 1j * simu._random(
            loc=0, scale=simu.delta_z, size=(simu.NY, simu.NX)
        )
        rand2 = simu._random(
            loc=0, scale=simu.delta_z, size=(simu.NY, simu.NX)
        ) + 1j * simu._random(
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
        F_pump_r: np.ndarray,
        F_pump_t: np.ndarray,
        F_probe_r: np.ndarray,
        F_probe_t: np.ndarray,
    ) -> None:
        """Add the pump and probe laser.

        This function adds a pump field with a spatial profile F_pump_r and a temporal
        profile F_pump_t and a probe field with a spatial profile F_probe_r and a
        temporal profile F_probe_t. The pump and probe fields are added to the
        cavity field at each propagation step.

        Args:
            simu (object): The simulation object.
            A (np.ndarray): The field array.
            t (float): The current solver time.
            i (int): The current solver step.
            F_pump_r (np.ndarray): The spatial profile of the pump field.
            F_pump_t (np.ndarray): The temporal profile of the pump field.
            F_probe_r (np.ndarray): The spatial profile of the probe field.
            F_probe_t (np.ndarray): The temporal profile of the probe field.
        """
        A[..., 1, :, :] -= F_pump_r * F_pump_t[i] * simu.delta_z * 1j
        A[..., 1, :, :] -= F_probe_r * F_probe_t[i] * simu.delta_z * 1j

    def _send_arrays_to_gpu(self) -> None:
        """
        Send arrays to GPU.
        """
        super()._send_arrays_to_gpu()
        # for broadcasting of parameters in case they are
        # not already on the GPU
        if isinstance(self.gamma, np.ndarray):
            self.gamma = cp.asarray(self.gamma)
        if isinstance(self.g, np.ndarray):
            self.g = cp.asarray(self.g)
        if isinstance(self.omega, np.ndarray):
            self.omega = cp.asarray(self.omega)
        if isinstance(self.k_z, np.ndarray):
            self.k_z = cp.asarray(self.k_z)
        if isinstance(self.omega_exc, np.ndarray):
            self.omega_exc = cp.asarray(self.omega_exc)
        if isinstance(self.omega_cav, np.ndarray):
            self.omega_cav = cp.asarray(self.omega_cav)
        if isinstance(self.detuning, np.ndarray):
            self.detuning = cp.asarray(self.detuning)
        if isinstance(self.omega_pump, np.ndarray):
            self.omega_pump = cp.asarray(self.omega_pump)

    def _retrieve_arrays_from_gpu(self) -> None:
        """
        Retrieve arrays from GPU.
        """
        super()._retrieve_arrays_from_gpu()
        if isinstance(self.gamma, cp.ndarray):
            self.gamma = self.gamma.get()
        if isinstance(self.g, cp.ndarray):
            self.g = self.g.get()
        if isinstance(self.omega, cp.ndarray):
            self.omega = self.omega.get()
        if isinstance(self.k_z, cp.ndarray):
            self.k_z = self.k_z.get()
        if isinstance(self.omega_exc, cp.ndarray):
            self.omega_exc = self.omega_exc.get()
        if isinstance(self.omega_cav, cp.ndarray):
            self.omega_cav = self.omega_cav.get()
        if isinstance(self.detuning, cp.ndarray):
            self.detuning = self.detuning.get()
        if isinstance(self.omega_pump, cp.ndarray):
            self.omega_pump = self.omega_pump.get()

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
                self.omega_cav
                * np.sqrt(1 + (self.Kxx**2 + self.Kyy**2) / self.k_z**2)
                - self.omega_pump
            )
            * self.delta_z
        ).astype(np.complex64)
        return np.array([propagator1, propagator2])
        pass

    def _prepare_output_array(
        self, E_in: np.ndarray, normalize: bool
    ) -> np.ndarray:
        """Prepare the output array depending on __BACKEND__.

        Args:
            E_in (np.ndarray): Input array
            normalize (bool): Normalize the field to the total power.
        Returns:
            np.ndarray: Output array
        """
        if self.backend == "GPU" and self.__CUPY_AVAILABLE__:
            A = cp.zeros_like(E_in)
            A_sq = cp.zeros_like(A, dtype=A.real.dtype)
            A[:] = cp.asarray(E_in)
        else:
            A = pyfftw.zeros_aligned(E_in.shape, dtype=E_in.dtype)
            A_sq = np.empty_like(A, dtype=A.real.dtype)
            A[:] = E_in
        if normalize:
            pass
        return A, A_sq

    def split_step(
        self,
        A: np.ndarray,
        A_sq: np.ndarray,
        V: np.ndarray,
        propagator: np.ndarray,
        plans: list,
        precision: str = "single",
    ) -> None:
        """Split step function for one propagation step

        Args:
            A (np.ndarray): Fields to propagate of shape (2, NY, NX)
            A_sq (np.ndarray): Squared modulus of the fields
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
            self._kernels.square_mod(A, A_sq)
            A_sq_1, A_sq_2 = self._take_components(A_sq)
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
                    self.I_sat,
                    self.I_sat2,
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
                    self.I_sat,
                    self.I_sat2,
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
        self._kernels.square_mod(A, A_sq)
        A_sq_1, A_sq_2 = self._take_components(A_sq)
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
                    self.I_sat,
                    self.I_sat2,
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
                    self.I_sat,
                    self.I_sat2,
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
                    self.I_sat,
                    self.I_sat2,
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
                    self.I_sat,
                    self.I_sat2,
                )
            if self.omega is not None:
                self._kernels.rabi_coupling(
                    A1, A2, self.delta_z, self.omega / 2
                )

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
        fig, ax = plt.subplots(2, 2, layout="constrained", figsize=(10, 10))
        fig.suptitle(rf"Field at $t$ = {t:} ps")
        ext_real = [
            np.min(self.X) * 1e3,
            np.max(self.X) * 1e3,
            np.min(self.Y) * 1e3,
            np.max(self.Y) * 1e3,
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
            phi0,
            extent=ext_real,
            cmap="twilight_shifted",
            vmin=-np.pi,
            vmax=np.pi,
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
            phi1,
            extent=ext_real,
            cmap="twilight_shifted",
            vmin=-np.pi,
            vmax=np.pi,
        )
        ax[1, 1].set_title(r"Phase $\mathrm{arg}(\psi_c)$")
        ax[1, 1].set_xlabel("x (mm)")
        ax[1, 1].set_ylabel("y (mm)")
        fig.colorbar(im3, ax=ax[1, 1], shrink=0.6, label="Phase (rad)")
        plt.show()

    def out_field(
        self,
        E_in: np.ndarray,
        t: float,
        laser_excitation: Union[callable, None],
        plot: bool = False,
        precision: str = "single",
        verbose: bool = True,
        callback: Union[list[callable], callable] = None,
        callback_args: Union[list[tuple], tuple] = None,
    ) -> np.ndarray:
        """Propagate a field to time T.

        Args:
            E_in (np.ndarray): Input field where E_in[0] is the exciton field and
            E_in[1] is the cavity field.
            t (float): Time to propagate to in s.
            laser_excitation (Union[callable, None]): The excitation function.
            This represents the laser pump and probe. Defaults to None which uses
            the static method defined in the class. In this case you still need
            to pass the correct arguments to the callback_args.
            plot (bool, optional): Whether to plot the results. Defaults to False.
            precision (str, optional): Whether to apply the nonlinear terms in a
            single or double step. Defaults to "single".
            verbose (bool, optional): Whether to print progress. Defaults to True.
            callback (Union[list[callable], callable], optional): A list of functions
            to execute at every solver step. Defaults to None.
            callback_args (Union[list[tuple], tuple], optional): A list of callback
            arguments passed to the callbacks. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        if laser_excitation is None:
            callback.insert(0, self.laser_excitation)
        else:
            callback.insert(0, laser_excitation)
        super().out_field(
            E_in=E_in,
            z=t,
            plot=plot,
            precision=precision,
            verbose=verbose,
            normalize=False,
            callback=callback,
            callback_args=callback_args,
        )
