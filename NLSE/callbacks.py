import numpy as np
from scipy.constants import c, epsilon_0

from .nlse import NLSE


def sample(
    simu: NLSE,
    A: np.ndarray,
    z: float,
    i: int,
    save_every: int,
    E_samples: np.ndarray,
) -> None:
    """Save samples of the field.

    This callback will save samples every save_every steps into the E_samples
    array.

    Args:
        simu (NLSE): Simulation object.
        A (np.ndarray): The current field.
        z (float): The current propagation distance.
        i (int): Step number.
        save_every (int): Number of propagation steps between each step.
        E_samples (np.ndarray): Array to store the samples.
    """
    if i % save_every == 0:
        E_samples[i // save_every] = A.copy()


def norm(
    simu: NLSE,
    A: np.ndarray,
    z: float,
    i: int,
    save_every: int,
    norms: np.ndarray,
) -> None:
    """Save the norm of the field.

    This callback will save the norm of the field every save_every steps into the
    E_samples array.

    Args:
        simu (NLSE): Simulation object.
        A (np.ndarray): The current field.
        z (float): The current propagation distance.
        i (int): Step number.
        save_every (int): Number of propagation steps between each step.
        E_samples (np.ndarray): Array to store the samples.
    """
    if i % save_every == 0:
        norms[i // save_every] = (A.real * A.real + A.imag * A.imag).sum()


def evaluate_delta_n(
    simu: NLSE,
    A: np.ndarray,
    z: float,
    i: int,
    save_every: int,
    delta_n: np.ndarray,
) -> None:
    """Evaluate the non-linear refractive index change.

    This will evaluate the weight of the non-linear refractive index change, allowing
    to adjust the step size accordingly.

    Args:
        simu (NLSE): Simulation object.
        A (np.ndarray): The current field.
        z (float): The current propagation distance.
        i (int): Step number.
        save_every (int): Number of propagation steps between each step.
        delta_n (np.ndarray): The array of delta_n values.
    """
    if i % save_every == 0:
        A_sq = A.real * A.real + A.imag * A.imag
        delta_n[i // save_every] = (
            c * epsilon_0 / 2 * simu.n2 * A_sq / (1 + A_sq / simu.I_sat)
        )


def adapt_delta_z(
    simu: NLSE,
    A: np.ndarray,
    z: float,
    i: int,
    update_every: int,
    delta_z: list,
) -> None:
    """Update the simulation step size.

    This callback will update the simulation step size every update_every steps by
    computing the nonlinear refractive index change and adjusting the step size
    accordingly.

    Args:
        simu (NLSE): Simulation object.
        A (np.ndarray): The current field.
        z (float): The current propagation distance.
        i (int): Step number.
        update_every (int): Update the step size every update_every steps.
        delta_z (list): A list to store the size of the steps.
    """
    delta_z.append(simu.delta_z)
    if i % update_every == 0:
        A_sq = A.real * A.real + A.imag * A.imag * c * epsilon_0 / 2
        delta_n = np.abs(simu.n2) * A_sq / (1 + A_sq / simu.I_sat)
        z_nl = float(1 / (simu.k * delta_n.max()))
        simu.delta_z = np.abs(z_nl) / 12
