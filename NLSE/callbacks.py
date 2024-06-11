import numpy as np
from nlse import NLSE


def sample(
    simu: NLSE, A: np.ndarray, z: float, i: int, save_every: int, E_samples: np.ndarray
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
    simu: NLSE, A: np.ndarray, z: float, i: int, save_every: int, norms: np.ndarray
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
        norms[i // save_every] = A.real @ A.real + A.imag @ A.imag


def evaluate_delta_n(
    simu: NLSE, A: np.ndarray, z: float, i: int, save_every: int, delta_n: np.ndarray
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
        delta_n[i // save_every] = simu.n2 * A_sq / (1 + A_sq / simu.Isat) ** 2
