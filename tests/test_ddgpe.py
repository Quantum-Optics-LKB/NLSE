from NLSE import DDGPE
import numpy as np
import matplotlib.pyplot as plt

if DDGPE.__CUPY_AVAILABLE__:
    import cupy as cp


def turn_on(
    F_laser_t: np.ndarray,
    time: np.ndarray,
    t_up=10,
):
    """A function to turn on the pump more or less adiabatically

    Args:
        F_laser_t (np.ndarray): self.F_pump_t as defined in class ggpe, cp.ones((int(self.t_max//self.dt)), dtype=cp.complex64)
        time (np.ndarray):  array with the value of the time at each discretized step
        t_up (int, optional): time taken to reach the maximum intensity (=F). Defaults to 200.
    """
    F_laser_t[time < t_up] = np.exp(
        -1 * (time[time < t_up] - t_up) ** 2 / (t_up / 2) ** 2
    )
    F_laser_t[time >= t_up] = 1


def callback_sample(
    simu: DDGPE,
    A: np.ndarray,
    z: float,
    i: int,
    save_every: int,
    sample1: list,
    sample2: list,
    sample3: list,
) -> None:
    if i % save_every == 0:
        sum_exc = (A[..., 0, :, :].real ** 2 + A[..., 0, :, :].imag ** 2).sum()
        sum_cav = (A[..., 1, :, :].real ** 2 + A[..., 1, :, :].imag ** 2).sum()
        sum_tot = sum_exc + sum_cav
        sample1[i // save_every] = sum_exc
        sample2[i // save_every] = sum_cav
        sample3[i // save_every] = sum_tot
