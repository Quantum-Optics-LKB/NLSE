from NLSE import DDGPE
import numpy as np
import cupy as cp

if DDGPE.__CUPY_AVAILABLE__:
    import cupy as cp

def turn_on(
    F_laser_t: np.ndarray, 
    time: np.ndarray, 
    t_up=200
):
    """A function to turn on the pump more or less adiabatically

    Args:
        F_laser_t (np.ndarray): self.F_pump_t as defined in class ggpe, cp.ones((int(self.t_max//self.dt)), dtype=cp.complex64)
        time (np.ndarray):  array with the value of the time at each discretized step
        t_up (int, optional): time taken to reach the maximum intensity (=F). Defaults to 200.
    """
    F_laser_t[time < t_up] = np.exp(-1 * (time[time < t_up] - t_up)**2 / (t_up / 2))
    F_laser_t[time >= t_up] = 1



def main():
    
    h_bar = 0.654 # (meV*ps)
    c = 2.9979*1e2 # (um/ps)
    omega = 0.5*5.07/h_bar # (meV/h_bar) linear coupling (Rabi split)
    g0 = (1e-2) /h_bar  # (frequency/density) (meV/hbar)/(1/um^2) nonlinear coupling constant 
    omega_exc = 1484.44 /h_bar # (meV/h_bar) exciton energy measured from the cavity energy #-0.5
    omega_cav = 1482.76 /h_bar # (meV/h_bar) cavity energy at k=0  original value: 1482.76 /h_bar
    omega_lp = (omega_exc + omega_cav) / 2 - 0.5 * np.sqrt((omega_exc - omega_cav) ** 2 + 4 * (omega) ** 2)
    detuning = 0.17/h_bar
    omega_pump = omega_lp + detuning
    omega_exc-=omega_pump
    omega_cav-=omega_pump
    k_z = 27
    gamma = 10 * 0.07 / h_bar
    puiss = 1
    waist = 50
    window = 4*waist
    g = 1e-2
    T = 6e2
    
    dd = DDGPE(gamma, puiss, window, g, omega, T, omega_exc, omega_cav, k_z, NX=256, NY=256)
    
    dd.delta_z = 2000/130000 #need to be adjusted automatically
    
    E0 = np.zeros((2, dd.NY, dd.NX), dtype=np.complex64)
    F_pump = 3
    F_pump_r = np.exp(-((dd.XX**2 + dd.YY**2) / waist**2)).astype(np.complex64)
    time = np.arange(
            dd.delta_z, T + dd.delta_z, step=dd.delta_z, dtype=E0.real.dtype
        )
    F_pump_t = np.ones(time.shape, dtype=np.complex64)
    turn_on(F_pump_t, time)
    
    callback = [DDGPE.add_noise]
    callback_args = [(0, 0)]
    dd.out_field(E0, F_pump, F_pump_r, F_pump_t, T, plot=True, callback=callback, callback_args=callback_args)

if __name__ == "__main__":
    main()