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
    F_laser_t[time < t_up] = np.exp(-1 * (time[time < t_up] - t_up)**2 / (t_up / 2)**2)
    F_laser_t[time >= t_up] = 1

def callback_sample(simu: DDGPE, A: np.ndarray, z: float, i: int, save_every: int, sample1: list, sample2: list, sample3: list) -> None:
    if i % save_every == 0:
        sum_exc = np.sum(np.abs(A[..., 0, :, :].get())**2)
        sum_cav = np.sum(np.abs(A[..., 1, :, :].get())**2)
        sum_tot = np.sum(np.abs(A[..., :, :, :].get())**2)
        sample1.append(sum_exc)
        sample2.append(sum_cav)
        sample3.append(sum_tot)
        
def main():
    
    h_bar = 0.654 # (meV*ps)
    c = 2.9979*1e2 # (um/ps)
    omega = 5.07 / h_bar # (meV/h_bar) linear coupling (Rabi split)
    omega_exc = 1484.44 / h_bar # (meV/h_bar) exciton energy 
    omega_cav = 1482.76 / h_bar # (meV/h_bar) cavity energy 
    omega_lp = (omega_exc + omega_cav) / 2 - 0.5 * np.sqrt((omega_exc - omega_cav) ** 2 + (omega) ** 2)
    detuning = 0.17 / h_bar
    k_z = 27
    gamma = 0*0.07 / h_bar
    puiss = 1
    waist = 50
    window = 256
    g = 1e-2 / h_bar
    T = 100
    
    dd = DDGPE(gamma, puiss, window, g, omega, T, omega_exc, omega_cav, detuning, k_z, NX=256, NY=256)
    

    dd.delta_z = 0.1/32 #need to be adjusted automatically
    time = np.arange(
            dd.delta_z, T + dd.delta_z, step=dd.delta_z, dtype=np.float32
        )

    E0 = np.zeros((2, dd.NY, dd.NX), dtype=np.complex64) 
    E0[...,0,:,:] = np.sqrt(detuning/g) * np.exp(-(dd.XX**2 + dd.YY**2) / waist**2)
    
    F_pump = 0
    F_pump_r = np.exp(-((dd.XX**2 + dd.YY**2) / waist**2)).astype(np.complex64)
    F_pump_t = np.zeros(time.shape, dtype=np.complex64)
    
    sample1, sample2, sample3 = [], [], []
    turn_on(F_pump_t, time, t_up=10)
    
    save_every = 1 #np.argwhere(time == 1)[0][0]
    callback = [callback_sample]
    callback_args = [[save_every, sample1, sample2, sample3]]
    dd.out_field(E0, F_pump, F_pump_r, F_pump_t, T, plot=True, callback=callback, callback_args=callback_args)

    plt.figure("pump and field norm sum")
    plt.plot(time, sample1/(np.max(sample3)+1e-30))
    plt.plot(time, sample2/(np.max(sample3)+1e-30))
    plt.plot(time, sample3/(np.max(sample3)+1e-30))
    plt.show()
    
if __name__ == "__main__":
    main()