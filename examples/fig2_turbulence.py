import numpy as np

from NLSE import NLSE

N = 1024
n2 = -1.6e-9
window = 8e-3
power = 1.05
Isat = 10e4
L = 20e-2
alpha = 20
waist = 2e-3
waist_d = 1e-3
backend = "GPU"
simu = NLSE(
    alpha, power, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend=backend
)
simu.delta_z = 1e-4
simu.V = 1e-4 * np.exp(-(np.hypot(simu.XX, simu.YY) ** 2) / waist_d**2)
kp = 2 * np.pi * 5e3
E0 = np.exp(-(np.hypot(simu.XX, simu.YY) ** 2) / waist**2).astype(np.complex64)
E0[0 : N // 2, :] *= np.exp(1j * kp * simu.XX[0 : N // 2, :])
E0[N // 2 :, :] *= np.exp(-1j * kp * simu.XX[N // 2 :, :])
simu.out_field(E0, L, plot=True)
