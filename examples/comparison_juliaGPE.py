from NLSE import NLSE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def vortex(x, y, xi=10e-6, ell=1):
    r = np.hypot(x, y)
    theta = np.arctan2(x, y)
    psi = r / np.sqrt(r**2 + (xi / 0.83) ** 2) * np.exp(1j * ell * theta)
    return psi


# Parameters
N = 1024
N_samples = 200
n2 = -1.6e-9
waist = 2.23e-3
window = 3.5 * waist
power = 1.05
intensity = power / (np.pi * waist**2)
Isat = np.inf  # saturation intensity in W/m^2
L = 20e-2
alpha = 0
simu = NLSE(
    alpha,
    power,
    window,
    n2,
    None,
    L,
    NX=N,
    NY=N,
    Isat=Isat,
    backend="GPU",
)
xi = 1 / (simu.k * np.sqrt(abs(n2) * intensity))
simu.delta_z = 0.5e-4
save_every = int((L / simu.delta_z) // N_samples)
E_0 = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(np.complex64)
E_samples = np.zeros((N_samples, N, N), dtype=np.complex64)


def callback_samples(sim, A, z, i):
    if i % save_every == 0:
        E_samples[i // save_every] = A.copy()


# Add a vortex phase
d = 1e-2 * waist
vortex_plus = vortex(simu.XX + d, simu.YY + d, xi=xi, ell=1)
vortex_minus = vortex(simu.XX - d, simu.YY - d, xi=xi, ell=-1)
E_0 *= vortex_plus
E_0 *= vortex_minus
# Hand tuned potential for Thomas-Fermi
simu.V = 4.311e-4 * np.exp(-2 * (simu.XX**2 + simu.YY**2) / waist**2)
simu.out_field(
    E_0, L, verbose=True, plot=False, precision="single", callback=callback_samples
)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
rho = np.abs(E_samples) ** 2
phi = np.angle(E_samples)
im0 = ax[0].imshow(rho[0], cmap="hot", interpolation="none")
ax[0].set_title("Density")
im1 = ax[1].imshow(phi[0], cmap="twilight_shifted", interpolation="none")
ax[1].set_title("Phase")


def animate(i):
    im0.set_data(rho[i])
    im0.set_clim(0, np.max(rho[i]))
    im1.set_data(phi[i])
    fig.suptitle(f"{i:02d}")
    return im0, im1


anim = FuncAnimation(fig, animate, frames=N_samples, interval=33, blit=True)
plt.show()
