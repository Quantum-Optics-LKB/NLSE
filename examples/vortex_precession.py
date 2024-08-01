import time

import numpy as np

from NLSE import NLSE


def vortex(x, y, xi=10e-6, ell=1):
    r = np.hypot(x, y)
    theta = np.arctan2(x, y)
    psi = r / np.sqrt(r**2 + (xi / 0.83) ** 2) * np.exp(1j * ell * theta)
    return psi


# Parameters
N = 256
n2 = -1.6e-10
waist = 750e-6
window = 3.5 * waist
power = 1.05
intensity = power / (np.pi * waist**2)
Isat = np.inf  # saturation intensity in W/m^2
L = 5e-2
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
    backend="CPU",
)
cs = np.sqrt(abs(n2) * intensity) / (1 + intensity / Isat)
delta_n = abs(n2) * intensity / (1 + intensity / Isat) ** 2
xi = 1 / (simu.k * cs)
z_nl = 1 / (simu.k * delta_n)
simu.L = 48 * z_nl
simu.delta_z = z_nl / 6
nsteps = simu.L // simu.delta_z
save_every = 2
N_samples = round(nsteps / save_every)
E_0 = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(np.complex64)
E_samples = np.zeros((N_samples, N, N), dtype=np.complex64)


def callback_samples(sim, A, z, i):
    if i % save_every == 0:
        E_samples[i // save_every] = A.copy()


sizes = [128, 256, 512, 1024, 2048, 4196, 8192]
navg = 10
ts = np.zeros((len(sizes), navg))
for i, n in enumerate(sizes):
    print(n)
    simu = NLSE(
        alpha,
        power,
        window,
        n2,
        None,
        L,
        NX=n,
        NY=n,
        Isat=Isat,
        backend="GPU",
    )
    simu.delta_z = z_nl / 6
    # Add a vortex phase
    d = 4 * xi
    vortex_plus = vortex(simu.XX + d / 2, simu.YY + d / 2, xi=xi, ell=1)
    vortex_minus = vortex(simu.XX - d / 2, simu.YY - d / 2, xi=xi, ell=1)
    E_0 = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(np.complex64)
    E_0 *= vortex_plus
    E_0 *= vortex_minus
    # Hand tuned potential for Thomas-Fermi
    simu.V = 4.31e-4 * np.exp(-2 * (simu.XX**2 + simu.YY**2) / waist**2).astype(
        np.float32
    )
    for _ in range(navg):
        t0 = time.perf_counter()
        simu.out_field(
            E_0,
            simu.L,
            verbose=False,
            plot=False,
            precision="single",
        )
        # # to plot the animation at the end
        # simu.out_field(
        #     E_0,
        #     simu.L,
        #     verbose=True,
        #     plot=False,
        #     precision="single",
        #     callback=callback_samples,
        # )
        ts[i, _] = time.perf_counter() - t0
    timing_string = f"Average time: {np.mean(ts[i]):.2f} s "
    timing_string += f"(min: {np.min(ts[i]):.2f} s, max: {np.max(ts[i]):.2f} s)"
    print(timing_string)
np.save(f"python_vortex_precession_{simu.backend}_times.npy", ts)
np.save(f"python_vortex_precession_{simu.backend}_sizes.npy", sizes)
# to plot the animation at the end
# fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
# rho = np.abs(E_samples) ** 2
# phi = np.angle(E_samples)
# ext = [simu.X.min() * 1e3, simu.X.max() * 1e3,
#        simu.Y.min() * 1e3, simu.Y.max() * 1e3]
# im0 = ax[0].imshow(rho[0], cmap="viridis", interpolation="none", extent=ext)
# ax[0].set_title("Density")
# im1 = ax[1].imshow(phi[0], cmap="twilight_shifted", interpolation="none",
#                    extent=ext)
# ax[1].set_title("Phase")
# for a in ax:
#     a.set_xlabel("x in mm")
#     a.set_ylabel("y in mm")


# def animate(i):
#     im0.set_data(rho[i])
#     im0.set_clim(0, np.max(rho[i]))
#     im1.set_data(phi[i])
#     fig.suptitle(f"{i:02d}")
#     return im0, im1


# anim = FuncAnimation(fig, animate, frames=N_samples, interval=60, blit=True)
# plt.show()
