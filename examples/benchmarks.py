import time

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from cycler import cycler

from NLSE import NLSE

# for plots
tab_colors = [
    "tab:blue",
    "tab:orange",
    "forestgreen",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "teal",
]
fills = [
    "lightsteelblue",
    "navajowhite",
    "darkseagreen",
    "lightcoral",
    "violet",
    "indianred",
    "lavenderblush",
    "lightgray",
    "darkkhaki",
    "darkturquoise",
]
edges = tab_colors
custom_cycler = (
    (cycler(color=tab_colors))
    + (cycler(markeredgecolor=edges))
    + (cycler(markerfacecolor=fills))
)
plt.rc("axes", prop_cycle=custom_cycler)

PRECISION = "single"
if PRECISION == "double":
    PRECISION_REAL = np.float64
    PRECISION_COMPLEX = np.complex128
else:
    PRECISION_REAL = np.float32
    PRECISION_COMPLEX = np.complex64

n2 = -1.99e-9
n12 = -0.75e-10
waist = 2.29e-3
waist_d = 70e-6
nl_length = 0
d_real = 3.76e-6
d_fourier = 5.5e-6
f_fourier = 200e-3
window = 3008 * d_real
puiss = 1.05
Isat = 3.92e4  # saturation intensity in W/m^2
L = 2e-3
alpha = 22
dn = None
N_avg = 2
sizes = np.logspace(6, 14, 9, base=2, dtype=int)
times = np.zeros((len(sizes), 3, N_avg))
pbar = tqdm.tqdm(total=np.prod(times.shape), desc="Benchmarks")
for i, size in enumerate(sizes):
    for j, backend in enumerate(["GPU", "CPU"]):
        simu0 = NLSE(
            alpha,
            puiss,
            window,
            n2,
            None,
            L,
            NX=size,
            NY=size,
            nl_length=nl_length,
            backend=backend,
        )
        simu0.I_sat = Isat
        simu0.delta_z = 1e-4
        if j == 0:
            E_0 = np.exp(
                -(np.hypot(simu0.XX, simu0.YY) ** 2) / waist**2
            ).astype(PRECISION_COMPLEX)
        for k in range(N_avg):
            t0 = time.perf_counter()
            simu0.out_field(E_0, L, verbose=False)
            times[i, j, k] = time.perf_counter() - t0
            pbar.update(1)
    # numpy naive implementation
    for k in range(N_avg):
        E1 = E_0.copy()
        t0 = time.perf_counter()
        for z in range(int(L / simu0.delta_z)):
            E1 = np.fft.fft2(E1)
            E1 *= np.exp(1j * simu0.delta_z * simu0.propagator / (2 * simu0.k))
            E1 = np.fft.ifft2(E1)
            E1 *= np.exp(
                1j
                * simu0.delta_z
                * simu0.k
                * simu0.n2
                * np.abs(E1) ** 2
                / (1 + np.abs(E1) ** 2 / Isat)
            )
            E1 *= np.exp(-simu0.alpha * simu0.delta_z)
        times[i, 2, k] = time.perf_counter() - t0
        pbar.update(1)
pbar.close()
np.save("benchmarks_times.npy", times)
np.save("benchmarks_sizes.npy", sizes)
err_gpu = [
    np.mean(times[:, 0, :], axis=-1) - np.min(times[:, 0, :], axis=-1),
    np.max(times[:, 0, :], axis=-1) - np.mean(times[:, 0, :], axis=-1),
]
err_cpu = [
    np.mean(times[:, 1, :], axis=-1) - np.min(times[:, 1, :], axis=-1),
    np.max(times[:, 1, :], axis=-1) - np.mean(times[:, 1, :], axis=-1),
]
err_np = [
    np.mean(times[:, 2, :], axis=-1) - np.min(times[:, 2, :], axis=-1),
    np.max(times[:, 2, :], axis=-1) - np.mean(times[:, 2, :], axis=-1),
]
fig, ax = plt.subplots()
ax.errorbar(
    np.log2(sizes).astype(int),
    np.mean(times[:, 0, :], axis=-1),
    yerr=err_gpu,
    label="GPU",
    marker="o",
    capsize=4,
)
ax.errorbar(
    np.log2(sizes).astype(int),
    np.mean(times[:, 1, :], axis=-1),
    yerr=err_cpu,
    label="CPU",
    marker="s",
    capsize=4,
)
ax.errorbar(
    np.log2(sizes).astype(int),
    np.mean(times[:, 2, :], axis=-1),
    yerr=err_cpu,
    label="Numpy",
    marker="^",
    capsize=4,
)
ax.legend()
ax.set_xticks(np.log2(sizes).astype(int))
ax.set_xlabel(r"Size of the system $2^N$")
ax.set_ylabel("Execution time in s")
ax.set_title("Execution time (lower is better)")
ax.set_yscale("log")
fig.savefig("benchmarks.svg", dpi=300)
plt.show()
