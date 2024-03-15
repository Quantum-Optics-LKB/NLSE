import matplotlib.pyplot as plt
import numpy as np
from nlse import NLSE
from cycler import cycler
import time

PRECISION = "single"
if PRECISION == "double":
    PRECISION_REAL = np.float64
    PRECISION_COMPLEX = np.complex128
else:
    PRECISION_REAL = np.float32
    PRECISION_COMPLEX = np.complex64

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
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 12})

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
L = 1e-2
alpha = 22
dn = None
N_avg = 10
sizes = np.logspace(6, 14, 9, base=2, dtype=int)
times = np.zeros((len(sizes), 2, N_avg))
for i, size in enumerate(sizes):
    for j, backend in enumerate(["GPU", "CPU"]):
        for k in range(N_avg):
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
            simu0.delta_z = 0.5e-4
            E_0 = np.exp(-(np.hypot(simu0.XX, simu0.YY) ** 2) / waist**2).astype(
                PRECISION_COMPLEX
            )
            t0 = time.perf_counter()
            simu0.out_field(E_0, L)
            times[i, j, k] = time.perf_counter() - t0
err_gpu = np.vstack([np.min(times[:, 0, :], axis=-1), np.max(times[:, 0, :], axis=-1)])
err_cpu = np.vstack([np.min(times[:, 1, :], axis=-1), np.max(times[:, 1, :], axis=-1)])
fig, ax = plt.subplots()
ax.errorbar(
    np.log2(sizes).astype(int),
    np.median(times[:, 0, :], axis=-1),
    yerr=err_gpu,
    label="GPU",
    marker="o",
)
ax.errorbar(
    np.log2(sizes).astype(int),
    np.median(times[:, 1, :], axis=-1),
    yerr=err_cpu,
    label="CPU",
    marker="s",
)
ax.legend()
ax.set_xticks(np.log2(sizes).astype(int))
ax.set_xlabel(r"Size of the system $2^N$")
ax.set_ylabel("Execution time (s)")
ax.set_yscale("log")
fig.savefig("benchmarks.pdf", dpi=300)
plt.show()
