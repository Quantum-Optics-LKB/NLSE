from NLSE import NLSE, callbacks
import numpy as np

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

N = 2048
n2 = -1.6e-9
waist = 2.23e-3
waist2 = 70e-6
window = 4 * waist
puiss = 1.05
Isat = 10e4  # saturation intensity in W/m^2
L = 10e-2
alpha = 20


def main():
    import matplotlib.pyplot as plt
    from scipy.constants import c, epsilon_0

    simu = NLSE(
        alpha,
        puiss,
        window,
        n2,
        None,
        L,
        NX=N,
        NY=N,
        Isat=Isat,
        backend="GPU",
    )
    simu.delta_z = 0.5e-4
    N_steps = int(simu.L / simu.delta_z) + 1
    norms = np.zeros(N_steps)
    E_0 = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(PRECISION_COMPLEX)
    simu.V = -1e-4 * np.exp(-(simu.XX**2 + simu.YY**2) / waist2**2).astype(
        PRECISION_COMPLEX
    )
    simu.out_field(
        E_0,
        L,
        verbose=True,
        plot=True,
        precision="single",
        callback=callbacks.norm,
        callback_args=(1, norms),
    )
    norms *= simu.delta_X * simu.delta_Y * c * epsilon_0 / 2
    plt.plot(np.arange(N_steps) * simu.delta_z * 1e3, norms)
    plt.xlabel("Propagation distance in mm")
    plt.ylabel("Total power in W")
    plt.title("Total power of the field")
    plt.show()
    simu.delta_z = 0.25e-4
    dzs = []
    simu.out_field(
        E_0,
        L,
        verbose=True,
        plot=True,
        precision="single",
        callback=callbacks.adapt_delta_z,
        callback_args=(10, dzs),
    )
    plt.plot(dzs)
    plt.xlabel("Iteration")
    plt.ylabel("Step size")
    plt.title("Adaptive step size")
    plt.show()


if __name__ == "__main__":
    main()
