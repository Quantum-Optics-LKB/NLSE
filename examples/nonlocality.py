from NLSE import NLSE
import numpy as np

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32x


def main():
    N = 2048
    n2 = -1.6e-9
    waist = 2.23e-3
    window = 4 * waist
    puiss = 1.05
    Isat = 10e4  # saturation intensity in W/m^2
    L = 1e-3
    alpha = 20
    nl_length = 60e-6
    simu_2d = NLSE(
        alpha,
        puiss,
        window,
        n2,
        None,
        L,
        NX=N,
        NY=N,
        Isat=Isat,
        nl_length=nl_length,
        backend="GPU",
    )
    simu_2d.delta_z = 1e-4
    E_0 = np.exp(-(simu_2d.XX**2 + simu_2d.YY**2) / waist**2).astype(PRECISION_COMPLEX)
    simu_2d.out_field(
        E_0,
        L,
        verbose=True,
        plot=True,
        precision="single",
    )


if __name__ == "__main__":
    main()
