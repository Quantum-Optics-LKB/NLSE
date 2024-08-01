import numpy as np
from scipy.constants import c

from NLSE import NLSE_3d

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32


N = 256
NZ = 128
n2 = -1.6e-9
D0 = 1e-16
vg = 1e-1 * c
waist = 2.23e-3
duration = 2e-6
waist2 = 70e-6
window = np.array(
    [4 * waist, 8 * duration]
)  # 4*waist transverse, 10e-6 s temporal
energy = 1.05 * duration
Isat = 10e4  # saturation intensity in W/m^2
L = 1e-2
alpha = 20


def main():
    simu = NLSE_3d(
        alpha=alpha,
        energy=energy,
        window=window,
        n2=n2,
        D0=D0,
        vg=vg,
        V=None,
        L=L,
        NX=N,
        NY=N,
        NZ=NZ,
        Isat=Isat,
        backend="GPU",
    )
    simu.delta_z = 0.25e-4
    E_0 = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(
        PRECISION_COMPLEX
    )
    E_0 *= np.exp(-(simu.TT**2) / duration**2)
    simu.V = -1e-4 * np.exp(-(simu.XX**2 + simu.YY**2) / waist2**2).astype(
        PRECISION_COMPLEX
    )
    simu.out_field(E_0, L, verbose=True, plot=True, precision="single")


if __name__ == "__main__":
    main()
