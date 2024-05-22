def main():
    from NLSE import DDGPE
    import numpy as np

    gamma = 1
    puiss = 1
    window = 1e-3
    g = 1
    omega = 1
    T = 10e-12
    dd = DDGPE(gamma, puiss, window, g, omega, T)
    dd.delta_z = 1e-14
    E0 = np.ones((2, dd.NY, dd.NX), dtype=np.complex64)
    dd.out_field(E0, T, plot=True)


if __name__ == "__main__":
    main()
