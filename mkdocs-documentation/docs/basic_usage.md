After installing `NLSE`, you can simply import one of the solvers and instantiate your problem as follows:

```python
N = 2048 # number of points in solver
n2 = -1.6e-9 # nonlinear index in m^2/W
waist = 2.23e-3 # initial beam waist in m
waist2 = 70e-6 # potential beam waist in m
window = 4*waist # total computational window size in m
puiss = 1.05 # input optical power in W
Isat = 10e4  # saturation intensity in W/m^2
L = 10e-3 # Length of the medium in m
alpha = 20 # linear losses coefficient in m^-1

simu = NLSE(
    alpha, puiss, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend=backend
)
# Define input field and potential
E_0 = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2)
V = -1e-4 * np.exp(-(simu.XX**2 + simu.YY**2) / waist2**2)
simu.out_field(E_0, L, verbose=True, plot=True, precision="single")
```

<!-- TODO ADD IMAGE !!! -->