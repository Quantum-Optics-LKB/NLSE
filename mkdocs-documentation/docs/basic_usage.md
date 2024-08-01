After installing `NLSE`, you can simply import one of the solvers and instantiate your problem as follows.

You first need to define the relevant physical parameters of your simulation:

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
backend = "GPU" # whether to run on the GPU or CPU
```

You can then instantiate the actual simulation object `simu` by giving the physical parameters as input of the initialization function:

```python
simu = NLSE(
    alpha, puiss, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend=backend
)
```

One can choose what field we want to propagate (here a simple gaussian), as well as the potential landscape in which the field will propagate:

```python
# Define input field and potential
E_0 = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2)
V = -1e-4 * np.exp(-(simu.XX**2 + simu.YY**2) / waist2**2)
```

Finally, in order to find the final state, one calls the `out_field` function:

```python
simu.out_field(E_0, L, verbose=True, plot=True, precision="single")
```

<!-- TODO ADD IMAGE !!! -->
