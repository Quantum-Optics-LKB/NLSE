## Physical situation

The code offers to solve a typical [non linear Schrödinger](https://en.wikipedia.org/wiki/Nonlinear_Schr%C3%B6dinger_equation) / [Gross-Pitaevskii](https://en.wikipedia.org/wiki/Gross%E2%80%93Pitaevskii_equation) equation of the type :
$$i\partial_{t}\psi = -\frac{1}{2}\nabla^2\psi+V\psi+g|\psi|^2\psi$$

In this particular instance, we solve in the formalism of the propagation of a pulse of light in a non linear medium.
Within the [paraxial approximation](https://en.wikipedia.org/wiki/Paraxial_approximation), the propagation equation for the field $E$ in V/m solved is:

$$
i\partial_{z}E = -\frac{1}{2k_0}\nabla_{\perp}^2 E +
\frac{D_0}{2}\partial^2_t E
-\frac{k_0}{2}\delta n(r) E - n_2 \frac{k_0}{2n}c\epsilon_0|E|^2E
$$

Here, the constants are defined as followed :

- $k_0$ : is the electric field [wavenumber](https://en.wikipedia.org/wiki/Wavenumber) in $m^{-1}$
- $D_0$ : is the [group velocity dispersion](https://en.wikipedia.org/wiki/Group-velocity_dispersion) (GVD) in $s^2/m$
- $\delta n(\mathbf{r})$ : the "potential" i.e a local change in linear index of refraction. Dimensionless.
- $n_2$ : the [non linear index of refraction](https://en.wikipedia.org/wiki/Kerr_effect) in $m^2/W$.
- $n$ is the linear [index of refraction](https://en.wikipedia.org/wiki/Refractive_index). In our case 1.
- $c,\epsilon_0$ : the speed of light and electric permittivity of vacuum.

In all generality, the interaction term can be *non-local* i.e $n_2=n_2(\mathbf{r})$.
This means usually that the response will be described as a convolution by some non-local kernel:

$$
n_2(\mathbf{r})|E|^2(\mathbf{r})=n_2\int_{\mathbb{R}^2}\mathrm{d}\mathbf{r}' K(\mathbf{r}-\mathbf{r}')|E|^2(\mathbf{r}'),
$$

where $K(\mathbf{r})$ is the non-local kernel, typically the Green function of some diffusion equation.

Please note that all of the code works with the **"God given" units** i.e **SI units** !
