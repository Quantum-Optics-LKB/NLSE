include("nlse.jl")
using Plots
using Metal

# Parameters
params = PhysicalParameters(alpha=20,
    power=1.0,
    window=10.0f-3,
    n2=-1.0f-9,
    V0=0.0,
    L=10.0f-2,
    NX=2048,
    NY=2048,
    Isat=Inf32,
    nl_length=0.0,
    k=2 * pi / 780.0f-9)
coords = Coordinates(parameters=params)
simu = NLSE(parameters=params, coordinates=coords)
w0 = 1.5f-3
A0 = exp.(-(coords.xx .^ 2 + coords.yy .^ 2) / w0^2) + 1im * zeros(Float32, (params.NX, params.NY))
A0 = MtlArray(A0)
A = out_field(simu, A0, simu.parameters.L)
rho = abs2.(A) * c * epsilon_0 / 2
phi = angle.(A)
gr()
h1 = heatmap(simu.coordinates.x, simu.coordinates.y, rho, aspect_ratio=1.0)
h2 = heatmap(simu.coordinates.x, simu.coordinates.y, phi, aspect_ratio=1.0)
plot(h1, h2, layout=2, show=true)