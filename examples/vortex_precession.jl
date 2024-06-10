using FourierGPE, Statistics, NPZ

## Initialize simulation
L = (18.0,18.0);
n = 256
N = (n,n);
sim = Sim(L,N);
@unpack_Sim sim;
## set simulation parameters
μ = 12.0

# Time dependent potential function (here trivial t dep)
import FourierGPE.V
V(x,y,t) = 0.5*(x^2 + y^2)
ψ0(x,y,μ,g) = sqrt(μ/g)*sqrt(max(1.0-V(x,y,0.0)/μ,0.0)+im*0.0)

## make initial state
x,y = X
ψi = ψ0.(x,y',μ,g)
ϕi = kspace(ψi,sim)
@pack_Sim! sim

## evolve
@time sol = runsim(sim)

## ground state
ϕg = sol[end]
ψg = xspace(ϕg,sim)

## set simulation parameters
γ = 0.0
t = LinRange(ti,tf,Nt)
ϕi = kspace(ψg,sim)
# reltol = 1e-7
# alg = DP5()

## vortex
using VortexDistributions
R(w) = sqrt(2*μ/w^2)
R(1)
rv = 3.
healinglength(x,y,μ,g) = 1/sqrt(g*abs2(ψ0(x,y,μ,g)))
ξ0 = healinglength.(0.,0.,μ,g)
ξ = healinglength(rv,0.,μ,g)

pv = PointVortex(rv,0.,1)
vi = ScalarVortex(ξ,pv)


psi = Torus(copy(ψg),x,y)
vortex!(psi,vi)

ψi .= psi.ψ
ϕi = kspace(ψi,sim)
## compare precession with Fetter JLTP 2010
ξ = 1/sqrt(μ)
Rtf = R(1)
Ωm = 3*log(Rtf/ξ/sqrt(2))/2/Rtf^2
Ωv = Ωm/(1-rv^2/Rtf^2)

@pack_Sim! sim
## evolve
sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
navg = 10;
ts = zeros(size(sizes, 1), navg);
for (i, n) in enumerate(sizes)
     println("Running simulation with n = $n")
     @unpack_Sim sim;
     sim.N = (n,n)
     @pack_Sim! sim
     for j = 1:navg
          t0 = @timed runsim(sim);
          ts[i, j] = t0.time
     end
     println("Average time to solve : $(mean(ts[i, :])) s (min $(minimum(ts[i, :])) / max $(maximum(ts[i, :])))")
end
npzwrite("julia_vortex_precession_times.npy", ts)
npzwrite("julia_vortex_precession_sizes.npy", sizes)
