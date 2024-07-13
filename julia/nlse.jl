using FFTW
using Metal
const PRECISION = Float32
const CPRECISION = Complex{PRECISION}
const c = 299792458.0
const epsilon_0 = 8.854187817e-12

FFTW.set_num_threads(Threads.nthreads())

if isfile("fft_julia.wisdom")
    FFTW.import_wisdom("fft_julia.wisdom")
end

Base.@kwdef mutable struct PhysicalParameters
    # Parameters
    alpha::PRECISION
    power::PRECISION
    window::PRECISION
    n2::PRECISION
    V0::PRECISION
    L::PRECISION
    NX::Int = 256
    NY::Int = 256
    Isat::PRECISION = Inf
    nl_length::PRECISION = 0.0
    k::PRECISION = 2 * pi / 780e-9
end

Base.@kwdef mutable struct Coordinates
    # Simulation
    parameters::PhysicalParameters
    dx::PRECISION = params.window / params.NX
    dy::PRECISION = params.window / params.NY
    dz::PRECISION = params.L / 1000
    x::Vector{PRECISION} = collect(range(-params.window / 2, params.window / 2, params.NX))
    y::Vector{PRECISION} = collect(range(-params.window / 2, params.window / 2, params.NY))
    kx::Vector{PRECISION} = 2 * pi * FFTW.fftfreq(params.NX, 1 / dx)
    ky::Vector{PRECISION} = 2 * pi * FFTW.fftfreq(params.NY, 1 / dy)
    xx::Matrix{PRECISION} = x' .* ones(params.NY) # this does a meshgrid
    yy::Matrix{PRECISION} = ones(params.NX)' .* y
    kxx::Matrix{PRECISION} = kx' .* ones(params.NY)
    kyy::Matrix{PRECISION} = ones(params.NX)' .* ky
    V::Matrix{CPRECISION} = ones(params.NX, params.NY) .* params.V0
end

Base.@kwdef mutable struct NLSE
    parameters::PhysicalParameters
    coordinates::Coordinates
    propagator::Matrix{CPRECISION} = exp.(-1im * 0.5 * coordinates.dz .* (coordinates.kxx .^ 2 + coordinates.kyy .^ 2) ./ parameters.k)
    plan_fft::FFTW.Plan{CPRECISION} = FFTW.plan_fft!(ones(CPRECISION, (params.NX, params.NY)))
    plan_ifft::FFTW.Plan{CPRECISION} = FFTW.plan_ifft!(ones(CPRECISION, (params.NX, params.NY)))
end

function nl_prop!(A::Matrix{CPRECISION}, dz::PRECISION, alpha::PRECISION, V::Matrix{CPRECISION}, g::PRECISION, Isat::PRECISION)
    A_sq = abs2.(A)
    # saturation
    sat = 1 .+ A_sq ./ Isat
    sat .= 1 ./ sat
    # interactions
    arg = 1im * g .* A_sq .* sat
    # losses
    arg .+= -alpha .* sat
    # potential
    arg .+= 1im .* V
    arg .*= dz
    arg .= exp.(arg)
    A .*= arg
end

function split_step!(simu::NLSE, A::Matrix{CPRECISION}, plan_fft::FFTW.Plan{CPRECISION}, plan_ifft::FFTW.Plan{CPRECISION})
    # Fourier transform
    plan_fft * A
    # apply propagator
    A .*= simu.propagator
    # inverse Fourier transform
    plan_ifft * A
    # apply nonlinearity
    nl_prop!(A, simu.coordinates.dz,
        simu.parameters.alpha / 2,
        simu.parameters.k / 2 .* simu.coordinates.V,
        simu.parameters.k / 2 * c * epsilon_0 * simu.parameters.n2,
        2 * simu.parameters.Isat / (c * epsilon_0))
end

function out_field(simu::NLSE, A::Matrix{CPRECISION}, z::PRECISION, normalize::Bool=true)
    simu.plan_fft = FFTW.plan_fft!(A)
    simu.plan_ifft = FFTW.plan_ifft!(A)
    FFTW.export_wisdom("fft_julia.wisdom")
    if normalize
        norm = sum(abs2.(A)) * simu.coordinates.dx * simu.coordinates.dy
        norm *= c * epsilon_0 / 2
        A .*= sqrt(simu.parameters.power / norm)
    end
    for i = 1:round(Int, z / simu.coordinates.dz)
        split_step!(simu, A, simu.plan_fft, simu.plan_ifft)
    end
    return A
end

function out_field(simu::NLSE, A::MtlMatrix{ComplexF32}, z::PRECISION, normalize::Bool=true)
    simu.plan_fft = FFTW.plan_fft!(A)
    simu.plan_ifft = FFTW.plan_ifft!(A)
    FFTW.export_wisdom("fft_julia.wisdom")
    if normalize
        norm = sum(abs2.(A)) * simu.coordinates.dx * simu.coordinates.dy
        norm *= c * epsilon_0 / 2
        A .*= sqrt(simu.parameters.power / norm)
    end
    for i = 1:round(Int, z / simu.coordinates.dz)
        split_step!(simu, A, simu.plan_fft, simu.plan_ifft)
    end
    return A
end
