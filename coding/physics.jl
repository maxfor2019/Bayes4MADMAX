"""
    All relevant fundamental constants.
"""
struct Constants
    c::Float64
    h_eV::Float64
    h_J::Float64
    qe::Float64
    eps0::Float64
    hbar_J::Float64
    hbar_eV::Float64
end

function SeedConstants()
    h_eV = 4.135667696e-15 # [eVs]
    h_J = 6.62607015e-34 # [Js]
    c = 299792458.0 # [m/s]
    qe = 1.602176634e-19 # [C]
    eps0 = 8.8541878128e-12 # [F/m]
    hbar_J = 1.054571817e-34 # [Js]
    hbar_eV = 6.582119569e-16 # [eVs]
    return Constants(c,h_eV,h_J,qe,eps0,hbar_J, hbar_eV)
end

"""
    All relevant experimental parameters.
"""
struct Experiment
    Be::Float64 # [T]
    A::Float64 # [m^2]
    β::Float64 # []
    t_int::Float64 # [s]
    Δω::Float64 # [Hz]
end

function SeedExperiment(;Be=10.0, A=1.0, β=5.0e4, t_int=100.0, Δω=1.0e3)
    return Experiment(Be, A, β, t_int, Δω)
end

"""
    All relevant theoretical/ cosmological/ nature parameters.
"""
struct Theory
    ma::Float64
    rhoa::Float64
    σ_v::Float64
end

function SeedTheory(ma, rhoa, σ_v)
    return Theory(ma, rhoa, σ_v)
end

"""
    Calculates power from a given photon count (and frequency and integration time).
    Beware: Counts do always come within a certain frequency interval. This formula implies that f is the mean
    of this interval.
"""
function Power(counts, f, Δt; c::Constants=SeedConstants())
    return f .* c.h_J .* counts ./ Δt
end

"""
    Calculates photon counts for a given power observed within a given frequency interval f_interval over time Δt.
    Be careful to choose f_interval small enough, otherwise you need an integration!
"""
function Counts(power, f_interval, Δt; c::Constants=SeedConstants())
    f = mean(f_interval)
    Δf = maximum(f_interval) - minimum(f_interval)
    return power .* Δt ./ ( f .* c.h_J ) * Δf
end

function deBroglie(mass, v=1e-3; c::Constants=SeedConstants())
    # mass::[eV]
    # v::[c]

    λdB = (c.h_eV * c.c) ./ (mass .* v)
end

"""
    Inverse of mass().
    Relativistic implementation.
"""
function freq(mass; v=0.0, c::Constants=SeedConstants())
    # mass::[eV]
    # v:: [c]
    
    # Derived from (hν)² = E² = (pc)² + (m₀c²)²
    ν = sqrt.(v.^2.0 .+ 1) .* mass ./ (c.h_eV) # [Hz]
end

"""
    Inverse of freq()
"""
function mass(freq; v=0.0, c::Constants=SeedConstants())
    # freq::[Hz]
    # v:: [c]
    m = freq .* c.h_eV ./ sqrt.(v.^2.0 .+ 1) # [eV]
end

"""
    The same as above, but solved for v. If freq < mass will return 0 instead of error!
    Freq and mass must be floats!
"""
function velocity(freq, mass; c::Constants=SeedConstants())
    # freq::[Hz]
    # mass::[eV]
    #try
    #    sqrt.((freq ./ (mass / c.h_eV)).^2.0 .- 1.0)
    #catch
    #    0.0
    #end
    if freq >= mass / c.h_eV
        sqrt((freq / (mass / c.h_eV))^2.0 - 1.0)
    else
        0.0
    end
end

"""
    Determine prediction for central signal frequency depending on axion mass.
"""
function mu(logmass, f_ref; velo=0.0)
    # Calculate absolute frequency in Hz:
    f = freq(10.0.^logmass, v=velo)
    # Subtract reference frequency
    return f - f_ref
end

"""
    Calculate axion speed from logma and f. Same formula as used for freq() and mass().
"""
function speed(logma, f, c::Constants=SeedConstants())
    return sqrt(abs((c.h_eV * f / 10.0^logma).^2.0 .-1.0))
end


"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
|                                                                                           |
|                             Axion Properties                                              |
|                                                                                           |
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

"""
    From 1511.02867.
"""
function ma(fa) #::[eV]
    # fa::[eV]
    return 5.7064e-6 * (1e21/fa) 
end

function fa(ma)
    return 1e21 * (5.7064e-6/ma)
end

function gaγγ(fa, EoverN)
    αem = c.qe^2 / (4*pi* c.eps0 * c.hbar_J * c.c)
    return αem / (2.0 * pi * fa) * abs(EoverN - 1.924)
end


"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
|                                                                                           |
|                             Signal shape, etc                                             |
|                                                                                           |
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

"""
    Galactic velocity distribution in 1D. Lab velocity not implemented.
    Formula 23 from 1701.03118v2.
    If change this to implement a different DM velocity model
"""
function velocity_distribution(v, σ_v) #::[1/c]
    # v::[c]
    # σ_v::[c]

    # Maxwell-Boltzmann distribution
    return pdf.(MaxwellBoltzmann(σ_v), v)
end

"""
    Computes dv/dω in [1/Hz].
"""
function dvdω(ω, ma; c::Constants=SeedConstants()) #::[1/Hz]
    # ω::[Hz]
    # ma::[eV]
    #try
    #    ω / ((ma/c.h_eV)^2. * sqrt((ω*c.h_eV/ma)^2.0 - 1.0))
    #catch
    #    0.0
    #end
    if ω >= mac/c.h_eV
        ω / ((ma/c.h_eV)^2. * sqrt((ω*c.h_eV/ma)^2.0 - 1.0))
    else
        0.0
    end
end

function signal_powerspectrum(ω, σ_v, rhoa, ma, ex::Experiment; c::Constants=SeedConstants()) #::[eV^2 1/Hz]
    # ω::[Hz]   
    # σ_v::[c]
    # rhoa::[GeV/cm^3]
    # ma::[eV]

    v = velocity.(ω, ma) # in []

    prefactor = signal_prefactor(rhoa, ma, ex; c=c)
    sp =  prefactor .* velocity_distribution(v, σ_v) .* dvdω.(ω, ma) #::[1e-27 J/s 1/Hz] (if scaling = 1e27)
    return sp  
end

function signal_prefactor(rhoa, ma, ex::Experiment; scaling=1e27, c::Constants=SeedConstants())
    # convert rhoa from [GeV/cm^3] to [eV^4]
    rhoa = rhoa * 1e9 * 1e6 * c.c^3.0 * c.hbar_eV^3.0

    # convert rhoa from [GeV/cm^3] to [eV^2/m^2]
    #rhoa = rhoa * 1e9 * 1e6 * c.c * c.hbar_eV

    gag = gaγγ(fa(ma),0.924) # [eV^-1]

    # convert Be from [T] to [eV^2]
    Be = ex.Be * sqrt(c.eps0 * c.c^5.0 * c.hbar_J^3.0) / c.qe^2.0 # 4 * pi * [eV^2]

    # convert A from [m^2] to [eV^-2]
    A = ex.A / (c.c^2.0 * c.hbar_eV^2.0) # [eV^-2]

    # This prefactor returns 1.708 for the values used to calculate 2.6 in Knircks Thesis. Result from Knirck: 1.6
    prefactor = rhoa/ma^2.0 * gag^2.0 * Be^2.0 * A * ex.β * (c.qe/c.hbar_eV) * scaling #::[1e-27 J/s] (if scaling = 1e27)
    return prefactor
end

"""
    Calculate number count of photons from signal (complete frequency range), assuming a fixed frequency of ma (i.e. v=0). Difference should not matter in the end, I hope!
"""
function signal_counts(ma, rhoa, ex::Experiment; c::Constants=SeedConstants())
    P_tot = signal_prefactor(rhoa, ma, ex; c=c, scaling=1.0)
    Eγ = freq(ma) * c.h_J
    Counts = P_tot / Eγ * ex.t_int
    return Counts
end

function signal_counts_bin(freqency, ma, rhoa, σ_v, ex::Experiment; c::Constants=SeedConstants()) # 799 μs
    prefactor = signal_prefactor(rhoa, ma, ex; c=c, scaling=1.0) # 232 ns
    Eγ = freq(ma) * c.h_J # This can be assumed constant, since the width of the peak is very small compared to f_ref. Parts far away from peak are 0 anyways...
    # cdf part is integral over maxwell distribution. freq = data[!,1] are bin_centers
    Counts = prefactor ./ Eγ .* ex.t_int .* (cdf.(MaxwellBoltzmann(σ_v), velocity.(freqency .+ ex.Δω/2, ma)) .- cdf.(MaxwellBoltzmann(σ_v), velocity.(freqency .- ex.Δω/2,ma)))
    return Counts
end

function background_powerspectrum(ω, kwarg_dict)
    return pdf.(Normal(kwarg_dict[:f_ref],1e6), ω) *0.0 #* 1e9
end