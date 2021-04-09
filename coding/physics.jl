struct Constants
    c::Float64
    h_eV::Float64
    h_J::Float64
end

function SeedConstants()
    h_eV = 4.135667696e-15 # [eVs]
    h_J = 6.62607015e-34 # [Js]
    c = 299792458.0 # [m/s]
    return Constants(c,h_eV,h_J)
end

"""
    Calculates power from a given photon count (and frequency and integration time).
    Beware that this ignores width of frequency intervals. This may have some effect!
"""
function Power(counts, f, Δt; c::Constants=SeedConstants())
    return f .* c.h_J .* counts ./ Δt
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
    # Hope I did not loose any 2π here!
    
    # Derived from (hν)² = E² = (pc)² + (m₀c²)²
    ν = sqrt.(v^2.0 .+ 1) .* mass ./ (c.h_eV) # [Hz]
end

"""
    Inverse of freq()
"""
function mass(freq; v=0.0, c::Constants=SeedConstants())
    # freq::[Hz]
    # v:: [c]
    m = freq .* c.h_eV ./ sqrt.(v^2.0 .+ 1) # [eV]
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
    Convert axion speed in units of [c] to frequency shift from mean in [Hz]
"""
function getsigma(vsig, logma, kwargs)
    vrange = [0.0, vsig]
    fmin = mu(logma, kwargs[:f_ref], velo=vrange[1])
    fmax = mu(logma, kwargs[:f_ref], velo=vrange[2])
    return abs(fmax - fmin)
end

function speed(logma, f, c::Constants=SeedConstants())
    return sqrt(abs((c.h_eV * f / 10.0^logma) -1))
end