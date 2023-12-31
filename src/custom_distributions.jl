#=
struct inv_sqrt_S{T <: AbstractFloat} <: Distribution{Univariate,Continuous}

    a::T
    b::T
    c::T

    # Normalized and extended to 10^-26 yr^-1  [a, b]
    # f(x) =

    μ::T
    var::T
    cov::Matrix{T}
    σ::T
end

export inv_sqrt_S


function inv_sqrt_S(a::Real, b::Real, T::DataType = Float64)

    c = 1.0 / (2.0 * (sqrt(b) - sqrt(a)))

    mean = (2.0 * c / 3.0) * (b^(3.0 / 2.0) - a^(3.0 / 2.0))
    var = (2.0 * c / 5.0) * (b^(5.0 / 2.0) - a^(5.0 / 2.0)) - mean^2


    d::inv_sqrt_S{T} = inv_sqrt_S{T}(
        a,
        b,
        c,
        mean,
        var,
        fill(var, 1, 1),
        sqrt(var)
    )
end


function Random.rand(rng::AbstractRNG, d::inv_sqrt_S{T})::T where {T <: AbstractFloat}
    res = (rand(rng) / (2.0 * d.c) + sqrt(d.a))^2
    return res
end


function Distributions.pdf(d::inv_sqrt_S{T}, x::Real)::T where {T <: AbstractFloat}
    return d.c / sqrt(x)
end

function Distributions.logpdf(d::inv_sqrt_S{T}, x::Real)::T where {T <: AbstractFloat}
    return log(d.c) - 0.5 * log(x)
end


function Distributions.cdf(d::inv_sqrt_S{T}, x::Real)::T where {T <: AbstractFloat}
    return (2.0 * d.c * (sqrt(x) - sqrt(d.a)))
end


function Distributions.minimum(d::inv_sqrt_S{T})::T where {T <: AbstractFloat}
    d.a
end

function Distributions.maximum(d::inv_sqrt_S{T})::T where {T <: AbstractFloat}
    d.b
end


function Distributions.insupport(d::inv_sqrt_S{T}, x::Real)::Bool where {T <: AbstractFloat}
    Distributions.minimum(d) <= x <= Distributions.maximum(d)
end

Base.eltype(d::inv_sqrt_S{T}) where {T <: AbstractFloat}= T

Statistics.mean(d::inv_sqrt_S) = d.μ

Statistics.var(d::inv_sqrt_S) = d.var

Statistics.cov(d::inv_sqrt_S) = d.cov
=#

"""
    Maxwell Boltzmann distribution rudimentary implementation. (May be necessary to expand.)
"""
struct MaxwellBoltzmann{T <: AbstractFloat} <: Distribution{Univariate,Continuous}
    a::Real

    μ::T
    var::T
    cov::Matrix{T}
    σ::T
end

function MaxwellBoltzmann(a::Real, T::DataType = Float64)
    mean = 2.0 * a * sqrt(2/pi)
    var = a^2.0 * (3 * pi - 8.0) / pi

    d::MaxwellBoltzmann{T} = MaxwellBoltzmann{T}(
        a,
        mean,
        var,
        fill(var, 1, 1),
        sqrt(var)
    )
end

function Distributions.pdf(d::MaxwellBoltzmann{T}, x::Real) where {T <: Real}
    return 4.0 * pi / sqrt(2.0 * pi)^3. * x^2. / d.a^3. * exp(-x^2. / (2.0*d.a^2.))
end

using SpecialFunctions

function Distributions.cdf(d::MaxwellBoltzmann{T}, x::Real) where {T <: Real} # 30 μs Dont broadcast if not absolutely necessary!
    return erf(x / (sqrt(2) * d.a)) - sqrt(2 / pi) * x * exp(-x^2 / (2 * d.a^2)) / d.a
end

function Random.rand(rng::AbstractRNG, d::MaxwellBoltzmann{T}) where {T <: Real}
    x = randn(rng, Float64) * d.a 
    y = randn(rng, Float64) * d.a 
    z = randn(rng, Float64) * d.a 
    vsq = x^2.0 + y^2.0 + z^2.0
    res = sqrt(vsq)
    return res
end

using SpecialFunctions

"""
    Boosted Maxwell Boltzmann distribution rudimentary implementation. (May be necessary to expand.)
    Sun velocity has to be taken into consideration! Effectively implements formula (25) in 1701.03118. See also discussion in 1706.08388 (formula (14)).
"""
struct BoostedMaxwellBoltzmann{T <: Real} <: Distribution{Univariate,Continuous}
    σv::T
    vlab::T
end

function BoostedMaxwellBoltzmann(σv::Real, vlab::Real, T::DataType = Float64)
    d::BoostedMaxwellBoltzmann{T} = BoostedMaxwellBoltzmann{T}(
        σv,
        vlab
    )
end

function Distributions.pdf(d::BoostedMaxwellBoltzmann, x::Real)
    return 4 * pi / sqrt(2 * pi)^3 * x / (d.σv * d.vlab) * exp(-(x^2 + d.vlab^2)/(2 * d.σv^2)) * sinh(x * d.vlab / d.σv^2)
end

function Distributions.cdf(d::BoostedMaxwellBoltzmann, x::Real)
    return d.σv / (sqrt(2*pi) * d.vlab) * (exp(-(x+d.vlab)^2 /(2 * d.σv^2)) - exp((2 * x * d.vlab - x^2 -d.vlab^2) /(2 * d.σv^2))) + 1 / 2 * (erf((x-d.vlab)/(sqrt(2)*d.σv)) + erf((x+d.vlab)/(sqrt(2)*d.σv)))
end

function Random.rand(rng::AbstractRNG, d::BoostedMaxwellBoltzmann)
    x = randn(rng, Float64) * d.σv + d.vlab
    y = randn(rng, Float64) * d.σv 
    z = randn(rng, Float64) * d.σv 
    vsq = x^2 + y^2 + z^2
    res = sqrt(vsq)
    return res
end
