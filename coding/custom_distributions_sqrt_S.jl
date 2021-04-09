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


# function Distributions.quantile(d::inv_sqrt_S{T}, x::Real)::T where {T <: AbstractFloat}
#     r::UnitRange{Int} = searchsorted(d.acc_prob, x)
#     idx::Int = min(r.start, r.stop)
#     p::T = d.acc_prob[ idx ]
#     q::T = d.edges[idx]
#     missing_p::T = x - p
#     inv_weight::T = d.inv_weights[idx]
#     if !isinf(inv_weight)
#         q += missing_p * inv_weight
#     end
#     return min(q, maximum(d))
# end


Base.eltype(d::inv_sqrt_S{T}) where {T <: AbstractFloat}= T


Statistics.mean(d::inv_sqrt_S) = d.μ

Statistics.var(d::inv_sqrt_S) = d.var

Statistics.cov(d::inv_sqrt_S) = d.cov
