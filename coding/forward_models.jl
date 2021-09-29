using BAT
using Random, LinearAlgebra
using Distributions
using ValueShapes
using Plots
using FFTW
import ForwardDiff
using DelimitedFiles
using PDMats
using MGVI
using Optim


###### UTILITY ####

function autodiff_linop(dp, linop)
    val_res = linop(ForwardDiff.value.(dp))
    psize = size(ForwardDiff.partials(dp[1]), 1)
    ps = x -> ForwardDiff.partials.(dp, x)
    val_ps = map((x -> linop(ps(x))), 1:psize)
    ForwardDiff.Dual{T}.(val_res, val_ps...)
end

######## MASK ######

function adjoint_mask(input, mask)
    function apply_adjoint_mask(input, mask)
        y = zeros(size(mask))
        y[mask] = x
        return y
    end
    function apply_adjoint_mask(input::Vector{ForwardDiff.Dual{T, V, N}}, mask) where {T,V,N}
        autodiff_linop(input, apply_adjoint_mask)
    end
    apply_adjoint_mask(input, mask)
end



######## CORRELATION STRUCTURE #####

function map_idx(idx::Real, idx_range::AbstractUnitRange{<:Integer})
    i = idx - minimum(idx_range)
    n = length(eachindex(idx_range))
    n_2 = n >> 1
    ifelse(i <= n_2, i, i - n)
end

function dist_k(idx::CartesianIndex, ax::NTuple{N,<:AbstractUnitRange{<:Integer}}, harmonic_distances::NTuple{N,<:Real}) where N
    mapped_idx = map(map_idx, Tuple(idx), ax)
    norm(map(*, mapped_idx, harmonic_distances))
end

function dist_array(dims::NTuple{N,<:Real}, harmonic_distances::NTuple{N,<:Real}) where N
    cart_idxs = CartesianIndices(map(Base.OneTo, dims))
    dist_k.(cart_idxs, Ref(axes(cart_idxs)), Ref(harmonic_distances))
end

function amplitude_forward_model(offset, slope, zero_mode, dist_array)
    corr = exp.(offset .+ slope .* log.(dist_array[2:end]))
    corr = vcat(zero_mode, corr)
    corr.^0.5 
end

######## HARMONIC TRANSFORM #######

function harmonic_transform(input, n_x_pad::Integer, harmonic_pad_distances::Tuple, ht::FFTW.r2rFFTWPlan)
    function apply_ht(ht::FFTW.r2rFFTWPlan, dp::Vector{Float64})
        ht * dp
    end
    function apply_ht(ht::FFTW.r2rFFTWPlan, dp::Vector{ForwardDiff.Dual{T, V, N}}) where {T,V,N}
        val_res = ht *  ForwardDiff.value.(dp)
        psize = size(ForwardDiff.partials(dp[1]), 1)
        ps = x -> ForwardDiff.partials.(dp, x)
        val_ps = map((x -> ht*ps(x)), 1:psize)
        ForwardDiff.Dual{T}.(val_res, val_ps...)
    end
    apply_ht(ht, input) * (harmonic_pad_distances[1]) #/ sqrt(n_x_pad)) #Zero mode is integrated position space!!!!1
end


######## GAUSSIAN PROCESS ######

function gp_forward_model(ξ, amplitude, harmonic_transform, nbin)
    harmonic_gp = amplitude .* ξ
    gp = harmonic_transform(harmonic_gp)
    gp[1:nbin]
end

###### GAUSSIAN SHAPE ######

function gaussian_shape_forward_model(amplitude::Real, mean::Real, std::Real, x::Vector{Float64})
    function eval_gauss(amplitude::Real, mean::Real, std::Real, x::Real)
        amplitude .* pdf(Normal(mean, std),x)
    end
    map(xx -> eval_gauss(amplitude, mean, std, xx), x)
end