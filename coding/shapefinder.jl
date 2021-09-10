using BAT
using Random, LinearAlgebra
using Distributions
using ValueShapes
using Plots
using FFTW
using ForwardDiff
using PDMats


distances = (1/100,)
dims = (100,)
harmonic_pad_distances = 1 ./ (2 .* dims .* distances)



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


function amplitude_spectrum(d::Real, zero_mode_dist::Normal, slope::Real, offset::Real)
    # R = float(promote_type(typeof(d), typeof(zero_mode), typeof(slope), typeof(offset)))
    # d ≈ 0 : R(zero_mode), R(exp(offset + slope * log(d)))
    ifelse(d ≈ 0, promote(std(zero_mode_dist), exp(offset + slope * log(d)))...)
end


rng = Random.default_rng()

trafo_dht = FFTW.plan_r2r(zeros(2 .* dims), FFTW.DHT)

D = dist_array(2 .* dims, harmonic_pad_distances)

x = collect(distances[1]:distances[1]:distances[1] * dims[1];)
x_pad = collect(distances[1]:distances[1]:distances[1] * dims[1]*2;)


function gaussian_shape(p::NamedTuple{(:α , :μ, :σ)}, x::Real)
    p.α.* pdf(Normal(p.μ,p.σ),x)
end

prior = NamedTupleDist(
    α = Uniform(1e-15, 0.5),
    μ = Uniform(0.1, 0.9),
    σ = Uniform(1e-15, 0.05)
)

truth = rand(prior)

truth = (
    α = 0.4,
    μ = 0.3,
    σ = 0.01
)

true_shape = map(xx -> gaussian_shape(truth, xx),x)


noise_mean = zeros(size(x_pad))
N = fill(1. / harmonic_pad_distances[1], size(x_pad))

mask = collect(1:length(x_pad))
mask = mask .<= length(mask)÷2

function adjoint_mask(x)
    y = zeros(size(mask))
    y[mask] = x
    return y
end

function adjoint_mask(dp::Vector{ForwardDiff.Dual{T, V, N}}) where {T,V,N}
    val_res = adjoint_mask(ForwardDiff.value.(dp))
    psize = size(ForwardDiff.partials(dp[1]), 1)
    ps = x -> ForwardDiff.partials.(dp, x)
    val_ps = map((x -> adjoint_mask(ps(x))), 1:psize)
    ForwardDiff.Dual{T}.(val_res, val_ps...)
end

# n = map(NN -> rand(Normal(0,NN)),N)

n_h = rand(MvNormal(noise_mean,N))



function kernel()
    corr = 1e20 ./(1 .+ (D/0.005).^6)
    corr[1] = 1000
    corr
end;

corr = kernel()

ht = FFTW.plan_r2r(zeros(length(x_pad)), FFTW.DHT) * (harmonic_pad_distances[1] / sqrt(length(x_pad)));
ξ_b = rand(Normal(0,1),length(corr))

b = ht * (corr.^0.5 .* ξ_b)
n = ht * n_h

function my_ht(dp::Vector{Float64})
    ht * dp
end


function my_ht(dp::Vector{ForwardDiff.Dual{T, V, N}}) where {T,V,N}
    val_res = ht *  ForwardDiff.value.(dp)
    psize = size(ForwardDiff.partials(dp[1]), 1)
    ps = x -> ForwardDiff.partials.(dp, x)
    val_ps = map((x -> ht*ps(x)), 1:psize)
    ForwardDiff.Dual{T}.(val_res, val_ps...)
end


data = true_shape + n[mask] + b[mask]
plot(x,true_shape, label="signal", title="Setup")
scatter!(x, data,marker=:x,color=:black, label="data" )
plot!(x,b[mask], label="background")
savefig("./data/tmp/setup.png")

cov_z = corr + N.^2

z_h = rand(MvNormal(noise_mean,cov_z.^0.5))

z = (ht * z_h)


a = rand(length(x))
b = rand(length(x_pad))


aa = my_ht(adjoint_mask(a))
bb = my_ht(b)[mask]

likelihood = let data = data, f = gaussian_shape, N = N, x = x
    params -> begin
        μ = map(xx -> f(params, xx),x)
        residual = data .- μ
        noise_weigted_residual = my_ht(((1 ./cov_z) .* (my_ht(adjoint_mask(residual)))))[mask]  #FIXME weights
        llhd = - 0.5 .* sum(residual .* noise_weigted_residual)
        # llhd = logpdf(MvNormal(μ,N),data)
        return LogDVal(llhd)
    end
end
# sampler =  MCMCSampling(mcalg =  MetropolisHastings(), nsteps = 10^4, store_burnin=true,
        #  nchains = 4, strict=false, burnin=MCMCMultiCycleBurnin(max_ncycles=100))

sampler =  MCMCSampling(mcalg = HamiltonianMC(), nsteps = 10^4, nchains = 4)
posterior = PosteriorDensity(likelihood, prior)
r = bat_sample(posterior, sampler)
samples = r.result
SampledDensity(posterior, samples)

plot(
    samples,
    mean = false, std = false, globalmode = true, marginalmode = false,
    nbins = 50,
)
savefig("./data/tmp/posterior.png")


plot(x, gaussian_shape, samples)
plot!(x, true_shape,label="truth", title="recovered signal")
savefig("./data/tmp/result.png")
