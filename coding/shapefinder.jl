using BAT
using Random
using Distributions
using ValueShapes
using Plots
using FFTW
using ForwardDiff

x = collect(0.01:0.01:1;)

function gaussian_shape(p::NamedTuple{(:α , :μ, :σ)}, x::Real)
    p.α.* pdf(Normal(p.μ,p.σ),x)
end

prior = NamedTupleDist(
    α = Uniform(0.05, 0.5),
    μ = Uniform(1e-3, 1),
    σ = Uniform(1e-3, 0.5)
)

truth = rand(prior)

truth = (
    α = 0.3,
    μ = 0.5,
    σ = 0.02
)

# Why not simply true_shape = gaussian_shape.(truth, x) oder so
true_shape = map(xx -> gaussian_shape(truth, xx),x) # i.e. [f(elem) for elem in x] with function f(xx) = gaussian_shape(truth, xx)


MvNormal(0,1)

noise_mean = zeros(size(x))
N = fill(1, size(x))
# n = map(NN -> rand(Normal(0,NN)),N)

# Draw from Mulitvariate Normal Distribution with mean 0, std 1
n_h = rand(MvNormal(noise_mean,N)) #shouldnt N have to be a correlation matrix here!?

k = collect(0:(length(x))÷2 -1);

# Spits out one array with declining vals from 1e7 and ascending again to 1e7
function kernel()
    pspec = 1000 ./(0.0001.+(k/0.1).^4)
    positive_modes = pspec
    negative_modes = positive_modes[end:-1:1]
    [positive_modes; negative_modes]
end;

corr = kernel()

# FFT plan. To FFT an array write ht * array
ht = FFTW.plan_r2r(zeros(length(x)), FFTW.DHT);
ξ_b = rand(Normal(0,1),length(corr))

b = ht * (corr.^0.5 .* ξ_b)./sqrt(length(x)) # background
n = (ht * n_h)./sqrt(length(x)) # noise

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


data = true_shape +b #+ n + b
plot(x,true_shape, label="signal", title="Setup")
scatter!(x, data,marker=:x,color=:black, label="data" )
plot!(x,b, label="background")
savefig("./data/tmp/setup.png")

cov_z = corr + N.^2

z_h = rand(MvNormal(noise_mean,cov_z.^0.5))

z = (ht * z_h)./sqrt(length(x))

truth2 = (
    α = 0.3,
    μ = 0.5,
    σ = 0.2
)

μ = map(xx -> gaussian_shape(truth2, xx),x)
residual = data #.- μ
tt  = my_ht((1 ./cov_z) .* my_ht(residual))
noise_weigted_residual = my_ht(((1 ./cov_z) .* (my_ht(residual))))./length(x)

plot(x, tt)


likelihood = let data = data, f = gaussian_shape, N = N, x = x
    params -> begin
        μ = map(xx -> f(params, xx),x) # gives sth that looks like the signal when params==truth
        residual = data .- μ
        noise_weigted_residual = my_ht(((1 ./cov_z) .* (my_ht(residual))))./length(x)  #FIXME weights
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
