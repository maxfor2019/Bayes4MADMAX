using BAT
using Random, LinearAlgebra
using Distributions
using ValueShapes
using Plots
using FFTW
using ForwardDiff
using PDMats
using HDF5
using LogExpFunctions
include("forward_models.jl")
rng = Random.default_rng()


########### DATA HANDLING ##########

data_raw = readdlm("./data/Fake_Axion_Data/Data_Set_1/Test00Osc01_17-01-24_0915.dat")#, '\t', Float32, '\n')
vals = data_raw[10000:10200,2]
vals = (vals .- mean(data_raw[:,2]))./std(data_raw[:,2])
dims = (length(vals),)
distances = (1/length(data_raw[:,1]),)
data = vals

######### HARMONIC TRANSFORM #########

harmonic_pad_distances = 1 ./ (2 .* dims .* distances)
volume = 2 * dims[1] * harmonic_pad_distances[1]^2

D = dist_array(2 .* dims, harmonic_pad_distances)

x = collect(distances[1]:distances[1]:distances[1] * (dims[1]);)
x_pad = collect(distances[1]:distances[1]:distances[1] * dims[1]*2;)

ht = inv(FFTW.plan_r2r(zeros(length(x_pad)), FFTW.DHT));

my_ht = let n_x_pad = length(x_pad), harmonic_pad_distances = (1,), ht=ht
    input -> harmonic_transform(input, n_x_pad, harmonic_pad_distances, ht)
end

my_inv_ht = let n_x_pad = length(x_pad), harmonic_pad_distances = (1,), ht=ht
    input -> harmonic_transform(input, n_x_pad, harmonic_pad_distances , inv(ht))
end

######## LOAD BACKGROUND AND NOISE #########

bg_fit_results = h5open("coding/background_fit.h5", "r") do file
    read(file)
end
noise_stds = bg_fit_results["n"]
harmonic_noise_vars = noise_stds.^2*2*dims[1]
offsets = bg_fit_results["offset"]
slopes = bg_fit_results["slope"]
zero_modes = bg_fit_results["zero_mode"]
bg_amplitudes = [harmonic_pad_distances[1] .* (2 * dims[1]) .* amplitude_forward_model(offsets[i],slopes[i],zero_modes[i],D) for i in 1:length(zero_modes)]
bg_noise_covariances = [0. .* bg_amplitudes[i].^2 .+ harmonic_noise_vars[i] for i in 1:length(zero_modes)] 

plot(D[2:dims[1]],bg_noise_covariances[1][2:dims[1]].^0.5, xaxis=:log, yaxis=:log)
plot!(D[2:dims[1]],bg_amplitudes[1][2:dims[1]], xaxis=:log, yaxis=:log)

####### BUILD FORWARD MODEL ######

mask = collect(1:length(x_pad))
mask = mask .<= length(mask)÷2

my_adjoint_mask = let mask = mask
    input -> adjoint_mask(input, mask)
end

my_gaussian_shape = let x = x
    parameters -> gaussian_shape_forward_model(parameters.α, parameters.μ, parameters.σ, x)
end

###### INJECT SIGNAL ######
truth = (
    α = 0.000003,
    μ = 0.002,
    σ = 0.0001
)

true_shape = my_gaussian_shape(truth)
data = my_ht(bg_amplitudes[1].*rand(Normal(),length(x_pad)))[mask]
data = data + true_shape

# Sanity check
a = rand(Normal(),length(x_pad))
my_bg_n =my_ht(bg_noise_covariances[1].^0.5 .* a)[1:length(x)]
my_bg =my_ht(bg_amplitudes[1].* a)[1:length(x)]
data = my_bg_n + true_shape
plot(my_bg_n + true_shape)
plot!(my_bg + true_shape)
my_n = my_bg_n - my_bg
println(std(my_n))
println(noise_stds[1])

#######  LIKELIHOOD #####

my_residuals = let data=data, forward_model=my_gaussian_shape,
                            adjoint_mask=my_adjoint_mask, harmonic_transform=my_ht,
                            inv_ht = my_inv_ht;
    parameters -> begin
        residual = data - forward_model(parameters) 
        harmonic_residual = (inv_ht ∘ adjoint_mask)(residual)
        return residual, harmonic_residual
    end
end

likelihood = let residuals = my_residuals, mask=mask, ht = my_ht, 
    harmonic_covs = bg_noise_covariances, volume=volume, distances=distances;
    parameters -> begin
        residual, harmonic_residual = residuals(parameters)
        noise_weighted_residual = ht(harmonic_residual ./harmonic_covs[1])[mask]
        llhd = - 0.5 .* sum(residual .* noise_weighted_residual)
        # noise_weighted_residual = harmonic_residual ./harmonic_covs[1] /volume^2
        # llhd = - 0.5 .* sum(harmonic_residual .* noise_weighted_residual)

        # llhd = logsumexp([- 0.5 * sum(residual.^2 ./ cov) for cov in harmonic_covs])
        # llhd = llhd / length(harmonic_covs)
        # llhd =  - 0.5 * sum(residual.^2 ./ harmonic_covs[1] / (2*dims[1])^2) #FIXME
        LogDVal(llhd)
    end
end



######## PRIOR ########

prior = NamedTupleDist(
    α = Uniform(1e-15, 0.00001),
    μ = Uniform(0.1*x[end], 0.9*x[end]),
    σ = Uniform(1e-15, 0.001)
)



sampler =  MCMCSampling(mcalg =  MetropolisHastings(), nsteps = 10^4, store_burnin=true,
         nchains = 4, strict=false, burnin=MCMCMultiCycleBurnin(max_ncycles=1000))

# sampler =  MCMCSampling(mcalg = HamiltonianMC(), nsteps = 10^4, nchains = 4)
posterior = PosteriorDensity(likelihood, prior)
r = bat_sample(posterior, sampler)
samples = r.result
SampledDensity(posterior, samples)

plot(
    samples,
    mean = false, std = false, globalmode = true, marginalmode = false,
    nbins = 50,
)
plot(x, true_shape,label="truth", title="recovered signal", color="red")

for i in 1:1000
    plot!(x,my_gaussian_shape(samples.v[i]), color="black",alpha=0.3)
end
plot!(x,my_gaussian_shape(samples.v[1]))

# savefig("./data/tmp/posterior.png")


# plot(x, my_gaussian_shape, samples)
# savefig("./data/tmp/result.png")
