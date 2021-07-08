println("Hello there!")

# ToDo Liste

# Get a better understanding of the background. I.e. implement realistic fit_function for background parameters
    # which functional form? Which parameters are meaningful?
# Include signal in this!
# Come up with realistic priors on axion mass and abundancy from theory parameters
using BAT
using Random, LinearAlgebra, Statistics, Distributions, StatsBase
include("custom_distributions.jl")
using Plots, LaTeXStrings
using ValueShapes
using IntervalSets
using FileIO, JLD2 # for saving the samples


include("physics.jl")
include("read_data.jl")
include("plotting.jl")

# Set up your experiment:
ex = SeedExperiment(Be=10.0, A=1.0, β=5e4, t_int=50.0, Δω=2e3) # careful not to accidentally ignore a few of the relevant parameters!

# rename to config options or sth
kwarg_dict=(
    # reference frequency
    f_ref = 11.0e9,
    scale_ω = 1e-5,
)

const c = SeedConstants()
σ_v = 218.0 # [km/s] +/- 6 according to 1209.0759
σ_v *= 1.0e3/c.c

nuisance = (mu=2.0e5, sigma=4.0e5) # will initialize gaussian background
model = (ma=45.49366806966277, rhoa=0.1, σ_v=σ_v) # μeV, GeV/cm^3, 1
# rudimentary fit on background with polynomial (at the bottom)
means = [2190.846044879241, 319.8237184038207, -89.23295673350981, 2.721724575537181]  # p_noise=500000
#means = [38.24640564539845, 9.499141922484778, 1.2171121325012888, -0.5683639960555529] #  p_noise=10000

data = dummy_data_right_signal(nuisance, model, ex; p_noise=500000, kwargs=kwarg_dict) # p_noise=500000 is not stable
plot_data(data)

include("prior.jl")

truth = (b=means, ma=model.ma, sig_v=σ_v, rhoa=model.rhoa)
println("truth = $truth")
#wrongth = (b=means, ma=model.ma+5e-4, sig_v=σ_v, rhoa=model.rhoa)

include("likelihood.jl")
plot_truths(truth,data,ex,kwarg_dict)

posterior = PosteriorDensity(likelihood, prior)

likelihood(truth)
sb = signal_counts_bin(data[1].+kwarg_dict.f_ref, model.ma*1e-6,model.rhoa, σ_v,ex)

# Make sure to set JULIA_NUM_THREADS=nchains for maximal speed (before starting up Julia), e.g. via VSC settings.
#samples = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(tuning=AdaptiveMHTuning()), nsteps = 10^5, nchains = 4, convergence=BrooksGelmanConvergence(10.0, false), burnin = MCMCMultiCycleBurnin(max_ncycles=30))).result
@time samples = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(tuning=AdaptiveMHTuning()), nsteps = 5*10^4, nchains = 4, burnin = MCMCMultiCycleBurnin(max_ncycles=100))).result

#using UltraNest
#@time samples = bat_sample(posterior, ReactiveNestedSampling()).result


FileIO.save("./data/samples/210706-test_faster.jld2", Dict("samples" => samples))
samples = FileIO.load("./data/samples/210706-test_faster.jld2", "samples")

# corner doesnt work anymore sadly
# corner(samples, 5:7, modify=false, truths=[m_true, σ_v, rhoa_true], savefig=nothing)
plot(samples, vsel=collect(5:2:7))

println("Mean: $(mean(samples))")
plot_fit(samples, data, ex, kwarg_dict, savefig=nothing)

#= If you want to get sensible values for the coefficients
using Polynomials

f1 = Polynomials.fit(data[!,1].*kwarg_dict[:scale_ω], data[!,2], 3)
a = f1[:]
testpars = (a=a,)
plot(data[!,1], f1.(data[!,1].*kwarg_dict[:scale_ω]))
plot!(data[!,1], data[!,2])
=#

