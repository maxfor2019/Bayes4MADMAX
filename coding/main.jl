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
Be = 10.0 # external magnetic field [T]
A = 1.0 # surface of dielectric disks [m^2]
βsq = 5.0e4 # Boost factor (assumed to be constant over this frequency range)
t_int = 100 # integration time [s]
Δω = 1e3 # integration frequency interval [Hz]
ex = SeedExperiment(Be=Be, A=A, β=5e4, t_int=50.0, Δω=5e3) # careful not to accidentally ignore a few of the relevant parameters!

kwarg_dict=Dict(
    # reference frequency
    :f_ref => 11.0e9,
    :scale_ω => 1e-5,
)

c = SeedConstants()
σ_v = 218.0 # [km/s] +/- 6 according to 1209.0759
σ_v *= 1.0e3/c.c

nuisance = (mu=2.0e5, sigma=4.0e5)
model = (ma=45.49366806966277, rhoa=0.3, σ_v=σ_v) # μeV, GeV/cm^3, 1
# rudimentary fit on background with polynomial
means = [2190.846044879241, 319.8237184038207, -89.23295673350981, 2.721724575537181]

data = dummy_data_right_signal(nuisance, model, ex; p_noise=500000, kwargs=kwarg_dict)
plot_data(data)

include("prior.jl")

m_true = model.ma
println("logm_true = $m_true")
sig_v_true = σ_v
println("sig_v_true = $sig_v_true")
rhoa_true = model.rhoa
println("rhoa_true = $rhoa_true")
truth = (b=means, ma=m_true, sig_v=sig_v_true, rhoa=rhoa_true)
wrongth = (b=means, ma=m_true+5e-4, sig_v=sig_v_true, rhoa=rhoa_true)

include("likelihood.jl")
plot_truths(truth,data,ex,kwarg_dict)

posterior = PosteriorDensity(likelihood, prior)

likelihood(truth)
sb = signal_counts_bin(data[!,1].+kwarg_dict[:f_ref], 10.0^m_true,rhoa_true, sig_v_true,ex)
plot(data[!,1],sb)

#samples = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(tuning=AdaptiveMHTuning()), nsteps = 10^5, nchains = 4, convergence=BrooksGelmanConvergence(1.1, false), burnin = MCMCMultiCycleBurnin(max_ncycles=50))).result
#@time samples = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(), nsteps = 5e4, nchains = 6, burnin = MCMCMultiCycleBurnin(max_ncycles=100))).result

FileIO.save("./data/210507-testsamples4.jld2", Dict("samples" => samples))
samples = FileIO.load("./data/210507-testsamples4.jld2", "samples")


corner(samples, 5:7, modify=false, truths=[m_true, σ_v, rhoa_true], savefig=nothing)
plot(samples)
plot(samples, vsels=collect(5:7))

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

