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
include("forward_models.jl")

data = gaussian_noise(1e6,20e6,2.034e3,scale=1e-24)
rel_freqs = data[:,1]
vals = data[:,2]

options=(
    # reference frequency
    f_ref = 11.0e9,
)

Δfreq = mean([rel_freqs[i] - rel_freqs[i-1] for i in 2:length(rel_freqs)])
freqs = rel_freqs .+ options.f_ref

ex = Experiment(Be=10.0, A=1.0, β=5e4, t_int=100.0, Δω=Δfreq) # careful not to accidentally ignore a few of the relevant parameters!

my_axion = let f = freqs, ex = ex
    function ax(parameters)
        sig = axion_forward_model(parameters.ma, parameters.ρa, parameters.σv, ex, f)
        if maximum(sig) > 0.0
            nothing
        else
            error("The specified axion model is not within the frequency range of your data. Fiddle around with signal.ma or options.f_ref!")
        end
        return sig
    end

end

# signal is roughly at 11e9+18e5 Hz for this mass value
# ma + 0.001 shifts the signal roughly by 4e5 Hz
signal = (
    ma=45.501, 
    ρa=0.3,
    σv=218.0
)

ax = my_axion(signal)
vals += ax
data = hcat(rel_freqs,vals)
#data = data[1:700,:]

plot(data[:,1],data[:,2])
ylims!((minimum(data[:,2]),maximum(data[:,2])))


include("prior.jl")

prior = make_prior(data, signal, options)

truth = (ma=signal.ma, sig_v=signal.σv, rhoa=signal.ρa)
println("truth = $truth")
#wrongth = (b=means, ma=model.ma+5e-4, sig_v=σ_v, rhoa=model.rhoa)

plot!(data[:,1],fit_function(truth,data[:, 1],ex, options))

include("likelihood.jl")
plot_truths(truth,data,ex, options)

posterior = PosteriorDensity(likelihood, prior)

likelihood(truth)
likelihood(mean(samples)[1])
likelihood((ma=45.50, sig_v=2., rhoa=0.8))

# Make sure to set JULIA_NUM_THREADS=nchains for maximal speed (before starting up Julia), e.g. via VSC settings.
#samples = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(tuning=AdaptiveMHTuning()), nsteps = 10^5, nchains = 4, convergence=BrooksGelmanConvergence(10.0, false), burnin = MCMCMultiCycleBurnin(max_ncycles=30))).result
@time output = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(tuning=AdaptiveMHTuning()), nsteps = 5*10^4, nchains = 4, burnin = MCMCMultiCycleBurnin(max_ncycles=100)))
@time output = bat_sample(posterior, MCMCSampling(mcalg = HamiltonianMC(), nsteps = 5*10^4, nchains = 4, burnin = MCMCMultiCycleBurnin(max_ncycles=20)))

sampling = MCMCSampling(mcalg = MetropolisHastings(tuning=AdaptiveMHTuning()), nsteps = 5*10^4, nchains = 4, burnin = MCMCMultiCycleBurnin(max_ncycles=100))

using UltraNest
@time output = bat_sample(posterior, ReactiveNestedSampling())

input = (
    data=data,
    ex=ex,
    options=options,
    signal=signal,
    prior=prior,
    likelihood=likelihood,
    posterior=posterior,
    MCMCsampler=sampling
)

run = Dict(
    "input" => input,
    "output" => output
)

FileIO.save("./data/samples/211019-test_noB_bigS.jld2", run)
output = FileIO.load("./data/samples/211019-test_noB_bigS.jld2", "output")

samples = output.result
# corner doesnt work anymore sadly
# corner(samples, 5:7, modify=false, truths=[m_true, σ_v, rhoa_true], savefig=nothing)
plot(samples)
#mysavefig("211018-test_noB_hugeS_full")

println("Mean: $(mean(samples))")
println("Std: $(std(samples))")
plot_fit(samples, data, ex, options, savefig="211018-test_noB_hugeS_full-fit")
#xlims!((2e6,2.3e6))
#mysavefig("211018-test_noB_hugeS_full-fit-peak")
#= If you want to get sensible values for the coefficients
using Polynomials

f1 = Polynomials.fit(data[!,1].*kwarg_dict[:scale_ω], data[!,2], 3)
a = f1[:]
testpars = (a=a,)
plot(data[!,1], f1.(data[!,1].*kwarg_dict[:scale_ω]))
plot!(data[!,1], data[!,2])
=#

