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

data = gaussian_noise(1e6,20e6,2.034e3,scale=18.9e-24)
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
        sig = axion_forward_model(parameters, ex, f)
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
signal = Theory(
    ma=45.501, 
    rhoa=0.3,
    EoverN=0.924,
    σ_v=218.0
)

ax = my_axion(signal)
vals += ax
data = hcat(rel_freqs,vals)
data = data[1:700,:]

maximum(ax)/18.9e-24#std(data[:,2])


plot(data[:,1],data[:,2])
ylims!((minimum(data[:,2]),maximum(data[:,2])))

include("prior.jl")
include("likelihood.jl")

prior = make_prior(data, signal, options,pow=:loggaγγ)

truth = (ma=signal.ma, sig_v=signal.σ_v, log_gag=log10(gaγγ(fa(scale_ma(signal.ma)),signal.EoverN)))
println("truth = $truth")
plot_truths(truth,data,ex, options)

posterior = PosteriorDensity(likelihood, prior)


likelihood(truth)

# Make sure to set JULIA_NUM_THREADS=nchains for maximal speed (before starting up Julia), e.g. via VSC settings.
#samples = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(tuning=AdaptiveMHTuning()), nsteps = 10^5, nchains = 4, convergence=BrooksGelmanConvergence(10.0, false), burnin = MCMCMultiCycleBurnin(max_ncycles=30))).result
sampling = MCMCSampling(mcalg = MetropolisHastings(tuning=AdaptiveMHTuning()), nsteps = 5*10^4, nchains = 4, burnin = MCMCMultiCycleBurnin(max_ncycles=2))
#sampling = MCMCSampling(mcalg = HamiltonianMC(), nsteps = 5*10^4, nchains = 4, burnin = MCMCMultiCycleBurnin(max_ncycles=20))
#using UltraNest
#sampling = ReactiveNestedSampling()

@time output = bat_sample(posterior, sampling)

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

samples_path = "/remote/ceph/user/d/diehl/MADMAXsamples/FakeAxion/"
#FileIO.save(samples_path*"211019-test_noB_SN1_loggag_full.jld2", run)
input = FileIO.load(samples_path*"211019-test_noB_SN1_loggag_full.jld2", "input")
data = input.data
options = input.options
ex=input.ex
likelihood=input.likelihood
prior=input.prior
signal=input.signal
posterior=input.posterior
sampling = input.MCMCsampler

run = FileIO.load(samples_path*"211018-test_noB_hugeS.jld2")

output = FileIO.load(samples_path*"211019-test_noB_SN1_loggag_full.jld2", "output")


samples = output.result
# corner doesnt work anymore sadly
# corner(samples, 5:7, modify=false, truths=[m_true, σ_v, rhoa_true], savefig=nothing)
plot(samples)
#mysavefig("211019-test_noB_SN1_loggag_full")

println("Mean: $(mean(samples))")
println("Std: $(std(samples))")
plot_fit(samples, data, ex, options, savefig=nothing)
xlims!((2e6,2.3e6))
#mysavefig("211019-test_noB_SN1_loggag_full-fit-peak")
#= If you want to get sensible values for the coefficients
using Polynomials

f1 = Polynomials.fit(data[!,1].*kwarg_dict[:scale_ω], data[!,2], 3)
a = f1[:]
testpars = (a=a,)
plot(data[!,1], f1.(data[!,1].*kwarg_dict[:scale_ω]))
plot!(data[!,1], data[!,2])
=#


us = unshaped.(samples.v)
loggags = [us[i][3] for i in 1:length(us)]
mas = [us[i][1] for i in 1:length(us)]

function produce_limit(mas, rhoas; frac=0.9)
    bins = range(minimum(mas),maximum(mas),length=100)
    bins_means = [(bins[i] + bins[i+1]) / 2.0 for i in 1:length(bins)-1]

    lims = [[] for i in 1:length(frac)]
    for i in 1:length(bins)-1
        rhoasort = sort(rhoas[bins[i] .< mas .< bins[i+1]])
        for j in 1:length(frac)
            append!(lims[j], rhoasort[Int(round(frac[j]*length(rhoasort)))])
        end
    end
    return bins_means, lims
end

fracs = [0.68,0.95, 0.998]
bm, l = produce_limit(mas[1:end], loggags[1:end], frac=fracs)
vcat(1, l[1], 1)
minimum(l, dims=1)
plot_exclusion(bm,l, fracs; signal=signal)
#mysavefig("211019-test_noB_SN1_loggag_full-limits")
ylims!((minimum(l), maximum(l)))
bm
log10.(1e9*gaγγ.(fa.(bm*1e-6),0.667))
plot()
a = 0.9
a[1]
length(a[1])