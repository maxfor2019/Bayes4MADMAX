
# may be necessary when running from the terminal
import Pkg
Pkg.activate("/home/th347/diehl/Documents/2103-Bayes")

println("Hello there!") # ;-) Check if anything is responding!


using BAT
using Random, LinearAlgebra, Statistics, Distributions, StatsBase
include("../src/custom_distributions.jl")
using Plots, LaTeXStrings
using ValueShapes
using IntervalSets
using FileIO, JLD2 # for saving the samples

using HDF5 # also for saving
using SavitzkyGolay # SG background fit
using ForwardDiff # to be able to define Theory so BAT can read the struct

include("../src/physics.jl")
include("../src/read_data.jl")
include("../src/plotting.jl")
include("../src/forward_models.jl")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                                                       #
#                            Initialize                                 #
#                                                                       #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

names_list = ["Het3_10K_0-15z_18-85GHz_20170413_160107_S0"*string(i)*".smp" for i in 1:4]
data = combine_data(names_list, path="./data/raw_data/measured/Olaf_2017/Data_Set_2/")

function initialize(data)
    options=(
        # reference frequency
        f_ref = 11.0e9,#+2.034e3,
    )

    rel_freqs = data[:,1]

    Δfreq = mean([rel_freqs[i] - rel_freqs[i-1] for i in 2:length(rel_freqs)])
    freqs = rel_freqs .+ options.f_ref

    ex = Experiment(Be=10.0, A=1.0, β=5e4, t_int=100.0, Δω=Δfreq)
    return options, ex, Δfreq
end


function sg_fit!(data)
    b=150
    e=24426
    sc = mean(data[:,2])

    rdata2 = deepcopy(data)
    rdata2[:,2] ./= sc
    sg = savitzky_golay(rdata2[:,2], 301, 4)
    rdata2 = rdata2[b:e,:]
    ft = sg.y[b:e]
    rdata2[:,2] = rdata2[:,2] - ft
    data = rdata2
    data[:,2] .*= sc
    return data
end

signal = Theory(
    ma=45.517, 
    rhoa=0.3,
    EoverN=0.1,
    σ_v=218.0,
    vlab=242.1
)

options, ex, Δfreq = initialize(data)
add_axion!(data, signal)
sg_fit!(data)

# generate dummy white noise of a specific length
#data = gaussian_noise(5e6,5e6+1000*Δfreq, Δfreq, scale=1e-23)

include("prior.jl")
include("likelihood.jl")

prior = make_prior(data, options,pow=:loggaγγ)
posterior = PosteriorDensity(likelihood, prior)

#truth = (ma=45.514, sig_v=59.9, log_gag=log10.(gaγγ(fa(scale_ma(signal.ma)),signal.EoverN)))  
#plot_truths(truth,data,ex, options)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                                                       #
#                            Run                                        #
#                                                                       #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Make sure to set JULIA_NUM_THREADS=nchains for maximal speed (before starting up Julia), e.g. via VSC settings.
#samples = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(tuning=AdaptiveMHTuning()), nsteps = 10^5, nchains = 4, convergence=BrooksGelmanConvergence(10.0, false), burnin = MCMCMultiCycleBurnin(max_ncycles=30))).result
sampling = MCMCSampling(mcalg = MetropolisHastings(tuning=AdaptiveMHTuning()), nsteps = 5*10^4, nchains = 4, burnin = MCMCMultiCycleBurnin(max_ncycles=500))
#sampling = MCMCSampling(mcalg = HamiltonianMC(), nsteps = 1*10^3, nchains = 4, burnin = MCMCMultiCycleBurnin(max_ncycles=3))
#using UltraNest
#sampling = ReactiveNestedSampling()

@time out = bat_sample(posterior, sampling)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                                                       #
#                            Save                                       #
#                                                                       #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

samples_path = "/remote/ceph/user/d/diehl/MADMAXsamples/FakeAxion/"
file_name = "test"

input = (
    data=data,
    ex=ex,
    options=options,
    #signal=signal,
    prior=prior,
    likelihood=likelihood,
    posterior=posterior,
    MCMCsampler=sampling
)

run2 = Dict(
    "input" => input,
    "samples" => out.result
)

FileIO.save(samples_path*file_name*".jld2", run2)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                                                       #
#                            Check                                      #
#                                                                       #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#=
input = FileIO.load(samples_path*file_name*".jld2", "input")
data = input.data
options = input.options
ex=input.ex
likelihood=input.likelihood
prior=input.prior
signal=input.signal
posterior=input.posterior
sampling = input.MCMCsampler
samples = FileIO.load(samples_path*file_name*".jld2", "samples")
plot(samples)

println("Mean: $(mean(samples))")
println("Std: $(std(samples))")
plot_fit(samples, data, ex, options, savefig=nothing)
xlims!((3.595e7,3.6075e7))
#xlims!((5.5e6,6.5e6))
mysavefig("220124-sg_loggag_OlafNoSignal_corner")
=#
