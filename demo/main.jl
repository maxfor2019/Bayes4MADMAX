
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

using DataFrames
using OrderedCollections
using HDF5 # also for saving
using SavitzkyGolay # SG background fit
using ForwardDiff # to be able to define Theory so BAT can read the struct

include("../src/physics.jl")
include("../src/read_data.jl")
include("../src/plotting.jl")
include("../src/forward_models.jl")

# Define where the data can be found / should be stored
DATASET = "test"
KEYWORD = "simulated"
TYPE = "raw_data"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                                                       #
#                            Read Data                                  #
#                                                                       #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Examples how to access datasets
"""
    Inputs:
        DATASET: [String] Name of dataset
        KEYWORD: [String] Always set to "simulated" when generating data!
        TYPE: [String] I suppose you want to generate "raw_data"!
        Needs to have a file called FILENAME.smp (and a file called meta-FILENAME.txt) in the specified folder DATASET.

        Might work differently when reading datasets from the experimentalists.

    Outputs:
        data: [DataFrame] Keys are :freq and :pow or :powwA depending on whether the dataset includes a simulated axion.
        ex: Experiment()
        signal: Theory()
        Saves all of the above to datafiles called "FILENAME.smp" and "meta-FILENAME.txt"
"""
filename = "myfile"
data = get_data(filename, DATASET, KEYWORD, TYPE)
ex = read_ex(DATASET, KEYWORD, TYPE)
sig = read_th(DATASET, KEYWORD, TYPE)

data = get_Olaf_2017("Data_Set_3")
# Calling the latter function changes DATASET and KEYWORD
println(DATASET*" "*KEYWORD)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                                                       #
#                       Manipulate Data                                 #
#                                                                       #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Do background subtraction on raw data (or whatever else you come up with in the future)


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
