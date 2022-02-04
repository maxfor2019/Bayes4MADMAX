
# may be necessary when running from the terminal
import Pkg
Pkg.activate("/home/th347/diehl/Documents/2103-Bayes")

println("Hello there!") # ;-) Check if anything is responding!


using BAT
using Random, LinearAlgebra, Statistics, StatsBase
include("../src/custom_distributions.jl")

using ValueShapes





using HDF5 # also for saving
using SavitzkyGolay # SG background fit









include("../src/forward_models.jl")

# Define where the data can be found / should be stored
#DATASET = "test"
#KEYWORD = "simulated"
#TYPE = "processed_data"

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
"""

using DelimitedFiles
using DataFrames
using ForwardDiff # to be able to define Theory so BAT can read the struct

include("../src/read_data.jl")
include("../src/physics.jl")

file_name = "myfile"
data = get_data(file_name, "test", "simulated", "raw_data")
ex = read_ex("test", "simulated", "raw_data")
signal = read_th("test", "simulated", "raw_data")

#=
data = get_Olaf_2017("Data_Set_3")
# Calling this function changes DATASET and KEYWORD
println(DATASET*" "*KEYWORD)
=#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                                                       #
#                       Manipulate Data                                 #
#                                                                       #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Do background subtraction on raw data (or whatever else you come up with in the future)
"""
    Inputs:
        data: [DataFrame] Keys are :freq and :pow or :powwA depending on whether the dataset includes a simulated axion.
        ex: Experiment()
        signal: Theory()

    Outputs: 
        data: [DataFrame] Keys are :freq and :pownoB or :powwAnoB depending on whether the dataset includes a simulated axion.
        ex: Experiment()
        signal: Theory()
        Saves all of the above to datafiles called "FILENAME_noB.smp" and "meta-FILENAME_noB.txt"
"""

using Plots, LaTeXStrings
using SavitzkyGolay
using Distributions
using OrderedCollections

include("../src/plotting.jl")
include("../src/backgrounds.jl")
include("../src/generate_data.jl")


# Data before background reduction
plot_data(data)

data = sg_fit(data, 4, 101; cut=true)

# Data after background reduction
plot_data(data)


save_data(data, ex, signal, "myfile_nobg", "test", "simulated", "processed_data")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                                                       #
#                        MCMC Analysis                                  #
#                                                                       #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Let MCMC run

using DelimitedFiles
using DataFrames
using ForwardDiff # to be able to define Theory so BAT can read the struct
using IntervalSets, Distributions, ValueShapes
using BAT
using FileIO, JLD2 # for saving the samples

include("../src/read_data.jl")
include("../src/physics.jl")

file_name = "myfile_nobg"
data = get_data(file_name, "test", "simulated", "processed_data")
ex = read_ex("test", "simulated", "processed_data")
signal = read_th("test", "simulated", "processed_data")

include("../src/prior.jl")
include("../src/likelihood.jl")

# Implement 
prior = make_prior(data, ex,pow=:loggaγγ)
posterior = PosteriorDensity(likelihood, prior)

likelihood((ma=signal.ma, sig_v=signal.σ_v, gag=gaγγ(fa(scale_ma(signal.ma)), signal.EoverN)))
plot_truths(data, signal, ex)


# Make sure to set JULIA_NUM_THREADS=nchains for maximal speed (before starting up Julia), e.g. via VSC settings.
# Below are alternatives for the sampling algorithm
#sampling = MCMCSampling(mcalg = MetropolisHastings(tuning=AdaptiveMHTuning()), nsteps = 5*10^4, nchains = 4, burnin = MCMCMultiCycleBurnin(max_ncycles=50))
sampling = MCMCSampling(mcalg = HamiltonianMC(), nsteps = 2*10^3, nchains = 4, burnin = MCMCMultiCycleBurnin(max_ncycles=3))
#using UltraNest
#sampling = ReactiveNestedSampling()

@time out = bat_sample(posterior, sampling)

save_samples(out, prior, "myfile", "test", "simulated")


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
