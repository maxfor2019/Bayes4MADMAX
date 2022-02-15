
# may be necessary when running from the terminal
import Pkg
Pkg.activate("/home/th347/diehl/Documents/2103-Bayes")

println("Hello there!") # ;-) Check if anything is responding!


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
ex = read_ex(file_name, "test", "simulated", "raw_data")
signal = read_th(file_name, "test", "simulated", "raw_data")
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
using ValueShapes

include("../src/plotting.jl")
include("../src/backgrounds.jl")
include("../src/generate_data.jl")


# Data before background reduction
plot_data(data)

data = sg_fit(data, 4, 301; cut=true)

# Data after background reduction
plot_data(data; key=:pownoB)


save_data(data, ex, signal, "myfile_nobg", "test", "simulated", "processed_data", overwrite=true)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                                                       #
#                        MCMC Analysis                                  #
#                                                                       #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Let MCMC run
"""
    Inputs:
        DATASET: [String] Name of dataset
        KEYWORD: [String] Always set to "simulated" when generating data!
        TYPE: [String] I suppose you want to read "processed_data"!
        Needs to have a file called FILENAME.smp (and a file called meta-FILENAME.txt) in the specified folder DATASET.

        Might work differently when reading datasets from the experimentalists.

    Outputs: 
        samples: [DensitySampleVector] BAT samples for further analysis. No other BAT output is saved!
        prior: [NamedTupleDist] As the type suggests named tuple containing prior distributions.
        Saves the above to a JLD2 file with keys "prior" and "samples".
"""


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
ex = read_ex(file_name, "test", "simulated", "processed_data")
signal = read_th(file_name, "test", "simulated", "processed_data")

include("../src/prior.jl")
include("../src/likelihood.jl")

# Implement 
prior = make_prior(data, ex,pow=:loggaγγ)
posterior = PosteriorDensity(likelihood, prior)

likelihood((ma=signal.ma, sig_v=signal.σ_v, gag=gaγγ(fa(scale_ma(signal.ma)), signal.EoverN)))
plot_truths(data, signal, ex)


# Make sure to set JULIA_NUM_THREADS=nchains for maximal speed (before starting up Julia), e.g. via VSC settings.
# Below are alternatives for the sampling algorithm
sampling = MCMCSampling(mcalg = MetropolisHastings(tuning=AdaptiveMHTuning()), nsteps = 5*10^4, nchains = 4, burnin = MCMCMultiCycleBurnin(max_ncycles=500))
#sampling = MCMCSampling(mcalg = HamiltonianMC(), nsteps = 2*10^3, nchains = 4, burnin = MCMCMultiCycleBurnin(max_ncycles=3))
#using UltraNest
#sampling = ReactiveNestedSampling()

@time out = bat_sample(posterior, sampling)

save_samples(out, prior, "myfile", "test", "simulated", overwrite=true)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                                                       #
#                            Check                                      #
#                                                                       #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Read and analyze MCMC output
"""
    Inputs:
        DATASET: [String] Name of dataset
        KEYWORD: [String] Always set to "simulated" when generating data!
        TYPE: [String] I suppose you want to read "processed_data"!
        Needs to have a file called FILENAME.smp (and a file called meta-FILENAME.txt) in the specified folder DATASET.

        Might work differently when reading datasets from the experimentalists.

    Outputs: 
        Use your imagination! I post means and stds of the MCMC parameters as well as a corner plot and a best fit plot.
"""



using BAT
using FileIO, JLD2 # for saving the samples
using DelimitedFiles, DataFrames, ValueShapes, OrderedCollections
using Random, Distributions, ForwardDiff
using Plots, LaTeXStrings

include("../src/read_data.jl")
include("../src/generate_data.jl")
include("../src/plotting.jl")
include("../src/custom_distributions.jl")
include("../src/physics.jl")

file_name = "myfile_nobg"
data = get_data(file_name, "test", "simulated", "processed_data")
ex = read_ex(file_name, "test", "simulated", "processed_data")
signal = read_th(file_name, "test", "simulated", "processed_data")

include("../src/likelihood.jl") # contains the fit_function. Make sure you use the same for the analysis as you used to run MCMC!!

samples = get_samples("myfile", "test", "simulated")
prior = get_prior("myfile", "test", "simulated")


plot(samples)
println("Means: $(mean(samples)[1])")
println("Stds: $(std(samples)[1])")
plot_fit(samples, data, ex) # Best fit plot using mean values