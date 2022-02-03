println("Hello there!") # ;-) Check if anything is responding!


using BAT
using Random, Distributions
using DataFrames
using OrderedCollections
using ForwardDiff # to be able to define Theory so BAT can read the struct

include("../src/custom_distributions.jl")
include("../src/physics.jl")
include("../src/read_data.jl")
include("../src/forward_models.jl")
include("../src/generate_data.jl")

# Define where the data can be found / should be stored
DATASET = "test"
KEYWORD = "simulated"
TYPE = "raw_data"


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                                                       #
#                         Generate Data                                 #
#                                                                       #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Example of how to write a simulated datafile
"""
    Inputs:
        DATASET: [String] Name of dataset
        KEYWORD: [String] Always set to "simulated" when generating data!
        TYPE: [String] I suppose you want to generate "raw_data"!
        Need to define frequencies you are looking at somehow (e.g. by defining inputs to gaussian_noise())
        Experiment: Struct containing all the data coming from the experiment, i.e.
            Be: external magnetic field
            A: Plate surface area
            β: Boost Factor
            t_int: Integration time of measurement
            Δω: (mean) frequency resolution of your dataset
            f_ref: Convert relative frequencies in dataset to absolute ones
    Optional:
        Theory: Struct containing all necessary theoretical parameters of the signal, if you have one
            ma: Mass of the axion
            rhoa: Local axion DM energy density
            EoverN: axion anomaly ratio
            σ_v: DM velocity dispersion
            vlab: Lab velocity relative to DM halo

    Outputs:
        data: [DataFrame] Keys are :freq and :pow or :powwA depending on whether the dataset includes a simulated axion.
        ex: Experiment()
        signal: Theory()
        Saves all of the above to datafiles called "FILENAME.smp" and "meta-FILENAME.txt"
"""

data = gaussian_noise(2e6,7e6,2e3)
ex = Experiment(Be=10.0, A=1.0, β=5e4, t_int=100.0, Δω=Δω(data), f_ref=11.0e9)

# optional
signal = Theory(
    ma=45.517, 
    rhoa=0.3,
    EoverN=0.1,
    σ_v=218.0,
    vlab=242.1
)

# optional
add_artificial_background!(data)
add_axion!(data, signal)

# Sometimes frustratingly slow. Fix this!
@time save_data(data, ex, signal, "myfile", DATASET, KEYWORD, TYPE)