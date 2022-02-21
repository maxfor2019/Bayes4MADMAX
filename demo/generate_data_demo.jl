println("Hello there!") # ;-) Check if anything is responding!


using BAT
using Random, Distributions
using DataFrames
using OrderedCollections, ValueShapes
using ForwardDiff # to be able to define Theory so BAT can read the struct
using Plots
using HDF5
using DelimitedFiles

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

data = gaussian_noise(2e6,7e6,2e3, scale=1e-24)
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
add_axion!(data, signal, ex)

plot(data[2:end,1],data[2:end,2],
    ylims = (minimum(data[2:end,2]),maximum(data[2:end,2]))
)

# If the folder does not yet exist, construct folder
PATH = data_path(DATASET, KEYWORD, TYPE)
mkdir(PATH)

# Sometimes frustratingly slow. Fix this!
@time save_data(data, ex, signal, "myfile", DATASET, KEYWORD, TYPE; overwrite=true)




#=
PATH = data_path("211201-MockData/Bg_fits_MGVI", "simulated", "processed_data")
info = data_path("211201-MockData/Bg_fits_id", "simulated", "processed_data")


for file_name in readdir(PATH)
    file_name = file_name[1:end-3]
    ex = Experiment(Be=10.0, A=1.0, β=5e4, t_int=100.0, Δω=2034.0, f_ref=11.0e9)
    signal = Theory(ma=45.517, rhoa=0.3,EoverN=0.1,σ_v=218.0,vlab=242.1)

    dinf = get_data(file_name[1:end-4]*"-id", "211201-MockData/Bg_fits_id", "simulated", "processed_data")
    df = DataFrame([dinf[!,:freq], dinf[!,:pow], dinf[!,:noise], dinf[!,:axion]], [:freq, :pow, :noise, :axion])

    N = h5read(PATH*file_name*".h5", "noise_std")
    NA = h5read(PATH*file_name*".h5", "background")
    df[!,:background] = vec(mean(NA, dims=2))
    df[!,:pownoB] = df[!, :pow] .- vec(mean(NA, dims=2))
    for i in 1:size(NA)[2]
        insertcols!(df, :background => NA[:,i], makeunique=true)
        insertcols!(df, :pownoB => df[!, :pow] .- NA[:,i], makeunique=true)
    end
    #df[!, :axion] = NA[:,2] - N[:,2]
    #df[!, :noise] = N[:,2]
    file_name *= "-MGVI"
    save_data(df, ex, signal, file_name, "211201-MockData/Bg_fits_MGVI", "simulated", "processed_data", overwrite=true)
end
NA[:,1]
file_name = readdir(PATH)[1231][1:end-3]
NA = h5read(PATH*file_name*".h5", "background")
dft[!,:bg]=vec(mean(NA,dims=2))
dinf = get_data("test5000-1000-id", "211201-MockData/Bg_fits_id", "simulated", "processed_data")
dinf = get_data(file_name[1:end-4]*"-id", "211201-MockData/Bg_fits_id", "simulated", "processed_data")
dft = DataFrame([dinf[!,:freq], dinf[!,:pow], dinf[!,:noise], dinf[!,:axion]], [:freq, :pow, :noise, :axion])
=#
