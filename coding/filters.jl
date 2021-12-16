println("Hello there!")

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

using SavitzkyGolay
using StatsPlots
using HDF5

function add_artificial_background!(data)
    data[:,2] .+=  (deepcopy(data[:,1]).-7e6).^3 .* 1e-42 .* (1. +randn()) .- (deepcopy(data[:,1]).-7e6).^2 .* 1e-35 .* (1. +randn()) .+ (deepcopy(data[:,1]).-7e6) .* 1e-29 .* (1. +randn()) .+ 1e-20  .+ 2e-24 .* (1. +randn()) .* sin.(deepcopy(data[:,1])./5e4) .+ 1e-23 .* (1. +randn()) .* sin.(deepcopy(data[:,1])./20e4)
    return data
end

function dummy_data()
    data = gaussian_noise(4e6,9e6,2.034e3,scale=9.4e-24)
end

function add_axion!(data, signal)
    rel_freqs = data[:,1]
    vals = data[:,2]

    # will have to cut half of the SG length

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
    ax = my_axion(signal)
    data[:,2] += ax
    return data
end

function my_normalize(data, scale=mean(data[:,2]))
    data[:,2] ./= scale
    return data, scale
end

signal = Theory(
    ma=45.517, 
    rhoa=0.3,
    EoverN=0.1,
    σ_v=218.0,
    vlab=242.1
)

# Obtains the peak power for different noise realisations without background reduction.
# This is the ideal case to test your filter against.
nr_of_samples = 5000
bin_list_ideal = ones(nr_of_samples,2)
@time for i in range(1,length(bin_list_ideal[:,1]))
    data = dummy_data()
    add_axion!(data, signal)
    data, scale = my_normalize(data, 1e-20)
    s_bin = 968
    bin_list_ideal[i,1] = data[:,2][s_bin] # Actual signal bin
    bin_list_ideal[i,2] = data[:,2][s_bin+20] # Some other bin, that should just give white noise with 0 mean. if it doesn't this can be used to further understand induced correlations.
end
h5write("./data/filters/211215-test5000.h5", "ideal", bin_list_ideal)
bin_list_ideal = h5read("./data/filters/211215-test5000.h5", "ideal")
bin_list_ideal
gauss_ideal = fit(Normal, bin_list_ideal[:,1]) # Fit a gaussian on the histogram and use this afterwards to calculate η.
histogram(bin_list_ideal[:,1], alpha=0.5, label="ideal")
plot!(gauss_ideal)

# This is the equivalent what I did for SG filter. You would need to adapt this to MGVI
bin_list_sg = ones(nr_of_samples,2)
@time for i in range(1,length(bin_list_sg[:,1]))
    data = dummy_data()
    add_axion!(data, signal)
    add_artificial_background!(data)
    data, scale = my_normalize(data, 1e-20)
    w = 201
    d=6
    sg = savitzky_golay(data[:,2], w, d)
    s_bin = 968
    data = data[w:end-w,:]
    myfit = sg.y[w:end-w]
    data[:,2] = data[:,2] - myfit
    bin_list_sg[i,1] = data[:,2][s_bin-w]
    bin_list_sg[i,2] = data[:,2][s_bin-w+21]
end
bin_list_sg
gauss_sg = fit(Normal, bin_list_sg[:,1])
histogram!(bin_list_sg[:,1], alpha=0.5, label="SG filter")
plot!(gauss_sg)

# These functions are what we care for in the end. The higher the η the better!
η(normal1, normal2) = normal1.μ / normal2.μ * normal2.σ / normal1.σ
ξ(normal1, normal2) = normal1.σ / normal2.σ
ξ(gauss_sg, gauss_ideal)
η(gauss_sg, gauss_ideal)

