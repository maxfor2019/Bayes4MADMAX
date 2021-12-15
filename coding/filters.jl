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

function add_artificial_background!(data)
    data[:,2] .+=  deepcopy(data[:,1]).^3 .* 2e-43 .- deepcopy(data[:,1]).^2 .* 1e-35 .+ deepcopy(data[:,1]) .* 1e-28 .+ 1e-20 .+ 1e-23 .* sin.(deepcopy(data[:,1])./5e4)
    return data
end

function dummy_data()
    data = gaussian_noise(1e6,20e6,2.034e3,scale=9.4e-24)
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
    σ_v=218.0
)

bin_list_ideal = ones(300,2)
@time for i in range(1,length(bin_list_ideal[:,1]))
    data = dummy_data()
    add_axion!(data, signal)
    data, scale = my_normalize(data, 1e-20)
    s_bin = 2442
    bin_list_ideal[i,1] = data[:,2][s_bin]
    bin_list_ideal[i,2] = data[:,2][s_bin+20]
end
bin_list_ideal
gauss_ideal = fit(Normal, bin_list_ideal[:,2])
histogram!(bin_list_ideal[:,1], alpha=0.5, label="ideal")
plot!(gauss_ideal)

function bin_means(hist)
    edges = hist.edges[1]
    means = edges[1:end-1] .+ (edges[2]-edges[1])/2
    return means
end


bin_list_sg = ones(300,2)
@time for i in range(1,length(bin_list_sg[:,1]))
    data = dummy_data()
    add_axion!(data, signal)
    add_artificial_background!(data)
    data, scale = my_normalize(data, 1e-20)
    w = 201
    d=6
    sg = savitzky_golay(data[:,2], w, d)
    s_bin = 2442
    data = data[w:end-w,:]
    myfit = sg.y[w:end-w]
    data[:,2] = data[:,2] - myfit
    bin_list_sg[i,1] = data[:,2][s_bin-w+1]
    bin_list_sg[i,2] = data[:,2][s_bin-w+21] # Why +1 ?!
end
bin_list_sg
gauss_sg = fit(Normal, bin_list_sg[:,2])
histogram!(bin_list_sg[:,1], alpha=0.5, label="SG filter")
plot!(gauss_sg)
η(normal1, normal2) = normal1.μ / normal2.μ * normal2.σ / normal1.σ
ξ(normal1, normal2) = normal1.σ / normal2.σ
ξ(gauss_sg, gauss_ideal)
η(gauss_sg, gauss_ideal)
gauss_sg.μ


data = dummy_data()
add_axion!(data, signal)
add_artificial_background!(data)
data, scale = my_normalize(data, 1e-20)
w = 201
d=6
sg = savitzky_golay(data[:,2], w, d)
s_bin = 2442
data = data[w:end-w,:]
myfit = sg.y[w:end-w]
data[:,2] = data[:,2] - myfit
data[:,2][s_bin-w+1]
plot(data[:,2][s_bin-w-5:s_bin-w+5])
plot!(data[:,2][s_bin-5:s_bin+5])

plot(data[:,1],data[:,2])

