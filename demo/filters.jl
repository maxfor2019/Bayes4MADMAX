println("Hello there!")

# Come up with realistic priors on axion mass and abundancy from theory parameters
using BAT
using Random, LinearAlgebra, Statistics, Distributions, StatsBase
include("../src/custom_distributions.jl")
using Plots, LaTeXStrings
using ValueShapes
using IntervalSets
using FileIO, JLD2 # for saving the samples
using ForwardDiff

include("../src/physics.jl")
include("../src/read_data.jl")
include("../src/plotting.jl")
include("../src/forward_models.jl")

using SavitzkyGolay
using StatsPlots
using HDF5

################ PART 1 - Definitions #####################################################

function add_artificial_background!(data)
    data[:,2] .+=  (deepcopy(data[:,1]).-7e6).^3 .* 1e-42 .* (1. +randn()) .- (deepcopy(data[:,1]).-7e6).^2 .* 1e-35 .* (1. +randn()) .- (deepcopy(data[:,1]).-7e6) .* 1e-29 .* (1. +randn()) .+ 1e-20  .+ 2e-24 .* (1. +randn()) .* sin.(deepcopy(data[:,1])./5e4) .+ 1e-23 .* (1. +randn()) .* sin.(deepcopy(data[:,1])./20e4)
    return data
end

function dummy_data(;scale=9.4e-24)
    data = gaussian_noise(4e6,9e6,2.034e3,scale=scale)
end

function add_axion!(data, signal)
    rel_freqs = data[:,1]

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

function my_normalize!(data, scale=mean(data[:,2]))
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

################ PART 2 - write data #####################################################
# do not execute this part
# runtime ~ 1h

# Writes mock data generated by the functions above into a set of files.
path = "/remote/ceph/user/d/diehl/MADMAXsamples/FilterTest/MockDatasets/"
nr_of_samples = 100
s_bin = 968 # signal bin, i.e. the most interesting bin for analysis
#=
bin_list_ideal = ones(nr_of_samples,2)
@time for i in range(1,length(bin_list_ideal[:,1]))
    println(i)
    scale = 9.4e-24
    data = dummy_data(scale=scale)
    rel_freqs = data[:,1]
    Δfreq = mean([rel_freqs[i] - rel_freqs[i-1] for i in 2:length(rel_freqs)])
    ex = "Experiment(Be=10.0, A=1.0, β=5e4, t_int=100.0, Δω="*string(Δfreq)*")"
    sig = "Theory(ma=45.517, rhoa=0.3,EoverN=0.1,σ_v=218.0,vlab=242.1)"
    options="(f_ref = 11.0e9,)"
    h5write(path*"test5000-"*string(i)*".h5", "scale", scale)
    h5write(path*"test5000-"*string(i)*".h5", "experiment", ex)
    h5write(path*"test5000-"*string(i)*".h5", "signal", sig)
    h5write(path*"test5000-"*string(i)*".h5", "options", options)
    h5write(path*"test5000-"*string(i)*".h5", "noise", data)
    add_axion!(data, signal)
    h5write(path*"test5000-"*string(i)*".h5", "noise+ax", data)
    add_artificial_background!(data)
    h5write(path*"test5000-"*string(i)*".h5", "noise+ax+bg", data)
    data, scale = my_normalize(data, 1e-20)
    i += 1
end
=#
################ PART 3 - look at data #####################################################
# read out basic properties from data written in Part 2.

# example data plot
data = h5read(path*"test5000-"*string(134)*".h5", "noise+ax+bg")
sig = h5read(path*"test5000-"*string(134)*".h5", "signal")
println(sig) # unfortunately this is a string and not a usable object. But at least I saved the metadata...
plot(data[:,1], data[:,2]/1e-20)


################ PART 4 - ideal analysis #####################################################
# runtime ~ seconds

function handle(data, s_bin; add=0)
    # just take maximal bin into consideration
    return data[s_bin+add,2]
    # sum over a couple of bins (maximizes μ/σ for the values given)
    #return sum(data[s_bin-2+add:s_bin+7+add,2])
    # implement HAYSTAC-like filter (weights corresponding to axion signal. This version only works for correct signal position)
    #s_bin = ideal_filter()
    return sum(data[:,2] .* s_bin[add+1:end-add,2])

end

function ideal_filter()
    axdata = h5read(path*"test5000-"*string(134)*".h5", "noise+ax")
    axdata[:,2] = zeros(size(axdata[:,2]))
    signal = Theory(ma=45.517, rhoa=0.3,EoverN=0.1,σ_v=218.0,vlab=242.1)
    add_axion!(axdata, signal)
    axdata[:,2] ./= sum(axdata[:,2])
    return s_bin = deepcopy(axdata)
end

# analysis for ideal case wo/ background
bin_list_ideal = []
@time for i in range(1,nr_of_samples)
    data = h5read(path*"test5000-"*string(i)*".h5", "noise+ax")
    integral_id = handle(data, s_bin) # this bin range optimizes gauss_ideal.μ / gauss_ideal.σ
    append!(bin_list_ideal, integral_id)
    i += 1
end
bin_list_ideal *= 1e20
gauss_ideal = fit(Normal, bin_list_ideal) # Fit a gaussian on the histogram and use this afterwards to calculate η.

################ PART 5 - SG #####################################################
# runtime ~ seconds

# This is the equivalent what I did for SG filter. You would need to adapt this to MGVI
bin_list_sg = []
bin_list_noise = []
@time for i in range(1,nr_of_samples)
    data = h5read(path*"test5000-"*string(i)*".h5", "noise+ax+bg")
    data, scale = my_normalize!(data, 1e-20)
    w = 301
    d=4
    sg = savitzky_golay(data[:,2], w, d)
    data = data[w+1:end-w,:]
    myfit = sg.y[w+1:end-w]
    data[:,2] = data[:,2] - myfit
    append!(bin_list_sg, handle(data, s_bin, add=-w))#1-w))
    append!(bin_list_noise, handle(data, s_bin, add=1))
end


################ PART 6 - MGVI #####################################################
# runtime ~ seconds

path_mgvi_fits = "/remote/ceph2/user/k/knollmue/madmax_fits/"
bin_list_mgvi1 = []
bin_list_mgvi2 = []
bin_list_mgvi3 = []
@time for i in range(1,nr_of_samples)
    try
        data = h5read(path*"test5000-"*string(i)*".h5", "noise+ax+bg")
        data1 = deepcopy(data)
        data2 = deepcopy(data)
        bg = h5read(path_mgvi_fits*"test5000-"*string(i)*"_fit.h5", "background")
        bg1 = bg[:,1]#(bg[:,1] .+ bg[:,2]) ./ 2.0
        bg2 = bg[:,2]
        data1[:,2] = data[:,2] - bg1
        data1, scale = my_normalize!(data1, 1e-20)
        append!(bin_list_mgvi1, handle(data1,s_bin))
        
        data2[:,2] = data[:,2] - bg2
        data2, scale = my_normalize!(data2, 1e-20)
        append!(bin_list_mgvi2, handle(data2,s_bin))

        bg = (bg[:,1] .+ bg[:,2]) ./ 2.0
        data[:,2] = data[:,2] - bg
        data, scale = my_normalize!(data, 1e-20)
        append!(bin_list_mgvi3, handle(data,s_bin))
    catch
        println(i)
    end
end
bin_list_mgvis = vcat(bin_list_mgvi1, bin_list_mgvi2)

# throw all bin_lists in there!
list_of_binlists = [bin_list_ideal, bin_list_sg, bin_list_noise, bin_list_mgvis, bin_list_mgvi1, bin_list_mgvi2, bin_list_mgvi3]

list_of_binlists = list_of_binlists ./ gauss_ideal.σ # convert Vector{Any} to Vector{Float}, rescale
list_of_gausses = fit.(Normal, list_of_binlists)


# replicate Fig. 6 of 1706.08388
function gauss_prediction(norm, edges, len)
    (cdf(norm, edges)[2:end] - cdf(norm, edges)[1:end-1]) .* len
end

function make_hist(bin_list)
    hi = fit(Histogram, bin_list, 0.2:0.2:9.0)
    m = (hi.edges[1] .+ (hi.edges[1][2] - hi.edges[1][1])/2.)[1:end-1]
    return hi, m
end

hists = make_hist.(list_of_binlists)

scatter(hists[1][2], hists[7][1].weights, yaxis=(:log, [0.4, :auto]), c=:red, label="MGVI sum first")

sumweights = (hists[5][1].weights .+ hists[6][1].weights) ./2

scatter!(hists[1][2], sumweights, yaxis=(:log, [0.4, :auto]), c=:red, markershape=:ltriangle, label="MGVI sum last")

scatter!(hists[1][2], hists[2][1].weights, yaxis=(:log, [0.9, :auto]), c=:black, label="SG filter")
scatter!(hists[1][2], hists[1][1].weights, yaxis=(:log, [0.9, :auto]), c=:royalblue, markershape=:utriangle, label="ideal")
#scatter!(mmgvi, himgvi.weights, yaxis=(:log, [0.9, :auto]), c=:red, markershape=:square, label="MGVI")
plot!(hists[1][2], gauss_prediction(list_of_gausses[2],hists[2][1].edges[1], length(bin_list_sg)), c=:black, label="", yaxis=(:log, [0.9, :auto]))
plot!(hists[1][2], gauss_prediction(list_of_gausses[1],hists[1][1].edges[1], length(bin_list_ideal)), c=:royalblue, label="", yaxis=(:log, [0.9, :auto]))
plot!(hists[1][2], gauss_prediction(list_of_gausses[7],hists[7][1].edges[1], length(bin_list_mgvi3)), c=:red, label="", yaxis=(:log, [0.4, :auto]))
plot!(hists[1][2], gauss_prediction(list_of_gausses[4],hists[4][1].edges[1], length(bin_list_mgvis)/2), c=:blue, label="", yaxis=(:log, [0.4, :auto]))

plot!(legend=:bottom)
xlabel!("Normalized power excess")
ylabel!("Count")

function μ(gauss)
    return round(gauss.μ; digits=2)
end

function σ(gauss)
    return round(gauss.σ; digits=2)
end

μs = μ.(list_of_gausses)
σs = σ.(list_of_gausses)

# These functions are what we care for in the end. The higher the η the better!
η(normal1, normal2) = round(normal1.μ / normal2.μ * normal2.σ / normal1.σ; digits=3)
ξ(normal1, normal2) = round(normal1.σ / normal2.σ; digits=3)

ξs = ξ.(list_of_gausses, list_of_gausses[1])
ηs = η.(list_of_gausses, list_of_gausses[1])

#annotate!(2,400,text(" μ = $μsg \n σ = $σsg", :left, 10))
#annotate!(2,200,text(" μ = $μmgvi \n σ = $σmgvi", :red, :left, 10))
#annotate!(2,100,text(" μ = $μid \n σ = $σid", :royalblue, :left, 10))
#annotate!(8,10,text(" ξ = $xisg \n η = $etasg", :left, 10))
#annotate!(8,5,text(" ξ = $ximgvi \n η = $etamgvi", :red, :left, 10))
#mysavefig("220125-HAYSTACfig6_compare_sumBGs_vs_sumBins")



############# Non-Gaussianity Tests ######################################

data_id = zeros(size(h5read(path*"test5000-"*string(2)*".h5", "noise+ax")))
@time for i in range(1,nr_of_samples)
    #data = h5read(path*"test5000-"*string(i)*".h5", "noise+ax")
    data_noax = h5read(path*"test5000-"*string(i)*".h5", "noise")
    #data_ax = data .- data_noax
    #integral_id = handle(data, s_bin) # this bin range optimizes gauss_ideal.μ / gauss_ideal.σ
    #append!(bin_list_ideal, integral_id)
    data_id .+= data_noax
end

plot(data_id[:,2] .*1e20, label="ideal")

w = 0#301
d=4
data_sg = zeros(size(h5read(path*"test5000-"*string(2)*".h5", "noise+ax")[w+1:end-w,:]))
@time for i in range(1,nr_of_samples)
    data = h5read(path*"test5000-"*string(i)*".h5", "noise+ax+bg")
    data_nobg = h5read(path*"test5000-"*string(i)*".h5", "noise+ax")
    data_noax = h5read(path*"test5000-"*string(i)*".h5", "noise")
    data_ax = data_nobg .- data_noax
    data, scale = my_normalize!(data, 1e-20)
    
    sg = savitzky_golay(data[:,2], 201, d)
    data = data[w+1:end-w,:]
    myfit = sg.y[w+1:end-w]
    data[:,2] = data[:,2] - myfit - data_ax[w+1:end-w,2]*1e20
    data_sg .+= data
end

plot!(data_sg[:,2], label="SG filter 2")


data_m1 = zeros(size(h5read(path*"test5000-"*string(2)*".h5", "noise+ax")))
data_m2 = zeros(size(h5read(path*"test5000-"*string(2)*".h5", "noise+ax")))
data_msum = zeros(size(h5read(path*"test5000-"*string(2)*".h5", "noise+ax")))
@time for i in range(1,5000)
    try
        data = h5read(path*"test5000-"*string(i)*".h5", "noise+ax+bg")
        data1 = deepcopy(data)
        data2 = deepcopy(data)

        data_nobg = h5read(path*"test5000-"*string(i)*".h5", "noise+ax")
        data_noax = h5read(path*"test5000-"*string(i)*".h5", "noise")
        data_ax = data_nobg .- data_noax

        bg = h5read(path_mgvi_fits*"test5000-"*string(i)*"_fit.h5", "background")
        bg1 = bg[:,1]#(bg[:,1] .+ bg[:,2]) ./ 2.0
        bg2 = bg[:,2]
        data1[:,2] = data[:,2] - bg1 - data_ax[:,2]
        data1, scale = my_normalize!(data1, 1e-20)
        data_m1 .+= data1
        
        data2[:,2] = data[:,2] - bg2 - data_ax[:,2]
        data2, scale = my_normalize!(data2, 1e-20)
        data_m2 .+= data2

        bg = (bg[:,1] .+ bg[:,2]) ./ 2.0
        data[:,2] = data[:,2] - bg - data_ax[:,2]
        data, scale = my_normalize!(data, 1e-20)
        data_msum .+= data

    catch
        println(i)
    end
end


plot(data_msum[:,2], label="MGVI")
plot!(data_m1[:,2], label="MGVI sample1")
plot!(data_m2[:,2], label="MGVI sample2")

plot!(legend=:bottomright)
#mysavefig("220126-gaussianity_mgvi_compare")
plot!(data_m1[:,1], data_m1[:,2])
plot!(data_m2[:,1], data_m2[:,2])