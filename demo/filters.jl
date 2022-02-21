println("Hello there!")

# Come up with realistic priors on axion mass and abundancy from theory parameters
using BAT
using Random, LinearAlgebra, Statistics, Distributions, StatsBase
include("../src/custom_distributions.jl")
using Plots, LaTeXStrings
using ValueShapes, DataFrames
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

################ PART 3 - look at data #####################################################
# read out basic properties from data written in Part 2.

# example data plot
data = get_data("test5000-"*string(134), "211201-MockData-cp", "simulated", "raw_data")#(path*"test5000-"*string(134)*".h5", "pow")
plot_data(data, key=:pow)


nr_of_samples = 5000
s_bin = 968 # signal bin, i.e. the most interesting bin for analysis


################ Generate processed datasets #####################################################
#=
ex = read_ex("test5000-1", "211201-MockData-cp", "simulated", "raw_data")
signal = read_th("test5000-1", "211201-MockData-cp", "simulated", "raw_data")

@time for i in range(1,nr_of_samples)
    data = get_data("test5000-"*string(i), "211201-MockData-cp", "simulated", "raw_data")
    data[!,:pownoB] = data[!,:noise] .+ data[!,:axion]
    save_data(data, ex, signal, "test5000-"*string(i)*"-id", "211201-MockData/Bg_fits_id", "simulated", "processed_data")
end

@time for i in range(1,nr_of_samples)
    data = get_data("test5000-"*string(i), "211201-MockData-cp", "simulated", "raw_data")
    data = sg_fit(data, 4, 301; cut=false)
    save_data(data, ex, signal, "test5000-"*string(i)*"-sg", "211201-MockData/Bg_fits_sg", "simulated", "processed_data")
end
=#

################ PART 4 - ideal analysis #####################################################
# runtime ~ seconds

function handle(data, s_bin; add=0)
    # just take maximal bin into consideration
    #=
    if in("pownoB_1", names(data)) == false
        return data[s_bin+add,:pownoB]
    else
        # assumes that pownoB is the mean and pownoB_i are the samples. If this is not the case, what were you thinking?!
        samps = []
        for j in 1:sum(occursin.("pownoB", names(data)))-1
            append!(samps, data[s_bin+add,Symbol("pownoB_"*string(j))])
        end
        return samps
    end
    =#

    # implement HAYSTAC-like filter (weights corresponding to axion signal. This version only works for correct signal position)
    s_bin = ideal_filter(data)
    if in("pownoB_1", names(data)) == false
        return sum(data[:,:pownoB] .* s_bin[add+1:end-add])
    else
        # assumes that pownoB is the mean and pownoB_i are the samples. If this is not the case, what were you thinking?!
        samps = []
        for j in 1:sum(occursin.("pownoB", names(data)))-1
            append!(samps, sum(data[!,Symbol("pownoB_"*string(j))] .* s_bin[add+1:end-add]))
        end
        return samps
    end
end

function ideal_filter(data)
    if in("axion", names(data)) == true
        return deepcopy(data[!,:axion])
    else
        error("Data does not contain axion. This filter is not yet implemented if you don't know where the axion is!")
    end
end

function get_binlist(key::String, nr_of_samples, s_bin)
    bin_list = []
    for i in range(1,nr_of_samples)
        try
            data = get_data("test5000-"*string(i)*"-"*key, "211201-MockData/Bg_fits_"*key, "simulated", "processed_data")
            integral = handle(data, s_bin) # this bin range optimizes gauss_ideal.μ / gauss_ideal.σ
            append!(bin_list, integral)
        catch
            println(i)
        end
    end
    bin_list *= 1e20
end

function get_gaussian(bin_list; norm=1.0)
    bl = deepcopy(bin_list) ./ norm
    gauss = fit(Normal, bl) # Fit a gaussian on the histogram and use this afterwards to calculate η.
end

BLid = get_binlist("id", nr_of_samples, s_bin)
BLsg = get_binlist("sg", nr_of_samples, s_bin)
BLmgvi = get_binlist("MGVI", nr_of_samples, s_bin)
BLn = get_binlist("id", nr_of_samples, 1021)

Gnorm = get_gaussian(BLid)
Gid = get_gaussian(BLid; norm=Gnorm.σ)
Gsg = get_gaussian(BLsg, norm=Gnorm.σ)
Gmgvi = get_gaussian(BLmgvi; norm=Gnorm.σ)
Gn = get_gaussian(BLn; norm=Gnorm.σ)


# throw all bin_lists in there!
list_of_binlists = [BLid, BLsg, BLmgvi] ./Gnorm.σ
list_of_gausses = [Gid, Gsg, Gmgvi]

# replicate Fig. 6 of 1706.08388
function gauss_prediction(norm, edges, len)
    (cdf(norm, edges)[2:end] - cdf(norm, edges)[1:end-1]) #.* len
end

function make_hist(bin_list)
    hi = fit(Histogram, bin_list, nbins=50)
    m = (hi.edges[1] .+ (hi.edges[1][2] - hi.edges[1][1])/2.)[1:end-1]
    #hi.weights = Vector{Real}(hi.weights ./ sum(hi.weights))
    return hi, m
end

hists = make_hist.(list_of_binlists)

scatter(hists[1][2], hists[1][1].weights ./ sum(hists[1][1].weights), yaxis=(:log, [0.0001, :auto]), c=:red, label="Ideal")
plot!(hists[1][2], gauss_prediction(list_of_gausses[1],hists[1][1].edges[1], length(list_of_binlists[1])), label="")#, yaxis=(:log, [0.9, :auto]))

name_list = ["SG filter", "MGVI"]#, "noise"]
# kwargs: c, markershape

for i in 2:length(list_of_binlists)
    scatter!(hists[i][2], hists[i][1].weights ./ sum(hists[i][1].weights), label=name_list[i-1])
    plot!(hists[i][2], gauss_prediction(list_of_gausses[i],hists[i][1].edges[1], length(list_of_binlists[i])), label="")
end
plot!(legend=:topleft)
xlabel!("Normalized power excess")
ylabel!("Normalized Count")

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
#=
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
=#