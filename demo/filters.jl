println("Hello there!")

# Come up with realistic priors on axion mass and abundancy from theory parameters
using BAT
using Random, LinearAlgebra, Statistics, Distributions, StatsBase
using Plots, LaTeXStrings
using ValueShapes, DataFrames
using IntervalSets, OrderedCollections, DelimitedFiles
using FileIO, JLD2 # for saving the samples
using ForwardDiff

include("../src/custom_distributions.jl")
include("../src/physics.jl")
include("../src/read_data.jl")
include("../src/generate_data.jl")
include("../src/plotting.jl")
include("../src/forward_models.jl")
include("../src/backgrounds.jl")

using SavitzkyGolay
using StatsPlots
using HDF5

################ PART 3 - look at data #####################################################
# read out basic properties from data written in Part 2.

#foo = h5open("./data/processed_data/simulated/211201-MockData/MGVI_test2/test5000-2403_fit.h5")



# example data plot
#data = get_data("gp_test5000-"*string(134), "211201-MockData/MGVI_test2", "simulated", "processed_data")#(path*"test5000-"*string(134)*".h5", "pow")
#plot_data(data, key=:pow)


nr_of_samples = 5000
s_bin = 984 # signal bin, i.e. the most interesting bin for analysis


################ Generate processed datasets #####################################################
=
ex = read_ex("test5k4-1", "220225-MockData4randphase", "simulated", "raw_data")
signal = read_th("test5k4-1", "220225-MockData4randphase", "simulated", "raw_data")

@time for i in range(101,nr_of_samples)
    data = get_data("test5k5-"*string(i), "220301-MockData5weakax", "simulated", "raw_data")
    data[!,:pownoB] = data[!,:noise] .+ data[!,:axion]
    save_data(data, ex, signal, "test5k5-"*string(i)*"-id", "220301-MockData5weakax/Bg_fits_id", "simulated", "processed_data", overwrite=false)
end
=#
=
@time for i in range(1,nr_of_samples)
    data = get_data("test5k5-"*string(i), "220301-MockData5weakax", "simulated", "raw_data")
    data = sg_fit(data, 4, 251; cut=true)
    save_data(data, ex, signal, "test5k5-"*string(i)*"-sg", "220301-MockData5weakax/Bg_fits_sg_251_4", "simulated", "processed_data", overwrite=false)
end
=#

################ PART 4 - ideal analysis #####################################################
# runtime ~ seconds

function handle(data, key, s_bin; add=0)
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
    s_bin = ideal_filter(data, key)
    if in("pownoB_1", names(data)) == false
        return sum(data[!,:pownoB] .* vcat(s_bin[1:end],s_bin[1:1-1])) #s_bin[add+1:end-add])
    else
        # assumes that pownoB is the mean and pownoB_i are the samples. If this is not the case, what were you thinking?!
        samps = []
        for j in 1:sum(occursin.("pownoB", names(data)))-1
            append!(samps, sum(data[!,Symbol("pownoB_"*string(j))] .* s_bin[add+1:end-add]))
        end
        return samps
    end
end

function ideal_filter(data, key)
    if in("axion", names(data)) == true
        if key == "sg"
            modax = savitzky_golay(deepcopy(data[!,:axion]) ./ maximum(data[!,:axion]), 251, 4)
            modax = deepcopy(data[!,:axion]) .- modax.y .* maximum(data[!,:axion])
            return modax ./ sum(abs2.(modax))
        elseif key == "id" || key == "gp"
            return deepcopy(data[!,:axion]) ./ sum(abs2.(data[!,:axion]))
        end
    else
        error("Data does not contain axion. This filter is not yet implemented if you don't know where the axion is!")
    end
end

function get_binlist(key::String, nr_of_samples, s_bin; filter=key)
    bin_list = []
    for i in range(1,nr_of_samples)
        try
            data = get_data("test5k4-"*string(i)*"-"*key, "220225-MockData4randphase/Bg_fits_"*key*"_251_4", "simulated", "processed_data")
            integral = handle(data, filter, s_bin) # this bin range optimizes gauss_ideal.μ / gauss_ideal.σ
            append!(bin_list, integral)
        catch
            println(i)
        end
    end
    return bin_list
end

function get_gaussian(bin_list; norm=1.0)
    bl = deepcopy(bin_list) ./ norm
    gauss = fit(Normal, bl) # Fit a gaussian on the histogram and use this afterwards to calculate η.
end

@time BLid = get_binlist("id", nr_of_samples, s_bin) # s_bin doesnt really matter for this implementation!
@time BLsg101 = get_binlist("sg", nr_of_samples, s_bin+221)
@time BLsg101id = get_binlist("sg", nr_of_samples, s_bin+221, filter="id")
@time BLsg501 = get_binlist("sg", nr_of_samples, s_bin+221)
@time BLsg501id = get_binlist("sg", nr_of_samples, s_bin+221, filter="id")
@time BLsg251 = get_binlist("sg", nr_of_samples, s_bin+221)
@time BLsg251id = get_binlist("sg", nr_of_samples, s_bin+221, filter="id")
BLmgvi = get_binlist("gp", nr_of_samples, s_bin)
BLn = get_binlist("sg", nr_of_samples, s_bin+221)
BLnid = get_binlist("sg", nr_of_samples, s_bin+221, filter="id")

data = get_data("test5k4-113-gp", "220225-MockData4randphase/Bg_fits_gp", "simulated", "processed_data")
ll = h5open("./data/processed_data/simulated/220225-MockData4randphase/Bg_fits_gp/test5k4-113-gp.h5")
h5read("./data/processed_data/simulated/220225-MockData4randphase/Bg_fits_gp/test5k4-113-gp.h5", "background_2")
close(ll)
filename="test5k4-113-gp"
PATH = data_path("220225-MockData4randphase/Bg_fits_gp", "simulated", "processed_data")
foo = h5open(PATH*filename*".h5", "r")
maximum(BLgp)
minimum(BLgp)
histogram(BLgp)#, bins=[-1, -0.8, -0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8])
BLgp = deepcopy(BLmgvi)
BLgp = BLgp[BLgp .> 0]
BLgp = BLgp[BLgp .< 1.6]
h5read(PATH*filename*".h5", "std")
DataFrame(h5read.(PATH*filename*".h5", keys(foo)), Symbol.(keys(foo)))


mean(data[!,:pownoB])
std(data[!,:noise])

Gnorm = get_gaussian(BLid)
Gid = get_gaussian(BLid; norm=Gnorm.σ)
Gsg101 = get_gaussian(BLsg101, norm=Gnorm.σ)
Gsg101id = get_gaussian(BLsg101id, norm=Gnorm.σ)
Gsg251 = get_gaussian(BLsg251, norm=Gnorm.σ)
Gsg251id = get_gaussian(BLsg251id, norm=Gnorm.σ)
Gsg501 = get_gaussian(BLsg501, norm=Gnorm.σ)
Gsg501id = get_gaussian(BLsg501id, norm=Gnorm.σ)
Gmgvi = get_gaussian(BLgp; norm=Gnorm.σ)
Gn = get_gaussian(BLn; norm=Gnorm.σ)
Gnid = get_gaussian(BLnid; norm=Gnorm.σ)

list_of_binlists = [BLid, BLsg251id, BLgp]./Gnorm.σ
list_of_gausses = [Gid, Gsg251id, Gmgvi]

# throw all bin_lists in there!
#list_of_binlists = [BLid, BLsg, BLmgvi] ./Gnorm.σ
#list_of_gausses = [Gid, Gsg, Gmgvi]

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
sum(hists[4][1].weights)

scatter(hists[1][2], hists[1][1].weights ./ sum(hists[1][1].weights), yaxis=(:log, [0.0001, :auto]), c=:red, label="Ideal")
plot!(hists[1][2], gauss_prediction(list_of_gausses[1],hists[1][1].edges[1], length(list_of_binlists[1])), label="")#, yaxis=(:log, [0.9, :auto]))

name_list = ["SG filter", "Gaussian Processes"]
# kwargs: c, markershape
clist = [:green, :purple]

for i in 2:length(list_of_binlists)
    scatter!(hists[i][2], hists[i][1].weights ./ sum(hists[i][1].weights), label=name_list[i-1], c=clist[i-1])
    plot!(hists[i][2], gauss_prediction(list_of_gausses[i],hists[i][1].edges[1], length(list_of_binlists[i-1])), c=clist[i-1], label="")
end
plot!(legend=:bottom)
plot!(title="Comparison SG vs GP")
plot!(size=(500,400))
xlabel!("Normalized power excess")
ylabel!("Normalized Count")
mysavefig("220306-SGvsGP", path="plots/")
annotate!(7.5,9.4e-2,text("S/N = 1.0", :red,:left,10))
annotate!(7.5,6.7e-2,text("S/N = 0.897", :green,:left,10))
annotate!(7.5,4.7e-2,text("S/N = 0.760", :purple,:left,10))


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
=


data_id = zeros(size(get_data("test5k3-"*string(2)*"-id", "220225-MockData3noghost/Bg_fits_id", "simulated", "processed_data")[!,:noise]))
@time for i in range(1,nr_of_samples)
    #data = h5read(path*"test5000-"*string(i)*".h5", "noise+ax")
    data_noax = get_data("test5k3-"*string(i)*"-id", "220225-MockData3noghost/Bg_fits_id", "simulated", "processed_data")[!,:noise]
    #data_ax = data .- data_noax
    #integral_id = handle(data, s_bin) # this bin range optimizes gauss_ideal.μ / gauss_ideal.σ
    #append!(bin_list_ideal, integral_id)
    data_id .+= data_noax
end

plot(data_id .*1e20, label="ideal")

get_data("test5k3-"*string(2)*"-sg", "220225-MockData3noghost/Bg_fits_sg_501_4", "simulated", "processed_data")
data_sg = zeros(size(get_data("test5k3-"*string(2)*"-sg", "220225-MockData3noghost/Bg_fits_sg_501_4", "simulated", "processed_data")[!,:background]))
@time for i in range(1,nr_of_samples)
    data = get_data("test5k3-"*string(2)*"-sg", "220225-MockData3noghost/Bg_fits_sg_501_4", "simulated", "processed_data")
    data_noax = data[!,:pownoB] #.- data[!,:axion]
    data_sg .+= data_noax
end

plot!(data_sg .* 1e20, label="SG filter")
ylims!((-2.,1.))

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
