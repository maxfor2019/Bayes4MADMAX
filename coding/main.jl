println("Hello there!")


# ToDo Liste

# Get a better understanding of the background. I.e. implement realistic fit_function for background parameters
    # which functional form? Which parameters are meaningful?
# Include signal in this!
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

names_list = ["Het3_10K_0-15z_20170308_191203_S0"*string(i)*".smp" for i in 1:4]
data = combine_data(names_list)

rel_freqs = data[:,1]
vals = data[:,2]

options=(
    # reference frequency
    f_ref = 11.0e9,#+2.034e3,
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

# signal is roughly at 11e9+18e5 Hz for this mass value
# ma + 0.001 shifts the signal roughly by 4e5 Hz
signal = Theory(
    ma=45.517, 
    rhoa=0.3,
    EoverN=0.1, # 1.92 produces no signal, the further away the bigger the signal
    σ_v=218.0,
    vlab=242.1
)

ax = my_axion(signal)
vals += ax
data = hcat(rel_freqs,vals)

ddict = Dict(
    "data" => data
)

FileIO.save("data/tmp/dataWaxion.jld2", ddict)


using HDF5
using SavitzkyGolay
######## LOAD BACKGROUND AND NOISE #########

bg_fit_results = h5open("data/bg_fits/background_fit_waxion.h5", "r") do file
    read(file)
end

#noise_stds = bg_fit_results["noise_std"]
mean_bg_fit = sum(bg_fit_results["background"],dims=2)/size(bg_fit_results["background"],2)

b=100
e=24476
scale = mean(data[:,2])
rdata1 = deepcopy(data)
rdata1[b:e,2] = (deepcopy(data[b:e,2]) .- mean_bg_fit[b:e])./scale
rdata1 = rdata1[b:e,:]

rdata2 = deepcopy(data)
rdata2[:,2] ./= scale
sg = savitzky_golay(rdata2[:,2], 201, 6)
rdata2 = rdata2[b:e,:]
fit = sg.y[b:e]
rdata2[:,2] = rdata2[:,2] -fit
#rdata2[:,2] .-= mean(rdata2[:,2])

plot(rdata2[:,1], rdata2[:,2], alpha=0.7, label="SG fit")
plot!(rdata1[:,1], rdata1[:,2], alpha=0.7, label="MGVI fit")
xlims!((5.8e6,6.2e6))


data = rdata2
data[:,2] .*= scale

include("prior.jl")
include("likelihood.jl")

prior = make_prior(data, signal, options,pow=:loggaγγ)

truth = (ma=signal.ma, sig_v=signal.σ_v, log_gag=log10.(gaγγ(fa(scale_ma(signal.ma)),signal.EoverN)))
println("truth = $truth")
plot_truths(truth,data,ex, options)
xlims!((5.5e6,6.5e6))
posterior = PosteriorDensity(likelihood, prior)


likelihood(truth)

# Make sure to set JULIA_NUM_THREADS=nchains for maximal speed (before starting up Julia), e.g. via VSC settings.
#samples = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(tuning=AdaptiveMHTuning()), nsteps = 10^5, nchains = 4, convergence=BrooksGelmanConvergence(10.0, false), burnin = MCMCMultiCycleBurnin(max_ncycles=30))).result
sampling = MCMCSampling(mcalg = MetropolisHastings(tuning=AdaptiveMHTuning()), nsteps = 5*10^4, nchains = 4, burnin = MCMCMultiCycleBurnin(max_ncycles=100))
#sampling = MCMCSampling(mcalg = HamiltonianMC(), nsteps = 5*10^4, nchains = 4, burnin = MCMCMultiCycleBurnin(max_ncycles=20))
#using UltraNest
#sampling = ReactiveNestedSampling()

@time out = bat_sample(posterior, sampling)


input = (
    data=data,
    ex=ex,
    options=options,
    signal=signal,
    prior=prior,
    likelihood=likelihood,
    posterior=posterior,
    MCMCsampler=sampling
)

run2 = Dict(
    "input" => input,
    "samples" => out.result
)

samples_path = "/remote/ceph/user/d/diehl/MADMAXsamples/FakeAxion/"
FileIO.save(samples_path*"211202-mgvi_loggag_newfit.jld2", run2)
input = FileIO.load(samples_path*"211104-test_noB_SN2_gag_full.jld2", "input")
data = input.data
options = input.options
ex=input.ex
likelihood=input.likelihood
prior=input.prior
signal=input.signal
posterior=input.posterior
sampling = input.MCMCsampler
signal.rhoa
run = FileIO.load(samples_path*"211019-test_noB_SN1_loggag_full.jld2")

samples = FileIO.load(samples_path*"211202-mgvi_loggag_newfit.jld2", "samples")
#sampleslg = FileIO.load(samples_path*"211027-test_noB_SN1_loggag_full.jld2", "samples")

samples = out.result

#mysavefig("211019-test_noB_SN1_loggag_full")
samples.v[1]
samples2 = deepcopy(samples)
samples2.v[1] = (ma=samples.v[1][:ma], sig_v=samples.v[1][:sig_v], gag=samples.v[1][:gag])
samples2

for i in 1:length(samples.v)
    samples2.v[i] = (ma=samples.v[i][:ma], sig_v=samples.v[i][:sig_v], gag=samples.v[i][:gag]*1e24)
end
plot(samples)

println("Mean: $(mean(samples))")
println("Std: $(std(samples))")
plot_fit(samples, data, ex, options, savefig=nothing)
xlims!((5e6,5.5e6))
mysavefig("211202-mgvi_loggag_newfit")
#= If you want to get sensible values for the coefficients
using Polynomials

f1 = Polynomials.fit(data[!,1].*kwarg_dict[:scale_ω], data[!,2], 3)
a = f1[:]
testpars = (a=a,)
plot(data[!,1], f1.(data[!,1].*kwarg_dict[:scale_ω]))
plot!(data[!,1], data[!,2])
=#

samples = FileIO.load(samples_path*"211027-test_noB_SN1_gag_full.jld2", "samples")
sampleslg = FileIO.load(samples_path*"211027-test_noB_SN1_loggag_full.jld2", "samples")
input = FileIO.load(samples_path*"211027-test_noB_SN1_gag_full.jld2", "input")
signal=input.signal

uslg = unshaped.(sampleslg.v)
lggags = [uslg[i][3] for i in 1:length(uslg)]
maslg = [uslg[i][1] for i in 1:length(uslg)]

us = unshaped.(samples.v)
gags = [us[i][3] for i in 1:length(us)]
mas = [us[i][1] for i in 1:length(us)]

function produce_limit(mas, rhoas; frac=0.9)
    bins = range(minimum(mas),maximum(mas),length=100)
    bins_means = [(bins[i] + bins[i+1]) / 2.0 for i in 1:length(bins)-1]

    lims = [[] for i in 1:length(frac)]
    for i in 1:length(bins)-1
        rhoasort = sort(rhoas[bins[i] .< mas .< bins[i+1]])
        for j in 1:length(frac)
            append!(lims[j], rhoasort[Int(round(frac[j]*length(rhoasort)))])
        end
    end
    return bins_means, lims
end

fracs = [0.68,0.95, 0.998]
bm, l = produce_limit(mas[1:end], gags[1:end], frac=fracs)
bmlg, llg = produce_limit(maslg[1:end], lggags[1:end], frac=fracs)
fracstot = vcat(fracs,fracs)
lrescale = [log10.(l[i]) for i in 1:length(l)]
ltot = vcat(lrescale,llg)
plot_exclusion(bm, ltot, fracstot; signal=signal)
plot_exclusion2(bm, llg, lrescale, fracs,fracs; signal=signal)
llg
lrescale
#mysavefig("211019-test_noB_SN1_loggag_full-limits")
ylims!((minimum(l), maximum(l)))
bm
log10.(1e9*gaγγ.(fa.(bm*1e-6),0.667))
plot()
a = 0.9
a[1]
length(a[1])

 




using HDF5
######## LOAD BACKGROUND AND NOISE #########

bg_fit_results = h5open("data/bg_fits/background_fit_waxion.h5", "r") do file
    read(file)
end

noise_stds = bg_fit_results["noise_std"]
bg_fit_results["background"]

plot(bg_fit_results["background"][:,1]*1e19)
plot!(data[:,2]*1e19)
vals = (data[:,2].-mean(data[:,2])) ./ std(data[:,2])
scatter(data[1000:1100,1], vals[1000:1100])
plot!(data[1000:1100,1],bg_fit_results["background"][1000:1100,1])
mean_bg_fit = sum(bg_fit_results["background"],dims=2)/20
plot(data[:,1], 1e19*(data[:,2] - mean_bg_fit))
b = 400
e=24476
b=2000
e=3000
plot(data[b:e,1], 1e19*(data[b:e,2] - mean_bg_fit[b:e]))