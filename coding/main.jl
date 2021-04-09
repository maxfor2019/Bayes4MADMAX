println("Hello there!")

# ToDo Liste

# Get a better understanding of the background. I.e. implement realistic fit_function for background parameters
    # which functional form? Which parameters are meaningful?
# Include signal in this!
# Come up with realistic priors on axion mass and abundancy from theory parameters

using BAT
using Random, LinearAlgebra, Statistics, Distributions, StatsBase
using Plots
using ValueShapes
using IntervalSets

include("physics.jl")
include("read_data.jl")
#data = simulated_data("Het3_10K_0-15z_20170308_191203_S01.smp")
n = (mu=2.0e5, sigma=4.0e5)
m = (x0=3.2e5, Gamma=1.0e4)
p_signal = 1000
p_noise = 50000

kwarg_dict=Dict(
    # reference frequency
    :f_ref => 11.0e9,
    # Dummy integration time
    :int_time => 10.0
)

data = dummy_data(n,m, p_signal, p_noise=p_noise, kwargs=kwarg_dict)


m_true = log10(mass(kwarg_dict[:f_ref] + m.x0))
println("logm_true = $m_true")
vsig_true = speed(m_true, kwarg_dict[:f_ref]+m.x0+m.Gamma)
println("vsig_true = $vsig_true")
truth = (b=means, logma=m_true, vsig=vsig_true, Ps=1e-22)

include("prior.jl")
include("likelihood.jl")
include("plotting.jl")

posterior = PosteriorDensity(likelihood, prior)

samples = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(tuning=AdaptiveMHTuning()), nsteps = 10^6, nchains = 4, convergence=BrooksGelmanConvergence(1.1, false), burnin = MCMCMultiCycleBurnin(max_ncycles=5000))).result

corner(samples, 5:7, modify=false, truths=[m_true, vsig_true, 1e-22], savefig=nothing)

println("Mean: $(mean(samples))")
plot_fit(samples, data, kwarg_dict, savefig=nothing)
plot_data(data)
plot!(data[!,1], fit_function(truth,data[!,1]; kwargs=kwargs))



#= If you want to get sensible values for the coefficients
using Polynomials

f1 = Polynomials.fit(data[!,1], data[!,2], 3)
a = f1[:]
testpars = (a=a,)
plot(data[!,1], f1.(data[!,1]))
plot!(data[!,1], data[!,2], yscale=:log10)
=#

mydist = MixtureModel([Normal(2,1), Normal(4,10)], [0.5,0.5])
data = rand(mydist, 1e4)


f_ref = 1e9 # [Hz]



function Power(counts, f, Δt; c::Constants=SeedConstants())
    return f .* c.h_J .* counts / Δt
end

co = 100
x(c) = Power(c,kwarg_dict[:f_ref],10)
sqrt(co)
x(co)
logpdf.(Normal.(x(co), x(sqrt(c(x(co))))), 3e-22)

c(p) =  Counts(p, kwarg_dict[:f_ref],10)

c(x(co))

function Counts(power, f, Δt; c::Constants=SeedConstants())
    return power .* Δt ./ ( f .* c.h_J )
end

expectation = data[!,2]
Normal.(expectation, Power.(sqrt.(Counts.(expectation, kwargs[:f_ref].+data[1:end,1], kwargs[:int_time])), kwargs[:f_ref].+data[1:end,1], kwargs[:int_time]))