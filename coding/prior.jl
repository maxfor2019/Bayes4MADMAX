preinf = true

#=
struct MyDist <: ContinuousUnivariateDistribution end
Distributions.pdf(::MyDist, x) = x * exp(-x)
Distributions.rand(rng::AbstractRNG, d::MyDist) = d.pdf(random(rng))

rand(MyDist)

abstract type MyDist <: ContinuousUnivariateDistribution end

abstract type MyDist2 <: Sampleable{Univariate,Continuous} end

function rand(rng::AbstractRNG, s::MyDist2)
    return 
end

sample([2,1,3])
sampler(Poisson())
Normal(3,4) + Normal(2,1)

Distributions.sampler(::MyDist) -> Sampleable
sampler(s::Sampleable) -> s

d = MyDist(3,4)
Normal.VariateForm

Distributions.pdf(::MyDist, x) = pdf(Normal(d.mu, 2*d.sigma), x)
Base.rand(d::MyDist) = rand(Normal(d.mu, 2*d.sigma))
Base.iterate(d::MyDist, x) = iterate(pdf(Normal(d.mu, 2*d.sigma),x))
sampler(d::MyDist) = sampler(Normal(d.mu, 2*d.sigma))
convert(AbstractDensity, MyDist)

typeof(MyDist(3,4))

struct marray
    count::Int
end

Base.iterate(A::marray, state=1) = state > A.count ? nothing : (state*state, state+1)

for item in marray(5)
    println(item)
end

pdf(MyDist(), 3)
using StatsPlots
plot(x->pdf(MyDist(3,4), x), xlims=(-20,30))


rhoa = 0.0..1.0
if preinf == true
    θ_i = Normal(3,2)#-pi..pi
    prior = NamedTupleDist(
        logma = -9.0..(-5.0),
        rhoa = rhoa
    )
end =#



"""
    Calculates possible ma values for considered frequency range.
    This essentially is a flat prior. No possibility for anything else yet!
    Everything else comes in by some sort of folding with this top hat distribution.
    I have no idea how this could work yet.
"""
function logma_prior(data, kwargs)
    fstart = kwargs[:f_ref] + minimum(data[!,1])
    fend = kwargs[:f_ref] + maximum(data[!,1])
    mstart = log10(mass(fstart) )#/ sqrt(2)) # sqrt(2) is the maximum factor contributed by velocity
    mend = log10(mass(fend))                # velocity effect can only add, so upper bound stays intact
    return mstart..mend
end

"""
Priors include
    b: background parameters without physical meaning
    logma: log10(mass) changes position of the peak
    vsig: v distribution not really taken care of. Right now: uninformative priors on vsig, which translates
        to the sigma in f as needed for the signal function. Conversion done by getsigma(). Note that
        getsigma depends on logma! Also non zero mean velocities as for streams are not implemented!
        This would lead to a shift in frequency for the signal peak, indistinguishable from logma effects.
        Actually the shape of v distribution has influence on the signal shape. Right now, this is just
        a gaussian, which should be more or less accurate for realistic velocities. However, this will
        become completely false, when two velocity components are considered (e.g. background+stream).
    
"""


println(logma_prior(data, kwarg_dict))

#means = [16.711904185025094, 2.34436200254552, -0.2812509292477855, 0.1, 0.1]#, -0.005376850430048517, -0.0010847980505079456]
#means = [15.558959729080668, 5.3637970522877194e-5, -7.763696363349609e-11, -4.9519966478436284e-17]
means = [1.2000525312954732e-22, 3.248341200155389e-28, -7.963426216150934e-34, 2.363739346435701e-40]
prior = NamedTupleDist(
    b = [Normal(means[1], 5.0*abs(means[1])), Normal(means[2], 5.0*abs(means[2])), Normal(means[3], 5.0*abs(means[3])), Normal(means[4], 2.0*abs(means[4]))],
    logma = logma_prior(data, kwarg_dict),
    vsig = 5.0e-4..5.0e-3,
    Ps = 1e-22..1e-21
)

