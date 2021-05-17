using BAT # I'll probs need that at some point!

using Random, LinearAlgebra, Statistics, Distributions, StatsBase
using Plots

data = vcat(
    rand(Normal(-1.0, 0.5), 500),
    rand(Normal( 2.0, 0.5), 1000)
)

hist = append!(Histogram(-2:0.1:4), data)

plot(
    normalize(hist, mode=:density),
    st = :steps, label = "Data",
    title = "Data"
)
#savefig("./data/tmp/tutorial-data.pdf")

function fit_function(p::NamedTuple{(:a, :mu, :sigma)}, x::Real)
    p.a[1] * pdf(Normal(p.mu[1], p.sigma), x) +
    p.a[2] * pdf(Normal(p.mu[2], p.sigma), x)
end

true_par_values = (a = [500, 1000], mu = (-1.0, 2.0), sigma = 0.5)
wrong_par_values = (a = [600, 700], mu = (-1.5, 1.0), sigma = 0.9)

plot(
    normalize(hist, mode=:density),
    st = :steps, label = "Data",
    title = "Data and True Statistical Model"
)
plot!(
    -4:0.01:4, x -> fit_function(true_par_values, x),
    label = "Truth"
)
#savefig("./data/tmp/tutorial-data-and-truth.pdf")

using IntervalSets

# let syntax essentially lets you define specific variables that may change for a new execution of that code snippet.
# faster and safer than global variables.
likelihood = let h = hist, f = fit_function
    # Histogram counts for each bin as an array:
    observed_counts = h.weights

    # Histogram binning:
    bin_edges = h.edges[1]
    bin_edges_left = bin_edges[1:end-1]
    bin_edges_right = bin_edges[2:end]
    bin_widths = bin_edges_right - bin_edges_left
    bin_centers = (bin_edges_right + bin_edges_left) / 2

    # Anonymous function depending on paramters params. Equivalent to function (params) ... end
    params -> begin
        # Log-likelihood for a single bin:
        function bin_log_likelihood(i)
            # Simple mid-point rule integration of fit function `f` over bin:
            expected_counts = bin_widths[i] * f(params, bin_centers[i])
            logpdf(Poisson(expected_counts), observed_counts[i])
        end

        # Sum log-likelihood over bins:
        idxs = eachindex(observed_counts)
        ll_value = bin_log_likelihood(idxs[1])
        for i in idxs[2:end]
            ll_value += bin_log_likelihood(i)
        end

        # Wrap `ll_value` in `LogDVal` so BAT knows it's a log density-value.
        return LogDVal(ll_value)
    end
end

using BenchmarkTools
@btime likelihood(wrong_par_values)

using ValueShapes

# Weibull distribution can be similar to 1/x, exp(x), gaussian, etc. so optimal if you have no idea about prior
# this choice does seem weird though. λ∼1.0 k>1.0 gives poisson like 
# Note that all elements are distributions you need to draw from using e.g. rand(Dist(), #samples)!
prior = NamedTupleDist(
    a = [Weibull(1.1, 5000), Weibull(1.1, 5000)],
    # -1..4 notation defines an interval. e.g. -1 in -2..0 returns true
    mu = [-2.0..0.0, 1.0..3.0],
    sigma = Weibull(1.2, 2)
)
#=
using StatsPlots
plot(Weibull(1.1,4))

println(prior)
NamedTupleDist{(:a, :mu, :sigma),
    Tuple{
        Product{Continuous,Weibull{Float64},Array{Weibull{Float64},1}}, # Every data type a can possibly be
        Product{Continuous,Uniform{Float64},Array{Uniform{Float64},1}}, # Every data type mu can possibly be
        Weibull{Float64}                                                # Every data type sigma can possibly be
        },
    Tuple{ValueAccessor{ArrayShape{Real,1}},ValueAccessor{ArrayShape{Real,1}},ValueAccessor{ScalarShape{Real}}} # Some additional description of data types
    }(
    _internal_distributions: (
        a = Product{Continuous,Weibull{Float64},Array{Weibull{Float64},1}}(
            v=Weibull{Float64}[Weibull{Float64}(α=1.1, θ=5000.0), Weibull{Float64}(α=1.1, θ=5000.0)]), # Actual prior values for a. Akward structure because array.
        mu = Product{Continuous,Uniform{Float64},Array{Uniform{Float64},1}}(
            v=Uniform{Float64}[Uniform{Float64}(a=-2.0, b=0.0), Uniform{Float64}(a=1.0, b=3.0)]), 
        sigma = Weibull{Float64}(α=1.2, θ=2.0)
        )
    _internal_shape: NamedTupleShape{(:a, :mu, :sigma),
        Tuple{ValueAccessor{ArrayShape{Real,1}},ValueAccessor{ArrayShape{Real,1}},ValueAccessor{ScalarShape{Real}}}
        }(
            (
                a = ValueAccessor{ArrayShape{Real,1}}(ArrayShape{Real,1}((2,)), 0, 2),  # First number is probably length of array
                mu = ValueAccessor{ArrayShape{Real,1}}(ArrayShape{Real,1}((2,)), 2, 2), # Second number ??
                sigma = ValueAccessor{ScalarShape{Real}}(ScalarShape{Real}(), 4, 1)     # Third number number of elements drawn??
            ),
        5) # sum over all elements?
)
=#


parshapes = varshape(prior)

posterior = PosteriorDensity(likelihood, prior)

#=
PosteriorDensity{
    BAT.GenericDensity{
        var"#28#29"{
            Array{Int64,1},
            StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},
            StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},
            typeof(fit_function)}
        },
    BAT.DistributionDensity{
        NamedTupleDist{(:a, :mu, :sigma),
            Tuple{
                Product{Continuous,Weibull{Float64},Array{Weibull{Float64},1}},
                Product{Continuous,Uniform{Float64},Array{Uniform{Float64},1}},
                Weibull{Float64}},
            Tuple{
                ValueAccessor{ArrayShape{Real,1}},
                ValueAccessor{ArrayShape{Real,1}},
                ValueAccessor{ScalarShape{Real}}}},
        BAT.HyperRectBounds{Float64}},
    BAT.HyperRectBounds{Float64},
    NamedTupleShape{(:a, :mu, :sigma),
        Tuple{
            ValueAccessor{ArrayShape{Real,1}},
            ValueAccessor{ArrayShape{Real,1}},
            ValueAccessor{ScalarShape{Real}}}}
}(
    BAT.GenericDensity{
        var"#28#29"{ # Specifies anonymous likelihood function
            Array{Int64,1},
            StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},
            StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},
            typeof(fit_function)}
    }(var"#28#29"{
        Array{Int64,1},
        StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},
        StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},
        typeof(fit_function)
        }( # Actual stuff that gets thrown into likelihood function.
        [8, 9, 16, 11, 17, 32, 37, 39, 43, 36  …  7, 8, 3, 1, 2, 1, 0, 0, 0, 0], # Counts for the histogram bins
        0.1:0.0:0.1,    # No idea?!
        -1.95:0.1:3.95, # Range of the histogram
        fit_function)
    ), 
    BAT.DistributionDensity{
        NamedTupleDist{(:a, :mu, :sigma),
            Tuple{
                Product{Continuous,Weibull{Float64},Array{Weibull{Float64},1}},
                Product{Continuous,Uniform{Float64},Array{Uniform{Float64},1}},
                Weibull{Float64}},
            Tuple{
                ValueAccessor{ArrayShape{Real,1}},
                ValueAccessor{ArrayShape{Real,1}},
                ValueAccessor{ScalarShape{Real}}}},
        BAT.HyperRectBounds{Float64}
    }(
        NamedTupleDist{(:a, :mu, :sigma),
            Tuple{
                Product{Continuous,Weibull{Float64},Array{Weibull{Float64},1}},
                Product{Continuous,Uniform{Float64},Array{Uniform{Float64},1}},
                Weibull{Float64}},
            Tuple{
                ValueAccessor{ArrayShape{Real,1}},
                ValueAccessor{ArrayShape{Real,1}},
                ValueAccessor{ScalarShape{Real}}}
        }(
            _internal_distributions: 
                (a = Product{Continuous,Weibull{Float64},Array{Weibull{Float64},1}}(
                    v=Weibull{Float64}[Weibull{Float64}(α=1.1, θ=5000.0), Weibull{Float64}(α=1.1, θ=5000.0)]), 
                mu = Product{Continuous,Uniform{Float64},Array{Uniform{Float64},1}}(
                    v=Uniform{Float64}[Uniform{Float64}(a=-2.0, b=0.0), Uniform{Float64}(a=1.0, b=3.0)]), 
                sigma = Weibull{Float64}(α=1.2, θ=2.0))
            _internal_shape: 
                NamedTupleShape{(:a, :mu, :sigma),
                    Tuple{
                        ValueAccessor{ArrayShape{Real,1}},
                        ValueAccessor{ArrayShape{Real,1}},
                        ValueAccessor{ScalarShape{Real}}}
                }((
                    a = ValueAccessor{ArrayShape{Real,1}}(ArrayShape{Real,1}((2,)), 0, 2), 
                    mu = ValueAccessor{ArrayShape{Real,1}}(ArrayShape{Real,1}((2,)), 2, 2), 
                    sigma = ValueAccessor{ScalarShape{Real}}(ScalarShape{Real}(), 4, 1)), 
                5)
        ), 
        BAT.HyperRectBounds{Float64}(
            BAT.HyperRectVolume{Float64}(
                [0.0, 0.0, -2.0, 1.0, 0.0], # These are the fixed lower limits for all the parameters
                [Inf, Inf, 0.0, 3.0, Inf])) # These are upper limits
    ), 
    BAT.HyperRectBounds{Float64}(
        BAT.HyperRectVolume{Float64}(
            [0.0, 0.0, -2.0, 1.0, 0.0], 
            [Inf, Inf, 0.0, 3.0, Inf])
    ), 
    NamedTupleShape{(:a, :mu, :sigma),
        Tuple{
            ValueAccessor{ArrayShape{Real,1}},
            ValueAccessor{ArrayShape{Real,1}},
            ValueAccessor{ScalarShape{Real}}}
    }((
        a = ValueAccessor{ArrayShape{Real,1}}(ArrayShape{Real,1}((2,)), 0, 2), 
        mu = ValueAccessor{ArrayShape{Real,1}}(ArrayShape{Real,1}((2,)), 2, 2), 
        sigma = ValueAccessor{ScalarShape{Real}}(ScalarShape{Real}(), 4, 1)), 
    5)
)
=#



samples = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10^5, nchains = 4)).result

using FileIO, JLD2
FileIO.save("./data/210507-testsamples2.jld2", Dict("samples" => samples))
samples = FileIO.load("./data/210507-testsamples2.jld2", "samples")

# HDF5 approach does not seem to work!!
#import HDF5
#bat_write("./data/210507-testsamples2.h5", samples)
#samples = bat_read("./data/210507-testsamples2.h5").result

SampledDensity(posterior, samples)

println("Truth: $true_par_values")
println("Mode: $(mode(samples))")
println("Mean: $(mean(samples))")
println("Stddev: $(std(samples))")

# v is essentially the part, which includes all the data!
unshaped.(samples.v)

parshapes = varshape(posterior)

par_cov = cov(unshaped.(samples))
println("Covariance: $par_cov")

par_cov[parshapes.mu, parshapes.sigma]

plot(
    samples, :(mu[1]),
    mean = true, std = true, globalmode = true, marginalmode = true,
    nbins = 50, title = "Marginalized Distribution for mu[1]"
)
#savefig("./data/tmp/tutorial-single-par.pdf")

plot(
    samples, (:(mu[1]), :sigma),
    mean = true, std = true, globalmode = true, marginalmode = true,
    nbins = 50, title = "Marginalized Distribution for mu[1] and sigma"
)
plot!(BAT.MCMCBasicStats(samples), (3, 5))
#savefig("./data/tmp/tutorial-param-pair.png")

plot(
    samples,
    mean = false, std = false, globalmode = true, marginalmode = false,
    nbins = 50
)
#savefig("./data/tmp/tutorial-all-params.png")

diag_dict = Dict(
  "xlabel" => "lel", # This does not work!
  "ylims" => (0.0, 0.00005)
)

# Need to play with the plotting a little bit more. Make it look like actual corner plot!
plot(
    samples, bins=50,
)

#plot(rand(Normal(3,1), 500), xshowaxis=true)