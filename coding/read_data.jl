using DataFrames
using CSV
using Distributions

"""
    Reads in simulated dataset with more or less realistic background
    (and axion signal).
"""
function simulated_data(filename, path="./data/Fake_Axion_Data/Data Set 3/")

    #df = CSV.read(path*"Het3_10K_0-15z_18-85GHz_20170413_160107_S03.smp", delim="\t")
    df = CSV.read(path*filename, delim="\t")
    data = df[!,2:3]
    hplanck = 6.62606957e-34 # [Js]

    # Convert the whole thing to photon counts.
    # Assumes data[!,1] is in Hz and data[!,2] in J.
    #data[!,2] ./= (hplanck .* data[!,1])
    data[!,2] .*= 1e22
    # Make frequencies more workable
    data[!,1] .*= 1e-6
    return data
end

"""
    Sets up naive dummy data where an axion can easily be obtained
    with few nuisance parameters.

    Take σ ≥ μ, since going down to 0 is hardcoded!
    Don't place signal more than 1σ away from mean, or it will not be in range!
    Don't make Γ < σ/100 !
"""
function dummy_data(nuisance::NamedTuple{(:mu,:sigma)}, model::NamedTuple{(:x0, :Gamma)}, p_signal; p_noise=5000, kwargs=Dict())
    data = vcat(
    rand(Normal(nuisance.mu, nuisance.sigma), Int(p_noise)),
    rand(Normal( model.x0, model.Gamma), Int(p_signal))
    )
    hist = append!(Histogram(0.0:nuisance.sigma/100.0:nuisance.sigma+nuisance.mu), data)

    observed_counts = hist.weights

    # Histogram binning:
    bin_edges = hist.edges[1]
    bin_edges_left = bin_edges[1:end-1]
    bin_edges_right = bin_edges[2:end]
    bin_widths = bin_edges_right - bin_edges_left
    bin_centers = (bin_edges_right + bin_edges_left) / 2

    observed_power = Power(observed_counts, kwargs[:f_ref] .+ bin_centers, kwargs[:int_time])

    data = DataFrame([bin_centers, observed_power])
    return data
end