using DataFrames
using CSV
using Distributions

"""
"""
function combine_data(names_list; path="./data/Fake_Axion_Data/Data_Set_3/")
    files = [readdlm(path*name) for name in names_list]
    l = length(names_list)
    if all([files[1][:,1] == files[i][:,1] for i in 1:l])
        data = zeros(size(files[1]))
        data[:,1] = deepcopy(files[1][:,1])
        for i in 1:l
            data[:,2] += files[i][:,2]
        end
        data[:,2] ./= l
    else
        error("Files seem to use different frequencies. Interpolation would be in order.")
    end
    return data
end

"""
    Reads in simulated dataset with more or less realistic background
    (and axion signal).
"""
function simulated_data(filename, path="./data/Fake_Axion_Data/Data Set 3/")

    #df = CSV.read(path*"Het3_10K_0-15z_18-85GHz_20170413_160107_S03.smp", delim="\t")
    df = DataFrame(CSV.File(path*filename, delim="\t"))
    data = [df[!,2], df[!,3]]
    hplanck = 6.62606957e-34 # [Js]

    # Convert the whole thing to photon counts.
    # Assumes data[1] is in Hz and data[2] in J.
    #data[!,2] ./= (hplanck .* data[!,1])
    data[2] .*= 1e22
    # Make frequencies more workable
    #data[1] .*= 1e-6
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

    observed_power = 1e23 .* Power(observed_counts, kwargs[:f_ref] .+ bin_centers, kwargs[:int_time])

    data = DataFrame([bin_centers, observed_power])
    return data
end

"""
"""
function dummy_data_right_signal(nuisance::NamedTuple{(:mu,:sigma)}, model::NamedTuple{(:ma, :rhoa, :σ_v)}, ex::Experiment; p_noise=5000, kwargs=Dict())
    # Draw from signal
    C_signal = signal_counts(model.ma.*1e-6, model.rhoa, ex)
    N = rand(MaxwellBoltzmann(model.σ_v), convert(Int64, round(C_signal)))
    data = freq(model.ma*1e-6, v=N) .- kwargs[:f_ref]

    # Draw from background
    append!(data, rand(Normal(nuisance.mu,nuisance.sigma), convert(Int64, round(p_noise))))

    # Make a histogram out of it!
    hist = append!(Histogram(0.0:ex.Δω:nuisance.sigma+nuisance.mu), data)

    observed_counts = hist.weights

    # Histogram binning:
    bin_edges = hist.edges[1]
    bin_edges_left = bin_edges[1:end-1]
    bin_edges_right = bin_edges[2:end]
    bin_widths = bin_edges_right - bin_edges_left
    bin_centers = (bin_edges_right + bin_edges_left) / 2

    #observed_power = 1e23 .* Power(observed_counts, kwargs[:f_ref] .+ bin_centers, kwargs[:int_time])

    data = (bin_centers, observed_counts)
    return data

end

"""
    Construct an array of constant gaussian noise with a certain frequency range.
"""
function gaussian_noise(f_ini, f_fin, Δf; scale=1.0)
    f_arr = collect(range(f_ini, f_fin, step=Δf))
    noise_pow = rand(BAT.StandardMvNormal(length(f_arr))) * scale
    noise = hcat(f_arr, noise_pow)
    return noise
end