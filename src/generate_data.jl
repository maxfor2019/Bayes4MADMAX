#=
    This file generates and saves simulated datasets. It also adds axions if you want to.
=#

"""
    Construct an array of constant gaussian noise with a certain frequency range.
"""
function gaussian_noise(f_ini, f_fin, Δf; scale=1.0)
    f_arr = collect(range(f_ini, f_fin, step=Δf))
    noise_pow = rand(BAT.StandardMvNormal(length(f_arr))) * scale
    noise = hcat(f_arr, noise_pow)
    noise = DataFrame(noise, [:freq, :pow])
    insertcols!(noise, :noise => noise_pow)
    return noise
end

"""
    Add an arbitrary 3rd order polynomial background function with two sines to any dataset.
"""
function add_artificial_background!(data)
    #bg =  (deepcopy(data[!, :freq]).-7e6).^3 .* 3e-40 .* (1. +randn()/5) .- (deepcopy(data[!, :freq]).-7e6).^2 .* 1e-34 .* (1. +randn()/5) .- (deepcopy(data[!, :freq]).-7e6) .* 3e-27 .* (1. +randn()/5) .+ 1e-20  .+ 2e-24 .* (1. +randn()) .* sin.(deepcopy(data[!, :freq])./5e4) .+ 1e-23 .* (1. +randn()) .* sin.(deepcopy(data[!, :freq])./20e4) #.* 
    bg = 1e-20 .* (erf.((data[!,:freq].-data[1,:freq])/5e5) .* (data[1,:freq] ./data[!,:freq]).^3 .+ exp.(-((data[!,:freq].-6.5e6 .* (1. +randn()/15))./2e6 ./ (1. +randn()/10)).^2)) .+ 5e-23 .* (1. +randn()) .* sin.(deepcopy(data[!, :freq])./10e4) .+ 4e-22 .* (1. +randn()) .* sin.(deepcopy(data[!, :freq])./25e4)
    bg[rand(1:length(data[!,:freq]), 10)] .+= 3e-23
    insertcols!(data, :background => bg)
    data[!, :pow] .+= bg
    return data
end

"""
    Add an axion specified by signal to your dataset.
"""
function add_axion!(data, signal, ex)

    my_axion = let f = data[!, :freq], ex = ex
        function ax(parameters)
            sig = axion_forward_model(parameters, ex, f.+ex.f_ref)
            if maximum(sig) > 0.0
                nothing
            else
                error("The specified axion model is not within the frequency range of your data. Fiddle around with signal.ma or options.f_ref!")
            end
            return sig
        end

    end
    ax = my_axion(signal)
    if in("axion", names(data))
        @warn "Your data seem to already contain an axion. You are now throwing another one in!
        Problem is: You will only be able to save metadata for one axion. That means you're on your own now!"
        data[!, :pow] += ax
        insertcols!(data, :axion => ax, makeunique=true)
    else
        data[!, :pow] += ax
        insertcols!(data, :axion => ax)
    end
    return data
end

function Δω(data::DataFrame) 
    Δωvec = [data[i,1] - data[i-1,1] for i in 2:length(data[:,1])]
    Δωvec2 = vcat(Δωvec[2:end], Δωvec[1])
    if Δωvec ≈ Δωvec2
        nothing
    else
        @warn "The distances between datapoints in your dataset seem not to be equal. This will not change much, but you might wanna look into that!"
    end
    return mean(Δωvec)
end

"""
    Save a dataset, that may involve an axion signal. Self simulated or constructed from measured data.
"""
function save_data(data, ex::Experiment, th::Theory, filename::String, DATASET::String, KEYWORD::String, TYPE::String; overwrite=false)
    PATH = _HAL9000(DATASET, KEYWORD, TYPE)
    if in("axion", names(data)) == false
        error("Your data does not seem to contain a fake signal, therefore you should not throw in a Theory() type!")
    end
    meta_dict = OrderedDict(
        "ex Expl" => fieldnames(Experiment),
        "ex" => ex,
        "signal Expl" => fieldnames(Theory),
        "signal" => th
    )

    _save_data(data, meta_dict, filename, PATH; overwrite=overwrite)
end

function save_data(data, ex::Experiment, filename::String, DATASET::String, KEYWORD::String, TYPE::String; overwrite=false)
    PATH = _HAL9000(DATASET, KEYWORD, TYPE)
    if in("axion", names(data)) == true
        error("Your data seems to contain a fake signal, therefore you should throw in a Theory() type!")
    end
    meta_dict = OrderedDict(
        "ex Expl" => fieldnames(Experiment),
        "ex" => ex,
    )

    _save_data(data, meta_dict, filename, PATH; overwrite=overwrite)
end

function save_samples(out, prior::NamedTupleDist, filename::String, DATASET::String, KEYWORD::String; overwrite=false)
    PATH = _HAL9000(DATASET, KEYWORD, "samples")
    run = Dict(
        "prior" => prior,
        "samples" => out.result
    )
    # need specific format unfortunately because prior and samples have weird formats!
    if overwrite == false && isfile(PATH*filename*".jld2")
        error("The file you want to write already exists. If you want to overwrite change kwarg overwrite to true! Exiting.")
    else
        FileIO.save(PATH*filename*".jld2", run)
    end
end


function _HAL9000(DATASET::String, KEYWORD::String, TYPE::String)
    PATH = data_path(DATASET, KEYWORD, TYPE)
    if KEYWORD == "measured" && TYPE == "raw_data"
        error("I'm sorry Dave, I can't let you do that! You tried to write over measured raw data! PATH = $PATH")
    else
        println("Writing to / Reading from "*PATH)
    end
    return PATH
end

function _save_data(data, meta_dict::OrderedDict, filename::String, PATH::String; overwrite=false)
    writedlm(PATH*"meta-"*filename*".txt", meta_dict)
    if overwrite == true
        foo = h5open(PATH*filename*".h5", "w")
        close(foo)
    end
    for name in permutedims(names(data))
        h5write(PATH*filename*".h5", name, data[!, name])
    end
end

