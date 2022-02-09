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
    return noise
end

"""
    Add an arbitrary 3rd order polynomial background function with two sines to any dataset.
"""
function add_artificial_background!(data)
    data[:,2] .+=  (deepcopy(data[:,1]).-7e6).^3 .* 1e-42 .* (1. +randn()) .- (deepcopy(data[:,1]).-7e6).^2 .* 1e-35 .* (1. +randn()) .- (deepcopy(data[:,1]).-7e6) .* 1e-29 .* (1. +randn()) .+ 1e-20  .+ 2e-24 .* (1. +randn()) .* sin.(deepcopy(data[:,1])./5e4) .+ 1e-23 .* (1. +randn()) .* sin.(deepcopy(data[:,1])./20e4)
    return data
end

"""
    Add an axion specified by signal to your dataset.
"""
function add_axion!(data, signal)

    ex = Experiment(Be=10.0, A=1.0, β=5e4, t_int=100.0, Δω=Δω(data), f_ref=11.0e9) # careful not to accidentally ignore a few of the relevant parameters!

    my_axion = let f = data[:,1], ex = ex
        f .+= ex.f_ref
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
    if in("pow", names(data))
        rename!(data,:pow => :powwA)
    elseif in("powwA", names(data))
        @warn "Your data seem to already contain an axion. You are now throwing another one in!"
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
function save_data(data, ex::Experiment, th::Theory, filename::String, DATASET::String, KEYWORD::String, TYPE::String)
    PATH = _HAL9000(DATASET, KEYWORD, TYPE)
    if occursin("wA", names(data)[2]) == false
        error("Your data does not seem to contain a fake signal, therefore you should not throw in a Theory() type!")
    end
    meta_dict = OrderedDict(
        "ex Expl" => fieldnames(Experiment),
        "ex" => ex,
        "signal Expl" => fieldnames(Theory),
        "signal" => th
    )

    _save_data_internal(data, meta_dict, filename, PATH)
end

function save_data(data, ex::Experiment, filename::String, DATASET::String, KEYWORD::String, TYPE::String)
    PATH = _HAL9000(DATASET, KEYWORD, TYPE)
    if occursin("wA", names(data)[2]) == true
        error("Your data seems to contain a fake signal, therefore you should throw in a Theory() type!")
    end
    meta_dict = OrderedDict(
        "ex Expl" => fieldnames(Experiment),
        "ex" => ex,
    )

    _save_data_internal(data, meta_dict, filename, PATH)
end

function save_samples(out, prior::NamedTupleDist, filename::String, DATASET::String, KEYWORD::String)
    PATH = _HAL9000(DATASET, KEYWORD, "samples")
    run = Dict(
        "prior" => prior,
        "samples" => out.result
    )

    FileIO.save(PATH*filename*".jld2", run)
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

function _save_data(data, meta_dict::OrderedDict, filename::String, PATH::String)
    writedlm(PATH*"meta-"*filename*".txt", meta_dict)
    writedlm(PATH*filename*".smp", permutedims(names(data)))
    open(PATH*filename*".smp", "a") do io
        writedlm(io, Matrix(data), "\t")
    end 
end





