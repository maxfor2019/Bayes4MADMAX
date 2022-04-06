#=
    This file generates and saves simulated datasets. It also adds axions if you want to.
=#

export initialize_dataset, gaussian_noise, add_artificial_background!, add_axion!, Δω, save_data, save_samples, read_struct, read_bgfit, read_samples, read_prior, read_data


"""
    Set up data structure for a new dataset including correct permissions.
"""
function initialize_dataset(dataset::String)
    if occursin("simulated", dataset) == false && occursin("measured", dataset) == false
        error("Specify dataset either via 'simulated/'*NAME or 'measured/'*NAME")
    end
    PATH = joinpath(_parent_path("data"), dataset)
    mkdir(PATH)
    mkdir(joinpath(PATH, "raw_data"))
    mkdir(joinpath(PATH, "processed_data"))
    mkdir(joinpath(PATH, "samples"))
    chmod(PATH, 0o770, recursive=true)
end

"""
Vertically (aka summing bins of same frequency) adding datasets together.
"""
function combine_vertically(names_list, PATH)
    files = [readdlm(PATH*name) for name in names_list]
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
    bg = 1e-20 .* (erf.((data[!,:freq].-data[1,:freq])/5e5) .* (data[1,:freq] ./data[!,:freq]).^3 .+ exp.(-((data[!,:freq].-6.5e6 .* (1. +randn()/15))./2e6 ./ (1. +randn()/10)).^2)) .+ 5e-23 .* (1. +randn()) .* sin.((deepcopy(data[!, :freq]) .+ randn() * data[1, :freq])./10e4) .+ 4e-22 .* (1. +randn()) .* sin.((deepcopy(data[!, :freq]) .+ randn() * data[1, :freq])./25e4)
    #bg[rand(1:length(data[!,:freq]), 10)] .+= 3e-23
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
function save_data(data, ex::Experiment, sig::Theory, fname::String, dataset::String; overwrite=false, newfolder="", BGfit_dict=nothing)
    if in("axion", names(data)) == false
        error("Your data does not seem to contain a fake signal, therefore you should not throw in a Theory() type!")
    end

    PATH = _check_path(data, fname, dataset; newfolder=newfolder)

    _save_powerspectrum(data, fname, PATH; overwrite=overwrite)
    _save_struct(ex, fname, PATH)
    _save_struct(sig, fname, PATH)
    _save_bgfit(data, BGfit_dict, fname, PATH)
end

function save_data(data, ex::Experiment, fname::String, dataset::String; overwrite=false, newfolder="", BGfit_dict=nothing)
    if in("axion", names(data)) == true
        error("Your data seems to contain a fake signal, therefore you should throw in a Theory() type!")
    end

    PATH = _check_path(data, fname, dataset; newfolder=newfolder)

    _save_powerspectrum(data, fname, PATH; overwrite=overwrite)
    _save_struct(ex, fname, PATH)
    _save_bgfit(data, BGfit_dict, fname, PATH)
end

function _save_powerspectrum(data, fname::String, PATH::String; overwrite=false)
    if overwrite == true
        foo = h5open(_file_path(PATH, fname), "w")
        close(foo)
    end

    for name in permutedims(names(data))
        h5write(_file_path(PATH, fname), "powerspectrum/"*name, data[!, name])
    end
    chmod(PATH, 0o770, recursive=true)
end

function _check_path(data, fname::String, dataset::String; newfolder="")
    if in("pownoB", names(data)) == true
        PATH=joinpath(_parent_path("data"), dataset, "processed_data", newfolder)
    else
        PATH=PATH=joinpath(_parent_path("data"), dataset, "raw_data", newfolder)
        if occursin("measured", dataset) == true
            error("You almost wrote over measured raw data @ PATH = $PATH. I require you to have the key 'pownoB' in your dataset when being done with data processing.")
        end
    end

    if isdir(PATH) == false
        mkdir(PATH)
    else
        @warn "The folder you want to write to already exists ($PATH). If it contains data with name $fname, the data may have been written over, if you are using the overwrite=true option."
    end
    
    println("Writing to $(joinpath(PATH, fname*".h5"))")
    return PATH
end

_dict_to_named_tuple(dic) = (; (Symbol(k) => v for (k,v) in dic)...)
_struct_to_named_tuple(p) = (; (v=>getfield(p, v) for v in fieldnames(typeof(p)))...)

function _save_struct(meta::Union{Theory, Experiment}, fname::String, PATH::String)
    tup = _struct_to_named_tuple(meta)
    for i in 1:length(tup)
        key, val = hcat(collect(keys(tup)), collect(tup))[i,:]
        h5write(_file_path(PATH, fname), "metadata/"*String(Symbol(typeof(meta)))*"/"*String(key), val)
    end
end

function _save_bgfit(data, BGfit_dict, fname::String, PATH::String)
    if in("pownoB", names(data)) == true && occursin("Dict", String(Symbol(typeof(BGfit_dict)))) == false
        error("Your data contains 'pownoB', which means you have done a background fit. Please specify how you did it by using the kwarg 'BGfit_dict'!")
    elseif in("pownoB", names(data)) == true && occursin("Dict", String(Symbol(typeof(BGfit_dict)))) == true
        tup = _dict_to_named_tuple(BGfit_dict)
        for i in 1:length(tup)
            key, val = hcat(collect(keys(tup)), collect(tup))[i,:]
            h5write(_file_path(PATH, fname), "metadata/BGfit/"*String(key), val)
        end
    else
        nothing
    end
end    


function save_samples(out, prior::NamedTupleDist, BGfit, fname::String, dataset::String; overwrite=false, newfolder="")
    PATH=joinpath(_parent_path("data"), dataset, "samples", newfolder)
    run = Dict(
        "prior" => prior,
        "samples" => out.result,
        "BGfit" => BGfit
    )
    # need specific format unfortunately because prior and samples have weird formats!
    if overwrite == false && isfile(_file_path(PATH,fname, type=".jld2"))
        error("The file you want to write already exists. If you want to overwrite change kwarg overwrite to true! Exiting.")
    else
        FileIO.save(_file_path(PATH, fname; type=".jld2"), run)
    end
    chmod(PATH, 0o770, recursive=true)
end

"""
    Read metadata from raw_data file. If you changed Theory or Experiment for the processed data, God have mercy on your soul!
"""
function read_struct(key::Union{DataType, String}, fname::String, dataset::String; type::String="raw_data")
    PATH=joinpath(_parent_path("data"), dataset, type)
    fid = h5open(_file_path(PATH, fname), "r")
    mdnt = _dict_to_named_tuple(read(fid["metadata"][String(Symbol(key))]))
    if typeof(key) != DataType
        key = eval(Meta.parse(key))
    end
    md = key(; mdnt...)
    close(fid)
    return md
end

"""
    Read metadata of background fit.
"""
function read_bgfit(fname::String, dataset::String, type::String)
    PATH=joinpath(_parent_path("data"), dataset, type)
    fid = h5open(_file_path(PATH, fname), "r")
    bgdict = read(fid["metadata"]["BGfit"])
    close(fid)
    return bgdict
end

"""
    Read powerspectrum data that is in standard format.
"""
function read_data(fname::String, dataset::String, type::String)
    if type != "raw_data" && occursin("processed_data", type) == false
        error("You need to use type='raw_data' or type='processed_data'!")
    end
    PATH=joinpath(_parent_path("data"), dataset, type)
    println("Reading from $PATH")

    foo = h5open(_file_path(PATH, fname), "r")
    tt = [size(obj) for obj in foo["powerspectrum"]]
    if all(y->y==tt[1],tt) == false
        error("Your 'powerspectrum' data seems to contain columns which have a different size() than others. This will lead to a messed up dataframe. Maybe you stored something here that belongs into 'metadata'?")
    end
    df = DataFrame(read(foo["powerspectrum"]))
    close(foo)
    return df
end

"""
"""
function read_samples(fname::String, dataset::String; folder="")
    PATH=joinpath(_parent_path("data"), dataset, "samples", folder)
    FileIO.load(_file_path(PATH, fname; type=".jld2"), "samples")
end

"""
"""
function read_prior(fname::String, dataset::String; folder="")
    PATH=joinpath(_parent_path("data"), dataset, "samples", folder)
    FileIO.load(_file_path(PATH, fname; type=".jld2"), "prior")
end

function _parent_path(parent::String)
    root = dirname(@__FILE__)
    root = normpath(root, "..")
    dpath = joinpath(root, parent)
end

function _file_path(PATH::String, fname::String; type=".h5")
    return joinpath(PATH, fname*type)
end