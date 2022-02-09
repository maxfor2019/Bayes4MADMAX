#=
    This file reads previously written data (and gives you the data path, given keywords).
    Use it to add methods for specific measured datasets.
=#

"""
    Construct path from input keywords.
"""
function data_path(DATASET, KEYWORD, TYPE)
    return DATA_PATH = "./data/"*TYPE*"/"*KEYWORD*"/"*DATASET*"/"
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
    Bring Olafs Dataset in the standard format.
    Changes global variables DATASET and KEYWORD! 
"""
function get_Olaf_2017(folder::String)
    global DATASET = "Olaf_2017"
    global KEYWORD = "measured"
    PATH = data_path(DATASET, KEYWORD, "raw_data")
    data = combine_vertically(readdir(PATH*folder), PATH*folder*"/")
    return DataFrame(data, [:freq, :pow])
end

"""
    Read data that is in standard format.
"""
function get_data(filename::String, DATASET::String, KEYWORD::String, TYPE::String)
    PATH = data_path(DATASET, KEYWORD, TYPE)
    full = readdlm(PATH*filename*".smp")
    return DataFrame(full[2:end,:], Symbol.(full[1,:]))
end


"""
    Metadata: Read out Experiment()
"""
function read_ex(DATASET::String, KEYWORD::String, TYPE::String)
    PATH = data_path(DATASET, KEYWORD, TYPE)
    META_PATH = check_meta(PATH)
    exdict = construct_dict(META_PATH, "ex")
    if length(exdict) != length(fieldnames(Experiment))
        @warn "The Experiment() in the metafile contains a different number of arguments than the current implementation. Will set some parameters to default!"
    end
    return Experiment(;exdict...)
end


"""
    Metadata: Read out Theory()
"""
function read_th(DATASET::String, KEYWORD::String, TYPE::String)
    PATH = data_path(DATASET, KEYWORD, TYPE)
    META_PATH = check_meta(PATH)
    try 
        thdict = construct_dict(META_PATH, "signal")
        if length(thdict) != length(fieldnames(Theory))
            @warn "The Theory() in the metafile contains a different number of arguments than the current implementation. Will set some parameters to default!"
        end    
        return Theory(;thdict...)
    catch
        error("Your metadata file seems to not contain a signal. Just wanted to let you know!")
    end
end

"""
    Metadata: Check if there is exactly one metafile in the folder and return its name.
"""
function check_meta(PATH::String)
    files = readdir(PATH)
    metafile = occursin.("meta", files)
    if sum(metafile) == 0
        error("The chosen directory ("*PATH*") does not contain a metadata file. Please specify one!")
    elseif sum(metafile) > 1
        error("The chosen directory ("*PATH*") contains more than one metadata file. I don't know which to choose. 
        If you want to set up multiple runs who differ in experimental or theoretical parameters, please put them in separate folders!")
    end
    META_PATH = PATH*files[metafile][1]
end

"""
    Metadata: String preparation to be able to convert values into Symbols or Floats.
"""
function construct_meta(META_PATH::String, key::String)
    strs = readdlm(META_PATH, '\t')
    metastr = strs[:,2][strs[:,1] .== key][1]
    metastr = split(metastr, "(")[2][1:end-1]
    metastr = String.(split(metastr, ","))
end

"""
    Metadata: Prepare dictionary to throw into Theory() or Experiment() structs.
"""
function construct_dict(META_PATH::String, key::String)
    sym = construct_meta(META_PATH, key*" Expl")
    sym = strip.(sym)
    sym = Symbol.([sym[i][2:end] for i in 1:length(sym)])
    str = construct_meta(META_PATH, key)
    args = parse.(Float64, str)
    return Dict(sym .=> args)
end

"""
"""
function get_samples(filename::String, DATASET::String, KEYWORD::String)
    PATH = _HAL9000(DATASET, KEYWORD, "samples")
    FileIO.load(PATH*filename*".jld2", "samples")
end

"""
"""
function get_prior(filename::String, DATASET::String, KEYWORD::String)
    PATH = _HAL9000(DATASET, KEYWORD, "samples")
    FileIO.load(PATH*filename*".jld2", "prior")
end