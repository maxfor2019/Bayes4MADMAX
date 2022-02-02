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
end

"""
    Read data that is in standard format.
"""
function get_data(filename::String, DATASET::String, KEYWORD::String, TYPE::String)
    PATH = data_path(DATASET, KEYWORD, TYPE)
    return readdlm(PATH*filename*".smp")
end
