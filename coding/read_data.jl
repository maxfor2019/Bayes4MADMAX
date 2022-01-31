"""
    Vertically (aka summing bins of same frequency) adding datasets together.
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
    Construct an array of constant gaussian noise with a certain frequency range.
"""
function gaussian_noise(f_ini, f_fin, Δf; scale=1.0)
    f_arr = collect(range(f_ini, f_fin, step=Δf))
    noise_pow = rand(BAT.StandardMvNormal(length(f_arr))) * scale
    noise = hcat(f_arr, noise_pow)
    return noise
end