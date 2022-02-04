#=
    All functions that manipulate raw data before throwing it into the main MCMC.
=#

"""
    Makes a Savitzky Golay polynomial fit and trims the dataset by len/2 at each side if you want to.
"""
function sg_fit(data, order, len; cut=true)
    b=Int(round(len/2))
    e=length(data[:,1]) - Int(round(len/2))

    # For stability reasons normalize data before SG fit
    sc = mean(data[:,2])
    data[:,2] ./= sc

    sg = savitzky_golay(data[:,2], len, order)
    
    if cut == true
        data = data[b:e,:]
        ft = sg.y[b:e]
    else
        ft = sg.y
    end
    data[:,2] = data[:,2] - ft
    data[:,2] .*= sc
    data = DataFrame(data)
    if in("pow", names(data))
        rename!(data,:pow => :pownoB)
    elseif in("powwA", names(data))
        rename!(data,:powwA => :powwAnoB)
    end
    return data
end

