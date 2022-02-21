#=
    All functions that manipulate raw data before throwing it into the main MCMC.
=#

"""
    Makes a Savitzky Golay polynomial fit and trims the dataset by len/2 at each side if you want to.
"""
function sg_fit(data, order, len; cut=true, overwrite=false)
    b=Int(round(len/2))
    e=length(data[!, :freq]) - Int(round(len/2))

    # For stability reasons normalize data before SG fit
    sc = mean(data[!, :pow])
    data[!, :pow] ./= sc

    sg = savitzky_golay(data[!, :pow], len, order)
    
    if cut == true
        data = data[b:e,:]
        ft = sg.y[b:e]
    else
        ft = sg.y
    end
    insertcols!(data, :pownoB => data[!, :pow] - ft; makeunique=overwrite)
    data[!, :pow] .*= sc
    data[!, :pownoB] .*= sc
    return data
end

