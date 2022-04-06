#=
    All functions that manipulate raw data before throwing it into the main MCMC.
=#

export id_fit, sg_fit

"""
    Subtract (known) ideal background from data.
"""
function id_fit(data)
    data[!,:pownoB] = data[!,:pow] .- data[!,:background]
    return data
end

"""
    Makes a Savitzky Golay polynomial fit and trims the dataset by width/2 at each side if you want to.
"""
function sg_fit(data, order, width; cut=true, overwrite=false)
    b=Int(trunc(width/2))
    e=length(data[!, :freq]) - Int(trunc(width/2))

    # For stability reasons normalize data before SG fit
    sc = mean(data[!, :pow])

    sg = savitzky_golay(data[!, :pow] ./ sc, width, order)
    
    if cut == true
        data = data[b:e,:]
        ft = sg.y[b:e]
    else
        ft = sg.y
    end
    insertcols!(data, :pownoB => data[!, :pow] ./ sc - ft; makeunique=overwrite)
    data[!, :pownoB] .*= sc
    return data
end

"""
    Always take uncorrelated samples!
"""
function sg_fit_samples(data, order, wid, nsamps)
    b=Int(round(wid/2))+1
    e=length(data[!, :freq]) - Int(round(wid/2))
    sc = mean(data[!, :pow])
    #y = deepcopy(data[!,:pownoB]) ./ sc\
    data[!,:pow] ./= sc
    data[!,:pownoB] ./= sc
    wh = Int(trunc(wid/2))
    sgsamps = Array{Any,2}(missing, (size(data[!,:pownoB])[1],nsamps*2))
    #fill!(sgsamps, NaN)
    for i in wh+1:length(data[!,:pownoB])-wh
        sgf = SG_poly_samples.([randn(order+1) for k in 1:nsamps], Ref(data), wid, order, i)
        for j in 1:length(sgf)
            sgsamps[i,j] = sgf[j][wh+1,1]
            sgsamps[i,j+length(sgf)] = sgf[j][wh+1,2]
        end
    end
    sgsamps = sgsamps[b:e,:]
    #y = y[b:e]
    data = data[b:e,:]
    #insertcols!(data, :pownoB => (y .- vec(mean(sgsamps, dims=2))) .* sc; makeunique=true)
    data[!,:pownoB] = (data[!,:pow] .- vec(mean(sgsamps, dims=2))) .* sc
    data[!, :background] = vec(mean(sgsamps, dims=2)) .* sc
    for j in 1:size(sgsamps)[2]
        insertcols!(data, :pownoB => (data[!,:pow] .- sgsamps[:,j]) .* sc; makeunique=true)
        insertcols!(data, :background => sgsamps[:,j] .* sc; makeunique=true)
    end
    data[!,:pow] .*= sc
    return data
end

function my_vandermonde(w::T0, order::T0; coeff::T0=0, rate::T1=1.0) where {T0 <: Signed, T1 <: Real}
    isodd(w) || throw(ArgumentError("w must be an even number."))
    w ≥ 1 || throw(ArgumentError("w must greater than or equal to 1."))
    w ≥ order + 2 || throw(ArgumentError("w too small for the polynomial order chosen (w ≥ order + 2)."))
    p = SGolay(w, order, coeff, rate)

    hw = Int64((p.w - 1) / 2) # half-window size
    V = zeros(2*hw + 1, length(0 : p.order))
    @inbounds for i in -hw:hw, j in 0 : p.order
        V[i+hw+1, j+1] = i^j
    end
    return V, p
end

function get_coefficients(w::T0, order::T0;
    coeff::T0=0, rate::T1=1.0,
    ) where {T0 <: Signed, T1 <: Real}
    V, p = my_vandermonde(w, order; coeff=coeff, rate=rate)
    order_range = 0 : p.order
    Vqr = qr(V')
    length(order_range) >= p.deriv + 1 || throw(ArgumentError("length of vector must be greater than the position"))
    oh = zeros(length(order_range))
    oh[p.deriv + 1] = 1.0
    c = Vqr.R \ (Vqr.Q' * oh)
    c .*= (p.rate)^(p.deriv) * factorial(p.deriv)
    return c
end

function SG_poly(y, w, order, i)
    wh = Int(trunc(width/2))
    avec = [get_coefficients(w,order,coeff=i) ./ factorial(i) for i in 0:order]
    params = [sum(a .* y[i-wh:i+wh]) for a in avec]
    z = -Int(trunc(length(y[i-wh:i+wh])/2)):1:Int(trunc(length(y[i-wh:i+wh])/2))
    sum(k -> params[k] .* z.^(k-1), 1:length(params))
end


function SG_poly_samples(rnr, data, width, order, i)
    wh = Int(trunc(width/2))
    # The diagonal will destroy everything if there are still correlations in your noise!
    inv_noise_cov = inv(Diagonal(ones(2*wh+1).*var(data[i-wh:i+wh,:pownoB])))
    J = my_vandermonde(width, order)[1]
    D = inv(transpose(J)*inv_noise_cov*J)
    paramsp =   eigvecs(D)*sqrt.(eigvals(D)) .* rnr +  D*transpose(J)*inv_noise_cov*data[i-wh:i+wh,:pow]
    paramsm =  - eigvecs(D)*sqrt.(eigvals(D)) .* rnr +  D*transpose(J)*inv_noise_cov*data[i-wh:i+wh,:pow]
    z = -Int(trunc(length(data[i-wh:i+wh,:pownoB])/2)):1:Int(trunc(length(data[i-wh:i+wh,:pownoB])/2))
    pp = sum(k -> paramsp[k] .* z.^(k-1), 1:length(paramsp))
    pm = sum(k -> paramsm[k] .* z.^(k-1), 1:length(paramsm))
    return hcat(pp, pm)
end

function make_matrix(v::Vector)
    mat = permutedims(hcat([vcat(v[i:end],v[1:i-1]) for i in 1:length(v)]...))
    cov(mat)
end

