
export fit_function, make_like

#function sig(freq, ex, options; rhoa=rhoa, ma=ma, vsig=vsig) # 80 μs
#    th = Theory(ma=ma,rhoa=rhoa,EoverN=0.924,σ_v=vsig)
#    return Power.(signal_counts_bin(freq .+ options.f_ref, th, ex), freq .+ options.f_ref, ex.t_int)
#end

#function sig(freq, ex, options; gag=gag, ma=ma, vsig=vsig) # 80 μs
#    EoN = EoverN(fa(scale_ma(ma)),gag)
#    th = Theory(ma=ma,rhoa=0.3,EoverN=EoN,σ_v=vsig)
#    return Power.(signal_counts_bin(freq .+ options.f_ref, th, ex), freq .+ options.f_ref, ex.t_int)
#end




#fit_function(p::NamedTuple{(:ma, :sig_v, :rhoa,)}, freq, ex::Experiment, options) = sig(freq, ex, options; rhoa=p.rhoa, ma=p.ma, vsig=p.sig_v) #.+ background(p.b, freq)
#fit_function(p::NamedTuple{(:ma, :sig_v, :log_gag,)}, freq, ex::Experiment) = sig(freq, ex, options; gag=10.0^p.log_gag, ma=p.ma, vsig=p.sig_v) #.+ background(p.b, freq)
#fit_function(p::NamedTuple{(:ma, :sig_v, :gag,)}, freq, ex::Experiment, options) = sig(freq, ex, options; gag=p.gag, ma=p.ma, vsig=p.sig_v) #.+ background(p.b, freq)

"""
    Compute the signal expectation. If BGfit is defined takes effect of BG fit on signal model into account.
"""
function fit_function(th::Theory, freq, ex::Experiment; BGfit=nothing)
    ff = Power.(signal_counts_bin(freq .+ ex.f_ref, th, ex), freq .+ ex.f_ref, ex.t_int)

    if BGfit === nothing || BGfit["type"] == "ID"
        nothing
    elseif BGfit["type"] == "SG"
        nff = savitzky_golay(ff ./ maximum(ff), BGfit["width"], BGfit["order"])
        ff .-= nff.y .* maximum(ff)
    else
        error("You used an unexpected argument for kwarg BGfit! Use dict with a 'type' parameter called 'ID' or 'SG'!")
    end
    return ff
end

function fit_function(p::NamedTuple, freq, ex::Experiment; BGfit=nothing)
    th = _prepare_th(p)
    fit_function(th, freq, ex; BGfit=BGfit)
end

function _prepare_th(p::NamedTuple{(:ma, :sig_v, :rhoa,)})
    return Theory(ma=p.ma,rhoa=p.rhoa,EoverN=0.924,σ_v=p.sig_v)
end

function _prepare_th(p::NamedTuple{(:ma, :sig_v, :gag,)})
    EoN = EoverN(fa(scale_ma(p.ma)),p.gag)
    return Theory(ma=p.ma,rhoa=0.3,EoverN=EoN,σ_v=p.sig_v)
end

function _prepare_th(p::NamedTuple{(:ma, :sig_v, :log_gag,)})
    gag=10.0^p.log_gag
    EoN = EoverN(fa(scale_ma(p.ma)),gag)
    return Theory(ma=p.ma,rhoa=0.3,EoverN=EoN,σ_v=p.sig_v)
end

function make_like(data, fit_function, ex, bg)
    likelihood = let freqs=collect(data[!, :freq]), observation=data[!, :pownoB], f=fit_function, ex=ex, BGfit=bg
        function logl(pars) 
            expectation = f(pars, freqs, ex, BGfit=bg)
            result = logpdf.(Normal.(expectation, std(observation)), observation)
            return LogDVal(sum(result))
        end
    end
    return likelihood
end


#=
using LogExpFunctions

samp_likelihood = let freqs=collect(data[!, :freq]), observation=Matrix(data[!, r"pownoB_"]), f=fit_function, ex=ex
    function logl(pars) 
        #println(pars)
        expectation = f(pars, freqs, ex) # 80 μs
        # Cant work with values below 0. Should only happen far away from likelihood maximum anyways
        #expectation[expectation .<= 0] .= eps(eltype(expectation))
        # Power.(sqrt.(Counts.(expectation, kwargs[:f_ref].+data[1:end,1], kwargs[:int_time])), kwargs[:f_ref].+data[1:end,1], kwargs[:int_time])
        result = logpdf.(Normal.(expectation, std(observation, dims=1)), observation) # 20 μs
        return LogDVal(sum(logsumexp(result, dims=2)))
    end
end
=#