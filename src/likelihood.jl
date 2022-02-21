
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

function fit_function(th::Theory, freq, ex::Experiment)
    return Power.(signal_counts_bin(freq .+ ex.f_ref, th, ex), freq .+ ex.f_ref, ex.t_int)
end

function fit_function(p::NamedTuple, freq, ex::Experiment)
    th = prepare_th(p)
    return Power.(signal_counts_bin(freq .+ ex.f_ref, th, ex), freq .+ ex.f_ref, ex.t_int)
end

function prepare_th(p::NamedTuple{(:ma, :sig_v, :rhoa,)})
    return Theory(ma=p.ma,rhoa=p.rhoa,EoverN=0.924,σ_v=p.sig_v)
end

function prepare_th(p::NamedTuple{(:ma, :sig_v, :gag,)})
    EoN = EoverN(fa(scale_ma(p.ma)),p.gag)
    return Theory(ma=p.ma,rhoa=0.3,EoverN=EoN,σ_v=p.sig_v)
end

function prepare_th(p::NamedTuple{(:ma, :sig_v, :log_gag,)})
    gag=10.0^p.log_gag
    EoN = EoverN(fa(scale_ma(p.ma)),gag)
    return Theory(ma=p.ma,rhoa=0.3,EoverN=EoN,σ_v=p.sig_v)
end



likelihood = let freqs=collect(data[!, :freq]), observation=data[!, :pownoB], f=fit_function, ex=ex
    function logl(pars) 
        #println(pars)
        expectation = f(pars, freqs, ex) # 80 μs
        # Cant work with values below 0. Should only happen far away from likelihood maximum anyways
        #expectation[expectation .<= 0] .= eps(eltype(expectation))
        # Power.(sqrt.(Counts.(expectation, kwargs[:f_ref].+data[1:end,1], kwargs[:int_time])), kwargs[:f_ref].+data[1:end,1], kwargs[:int_time])
        result = logpdf.(Normal.(expectation, std(observation)), observation) # 20 μs
        return LogDVal(sum(result))
    end
end