# Implement very rudimentary fit to have something to optimize.
# Datapoints < 1 are not good for Normal distribution!
# Thus rescale. Doesnt matter for the endresult.

background(a, freq) = a[1] .+ a[2] .* freq .+ a[3] .* freq.^2. + a[4] .* freq.^3. # .+ a[5] .* freq.^4. # 3 μs

#function sig(freq, ex, options; rhoa=rhoa, ma=ma, vsig=vsig) # 80 μs
#    th = Theory(ma=ma,rhoa=rhoa,EoverN=0.924,σ_v=vsig)
#    return Power.(signal_counts_bin(freq .+ options.f_ref, th, ex), freq .+ options.f_ref, ex.t_int)
#end

function sig(freq, ex, options; gag=gag, ma=ma, vsig=vsig) # 80 μs
    EoN = EoverN(fa(scale_ma(ma)),gag)
    th = Theory(ma=ma,rhoa=0.3,EoverN=EoN,σ_v=vsig)
    return Power.(signal_counts_bin(freq .+ options.f_ref, th, ex), freq .+ options.f_ref, ex.t_int)
end

#fit_function(p::NamedTuple{(:ma, :sig_v, :rhoa,)}, freq, ex::Experiment, options) = sig(freq, ex, options; rhoa=p.rhoa, ma=p.ma, vsig=p.sig_v) #.+ background(p.b, freq)
#fit_function(p::NamedTuple{(:ma, :sig_v, :log_gag,)}, freq, ex::Experiment, options) = sig(freq, ex, options; gag=10.0^p.log_gag, ma=p.ma, vsig=p.sig_v) #.+ background(p.b, freq)
fit_function(p::NamedTuple{(:ma, :sig_v, :gag,)}, freq, ex::Experiment, options) = sig(freq, ex, options; gag=p.gag, ma=p.ma, vsig=p.sig_v) #.+ background(p.b, freq)

likelihood = let freqs=collect(data[:, 1]), observation=data[:, 2], f=fit_function, options=options, ex=ex
    function logl(pars) 
        #println(pars)
        expectation = f(pars, freqs, ex, options) # 80 μs
        # Cant work with values below 0. Should only happen far away from likelihood maximum anyways
        #expectation[expectation .<= 0] .= eps(eltype(expectation))
        # Power.(sqrt.(Counts.(expectation, kwargs[:f_ref].+data[1:end,1], kwargs[:int_time])), kwargs[:f_ref].+data[1:end,1], kwargs[:int_time])
        result = logpdf.(Normal.(expectation, std(observation)), observation) # 20 μs
        return LogDVal(sum(result))
    end
end
