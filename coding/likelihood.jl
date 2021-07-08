# Implement very rudimentary fit to have something to optimize.
# Datapoints < 1 are not good for Normal distribution!
# Thus rescale. Doesnt matter for the endresult.

background(a, freq) = a[1] .+ a[2] .* freq .+ a[3] .* freq.^2. + a[4] .* freq.^3.# .+ a[5] .* freq.^4. # 3 μs

signal(rhoa, ma, vsig, freq, ex, kwargs) = signal_counts_bin((freq./kwargs.scale_ω) .+kwargs.f_ref, ma*1e-6,rhoa, σ_v,ex) # 80 μs

fit_function(p::NamedTuple{(:b, :ma, :sig_v, :rhoa,)}, freq, ex::Experiment; kwargs=Dict()) = signal(p.rhoa, p.ma, p.sig_v, freq, ex, kwargs) .+ background(p.b, freq)

likelihood = let freqs=collect(data[1]), observation=data[2], f=fit_function, kwargs=kwarg_dict, ex=ex
    function logl(pars) 
        #println(pars)
        expectation = f(pars, freqs .* kwargs.scale_ω, ex, kwargs=kwargs) # 80 μs
        # Cant work with values below 0. Should only happen far away from likelihood maximum anyways
        expectation[expectation .<= 0] .= eps(eltype(expectation))
        # Power.(sqrt.(Counts.(expectation, kwargs[:f_ref].+data[1:end,1], kwargs[:int_time])), kwargs[:f_ref].+data[1:end,1], kwargs[:int_time])
        result = logpdf.(Poisson.(expectation), observation) # 20 μs
        return LogDVal(sum(result))
    end
end
