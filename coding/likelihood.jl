# Implement very rudimentary fit to have something to optimize.
#fit_function(p::NamedTuple{(:a, :mu, :sigma, :A)}, freq) = p.A .* exp.(-0.5 .* (freq.-p.mu).^2. ./ p.sigma^2.) .+ p.a[1] .+ p.a[2] .* freq .+ p.a[3] .* freq.^2. + p.a[4] .* freq.^3. .+ p.a[5] .* freq.^4.
# Datapoints < 1 are not good for Normal distribution!
# Thus rescale. Doesnt matter for the endresult.
#data[!,2] *= 1e22

background(a, freq) = a[1] .+ a[2] .* freq .+ a[3] .* freq.^2. + a[4] .* freq.^3.# .+ a[5] .* freq.^4.
signal(rhoa, ma, vsig, freq, ex, kwargs) = signal_counts_bin((freq./kwargs[:scale_ω]) .+kwargs[:f_ref], ma*1e-6,rhoa, σ_v,ex)
# 1e-22 .* exp.(-0.5 .* (freq.-mu(logma, kwargs[:f_ref], velo=0.0)).^2. ./ getsigma(vsig,logma,kwargs)^2.)

fit_function(p::NamedTuple{(:b, :ma, :sig_v, :rhoa,)}, freq, ex::Experiment; kwargs=Dict()) = signal(p.rhoa, p.ma, p.sig_v, freq, ex, kwargs) .+ background(p.b, freq)

@time likelihood = let d=data, f=fit_function, kwargs=kwarg_dict, ex=ex
    function logl(pars) 
        #println(pars)
        expectation = f(pars, data[1:end,1] .* kwargs[:scale_ω], ex, kwargs=kwargs)
        # Cant work with values below 0. Should only happen far away from likelihood maximum anyways
        expectation[expectation .< 0] .= 1
        observation = data[1:end,2]
        # Power.(sqrt.(Counts.(expectation, kwargs[:f_ref].+data[1:end,1], kwargs[:int_time])), kwargs[:f_ref].+data[1:end,1], kwargs[:int_time])
        model = Poisson.(expectation)
        result = logpdf.(model, observation)
        result .= result .- maximum(result) # In case your loglikelihood becomes positive somewhere!
        return LogDVal(sum(result))
    end
end

