# Implement very rudimentary fit to have something to optimize.
#fit_function(p::NamedTuple{(:a, :mu, :sigma, :A)}, freq) = p.A .* exp.(-0.5 .* (freq.-p.mu).^2. ./ p.sigma^2.) .+ p.a[1] .+ p.a[2] .* freq .+ p.a[3] .* freq.^2. + p.a[4] .* freq.^3. .+ p.a[5] .* freq.^4.
# Datapoints < 1 are not good for Normal distribution!
# Thus rescale. Doesnt matter for the endresult.
#data[!,2] *= 1e22

background(a, freq) = a[1] .+ a[2] .* freq .+ a[3] .* freq.^2. + a[4] .* freq.^3.# .+ a[5] .* freq.^4.
signal(Ps, logma, vsig, freq, kwargs) = 1e-22 .* exp.(-0.5 .* (freq.-mu(logma, kwargs[:f_ref], velo=0.0)).^2. ./ getsigma(vsig,logma,kwargs)^2.)
#mu(logma, kwarg_dict[:f_ref], velo=1e-3)

fit_function(p::NamedTuple{(:b, :logma, :vsig, :Ps)}, freq; kwargs=Dict()) = signal(p.Ps, p.logma, p.vsig, freq, kwargs) .+ background(p.b, freq)

likelihood = let d=data, f=fit_function, kwargs=kwarg_dict
    function logl(pars) 
        # Sparesly sample dataset at this stage to reduce runtime

        expectation = f(pars, data[1:end,1], kwargs=kwargs)
        # Cant work with values below 0. Should only happen far away from likelihood maximum anyways
        expectation[expectation .< 0] .= 1
        observation = data[1:end,2]
        model = Normal.(expectation, Power.(sqrt.(Counts.(expectation, kwargs[:f_ref].+data[1:end,1], kwargs[:int_time])), kwargs[:f_ref].+data[1:end,1], kwargs[:int_time]))
        result = logpdf.(model, observation)
        result .= result .- maximum(result) # In case your loglikelihood becomes positive somewhere!
        return LogDVal(sum(result))
    end
end

# v must be implemented as signal shape!
"""
    Compute 
"""
function v_distribution(logma, sigma, kwargs, shape="Normal")
    if shape == "Normal"
        return Normal(mu(logma, kwargs[:f_ref], velo=0.0),sigma)
    elseif shape == "Flat"
        start = mu(logma, kwargs[:f_ref], velo=0.0)-sigma
        stop = mu(logma, kwargs[:f_ref], velo=0.0)+sigma
        return start..stop
    else
        println("I did not understand your input for shape. Use other distribution or try uppercase.")
    end
end

#=
# This really depends on logma! This is TERRIBLE!!!!!
getsigma(1.0e-3, logma, kwarg_dict)

logma = -4.3420616002786065
logma2 = -4.3420
sigma = 1e4
f = 1.3e5
A = 7.895015824668339e-6 / 0.19789869894772844

pdf(v_distribution(logma, sigma, kwarg_dict), f)
signal(A, logma, sigma, f, kwarg_dict)
f-mu(logma, kwarg_dict[:f_ref], velo=0.0)
minimum(data[!,1])
fstart = kwarg_dict[:f_ref] + minimum(data[!,1])
mass(fstart)
mstart = log10(mass(fstart))
freq(10.0.^mstart, v=0.0)
kwarg_dict[:f_ref] - freq(10.0.^mstart, v=0.0)
mu(logma, kwarg_dict[:f_ref], velo=0.0)
=#