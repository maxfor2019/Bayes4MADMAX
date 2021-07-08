"""
    Calculates possible ma values for considered frequency range.
    This essentially is a flat prior. No possibility for anything else yet!
    Everything else comes in by some sort of folding with this top hat distribution.
    I have no idea how this could work yet.
"""
function ma_prior(data, kwargs)
    fstart = kwargs.f_ref + minimum(data[1])
    fend = kwargs.f_ref + maximum(data[1])
    mstart = mass(fstart) .* 1e6
    mend = mass(fend) .* 1e6     
    return mstart..mend
end

"""
    Rho prior. Later this should involve E/N and true DM uncertainty.
"""
function rhoa_prior(rhoa_max)
    return 0.0..rhoa_max
end

"""
Priors include
    b: background parameters without physical meaning
    ma: mass changes position of the peak, confined to range of data studied
    rhoa: flat prior right now, should include actual knowledge of DM distribution (probably gaussian?)
    E/N: MISSING!
    sig_v: DM velocity dispersion taken from 1209.0759
"""
prior = NamedTupleDist(
    b = [Normal(means[1], 5.0*abs(means[1])), Normal(means[2], 5.0*abs(means[2])), Normal(means[3], 5.0*abs(means[3])), Normal(means[4], 2.0*abs(means[4]))],
    ma = ma_prior(data, kwarg_dict),
    sig_v = Normal(model.σ_v, 6.0 * 1.0e3/c.c),
    rhoa = rhoa_prior(model.rhoa+0.15)
)

