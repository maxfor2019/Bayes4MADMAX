"""
    Calculates possible ma values for considered frequency range.
    This essentially is a flat prior. No possibility for anything else yet!
    Everything else comes in by some sort of folding with this top hat distribution.
    I have no idea how this could work yet.
"""
function ma_prior(data, kwargs)
    fstart = kwargs.f_ref + minimum(data[:,1])
    fend = kwargs.f_ref + maximum(data[:,1])
    mstart = mass(fstart) .* 1e6
    mend = mass(fend) .* 1e6     
    return mstart..mend
end

"""
    Rho prior. Later this should involve true DM uncertainty.
"""
function rhoa_prior(rhoa_max)
    return 0.0..rhoa_max
end

"""
    gaγγ prior.
"""
function gaγγ_prior(gag_range)
    return Uniform(gag_range[1],gag_range[2])
end

function log_gaγγ_prior(gag_range)
    return Uniform(gag_range[1],gag_range[2])
end


"""
Priors include
    b: background parameters without physical meaning
    ma: mass changes position of the peak, confined to range of data studied
    rhoa: flat prior right now, should include actual knowledge of DM distribution (probably gaussian?)
    E/N: MISSING!
    sig_v: DM velocity dispersion taken from 1209.0759
"""
#prior = NamedTupleDist(
#    ma = ma_prior(data, options),
#    sig_v = Normal(signal.σv,6.0),
#    rhoa = rhoa_prior(signal.ρa+0.15)
#)

function make_prior(data, signal, options; pow=:rhoa)
    ma = ma_prior(data, options)
    sig_v = Normal(signal.σ_v,6.0)
    if pow==:rhoa
        rhoa = rhoa_prior(signal.rhoa+0.15)
        return NamedTupleDist(ma=ma, sig_v=sig_v, rhoa=rhoa)
    elseif pow==:gaγγ
        gag = gaγγ_prior([0,1e-19])
        return NamedTupleDist(ma=ma, sig_v=sig_v, gag=gag)
    elseif pow==:loggaγγ
        loggag = log_gaγγ_prior([-26,-19])
        return NamedTupleDist(ma=ma, sig_v=sig_v, log_gag=loggag)
    else
        error("The specified keyword for pow does not exist! Use :rhoa, :gaγγ or :loggaγγ instead.")
    end
end

