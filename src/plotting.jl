export plot_data, plot_fit, plot_truths

"""
    Modifies standard BAT output to nice corner plot.
    Not yet, but hopefully soon!
"""
function corner(samples, params; modify=true, truths=nothing, savefig=nothing)

    lp = length(params)
    top_plots = [i+1+(i-1)*lp:i*lp for i in 1:lp-1]
    bottom_plots = [i*lp+1:i*(lp+1) for i in 1:lp-1]
    diag_plots = [i*(lp+1)+1:i*(lp+1)+1 for i in 0:lp-1]

    plot(
        samples, vsel=collect(params), bins=50, globalmode=true
    )
    if modify == true
        # To modify the plot, you can just overplot. Do something like the following:
        for range in top_plots
            for i in range
                plot!(framestyle=:none, showaxis=false, xlabel="", ylabel="", subplot=i)
            end
        end
        for range in bottom_plots
            for i in range[2:end]
                plot!(ylabel="", yformatter=_->"", subplot=i)
            end
        end
        for range in bottom_plots[1:end-1]
            for i in range
                plot!(xlabel="", xformatter=_->"", subplot=i)
            end
        end
        for range in diag_plots[1:end-1]
            for i in range
                plot!(xlabel="", ylabel="", xformatter=_->"", yformatter=_->"", subplot=i)
            end
        end
        for range in diag_plots[end-1:end]
            for i in range
                plot!(ylabel="", yformatter=_->"", subplot=i)
            end
        end
    else
        nothing
    end
    if truths === nothing
        nothing
    else
        length(truths) == lp || throw(DimensionMismatch("length of truths does not match number of parameters plotted"))
        j = 0
        for range in diag_plots
            j += 1
            for i in range
                vline!([truths[j]], color="blue", ls=:dot, lw=2, subplot=i)
            end
        end
    end
    ma_label = L"m_a [\mu\textrm{eV}]"
    rhoa_label = L"\rho_a [\textrm{GeV}/\textrm{cm}^3]"
    sigv_label = L"\sigma_v [c]"
    ma_range = 45.493660:16e-6:45.493676
    sigv_range = 6.5e-4:5e-5:8e-4
    plot!(xlabel=ma_label, ylabel=L"\textrm{P}(m_a)", xticks=ma_range, subplot=1)
    plot!(xlabel=ma_label, ylabel=sigv_label, xticks=ma_range, subplot=2)
    plot!(xlabel=ma_label, ylabel=rhoa_label, xticks=ma_range, subplot=3)
    plot!(xlabel=ma_label, ylabel=sigv_label, xticks=ma_range, subplot=4)
    plot!(xlabel=sigv_label, ylabel=L"\textrm{P}(\sigma_v)", xticks=sigv_range, subplot=5)
    plot!(xlabel=sigv_label, ylabel=rhoa_label, subplot=6)
    plot!(xlabel=ma_label, ylabel=rhoa_label, xticks=ma_range, subplot=7)
    plot!(xlabel=sigv_label, ylabel=rhoa_label, subplot=8)
    plot!(xlabel=rhoa_label, ylabel=L"\textrm{P}(\rho_a)", subplot=9)

    mysavefig(savefig)
    # Julia is weird with for clauses. To make everything show up, add this in the end!
    plot!()
end

function plot_fit(samples, data, ex; scaling=1.0, truths=nothing, savefig=nothing, kwargs...)
    #power_o = scaling .* Power(data[:,2], data[:,1] .+ kwargs[:f_ref], ex.t_int)
    #data_tmp = (data[1], power_o)
    plot_data(data; key=:pownoB, label="mock data", kwargs...)
    testpars = mode(samples) # should be mode(samples) but that seems to not work with BATv3.0.0
    pow_p = fit_function(testpars,data[!, :freq],ex)
    #pow_p = Power.(counts_p, data[:,1] .+ kwargs.f_ref, ex.t_int)
    println(testpars)
    plot!(data[!, :freq], scaling .* pow_p, label="fit",legend=:bottomleft)
    if truths != nothing
        plot!(data[!, :freq], fit_function(truths,data[!, :freq],ex); label="true signal")
    end
    xlabel!(L"f - f_{\textrm{ref}}")
    ylabel!(L"\textrm{Power}\;[10^{-23} \textrm{W}]")
    mysavefig(savefig)
    plot!()
end

function plot_truths(data, truths::Theory, ex::Experiment; savefig=nothing, kwargs...)
    plot_data(data,; key=:pownoB, label="data", kwargs...)
    plot!(data[!, :freq], fit_function(truths,data[!, :freq],ex); label="true signal")
    ylims!((minimum(data[!, :pownoB]),maximum(data[!, :pownoB]*1.2)))
    mysavefig(savefig)
    xlabel!("\$ f - f_{ref} \$")
    ylabel!("Power")
    plot!()
end

function plot_data(data; key=:pow, label="", savefig=nothing, overplot=false, kwargs...)
    if overplot == false
        plot(data[!, :freq], data[!, key]; label=label, kwargs...)
    elseif overplot == true
        plot!(data[!, :freq], data[!, key]; label=label, kwargs...)
    end
    ylims!((minimum(data[!, key]),maximum(data[!, key])))
    xlabel!("\$ f - f_{ref} \$")
    ylabel!("Power / bin")
    #xlims!((3.3e7,3.5e7))
    #ylims!((2200,2300))
    #mysavefig(savefig)
    #plot!()
end

function plot_exclusion(mass_bins, exclusion, fracs; signal=nothing)
    plot()
    plot( mass_bins, exclusion[1].+9.0, fillrange=fill(-13.0,length(exclusion)), fillalpha=0.35, label=string(fracs[1]))
    if size(exclusion)[1] > 1
        for i in 2:size(exclusion)[1]
            plot!(mass_bins, exclusion[i].+9.0, fillrange=fill(-13.0,length(exclusion)), fillalpha=0.35, label=string(fracs[i]))
        end
    end

    # E/N = 0.97 for KSVZ and 0.36 for DFSZ according to Haystac Palken et al
    plot!(mass_bins, log10.(1e9*gaγγ.(fa.(mass_bins*1e-6),8.0/3.0)), label="E/N = 8/3", linewidth=3, ls=:dash)
    if signal !== nothing
        scatter!([signal.ma], [log10.(1e9*gaγγ.(fa.(signal.ma*1e-6),signal.EoverN))], label="signal")
    end
    xlabel!(L"m_a\;[\mu \textrm{eV}]")
    ylabel!(L"\log (g_{a\gamma\gamma})\;[GeV^{-1}]")
    plot!()
    #mysavefig(savefig)
end

function plot_exclusion2(mass_bins, exclusion, exclusion2, fracs, fracs2; signal=nothing)
    plot()
    plot(mass_bins, exclusion[1].+9.0, fillrange=fill(-13.0,length(exclusion)), fillalpha=0.35, label="prior -28 "*string(fracs2[1]),c=:red, legend=:bottomright,ls=:dot)
    plot!(mass_bins, exclusion2[1].+9.0, fillrange=fill(-13.0,length(exclusion2)), fillalpha=0.0, label="prior -26 "*string(fracs2[1]),c=:red, lw=2)
    cs = [:blue,:green]
    if size(exclusion)[1] > 1
        for i in 2:size(exclusion)[1]
            plot!(mass_bins, exclusion[i].+9.0, fillrange=fill(-13.0,length(exclusion)), fillalpha=0.35, label="prior -28 "*string(fracs[i]),c=cs[i-1],ls=:dot)
            plot!(mass_bins, exclusion2[i].+9.0, fillrange=fill(-13.0,length(exclusion2)), fillalpha=0.0, label="prior -26 "*string(fracs[i]),c=cs[i-1], lw=2)
        end
    end

    # E/N = 0.97 for KSVZ and 0.36 for DFSZ according to Haystac Palken et al
    #plot!(mass_bins, log10.(1e9*gaγγ.(fa.(mass_bins*1e-6),5.0/3.0)), label="E/N = 5/3", linewidth=3, ls=:dash, c=:black)
    #if signal !== nothing
    #    scatter!([signal.ma], [log10.(1e9*gaγγ.(fa.(signal.ma*1e-6),signal.EoverN))], label="signal")
    #end
    #xlabel!(L"m_a\;[\mu \textrm{eV}]")
    #ylabel!(L"\log (g_{a\gamma\gamma})\;[GeV^{-1}]")
    plot!()
    #mysavefig(savefig)
end



function mysavefig(name; index=nothing, path="data/plots/", form=".pdf")
    if name !== nothing
        if index !== nothing
            save = path*name*string(index)*form
        else
            save = path*name*form
        end
        savefig(save)
    end
end