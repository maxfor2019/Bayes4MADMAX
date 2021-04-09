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
                vline!([truths[j]], color="blue", ls=:dot, subplot=i)
            end
        end
    end
    mysavefig(savefig)
    # Julia is weird with for clauses. To make everything show up, add this in the end!
    plot!()
end

function plot_fit(samples, data, kwargs; savefig=nothing)
    plot_data(data)
    testpars = mean(samples)[1]
    plot!(data[!,1], fit_function(testpars,data[!,1]; kwargs=kwargs))
    mysavefig(savefig)
    plot!()
end

function plot_data(data; savefig=nothing)
    plot(data[!,1], data[!,2], yscale=:log10)
    mysavefig(savefig)
    plot!()
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