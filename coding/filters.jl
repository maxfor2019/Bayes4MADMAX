using SavitzkyGolay
using Plots

t = LinRange(-4, 4, 5000)
y = exp.(-t.^2) .+ 0.05 .* (1.0 .+ randn(length(t)))
sg = savitzky_golay(y, 101, 5)
plot(t, [y sg.y], label=["Original signal" "Filtered signal"])

function add_artificial_background(data)
    data[:,2] .+=  deepcopy(data[:,1]).^3 .* 2e-43 .- deepcopy(data[:,1]).^2 .* 1e-35 .+ deepcopy(data[:,1]) .* 1e-28 .+ 1e-20 .+ 1e-23 .* sin.(deepcopy(data[:,1])./5e4)
    return data
end