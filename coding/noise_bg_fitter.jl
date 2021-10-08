using BAT
using Random, LinearAlgebra
using Distributions
using ValueShapes
using Plots
using FFTW
import ForwardDiff
using DelimitedFiles
using PDMats
using MGVI
using Optim
using HDF5
# using Zygote: @adjoint, @ignore, gradient
using Zygote
include("forward_models.jl")

datafiles = ["Het3_10K_0-15z_20170308_191203_S0"*string(i)*".smp" for i in 1:4]
data = combine_data(datafiles)


vals = data[10001:11000,2]
vals = (vals .- mean(data[:,2]))./std(data[:,2])
dists = (data[:,1] - circshift(data[:,1],1))

dims = (length(vals),)
distances = (1/length(data[:,1]),)
data = vals
harmonic_pad_distances = 1 ./ (2 .* dims .* distances)

x = collect(distances[1]:distances[1]:distances[1] * dims[1];)
x_pad = collect(distances[1]:distances[1]:distances[1] * dims[1]*2;)

ht = FFTW.plan_r2r(zeros(length(x_pad)), FFTW.DHT);
D = dist_array(2 .* dims, harmonic_pad_distances)


my_amplitude = let D = D
    parameters -> amplitude_forward_model(parameters.offset, parameters.slope, 
                                            parameters.zero_mode, D)
end

ht = FFTW.plan_r2r(zeros(length(x_pad)), FFTW.DHT);

my_ht = let n_x_pad = length(x_pad), harmonic_pad_distances = harmonic_pad_distances, ht=ht
    input -> harmonic_transform(input, n_x_pad, harmonic_pad_distances, ht)
end

my_gp = let nbin = length(data), harmonic_transform = my_ht, amplitude = my_amplitude
        parameters -> gp_forward_model(parameters.ξ, amplitude(parameters), my_ht, nbin)
end

rng = Random.default_rng()

prior = NamedTupleDist(
    ξ = BAT.StandardMvNormal(length(D)), 
    offset = Uniform(-3, 3),
    zero_mode = Uniform(0.001, 1),
    slope = Uniform(-8, -2),
    n = Uniform(0.00001,0.005)
)

truth = rand(prior)

data = vals

bwd_trafo = BAT.DistributionTransform(Normal, prior)
standard_truth = bwd_trafo(truth)

fwd_trafo = inv(bwd_trafo)

standard_prior = BAT.StandardMvNormal(length(standard_truth))

model = let prior_trafo = fwd_trafo, forward_model = my_gp
    function (stand_pars)
        parameters = prior_trafo(stand_pars)[]
        Product(Normal.(forward_model(parameters), parameters.n))
    end
end


starting_point = 0.1*bwd_trafo(rand(prior))
next_iteration = mgvi_kl_optimize_step(rng,
                                        model, data,
                                        starting_point;                                     
                                        jacobian_func=FwdRevADJacobianFunc,
                                        residual_sampler=ImplicitResidualSampler,
                                        num_residuals=5,
                                        optim_options=Optim.Options(iterations=5, show_trace=true),
                                        residual_sampler_options=(;cg_params=(;maxiter=100)))


N_samps = 5
for i in 1:10
    global next_iteration = mgvi_kl_optimize_step(rng,
                                                  model, data,
                                                  next_iteration.result;
                                                  jacobian_func=FwdRevADJacobianFunc,
                                                  residual_sampler=ImplicitResidualSampler,
                                                  num_residuals=N_samps,
                                                  optim_options=Optim.Options(iterations=20, show_trace=true),
                                                  residual_sampler_options=(;cg_params=(;maxiter=100)))
end

########## SAVE RESULTS ##########
sample_dict = Dict()
for key in keys(prior)
    sample_dict[key] = Vector{Float64}[]
end

for i in 1:size(next_iteration.samples)[end]
    samp = fwd_trafo(next_iteration.samples[:,i])[1]
    for key in keys(sample_dict)
        s =  samp[key]
        if  ~isa(s,Vector)
            s = [s]
        end
        push!(sample_dict[key], s)
    end
end

for key in keys(sample_dict)
    h5open("coding/background_fit.h5", "cw") do file
        s = hcat(sample_dict[key]...)
        write(file,"$(String(key))", s)
    end
end

res = fwd_trafo(next_iteration.result)[1]
plot(data,label="data",seriestype = :scatter)
for i in 1:N_samps*2
    plot!(my_gp(fwd_trafo(next_iteration.samples[:,i])),color="black",alpha=0.3)
end
plot!(my_gp(res),label="mean", color="red")
# # savefig("full_bg.png")

# for i in 1:(length(data)/10):length(data)
    # ii = Int64(round(i))
    # plot(data[ii:ii+100],label="data",seriestype = :scatter)
    # for i in 1:N_samps*2
    #     plot!(my_gp(fwd_trafo(next_iteration.samples[:,i]))[ii:ii+100],color="black",alpha=0.3)
    # end
    # plot!(my_gp(res)[ii:ii+100],label="mean", color="red")
#     # savefig("bg_cut_$ii.png")
# end

