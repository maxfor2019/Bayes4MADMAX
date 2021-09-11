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
# using Zygote: @adjoint, @ignore, gradient
using Zygote
data = readdlm("./data/Fake_Axion_Data/Data_Set_1/Test00Osc01_17-01-24_0915.dat")#, '\t', Float32, '\n')


vals = data[11001:11100,2]
vals = (vals .- mean(vals))./std(vals)
dists = (data[:,1] - circshift(data[:,1],1))
# vals = vals[12001:12100]



plot(data[:,1],data[:,2], ylims=(minimum(vals),maximum(vals)))
# data = data[1:100]
data = vals
# distances = (mean(dists[2:end]),)
# distances = (1/100,)
dims = (length(data[:,1]),)
distances = (1/dims[1],)

harmonic_pad_distances = 1 ./ (2 .* dims .* distances)



function map_idx(idx::Real, idx_range::AbstractUnitRange{<:Integer})
    i = idx - minimum(idx_range)
    n = length(eachindex(idx_range))
    n_2 = n >> 1
    ifelse(i <= n_2, i, i - n)
end

function dist_k(idx::CartesianIndex, ax::NTuple{N,<:AbstractUnitRange{<:Integer}}, harmonic_distances::NTuple{N,<:Real}) where N
    mapped_idx = map(map_idx, Tuple(idx), ax)
    norm(map(*, mapped_idx, harmonic_distances))
end

function dist_array(dims::NTuple{N,<:Real}, harmonic_distances::NTuple{N,<:Real}) where N
    cart_idxs = CartesianIndices(map(Base.OneTo, dims))
    dist_k.(cart_idxs, Ref(axes(cart_idxs)), Ref(harmonic_distances))
end


function amplitude_spectrum(d::Real, zero_mode_dist::Normal, slope::Real, offset::Real)
    # R = float(promote_type(typeof(d), typeof(zero_mode), typeof(slope), typeof(offset)))
    # d ≈ 0 : R(zero_mode), R(exp(offset + slope * log(d)))
    ifelse(d ≈ 0, promote(std(zero_mode_dist), exp(offset + slope * log(d)))...)
end


function my_mask(x, n_data::Integer)
    #  y = x[mask]
     y = x[1:n_data] #FIXME
     y
end

# function adjoint_mask(x)
#     y = zeros(size(mask))
#     y[mask] = x
#     return y
# end

# @adjoint my_mask(x) = my_mask(x), y -> (adjoint_mask(y),)




# function adjoint_mask(dp::Vector{ForwardDiff.Dual{T, V, N}}) where {T,V,N}
#     val_res = adjoint_mask(ForwardDiff.value.(dp))
#     psize = size(ForwardDiff.partials(dp[1]), 1)
#     ps = x -> ForwardDiff.partials.(dp, x)
#     val_ps = map((x -> adjoint_mask(ps(x))), 1:psize)
#     ForwardDiff.Dual{T}.(val_res, val_ps...)
# end


function amplitude_forward_model(parameters)
    corr = (parameters.fluctuations ./(1 .+ (D./0.1).^(-parameters.slope)))[2:end]
    corr = vcat(parameters.zero_mode, corr)
    # corr[1] = parameters.zero_mode
    corr.^0.5 
end


function gp_forward_model(parameters::NamedTuple, n_data::Integer, n_x_pad::Integer, harmonic_pad_distances::Tuple, ht::FFTW.r2rFFTWPlan)
    # amplitude = amplitude_forward_model(parameters)
    amplitude = 1.
    harmonic_gp = amplitude .* parameters.ξ
    gp = apply_ht(ht, harmonic_gp) * (harmonic_pad_distances[1] / sqrt(n_x_pad))
    my_mask(gp, n_data)
end

function apply_ht(ht::FFTW.r2rFFTWPlan, dp::Vector{Float64})
    ht * dp
end


function apply_ht(ht::FFTW.r2rFFTWPlan, dp::Vector{ForwardDiff.Dual{T, V, N}}) where {T,V,N}
    val_res = ht *  ForwardDiff.value.(dp)
    psize = size(ForwardDiff.partials(dp[1]), 1)
    ps = x -> ForwardDiff.partials.(dp, x)
    val_ps = map((x -> ht*ps(x)), 1:psize)
    ForwardDiff.Dual{T}.(val_res, val_ps...)
end


rng = Random.default_rng()

# trafo_dht = FFTW.plan_r2r(zeros(2 .* dims), FFTW.DHT)
x = collect(distances[1]:distances[1]:distances[1] * dims[1];)
x_pad = collect(distances[1]:distances[1]:distances[1] * dims[1]*2;)

ht = FFTW.plan_r2r(zeros(length(x_pad)), FFTW.DHT);
D = dist_array(2 .* dims, harmonic_pad_distances)


mask = collect(1:length(x_pad))
mask = mask .<= length(mask)÷2

prior = NamedTupleDist(
    ξ = BAT.StandardMvNormal(length(D)), 
    # fluctuations = Uniform(0.1, 90),
    # zero_mode = Uniform(0.1, 200),
    # slope = Uniform(-6, -2),
    n = Uniform(1e-4,10)
)

truth = rand(prior)
# data = gp_forward_model(truth)[mask]
data = vals
# likelihood = let data = data, f = gp_forward_model
#     params -> begin
#         residual = data .- f(params)[mask]
#         noise_weigted_residual = residual ./ 1.0e-4  #FIXME weights
#         llhd = - 0.5 .* sum(residual .* noise_weigted_residual)
#         # llhd = logpdf(MvNormal(μ,N),data)
#         return LogDVal(llhd)
#     end
# end

bwd_trafo = BAT.DistributionTransform(Normal, prior)
standard_truth = bwd_trafo(truth)

fwd_trafo = inv(bwd_trafo)

standard_prior = BAT.StandardMvNormal(length(standard_truth))

model = let fwd_trafo = fwd_trafo, n_data = length(data), n_x_pad = length(x_pad), harmonic_pad_distances = harmonic_pad_distances, ht = ht
    function (stand_pars)
        parameters = fwd_trafo(stand_pars)[]
        Product(Normal.(gp_forward_model(parameters, n_data, n_x_pad, harmonic_pad_distances, ht), parameters.n))
    end
end


# sampler =  MCMCSampling(mcalg = HamiltonianMC(), nsteps = 10^4, nchains = 4)
# r = bat_sample(posterior, sampler)




starting_point = bwd_trafo(rand(prior))
first_iteration = mgvi_kl_optimize_step(rng,
                                        model, data,
                                        starting_point;
                                        # jacobian_func=FullJacobianFunc,   
                                        # jacobian_func=FwdDerJacobianFunc,                                        
                                        jacobian_func=FwdRevADJacobianFunc,
                                        residual_sampler=ImplicitResidualSampler,
                                        optim_options=Optim.Options(iterations=10, show_trace=true),
                                        residual_sampler_options=(;cg_params=(;maxiter=10)))


likelihood = x -> LogDVal(MGVI.posterior_loglike(model, x, data))
posterior = PosteriorDensity(likelihood, standard_prior)

findmode_result = bat_findmode(posterior,MaxDensityLBFGS()).result

max_posterior = Optim.optimize(x -> -MGVI.posterior_loglike(model, x, data),
                 starting_point, LBFGS(), Optim.Options(show_trace=false, g_tol=1E-10, iterations=300));

plot(data)
plot!(gp_forward_model(fwd_trafo(Optim.minimizer(max_posterior))[1], length(data), length(x_pad), harmonic_pad_distances, ht))
plot!(gp_forward_model(fwd_trafo(findmode_result)[1], length(data), length(x_pad), harmonic_pad_distances, ht))
