
__precompile__()

module Bayes4MADMAX

using BAT
using DataFrames
using ForwardDiff
using ValueShapes, IntervalSets
using Distributions, SpecialFunctions, Random
using FileIO, JLD2, HDF5
using Plots, LaTeXStrings
using Statistics
using LinearAlgebra
using SavitzkyGolay

include("physics.jl")
include("custom_distributions.jl")
include("generate_data.jl")
include("backgrounds.jl")
include("forward_models.jl")
include("plotting.jl")
include("likelihood.jl")
include("prior.jl")


end # module